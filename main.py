import os
import re
import json
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from flask import Flask, request
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

TEHRAN_TZ = ZoneInfo("Asia/Tehran")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def require_env(name):
    value = os.environ.get(name)
    if not value:
        logging.error(f"❌ متغیر محیطی {name} تنظیم نشده یا خالیه!")
        raise SystemExit(1)
    return value


def optional_env(name):
    return os.environ.get(name)


TELEGRAM_TOKEN = require_env("TELEGRAM_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

SUPABASE_URL = require_env("SUPABASE_URL").rstrip("/")
SUPABASE_KEY = require_env("SUPABASE_SERVICE_KEY")
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

GEMINI_API_KEY = require_env("GEMINI_API_KEY")
REMINDER_SECRET = require_env("REMINDER_SECRET")

# اختیاری: اگر ست نشده باشد، مدل سوم به‌سادگی رد می‌شود (بدون کرش کردن کل بات)
GROQ_API_KEY = optional_env("GROQ_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# از alias استفاده می‌کنیم نه اسم دقیق نسخه، تا وقتی گوگل یک مدل را retire کرد
# مجبور به ویرایش دستی کد نشویم (همان مشکلی که با gemini-2.0-flash پیش آمد).
PRIMARY_MODEL = "gemini-flash-latest"
SECONDARY_MODEL = "gemini-flash-lite-latest"
GROQ_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Circuit breaker ساده (درون‌حافظه‌ای)
# وقتی مدلی خطای quota/rate-limit بدهد، برای مدتی مستقیم سراغش نمی‌رویم
# تا هر پیام کاربر مجبور به «تلاش بی‌فایده روی مدل خراب» نشود.
# ---------------------------------------------------------------------------

CIRCUIT_COOLDOWN_SECONDS = 10 * 60  # ۱۰ دقیقه
_circuit_state = {}  # {model_name: unblock_timestamp}


def circuit_is_open(model_name):
    unblock_at = _circuit_state.get(model_name)
    if unblock_at is None:
        return False
    if time.time() >= unblock_at:
        _circuit_state.pop(model_name, None)
        return False
    return True


def circuit_trip(model_name):
    _circuit_state[model_name] = time.time() + CIRCUIT_COOLDOWN_SECONDS
    logging.warning(f"⚡ Circuit breaker فعال شد برای {model_name} تا {CIRCUIT_COOLDOWN_SECONDS}s")


FALLBACK_KEYWORDS = [
    "429", "quota", "rate limit", "resource_exhausted", "resource exhausted",
    "503", "500", "unavailable", "timeout", "overloaded", "404", "not found",
]


def should_fallback(error):
    error_text = str(error).lower()
    return any(keyword in error_text for keyword in FALLBACK_KEYWORDS)


# ---------------------------------------------------------------------------
# پارسر مقاوم JSON
# مدل‌های مختلف (به‌خصوص Groq) گاهی دور JSON کدبلاک مارک‌داون یا متن اضافه می‌گذارند.
# ---------------------------------------------------------------------------

def extract_json(raw_text):
    text = raw_text.strip()

    # حذف fence های مارک‌داون مثل ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # آخرین تلاش: پیدا کردن اولین { و آخرین } در متن
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"نتوانستم JSON معتبر از پاسخ مدل استخراج کنم: {raw_text[:200]}")


# ---------------------------------------------------------------------------
# فراخوانی مدل‌ها
# ---------------------------------------------------------------------------

def generate_with_gemini(model_name, contents, system_instruction):
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
        ),
    )
    return response.text


def generate_with_groq(contents, system_instruction):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY تنظیم نشده - مدل سوم در دسترس نیست")

    messages = [{"role": "system", "content": system_instruction}]
    for item in contents:
        messages.append({"role": item["role"], "content": item["text"]})

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def generate_ai_response(contents, system_instruction):
    """
    زنجیره fallback:
    1. Gemini Flash (alias جدید)
    2. Gemini Flash-Lite
    3. Groq (اگر تنظیم شده باشد)
    هر مدلی که به‌تازگی quota/خطا داده باشد (circuit باز) رد می‌شود.
    """
    chain = [
        ("gemini_primary", PRIMARY_MODEL, lambda: generate_with_gemini(PRIMARY_MODEL, contents, system_instruction)),
        ("gemini_secondary", SECONDARY_MODEL, lambda: generate_with_gemini(SECONDARY_MODEL, contents, system_instruction)),
        ("groq", GROQ_MODEL, lambda: generate_with_groq(contents, system_instruction)),
    ]

    last_error = None
    for circuit_key, model_name, call in chain:
        if circuit_is_open(circuit_key):
            logging.info(f"⏭️ رد شدن از {model_name} (circuit هنوز باز است)")
            continue
        try:
            raw_text = call()
            parsed = extract_json(raw_text)
            return parsed
        except Exception as e:
            last_error = e
            logging.warning(f"❌ {model_name} شکست خورد: {e}")
            if should_fallback(e):
                circuit_trip(circuit_key)
            continue

    raise RuntimeError(f"هر سه مدل شکست خوردند. آخرین خطا: {last_error}")


# ---------------------------------------------------------------------------
# Supabase - تاریخچه مکالمه
# ---------------------------------------------------------------------------

def save_message(chat_id, role, text):
    requests.post(
        f"{SUPABASE_URL}/rest/v1/conversations",
        headers=SUPABASE_HEADERS,
        json={"chat_id": chat_id, "role": role, "content": text},
        timeout=15,
    )


def get_history(chat_id, limit=15):
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/conversations",
        headers=SUPABASE_HEADERS,
        params={
            "chat_id": f"eq.{chat_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        },
        timeout=15,
    )
    resp.raise_for_status()
    rows = resp.json()
    return list(reversed(rows))


# ---------------------------------------------------------------------------
# Supabase - حافظه بلندمدت (فکت‌ها)
# ---------------------------------------------------------------------------

def save_fact(chat_id, fact_text):
    requests.post(
        f"{SUPABASE_URL}/rest/v1/memory_facts",
        headers=SUPABASE_HEADERS,
        json={"chat_id": chat_id, "fact": fact_text},
        timeout=15,
    )


def get_facts(chat_id):
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/memory_facts",
        headers=SUPABASE_HEADERS,
        params={"chat_id": f"eq.{chat_id}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Supabase - مدیریت کارها
# ---------------------------------------------------------------------------

def create_task(chat_id, title, date_str, time_str):
    requests.post(
        f"{SUPABASE_URL}/rest/v1/tasks",
        headers=SUPABASE_HEADERS,
        json={
            "chat_id": chat_id,
            "title": title,
            "due_date": date_str,
            "due_time": time_str,
            "status": "pending",
        },
        timeout=15,
    )


def update_task_status(task_id, status):
    requests.patch(
        f"{SUPABASE_URL}/rest/v1/tasks",
        headers=SUPABASE_HEADERS,
        params={"id": f"eq.{task_id}"},
        json={"status": status},
        timeout=15,
    )


def delete_task(task_id):
    requests.delete(
        f"{SUPABASE_URL}/rest/v1/tasks",
        headers=SUPABASE_HEADERS,
        params={"id": f"eq.{task_id}"},
        timeout=15,
    )


def get_pending_tasks(chat_id):
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/tasks",
        headers=SUPABASE_HEADERS,
        params={"chat_id": f"eq.{chat_id}", "status": "eq.pending"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_due_tasks(now_iso):
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/tasks",
        headers=SUPABASE_HEADERS,
        params={
            "status": "eq.pending",
            "reminded": "eq.false",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def mark_task_reminded(task_id):
    requests.patch(
        f"{SUPABASE_URL}/rest/v1/tasks",
        headers=SUPABASE_HEADERS,
        params={"id": f"eq.{task_id}"},
        json={"reminded": True},
        timeout=15,
    )


# ---------------------------------------------------------------------------
# Idempotency - جلوگیری از پردازش دوباره‌ی یک آپدیت تلگرام
# (اگر Flask دیر جواب بدهد، تلگرام همان پیام را دوباره می‌فرستد)
# ---------------------------------------------------------------------------

def already_processed(update_id):
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/processed_updates",
        headers=SUPABASE_HEADERS,
        params={"update_id": f"eq.{update_id}"},
        timeout=15,
    )
    resp.raise_for_status()
    return len(resp.json()) > 0


def mark_processed(update_id):
    requests.post(
        f"{SUPABASE_URL}/rest/v1/processed_updates",
        headers=SUPABASE_HEADERS,
        json={"update_id": update_id},
        timeout=15,
    )


# ---------------------------------------------------------------------------
# تلگرام
# ---------------------------------------------------------------------------

def send_telegram_message(chat_id, text):
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=15,
    )


# ---------------------------------------------------------------------------
# System instruction و منطق اصلی پردازش پیام
# ---------------------------------------------------------------------------

def build_system_instruction(chat_id, now_tehran):
    facts = get_facts(chat_id)
    facts_text = "\n".join(f"- {f['fact']}" for f in facts) if facts else "(هنوز فکتی ثبت نشده)"

    pending = get_pending_tasks(chat_id)
    tasks_text = "\n".join(
        f"- id={t['id']} | {t['title']} | {t.get('due_date')} {t.get('due_time')}"
        for t in pending
    ) if pending else "(کار در انتظاری نیست)"

    return f"""
تو یک دستیار شخصی فارسی‌زبان هستی که داخل تلگرام کار می‌کنی.
تاریخ و ساعت فعلی به وقت تهران: {now_tehran.strftime('%Y-%m-%d %H:%M')}

فکت‌های شناخته‌شده درباره‌ی کاربر:
{facts_text}

کارهای در انتظار کاربر:
{tasks_text}

همیشه فقط و فقط یک JSON معتبر با این ساختار برگردان (بدون هیچ متن اضافه، بدون کدبلاک مارک‌داون):

{{
  "reply": "متن جواب به فارسی",
  "suggestions": ["پیشنهاد اول", "پیشنهاد دوم"],
  "new_facts": ["فکت مهم جدید در صورت وجود"],
  "task_action": {{
    "action": "create | complete | delete | none",
    "task_id": "فقط برای complete/delete",
    "title": "فقط برای create",
    "date": "YYYY-MM-DD فقط برای create",
    "time": "HH:MM فقط برای create"
  }}
}}

اگر کاربر تاریخ نسبی فارسی گفت (مثل «فردا»، «پس‌فردا»، «سه‌شنبه آینده»)، آن را بر اساس تاریخ امروز که بالا داده شد
به تاریخ دقیق میلادی YYYY-MM-DD تبدیل کن. اگر کاری برای انجام دادن نیست، task_action.action را "none" بگذار.
"""


def handle_user_message(chat_id, user_text):
    now_tehran = datetime.now(TEHRAN_TZ)
    system_instruction = build_system_instruction(chat_id, now_tehran)

    history = get_history(chat_id)
    contents = [{"role": row["role"], "text": row["content"]} for row in history]
    contents.append({"role": "user", "text": user_text})

    save_message(chat_id, "user", user_text)

    try:
        result = generate_ai_response(contents, system_instruction)
    except Exception as e:
        logging.error(f"🔥 هر سه مدل شکست خوردند: {e}")
        send_telegram_message(chat_id, "الان همه‌ی سرویس‌های هوش مصنوعی مشکل دارن، چند دقیقه دیگه دوباره امتحان کن.")
        return

    reply_text = result.get("reply", "")
    suggestions = result.get("suggestions") or []
    new_facts = result.get("new_facts") or []
    task_action = result.get("task_action") or {"action": "none"}

    save_message(chat_id, "model", reply_text)

    for fact in new_facts:
        save_fact(chat_id, fact)

    action = task_action.get("action", "none")
    if action == "create":
        create_task(
            chat_id,
            task_action.get("title", ""),
            task_action.get("date"),
            task_action.get("time"),
        )
    elif action == "complete":
        task_id = task_action.get("task_id")
        if task_id:
            update_task_status(task_id, "done")
    elif action == "delete":
        task_id = task_action.get("task_id")
        if task_id:
            delete_task(task_id)

    final_text = reply_text
    if suggestions:
        final_text += "\n\n" + "\n".join(f"• {s}" for s in suggestions)

    send_telegram_message(chat_id, final_text)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json(force=True, silent=True) or {}
    update_id = update.get("update_id")

    if update_id is not None:
        try:
            if already_processed(update_id):
                logging.info(f"↩️ آپدیت {update_id} قبلاً پردازش شده، رد می‌شود")
                return "ok", 200
            mark_processed(update_id)
        except Exception as e:
            logging.warning(f"⚠️ خطا در چک idempotency: {e}")

    message = update.get("message") or {}
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text")

    if not chat_id or not text:
        return "ok", 200

    if text.strip() == "/start":
        send_telegram_message(chat_id, "سلام! من دستیار شخصی توام. هر کاری، یادآوری یا سوالی داری بگو.")
        return "ok", 200

    handle_user_message(chat_id, text)
    return "ok", 200


@app.route("/check-reminders", methods=["GET", "POST"])
def check_reminders():
    secret = request.args.get("secret")
    if secret != REMINDER_SECRET:
        return "forbidden", 403

    now_tehran = datetime.now(TEHRAN_TZ)
    due_tasks = get_due_tasks(now_tehran.isoformat())

    sent_count = 0
    for task in due_tasks:
        due_date = task.get("due_date")
        due_time = task.get("due_time")
        if not due_date or not due_time:
            continue

        try:
            due_dt = datetime.fromisoformat(f"{due_date}T{due_time}").replace(tzinfo=TEHRAN_TZ)
        except ValueError:
            continue

        if due_dt <= now_tehran:
            send_telegram_message(task["chat_id"], f"⏰ یادآوری: {task['title']}")
            mark_task_reminded(task["id"])
            sent_count += 1

    return {"sent": sent_count}, 200


@app.route("/", methods=["GET"])
def health():
    return "bot is alive", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
