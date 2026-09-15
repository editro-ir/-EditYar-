import os
import json
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, request
from google import genai
from google.genai import types
from google.genai.errors import ServerError

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

TEHRAN_TZ = ZoneInfo("Asia/Tehran")


def require_env(name):
    value = os.environ.get(name)
    if not value:
        logging.error(f"❌ متغیر محیطی {name} تنظیم نشده یا خالیه! برو تو Render > Environment و اضافه‌ش کن.")
        raise SystemExit(1)
    return value


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

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

BASE_SYSTEM_INSTRUCTION = """تو دستیار شخصی علی مسجدی هستی. هر جا لازم بود خودت رو معرفی کنی، بگو «من دستیار شخصی علی مسجدی هستم».

همیشه فقط و فقط یک شیء JSON با همین ساختار برگردون، بدون هیچ متن اضافه و بدون بک‌تیک:
{
  "reply": "متن جواب تو به زبان فارسی",
  "suggestions": ["پیشنهاد کوتاه اول", "پیشنهاد کوتاه دوم"],
  "new_facts": ["فکت مهم جدید"],
  "task_action": {"action": "create یا complete یا delete یا none", "title": "عنوان کار", "date": "YYYY-MM-DD یا null", "time": "HH:MM یا null"}
}

راهنمای هر بخش:
- suggestions: حداکثر ۲ پیشنهاد کوتاه (حداکثر ۶-۷ کلمه) برای جملهٔ بعدی که کاربر ممکنه بخواد بفرسته.
- new_facts: اگه کاربر یه اطلاعات ماندگار مهم دربارهٔ خودش گفت (اسم، علاقه، عادت، شغل...) که ارزش داره برای همیشه یادت بمونه، به‌صورت جملهٔ کوتاه بنویس. اگه چیز جدیدی نبود، [] بذار. تکراری ننویس.
- task_action: اگه کاربر خواست کاری/یادآوری/جلسه‌ای رو ثبت کنه → action=create با title و date/time (اگه ساعت یا تاریخ نگفت، همون null بذار). اگه گفت کاری رو انجام داده/تمومش کرده → action=complete با title (باید با یکی از کارهای فعلی لیست‌شده مطابقت داشته باشه). اگه خواست کاری رو حذف کنه → action=delete با title. در غیر این صورت action=none.
- تاریخ‌ها رو همیشه به فرم میلادی YYYY-MM-DD و ساعت رو به فرم ۲۴ساعته HH:MM بنویس، حتی اگه کاربر «فردا» یا «سه‌شنبه» گفته باشه (بر اساس تاریخ امروز که بهت داده می‌شه محاسبه کن)."""


def save_message(chat_id, role, content):
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/conversations",
            headers=SUPABASE_HEADERS,
            json={"chat_id": chat_id, "role": role, "content": content},
            timeout=10,
        )
    except Exception:
        logging.exception("خطا در ذخیرهٔ پیام در Supabase")


def get_history(chat_id, limit=30):
    try:
        params = {"chat_id": f"eq.{chat_id}", "order": "created_at.desc", "limit": str(limit)}
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/conversations", headers=SUPABASE_HEADERS, params=params, timeout=10)
        rows = resp.json() if resp.ok else []
        rows.reverse()
        return rows
    except Exception:
        logging.exception("خطا در خوندن تاریخچه از Supabase")
        return []


def save_fact(chat_id, content):
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/memory_facts",
            headers=SUPABASE_HEADERS,
            json={"chat_id": chat_id, "content": content},
            timeout=10,
        )
    except Exception:
        logging.exception("خطا در ذخیرهٔ فکت در Supabase")


def get_facts(chat_id):
    try:
        params = {"chat_id": f"eq.{chat_id}", "order": "created_at.asc"}
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/memory_facts", headers=SUPABASE_HEADERS, params=params, timeout=10)
        return resp.json() if resp.ok else []
    except Exception:
        logging.exception("خطا در خوندن حافظهٔ بلندمدت از Supabase")
        return []


def get_pending_tasks(chat_id):
    try:
        params = {"chat_id": f"eq.{chat_id}", "status": "eq.pending", "order": "task_date.asc,task_time.asc"}
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/tasks", headers=SUPABASE_HEADERS, params=params, timeout=10)
        return resp.json() if resp.ok else []
    except Exception:
        logging.exception("خطا در خوندن کارها از Supabase")
        return []


def create_task(chat_id, title, date, time_):
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/tasks",
            headers=SUPABASE_HEADERS,
            json={"chat_id": chat_id, "title": title, "task_date": date, "task_time": time_},
            timeout=10,
        )
    except Exception:
        logging.exception("خطا در ثبت کار در Supabase")


def update_task_status(chat_id, title, status):
    try:
        tasks = get_pending_tasks(chat_id)
        match = next((t for t in tasks if title.strip() in t.get("title", "") or t.get("title", "") in title.strip()), None)
        if match:
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/tasks",
                headers=SUPABASE_HEADERS,
                params={"id": f"eq.{match['id']}"},
                json={"status": status},
                timeout=10,
            )
    except Exception:
        logging.exception("خطا در آپدیت کار در Supabase")


def delete_task(chat_id, title):
    try:
        tasks = get_pending_tasks(chat_id)
        match = next((t for t in tasks if title.strip() in t.get("title", "") or t.get("title", "") in title.strip()), None)
        if match:
            requests.delete(
                f"{SUPABASE_URL}/rest/v1/tasks",
                headers=SUPABASE_HEADERS,
                params={"id": f"eq.{match['id']}"},
                timeout=10,
            )
    except Exception:
        logging.exception("خطا در حذف کار در Supabase")


def send_message(chat_id, text, suggestions=None):
    payload = {"chat_id": chat_id, "text": text}
    if suggestions:
        payload["reply_markup"] = json.dumps({
            "keyboard": [[s] for s in suggestions],
            "resize_keyboard": True,
            "one_time_keyboard": True,
        })
    requests.post(f"{TELEGRAM_API}/sendMessage", json=payload)


@app.route("/")
def home():
    return "بات روشنه و کار می‌کنه ✅"


@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json(silent=True) or {}
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    user_text = message.get("text")

    if chat_id is None or not user_text:
        return "ok"

    if user_text == "/start":
        send_message(chat_id, "سلام! من دستیار شخصی علی مسجدی هستم. هر چی بخوای بپرس 🙂")
        return "ok"

    save_message(chat_id, "user", user_text)
    history = get_history(chat_id)
    facts = get_facts(chat_id)
    tasks = get_pending_tasks(chat_id)
    now_tehran = datetime.now(TEHRAN_TZ)

    system_instruction = BASE_SYSTEM_INSTRUCTION
    system_instruction += f"\n\nتاریخ و ساعت الان: {now_tehran.strftime('%Y-%m-%d %H:%M')} (به وقت تهران)"

    if facts:
        facts_text = "\n".join(f"- {f.get('content', '')}" for f in facts)
        system_instruction += f"\n\nاطلاعاتی که قبلاً دربارهٔ کاربر یاد گرفتی:\n{facts_text}"

    if tasks:
        tasks_text = "\n".join(
            f"- {t.get('title')} (تاریخ: {t.get('task_date') or '-'}، ساعت: {t.get('task_time') or '-'})"
            for t in tasks
        )
        system_instruction += f"\n\nکارهای فعلی که هنوز انجام نشدن:\n{tasks_text}"
    else:
        system_instruction += "\n\nهیچ کار ثبت‌نشده‌ای فعلاً وجود نداره."

    contents = []
    for row in history:
        role = "user" if row.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=row.get("content", ""))]))

    reply_text = ""
    suggestions = []
    new_facts = []
    task_action = {}
    try:
        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=system_instruction),
                )
                break
            except ServerError as e:
                last_error = e
                logging.warning(f"مدل موقتاً در دسترس نیست، تلاش دوباره... ({attempt + 1}/3)")
                time.sleep(3)
        if response is None:
            raise last_error

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        reply_text = data.get("reply", raw)
        suggestions = data.get("suggestions", [])
        new_facts = data.get("new_facts", [])
        task_action = data.get("task_action", {}) or {}
    except Exception as e:
        logging.exception("خطا در ارتباط با Gemini")
        reply_text = f"یه خطا پیش اومد: {e}"

    save_message(chat_id, "model", reply_text)

    for fact in new_facts:
        if fact:
            save_fact(chat_id, fact)

    action = task_action.get("action")
    title = task_action.get("title")
    if action == "create" and title:
        create_task(chat_id, title, task_action.get("date"), task_action.get("time"))
    elif action == "complete" and title:
        update_task_status(chat_id, title, "done")
    elif action == "delete" and title:
        delete_task(chat_id, title)

    send_message(chat_id, reply_text, suggestions)
    return "ok"


@app.route("/check-reminders")
def check_reminders():
    secret = request.args.get("secret")
    if secret != REMINDER_SECRET:
        return "forbidden", 403

    now_tehran = datetime.now(TEHRAN_TZ)
    try:
        params = {
            "status": "eq.pending",
            "notified": "eq.false",
            "task_date": f"eq.{now_tehran.strftime('%Y-%m-%d')}",
        }
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/tasks", headers=SUPABASE_HEADERS, params=params, timeout=10)
        due_candidates = resp.json() if resp.ok else []
    except Exception:
        logging.exception("خطا در خوندن کارها برای یادآوری")
        return "error", 500

    sent = 0
    current_hm = now_tehran.strftime("%H:%M")
    for t in due_candidates:
        task_time = t.get("task_time")
        if not task_time:
            continue
        # task_time از Supabase به شکل HH:MM:SS برمی‌گرده
        task_hm = task_time[:5]
        if task_hm <= current_hm:
            send_message(t["chat_id"], f"⏰ یادآوری: {t.get('title')}")
            try:
                requests.patch(
                    f"{SUPABASE_URL}/rest/v1/tasks",
                    headers=SUPABASE_HEADERS,
                    params={"id": f"eq.{t['id']}"},
                    json={"notified": True},
                    timeout=10,
                )
            except Exception:
                logging.exception("خطا در آپدیت وضعیت اعلان")
            sent += 1

    return {"checked": len(due_candidates), "sent": sent}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
