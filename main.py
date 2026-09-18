import os
import json
import time
import logging
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, request

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

OPENROUTER_API_KEY = require_env("OPENROUTER_API_KEY")
GROQ_API_KEY = require_env("GROQ_API_KEY")
REMINDER_SECRET = require_env("REMINDER_SECRET")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# روتر رایگان: خودش هر بار یکی از مدل‌های رایگان دردسترس رو انتخاب می‌کنه
MODEL_NAME = "openrouter/free"

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

VOICE_DIR = "/tmp/piper_voice"
VOICE_ONNX = os.path.join(VOICE_DIR, "fa_IR-amir-medium.onnx")
VOICE_JSON = os.path.join(VOICE_DIR, "fa_IR-amir-medium.onnx.json")
VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium"


def ensure_voice_model():
    os.makedirs(VOICE_DIR, exist_ok=True)
    for local_path, filename in [(VOICE_ONNX, "fa_IR-amir-medium.onnx"), (VOICE_JSON, "fa_IR-amir-medium.onnx.json")]:
        if not os.path.exists(local_path):
            try:
                logging.info(f"در حال دانلود مدل صدای فارسی: {filename} ...")
                resp = requests.get(f"{VOICE_BASE_URL}/{filename}", timeout=120)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                logging.info(f"✅ دانلود شد: {filename}")
            except Exception:
                logging.exception(f"خطا در دانلود مدل صدا: {filename}")

BASE_SYSTEM_INSTRUCTION = """تو دستیار شخصی علی مسجدی هستی. هر جا لازم بود خودت رو معرفی کنی، بگو «من دستیار شخصی علی مسجدی هستم».

همیشه فقط و فقط یک شیء JSON با همین ساختار برگردون، بدون هیچ متن اضافه و بدون بک‌تیک:
{
  "reply": "متن جواب تو به زبان فارسی",
  "suggestions": ["پیشنهاد کوتاه اول", "پیشنهاد کوتاه دوم"],
  "new_facts": ["فکت مهم جدید"],
  "task_action": {"action": "create یا complete یا delete یا none", "title": "عنوان کار", "date": "YYYY-MM-DD یا null", "time": "HH:MM یا null", "project": "نام پروژهٔ مرتبط یا null"},
  "note_action": {"action": "create یا none", "content": "متن یادداشت"},
  "project_action": {"action": "create یا none", "name": "نام پروژه", "description": "توضیح کوتاه یا null"}
}

راهنمای هر بخش:
- suggestions: حداکثر ۲ پیشنهاد کوتاه (حداکثر ۶-۷ کلمه) برای جملهٔ بعدی که کاربر ممکنه بخواد بفرسته.
- new_facts: اگه کاربر یه اطلاعات ماندگار مهم دربارهٔ خودش گفت (اسم، علاقه، عادت، شغل...) که ارزش داره برای همیشه یادت بمونه، به‌صورت جملهٔ کوتاه بنویس. اگه چیز جدیدی نبود، [] بذار. تکراری ننویس.
- task_action: اگه کاربر خواست کاری/یادآوری/جلسه‌ای رو ثبت کنه → action=create با title و date/time (اگه ساعت یا تاریخ نگفت، همون null بذار). اگه گفت کاری رو انجام داده/تمومش کرده → action=complete با title (باید با یکی از کارهای فعلی لیست‌شده مطابقت داشته باشه). اگه خواست کاری رو حذف کنه → action=delete با title. اگه کار به یه پروژهٔ موجود مربوطه، اسمش رو تو project بذار وگرنه null. در غیر این صورت action=none.
- note_action: اگه کاربر خواست یه نکته/یادداشت/ایده رو ثبت کنی (نه یه کار زمان‌دار) → action=create با content. در غیر این صورت action=none.
- project_action: اگه کاربر خواست یه پروژهٔ جدید بسازه (یه مجموعه کار بزرگ‌تر با اسم مشخص) → action=create با name و description. در غیر این صورت action=none.
- تاریخ‌ها رو همیشه به فرم میلادی YYYY-MM-DD و ساعت رو به فرم ۲۴ساعته HH:MM بنویس، حتی اگه کاربر «فردا» یا «سه‌شنبه» گفته باشه (بر اساس تاریخ امروز که بهت داده می‌شه محاسبه کن).
- خروجی تو فقط همون یک شیء JSON باشه، هیچ متن قبل یا بعدش ننویس."""


def save_message(chat_id, role, content):
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/conversations",
            headers=SUPABASE_HEADERS,
            json={"chat_id": chat_id, "role": role, "content": content},
            timeout=10,
        )
        if not resp.ok:
            logging.error(f"❌ Supabase رد کرد (save_message): {resp.status_code} - {resp.text}")
    except Exception:
        logging.exception("خطا در ذخیرهٔ پیام در Supabase")


def get_history(chat_id, limit=30):
    try:
        params = {"chat_id": f"eq.{chat_id}", "order": "created_at.desc", "limit": str(limit)}
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/conversations", headers=SUPABASE_HEADERS, params=params, timeout=10)
        if not resp.ok:
            logging.error(f"❌ Supabase رد کرد (get_history): {resp.status_code} - {resp.text}")
            return []
        rows = resp.json()
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


def create_task(chat_id, title, date, time_, project_name=None):
    try:
        payload = {"chat_id": chat_id, "title": title, "task_date": date, "task_time": time_}
        if project_name:
            payload["project_name"] = project_name
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/tasks",
            headers=SUPABASE_HEADERS,
            json=payload,
            timeout=10,
        )
        if resp.ok:
            logging.info(f"✅ کار ثبت شد: {title} ({date} {time_})")
        else:
            logging.error(f"❌ Supabase رد کرد (create_task): {resp.status_code} - {resp.text}")
    except Exception:
        logging.exception("خطا در ثبت کار در Supabase")


def create_note(chat_id, content):
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/notes",
            headers=SUPABASE_HEADERS,
            json={"chat_id": chat_id, "content": content},
            timeout=10,
        )
        if not resp.ok:
            logging.error(f"❌ Supabase رد کرد (create_note): {resp.status_code} - {resp.text}")
    except Exception:
        logging.exception("خطا در ثبت یادداشت در Supabase")


def get_notes(chat_id):
    try:
        params = {"chat_id": f"eq.{chat_id}", "order": "created_at.desc", "limit": "50"}
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/notes", headers=SUPABASE_HEADERS, params=params, timeout=10)
        return resp.json() if resp.ok else []
    except Exception:
        logging.exception("خطا در خوندن یادداشت‌ها از Supabase")
        return []


def create_project(chat_id, name, description):
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/projects",
            headers=SUPABASE_HEADERS,
            json={"chat_id": chat_id, "name": name, "description": description},
            timeout=10,
        )
        if not resp.ok:
            logging.error(f"❌ Supabase رد کرد (create_project): {resp.status_code} - {resp.text}")
    except Exception:
        logging.exception("خطا در ثبت پروژه در Supabase")


def get_projects(chat_id):
    try:
        params = {"chat_id": f"eq.{chat_id}", "order": "created_at.desc"}
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/projects", headers=SUPABASE_HEADERS, params=params, timeout=10)
        return resp.json() if resp.ok else []
    except Exception:
        logging.exception("خطا در خوندن پروژه‌ها از Supabase")
        return []


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


def transcribe_voice(file_id):
    try:
        r = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=15)
        file_path = r.json()["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
        audio_resp = requests.get(file_url, timeout=30)

        files = {"file": ("voice.ogg", audio_resp.content, "audio/ogg")}
        data = {"model": "whisper-large-v3-turbo"}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        resp = requests.post(GROQ_STT_URL, headers=headers, files=files, data=data, timeout=30)
        if resp.ok:
            return resp.json().get("text", "").strip()
        else:
            logging.error(f"❌ Groq رد کرد (transcribe): {resp.status_code} - {resp.text}")
            return None
    except Exception:
        logging.exception("خطا در تبدیل صدا به متن")
        return None


def text_to_speech(text):
    """با Piper (متن‌باز، محلی) متن رو به فایل صوتی OGG تبدیل می‌کنه."""
    try:
        ensure_voice_model()
        if not (os.path.exists(VOICE_ONNX) and os.path.exists(VOICE_JSON)):
            logging.error("❌ مدل صدای Piper موجود نیست")
            return None

        wav_path = "/tmp/reply.wav"
        ogg_path = "/tmp/reply.ogg"

        result = subprocess.run(
            ["piper", "--model", VOICE_ONNX, "--output_file", wav_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            logging.error(f"❌ خطای Piper: {result.stderr.decode(errors='ignore')}")
            return None

        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        conv = subprocess.run(
            [ffmpeg_exe, "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", ogg_path],
            capture_output=True,
            timeout=60,
        )
        if conv.returncode != 0:
            logging.error(f"❌ خطای تبدیل فرمت صدا: {conv.stderr.decode(errors='ignore')}")
            return None

        with open(ogg_path, "rb") as f:
            return f.read()
    except Exception:
        logging.exception("خطا در تولید صدا با Piper")
        return None


def send_voice_reply(chat_id, ogg_bytes, suggestions=None):
    try:
        files = {"voice": ("reply.ogg", ogg_bytes, "audio/ogg")}
        data = {"chat_id": chat_id}
        if suggestions:
            data["reply_markup"] = json.dumps({
                "keyboard": [[s] for s in suggestions],
                "resize_keyboard": True,
                "one_time_keyboard": True,
            })
        requests.post(f"{TELEGRAM_API}/sendVoice", data=data, files=files, timeout=30)
    except Exception:
        logging.exception("خطا در ارسال پیام صوتی")


def call_ai(system_instruction, messages_history):
    """messages_history: لیستی از دیکشنری‌های {"role": "user"/"assistant", "content": "..."}"""
    messages = [{"role": "system", "content": system_instruction}] + messages_history
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {"model": MODEL_NAME, "messages": messages}

    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                last_error = f"{resp.status_code} - {resp.text}"
                logging.warning(f"OpenRouter خطا داد، تلاش دوباره... ({attempt + 1}/3): {last_error}")
                time.sleep(3)
        except Exception as e:
            last_error = str(e)
            logging.warning(f"خطای اتصال به OpenRouter، تلاش دوباره... ({attempt + 1}/3): {last_error}")
            time.sleep(3)

    raise RuntimeError(f"OpenRouter بعد از ۳ تلاش جواب نداد: {last_error}")


@app.route("/")
def home():
    return "بات روشنه و کار می‌کنه ✅"


@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json(silent=True) or {}
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    user_text = message.get("text")
    voice = message.get("voice")
    is_voice_message = False

    if chat_id is None:
        return "ok"

    if voice:
        transcribed = transcribe_voice(voice["file_id"])
        if not transcribed:
            send_message(chat_id, "متأسفم، نتونستم صدات رو تشخیص بدم. می‌شه دوباره امتحان کنی یا با متن بنویسی؟")
            return "ok"
        user_text = transcribed
        is_voice_message = True

    if not user_text:
        return "ok"

    if user_text == "/start":
        send_message(chat_id, "سلام! من دستیار شخصی علی مسجدی هستم. هر چی بخوای بپرس 🙂")
        return "ok"

    save_message(chat_id, "user", user_text)
    history = get_history(chat_id)
    facts = get_facts(chat_id)
    tasks = get_pending_tasks(chat_id)
    notes = get_notes(chat_id)
    projects = get_projects(chat_id)
    now_tehran = datetime.now(TEHRAN_TZ)

    system_instruction = BASE_SYSTEM_INSTRUCTION
    system_instruction += f"\n\nتاریخ و ساعت الان: {now_tehran.strftime('%Y-%m-%d %H:%M')} (به وقت تهران)"

    if facts:
        facts_text = "\n".join(f"- {f.get('content', '')}" for f in facts)
        system_instruction += f"\n\nاطلاعاتی که قبلاً دربارهٔ کاربر یاد گرفتی:\n{facts_text}"

    if tasks:
        tasks_text = "\n".join(
            f"- {t.get('title')} (تاریخ: {t.get('task_date') or '-'}، ساعت: {t.get('task_time') or '-'}، پروژه: {t.get('project_name') or '-'})"
            for t in tasks
        )
        system_instruction += f"\n\nکارهای فعلی که هنوز انجام نشدن:\n{tasks_text}"
    else:
        system_instruction += "\n\nهیچ کار ثبت‌نشده‌ای فعلاً وجود نداره."

    if projects:
        projects_text = "\n".join(f"- {p.get('name')}: {p.get('description') or '-'}" for p in projects)
        system_instruction += f"\n\nپروژه‌های فعلی کاربر:\n{projects_text}"
    else:
        system_instruction += "\n\nهیچ پروژه‌ای فعلاً ثبت نشده."

    if notes:
        notes_text = "\n".join(f"- {n.get('content')}" for n in notes)
        system_instruction += f"\n\nیادداشت‌های ثبت‌شدهٔ کاربر:\n{notes_text}"
    else:
        system_instruction += "\n\nهیچ یادداشتی فعلاً ثبت نشده."

    messages_history = []
    for row in history:
        role = "user" if row.get("role") == "user" else "assistant"
        messages_history.append({"role": role, "content": row.get("content", "")})

    if not messages_history:
        messages_history.append({"role": "user", "content": user_text})

    reply_text = ""
    suggestions = []
    new_facts = []
    task_action = {}
    note_action = {}
    project_action = {}
    try:
        raw = call_ai(system_instruction, messages_history).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        reply_text = data.get("reply", raw)
        suggestions = data.get("suggestions", [])
        new_facts = data.get("new_facts", [])
        task_action = data.get("task_action", {}) or {}
        note_action = data.get("note_action", {}) or {}
        project_action = data.get("project_action", {}) or {}
        logging.info(f"🔍 task_action: {task_action} | note_action: {note_action} | project_action: {project_action}")
    except Exception as e:
        logging.exception("خطا در ارتباط با هوش مصنوعی")
        reply_text = f"یه خطا پیش اومد: {e}"

    save_message(chat_id, "model", reply_text)

    for fact in new_facts:
        if fact:
            save_fact(chat_id, fact)

    action = task_action.get("action")
    title = task_action.get("title")
    if action == "create" and title:
        create_task(chat_id, title, task_action.get("date"), task_action.get("time"), task_action.get("project"))
    elif action == "complete" and title:
        update_task_status(chat_id, title, "done")
    elif action == "delete" and title:
        delete_task(chat_id, title)

    if note_action.get("action") == "create" and note_action.get("content"):
        create_note(chat_id, note_action["content"])

    if project_action.get("action") == "create" and project_action.get("name"):
        create_project(chat_id, project_action["name"], project_action.get("description"))

    if is_voice_message:
        audio_bytes = text_to_speech(reply_text)
        if audio_bytes:
            send_voice_reply(chat_id, audio_bytes, suggestions)
        else:
            # اگه ساخت صدا شکست خورد، لااقل متن رو بفرست که بی‌جواب نمونه
            send_message(chat_id, reply_text, suggestions)
    else:
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
