import os
import json
import time
import logging
import wave
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, request
from piper import PiperVoice
import imageio_ffmpeg


logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

TEHRAN_TZ = ZoneInfo("Asia/Tehran")


def require_env(name):
    value = os.environ.get(name)
    if not value:
        logging.error(
            f"❌ متغیر محیطی {name} تنظیم نشده یا خالیه! "
            f"برو تو Render > Environment و اضافه‌ش کن."
        )
        raise SystemExit(1)
    return value


# =========================
# ENV
# =========================

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

MODEL_NAME = "openrouter/free"

GROQ_TRANSCRIPTION_URL = (
    "https://api.groq.com/openai/v1/audio/transcriptions"
)


# =========================
# PIPER / MANA
# =========================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fa_IR-mana-medium.onnx"
)

MODEL_CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "fa_IR-mana-medium.onnx.json"
)

MODEL_URL = (
    "https://huggingface.co/MahtaFetrat/"
    "Mana-Persian-Piper/resolve/main/"
    "fa_IR-mana-medium.onnx"
)

MODEL_CONFIG_URL = (
    "https://huggingface.co/MahtaFetrat/"
    "Mana-Persian-Piper/resolve/main/"
    "fa_IR-mana-medium.onnx.json"
)

piper_voice = None


def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return

    logging.info(f"⬇️ دانلود فایل: {path}")

    response = requests.get(
        url,
        stream=True,
        timeout=120
    )
    response.raise_for_status()

    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    logging.info(f"✅ دانلود شد: {path}")


def load_piper():
    global piper_voice

    if piper_voice is not None:
        return piper_voice

    download_file(MODEL_URL, MODEL_PATH)
    download_file(MODEL_CONFIG_URL, MODEL_CONFIG_PATH)

    logging.info("🔊 در حال بارگذاری مدل فارسی Piper...")

    piper_voice = PiperVoice.load(MODEL_PATH)

    logging.info("✅ مدل فارسی Piper آماده است.")

    return piper_voice


# =========================
# PERSIAN TTS
# =========================

def text_to_speech(text):
    """
    متن فارسی را به WAV تبدیل می‌کند
    و سپس به OGG/OPUS تبدیل می‌کند تا
    تلگرام بتواند آن را به شکل Voice ارسال کند.
    """

    try:
        voice = load_piper()

        os.makedirs("/tmp/edit_yar_voice", exist_ok=True)

        wav_path = "/tmp/edit_yar_voice/output.wav"
        ogg_path = "/tmp/edit_yar_voice/output.ogg"

        # پاک کردن فایل‌های قبلی
        for path in [wav_path, ogg_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        # Piper → WAV
        with wave.open(wav_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        # WAV → OGG/OPUS
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-y",
            "-i",
            wav_path,
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-vbr",
            "on",
            ogg_path,
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if not os.path.exists(ogg_path):
            raise RuntimeError("فایل صوتی OGG ساخته نشد.")

        return ogg_path

    except Exception:
        logging.exception("❌ خطا در تبدیل متن به صدا")
        return None


# =========================
# TELEGRAM
# =========================

def send_message(chat_id, text, suggestions=None):
    payload = {
        "chat_id": chat_id,
        "text": text
    }

    if suggestions:
        payload["reply_markup"] = json.dumps({
            "keyboard": [[s] for s in suggestions],
            "resize_keyboard": True,
            "one_time_keyboard": True,
        })

    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json=payload,
            timeout=20
        )
    except Exception:
        logging.exception("خطا در ارسال پیام متنی")


def send_voice(chat_id, voice_path):
    try:
        with open(voice_path, "rb") as voice_file:
            response = requests.post(
                f"{TELEGRAM_API}/sendVoice",
                data={
                    "chat_id": chat_id
                },
                files={
                    "voice": (
                        "edit_yar.ogg",
                        voice_file,
                        "audio/ogg"
                    )
                },
                timeout=60
            )

        if not response.ok:
            logging.error(
                f"❌ Telegram sendVoice: "
                f"{response.status_code} - {response.text}"
            )
            return False

        return True

    except Exception:
        logging.exception("❌ خطا در ارسال Voice")
        return False


def get_telegram_file(file_id):
    try:
        response = requests.get(
            f"{TELEGRAM_API}/getFile",
            params={"file_id": file_id},
            timeout=20
        )

        if not response.ok:
            logging.error(
                f"❌ getFile: {response.status_code} - {response.text}"
            )
            return None

        data = response.json()

        if not data.get("ok"):
            return None

        return data["result"].get("file_path")

    except Exception:
        logging.exception("خطا در دریافت اطلاعات فایل تلگرام")
        return None


def download_telegram_file(file_path):
    try:
        url = (
            f"https://api.telegram.org/file/bot"
            f"{TELEGRAM_TOKEN}/{file_path}"
        )

        response = requests.get(
            url,
            timeout=60
        )

        if not response.ok:
            logging.error(
                f"❌ دانلود فایل تلگرام: "
                f"{response.status_code}"
            )
            return None

        os.makedirs("/tmp/edit_yar_voice", exist_ok=True)

        local_path = "/tmp/edit_yar_voice/input.ogg"

        with open(local_path, "wb") as f:
            f.write(response.content)

        return local_path

    except Exception:
        logging.exception("خطا در دانلود فایل صوتی")
        return None


# =========================
# GROQ STT
# =========================

def speech_to_text(audio_path):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (
                    "voice.ogg",
                    audio_file,
                    "audio/ogg"
                )
            }

            data = {
                "model": "whisper-large-v3-turbo",
                "language": "fa",
                "response_format": "json",
                "temperature": "0",
            }

            response = requests.post(
                GROQ_TRANSCRIPTION_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )

        if not response.ok:
            logging.error(
                f"❌ Groq STT: "
                f"{response.status_code} - {response.text}"
            )
            return None

        result = response.json()

        text = result.get("text", "").strip()

        logging.info(f"🎤 متن تشخیص داده‌شده: {text}")

        return text if text else None

    except Exception:
        logging.exception("❌ خطا در تبدیل Voice به متن")
        return None


# =========================
# SUPABASE
# =========================

def save_message(chat_id, role, content):
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/conversations",
            headers=SUPABASE_HEADERS,
            json={
                "chat_id": chat_id,
                "role": role,
                "content": content
            },
            timeout=10,
        )

        if not resp.ok:
            logging.error(
                f"❌ Supabase رد کرد (save_message): "
                f"{resp.status_code} - {resp.text}"
            )

    except Exception:
        logging.exception(
            "خطا در ذخیرهٔ پیام در Supabase"
        )


def get_history(chat_id, limit=30):
    try:
        params = {
            "chat_id": f"eq.{chat_id}",
            "order": "created_at.desc",
            "limit": str(limit)
        }

        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/conversations",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10
        )

        if not resp.ok:
            logging.error(
                f"❌ Supabase رد کرد (get_history): "
                f"{resp.status_code} - {resp.text}"
            )
            return []

        rows = resp.json()
        rows.reverse()

        return rows

    except Exception:
        logging.exception(
            "خطا در خوندن تاریخچه از Supabase"
        )
        return []


def save_fact(chat_id, content):
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/memory_facts",
            headers=SUPABASE_HEADERS,
            json={
                "chat_id": chat_id,
                "content": content
            },
            timeout=10,
        )

    except Exception:
        logging.exception(
            "خطا در ذخیرهٔ فکت در Supabase"
        )


def get_facts(chat_id):
    try:
        params = {
            "chat_id": f"eq.{chat_id}",
            "order": "created_at.asc"
        }

        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/memory_facts",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10
        )

        return resp.json() if resp.ok else []

    except Exception:
        logging.exception(
            "خطا در خوندن حافظهٔ بلندمدت از Supabase"
        )
        return []


def get_pending_tasks(chat_id):
    try:
        params = {
            "chat_id": f"eq.{chat_id}",
            "status": "eq.pending",
            "order": "task_date.asc,task_time.asc"
        }

        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/tasks",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10
        )

        return resp.json() if resp.ok else []

    except Exception:
        logging.exception(
            "خطا در خوندن کارها از Supabase"
        )
        return []


def create_task(chat_id, title, date, time_, project_name=None):
    try:
        payload = {
            "chat_id": chat_id,
            "title": title,
            "task_date": date,
            "task_time": time_
        }

        if project_name:
            payload["project_name"] = project_name

        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/tasks",
            headers=SUPABASE_HEADERS,
            json=payload,
            timeout=10,
        )

        if resp.ok:
            logging.info(
                f"✅ کار ثبت شد: {title} ({date} {time_})"
            )
        else:
            logging.error(
                f"❌ create_task: "
                f"{resp.status_code} - {resp.text}"
            )

    except Exception:
        logging.exception(
            "خطا در ثبت کار در Supabase"
        )


def create_note(chat_id, content):
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/notes",
            headers=SUPABASE_HEADERS,
            json={
                "chat_id": chat_id,
                "content": content
            },
            timeout=10,
        )

        if not resp.ok:
            logging.error(
                f"❌ create_note: "
                f"{resp.status_code} - {resp.text}"
            )

    except Exception:
        logging.exception(
            "خطا در ثبت یادداشت در Supabase"
        )


def get_notes(chat_id):
    try:
        params = {
            "chat_id": f"eq.{chat_id}",
            "order": "created_at.desc",
            "limit": "50"
        }

        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/notes",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10
        )

        return resp.json() if resp.ok else []

    except Exception:
        logging.exception(
            "خطا در خوندن یادداشت‌ها از Supabase"
        )
        return []


def create_project(chat_id, name, description):
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/projects",
            headers=SUPABASE_HEADERS,
            json={
                "chat_id": chat_id,
                "name": name,
                "description": description
            },
            timeout=10,
        )

        if not resp.ok:
            logging.error(
                f"❌ create_project: "
                f"{resp.status_code} - {resp.text}"
            )

    except Exception:
        logging.exception(
            "خطا در ثبت پروژه در Supabase"
        )


def get_projects(chat_id):
    try:
        params = {
            "chat_id": f"eq.{chat_id}",
            "order": "created_at.desc"
        }

        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/projects",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10
        )

        return resp.json() if resp.ok else []

    except Exception:
        logging.exception(
            "خطا در خوندن پروژه‌ها از Supabase"
        )
        return []


def update_task_status(chat_id, title, status):
    try:
        tasks = get_pending_tasks(chat_id)

        match = next(
            (
                t for t in tasks
                if title.strip() in t.get("title", "")
                or t.get("title", "") in title.strip()
            ),
            None
        )

        if match:
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/tasks",
                headers=SUPABASE_HEADERS,
                params={
                    "id": f"eq.{match['id']}"
                },
                json={
                    "status": status
                },
                timeout=10,
            )

    except Exception:
        logging.exception(
            "خطا در آپدیت کار در Supabase"
        )


def delete_task(chat_id, title):
    try:
        tasks = get_pending_tasks(chat_id)

        match = next(
            (
                t for t in tasks
                if title.strip() in t.get("title", "")
                or t.get("title", "") in title.strip()
            ),
            None
        )

        if match:
            requests.delete(
                f"{SUPABASE_URL}/rest/v1/tasks",
                headers=SUPABASE_HEADERS,
                params={
                    "id": f"eq.{match['id']}"
                },
                timeout=10,
            )

    except Exception:
        logging.exception(
            "خطا در حذف کار در Supabase"
        )


# =========================
# AI
# =========================

BASE_SYSTEM_INSTRUCTION = """تو دستیار شخصی علی مسجدی هستی. هر جا لازم بود خودت رو معرفی کنی، بگو «من دستیار شخصی علی مسجدی هستم».

همیشه فقط و فقط یک شیء JSON با همین ساختار برگردون، بدون هیچ متن اضافه و بدون بک‌تیک:

{
  "reply": "متن جواب تو به زبان فارسی",
  "suggestions": ["پیشنهاد کوتاه اول", "پیشنهاد کوتاه دوم"],
  "new_facts": ["فکت مهم جدید"],
  "task_action": {
    "action": "create یا complete یا delete یا none",
    "title": "عنوان کار",
    "date": "YYYY-MM-DD یا null",
    "time": "HH:MM یا null",
    "project": "نام پروژهٔ مرتبط یا null"
  },
  "note_action": {
    "action": "create یا none",
    "content": "متن یادداشت"
  },
  "project_action": {
    "action": "create یا none",
    "name": "نام پروژه",
    "description": "توضیح کوتاه یا null"
  }
}

راهنمای هر بخش:

- suggestions: حداکثر ۲ پیشنهاد کوتاه.
- new_facts: فقط اطلاعات مهم و ماندگار دربارهٔ کاربر.
- task_action: برای ساخت، تکمیل یا حذف کار.
- note_action: برای ثبت یادداشت یا ایده.
- project_action: برای ساخت پروژه.
- تاریخ‌ها را همیشه به فرم میلادی YYYY-MM-DD بنویس.
- ساعت را همیشه به فرم ۲۴ساعته HH:MM بنویس.
- خروجی فقط JSON باشد."""


def call_ai(system_instruction, messages_history):
    messages = [
        {
            "role": "system",
            "content": system_instruction
        }
    ] + messages_history

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": MODEL_NAME,
        "messages": messages
    }

    last_error = None

    for attempt in range(3):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=body,
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()

                return data["choices"][0]["message"]["content"]

            last_error = (
                f"{resp.status_code} - {resp.text}"
            )

            logging.warning(
                f"OpenRouter خطا داد، تلاش دوباره... "
                f"({attempt + 1}/3): {last_error}"
            )

            time.sleep(3)

        except Exception as e:
            last_error = str(e)

            logging.warning(
                f"خطای اتصال به OpenRouter، تلاش دوباره... "
                f"({attempt + 1}/3): {last_error}"
            )

            time.sleep(3)

    raise RuntimeError(
        f"OpenRouter بعد از ۳ تلاش جواب نداد: {last_error}"
    )


# =========================
# PROCESS USER TEXT
# =========================

def process_user_text(chat_id, user_text):
    save_message(
        chat_id,
        "user",
        user_text
    )

    history = get_history(chat_id)

    facts = get_facts(chat_id)
    tasks = get_pending_tasks(chat_id)
    notes = get_notes(chat_id)
    projects = get_projects(chat_id)

    now_tehran = datetime.now(TEHRAN_TZ)

    system_instruction = BASE_SYSTEM_INSTRUCTION

    system_instruction += (
        f"\n\nتاریخ و ساعت الان: "
        f"{now_tehran.strftime('%Y-%m-%d %H:%M')} "
        f"(به وقت تهران)"
    )

    if facts:
        facts_text = "\n".join(
            f"- {f.get('content', '')}"
            for f in facts
        )

        system_instruction += (
            "\n\nاطلاعاتی که قبلاً دربارهٔ کاربر "
            "یاد گرفتی:\n"
            f"{facts_text}"
        )

    if tasks:
        tasks_text = "\n".join(
            f"- {t.get('title')} "
            f"(تاریخ: {t.get('task_date') or '-'}، "
            f"ساعت: {t.get('task_time') or '-'}، "
            f"پروژه: {t.get('project_name') or '-'})"
            for t in tasks
        )

        system_instruction += (
            "\n\nکارهای فعلی که هنوز انجام نشدن:\n"
            f"{tasks_text}"
        )

    else:
        system_instruction += (
            "\n\nهیچ کار ثبت‌نشده‌ای فعلاً وجود نداره."
        )

    if projects:
        projects_text = "\n".join(
            f"- {p.get('name')}: "
            f"{p.get('description') or '-'}"
            for p in projects
        )

        system_instruction += (
            "\n\nپروژه‌های فعلی کاربر:\n"
            f"{projects_text}"
        )

    else:
        system_instruction += (
            "\n\nهیچ پروژه‌ای فعلاً ثبت نشده."
        )

    if notes:
        notes_text = "\n".join(
            f"- {n.get('content')}"
            for n in notes
        )

        system_instruction += (
            "\n\nیادداشت‌های ثبت‌شدهٔ کاربر:\n"
            f"{notes_text}"
        )

    else:
        system_instruction += (
            "\n\nهیچ یادداشتی فعلاً ثبت نشده."
        )

    messages_history = []

    for row in history:
        role = (
            "user"
            if row.get("role") == "user"
            else "assistant"
        )

        messages_history.append({
            "role": role,
            "content": row.get("content", "")
        })

    reply_text = ""
    suggestions = []
    new_facts = []
    task_action = {}
    note_action = {}
    project_action = {}

    try:
        raw = call_ai(
            system_instruction,
            messages_history
        ).strip()

        if raw.startswith("```"):
            raw = raw.strip("`")

            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        data = json.loads(raw)

        reply_text = data.get("reply", raw)

        suggestions = data.get(
            "suggestions",
            []
        )

        new_facts = data.get(
            "new_facts",
            []
        )

        task_action = (
            data.get("task_action", {})
            or {}
        )

        note_action = (
            data.get("note_action", {})
            or {}
        )

        project_action = (
            data.get("project_action", {})
            or {}
        )

        logging.info(
            f"🔍 task_action: {task_action} | "
            f"note_action: {note_action} | "
            f"project_action: {project_action}"
        )

    except Exception as e:
        logging.exception(
            "خطا در ارتباط با هوش مصنوعی"
        )

        reply_text = (
            f"یه خطا پیش اومد: {e}"
        )

    save_message(
        chat_id,
        "model",
        reply_text
    )

    # حافظه
    for fact in new_facts:
        if fact:
            save_fact(
                chat_id,
                fact
            )

    # Task
    action = task_action.get("action")
    title = task_action.get("title")

    if action == "create" and title:
        create_task(
            chat_id,
            title,
            task_action.get("date"),
            task_action.get("time"),
            task_action.get("project")
        )

    elif action == "complete" and title:
        update_task_status(
            chat_id,
            title,
            "done"
        )

    elif action == "delete" and title:
        delete_task(
            chat_id,
            title
        )

    # Note
    if (
        note_action.get("action") == "create"
        and note_action.get("content")
    ):
        create_note(
            chat_id,
            note_action["content"]
        )

    # Project
    if (
        project_action.get("action") == "create"
        and project_action.get("name")
    ):
        create_project(
            chat_id,
            project_action["name"],
            project_action.get("description")
        )

    return reply_text, suggestions


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "بات روشنه و کار می‌کنه ✅"


@app.route("/webhook", methods=["POST"])
def webhook():

    update = request.get_json(
        silent=True
    ) or {}

    message = update.get(
        "message",
        {}
    )

    chat_id = message.get(
        "chat",
        {}
    ).get("id")

    if chat_id is None:
        return "ok"

    # =====================
    # TEXT
    # =====================

    user_text = message.get("text")

    if user_text:

        if user_text == "/start":
            send_message(
                chat_id,
                "سلام! من دستیار شخصی علی مسجدی هستم. "
                "هر چی بخوای بپرس."
            )
            return "ok"

        try:
            reply_text, suggestions = process_user_text(
                chat_id,
                user_text
            )

            send_message(
                chat_id,
                reply_text,
                suggestions
            )

        except Exception:
            logging.exception(
                "خطای کلی در پردازش پیام متنی"
            )

            send_message(
                chat_id,
                "در پردازش پیام مشکلی پیش اومد."
            )

        return "ok"

    # =====================
    # VOICE
    # =====================

    voice = message.get("voice")

    if voice:

        try:
            send_message(
                chat_id,
                "🎤 دارم گوش می‌کنم..."
            )

            file_id = voice.get("file_id")

            file_path = get_telegram_file(
                file_id
            )

            if not file_path:
                send_message(
                    chat_id,
                    "نتونستم فایل صوتی رو دریافت کنم."
                )
                return "ok"

            audio_path = download_telegram_file(
                file_path
            )

            if not audio_path:
                send_message(
                    chat_id,
                    "نتونستم فایل صوتی رو دانلود کنم."
                )
                return "ok"

            # Voice → Text
            user_text = speech_to_text(
                audio_path
            )

            if not user_text:
                send_message(
                    chat_id,
                    "صدات رو نتونستم تشخیص بدم. دوباره بفرست."
                )
                return "ok"

            logging.info(
                f"🎤 کاربر گفت: {user_text}"
            )

            # Text → Existing AI
            reply_text, suggestions = process_user_text(
                chat_id,
                user_text
            )

            # فعلاً جواب متنی را هم ارسال می‌کنیم
            send_message(
                chat_id,
                reply_text,
                suggestions
            )

            # AI Text → Persian Voice
            voice_path = text_to_speech(
                reply_text
            )

            if voice_path:

                sent = send_voice(
                    chat_id,
                    voice_path
                )

                if not sent:
                    logging.error(
                        "❌ Voice ارسال نشد."
                    )

            else:
                logging.error(
                    "❌ تولید Voice شکست خورد."
                )

        except Exception:
            logging.exception(
                "❌ خطای کلی در پردازش Voice"
            )

            send_message(
                chat_id,
                "در پردازش پیام صوتی مشکلی پیش اومد."
            )

        return "ok"

    return "ok"


# =========================
# REMINDERS
# =========================

@app.route("/check-reminders")
def check_reminders():

    secret = request.args.get(
        "secret"
    )

    if secret != REMINDER_SECRET:
        return "forbidden", 403

    now_tehran = datetime.now(
        TEHRAN_TZ
    )

    try:
        params = {
            "status": "eq.pending",
            "notified": "eq.false",
            "task_date": (
                f"eq.{now_tehran.strftime('%Y-%m-%d')}"
            ),
        }

        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/tasks",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10
        )

        due_candidates = (
            resp.json()
            if resp.ok
            else []
        )

    except Exception:
        logging.exception(
            "خطا در خوندن کارها برای یادآوری"
        )
        return "error", 500

    sent = 0

    current_hm = now_tehran.strftime(
        "%H:%M"
    )

    for t in due_candidates:

        task_time = t.get(
            "task_time"
        )

        if not task_time:
            continue

        task_hm = task_time[:5]

        if task_hm <= current_hm:

            send_message(
                t["chat_id"],
                f"⏰ یادآوری: {t.get('title')}"
            )

            try:
                requests.patch(
                    f"{SUPABASE_URL}/rest/v1/tasks",
                    headers=SUPABASE_HEADERS,
                    params={
                        "id": f"eq.{t['id']}"
                    },
                    json={
                        "notified": True
                    },
                    timeout=10,
                )

            except Exception:
                logging.exception(
                    "خطا در آپدیت وضعیت اعلان"
                )

            sent += 1

    return {
        "checked": len(due_candidates),
        "sent": sent
    }


# =========================
# RUN
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
        )
