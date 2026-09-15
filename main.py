import os
import json
import time
import logging
import requests
from flask import Flask, request
from google import genai
from google.genai import types
from google.genai.errors import ServerError

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)


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
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

BASE_SYSTEM_INSTRUCTION = """تو دستیار شخصی علی مسجدی هستی. هر جا لازم بود خودت رو معرفی کنی، بگو «من دستیار شخصی علی مسجدی هستم».

همیشه فقط و فقط یک شیء JSON با همین ساختار برگردون، بدون هیچ متن اضافه و بدون بک‌تیک:
{"reply": "متن جواب تو به زبان فارسی", "suggestions": ["پیشنهاد کوتاه اول", "پیشنهاد کوتاه دوم"], "new_facts": ["فکت مهم جدید"]}

- suggestions: حداکثر ۲ تا پیشنهاد کوتاه (حداکثر ۶-۷ کلمه) برای جملهٔ بعدی که کاربر ممکنه بخواد بفرسته.
- new_facts: اگه کاربر توی همین پیام یه اطلاعات ماندگار و مهم دربارهٔ خودش گفت (اسم، علاقه، عادت، شغل، اسم دوستان و خانواده، ترجیحات) که ارزش داره برای همیشه یادت بمونه، به‌صورت یک جملهٔ کوتاه و مستقل بنویسش. اگه چیز جدیدی نبود، آرایهٔ خالی [] بذار. هیچ‌وقت اطلاعات تکراری که قبلاً داری رو دوباره ننویس."""


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
        params = {
            "chat_id": f"eq.{chat_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        }
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/conversations",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10,
        )
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
        params = {
            "chat_id": f"eq.{chat_id}",
            "order": "created_at.asc",
        }
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/memory_facts",
            headers=SUPABASE_HEADERS,
            params=params,
            timeout=10,
        )
        return resp.json() if resp.ok else []
    except Exception:
        logging.exception("خطا در خوندن حافظهٔ بلندمدت از Supabase")
        return []


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

    system_instruction = BASE_SYSTEM_INSTRUCTION
    if facts:
        facts_text = "\n".join(f"- {f.get('content', '')}" for f in facts)
        system_instruction += f"\n\nاطلاعاتی که قبلاً دربارهٔ کاربر یاد گرفتی و باید در نظر بگیری:\n{facts_text}"

    contents = []
    for row in history:
        role = "user" if row.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=row.get("content", ""))]))

    reply_text = ""
    suggestions = []
    new_facts = []
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
    except Exception as e:
        logging.exception("خطا در ارتباط با Gemini")
        reply_text = f"یه خطا پیش اومد: {e}"

    save_message(chat_id, "model", reply_text)
    for fact in new_facts:
        if fact:
            save_fact(chat_id, fact)

    send_message(chat_id, reply_text, suggestions)
    return "ok"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
