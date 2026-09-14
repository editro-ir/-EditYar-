import os
import logging
from flask import Flask, request
import requests
from google import genai

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

client = genai.Client()
MODEL_NAME = "gemini-3-flash-preview"


def send_message(chat_id, text):
    requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text})


@app.route("/")
def home():
    # صفحه‌ای که وقتی خودمون آدرس سایت رو باز کنیم نشون داده میشه، فقط برای تست
    return "بات روشنه و کار می‌کنه ✅"


@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json(silent=True) or {}
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    user_text = message.get("text")

    if chat_id is None:
        return "ok"

    if user_text == "/start":
        send_message(chat_id, "سلام! من دستیار هوش مصنوعی تو هستم. هر چی بخوای بپرس 🙂")
        return "ok"

    if user_text:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_text,
            )
            send_message(chat_id, response.text)
        except Exception as e:
            logging.exception("خطا در ارتباط با Gemini")
            send_message(chat_id, f"یه خطا پیش اومد: {e}")

    return "ok"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
