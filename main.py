import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from google import genai

logging.basicConfig(level=logging.INFO)

# این دو خط، مقادیر رو از تنظیمات Render (Environment Variables) می‌خونن
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
# کلید Gemini خودش به‌صورت خودکار از متغیر GEMINI_API_KEY خونده میشه

client = genai.Client()
MODEL_NAME = "gemini-3-flash-preview"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! من دستیار هوش مصنوعی تو هستم. هر چی بخوای بپرس 🙂")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_text,
        )
        await update.message.reply_text(response.text)
    except Exception as e:
        logging.exception("خطا در ارتباط با Gemini")
        await update.message.reply_text(f"یه خطا پیش اومد: {e}")


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    app.run_polling()


if __name__ == "__main__":
    main()
