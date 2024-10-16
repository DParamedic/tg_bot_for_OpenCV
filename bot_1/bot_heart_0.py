import asyncio
from typing import NoReturn
from telegram.ext import Application, MessageHandler, ContextTypes, CallbackContext, CommandHandler, filters
from telegram import Update, Bot


async def start_callback(update=Update, context=ContextTypes.DEFAULT_TYPE):
    user_says = " ".join(context.args)
    await update.message.reply_text("You said: " + user_says)

async def downloader(name: str, exp: str, chat_id: str=None,  mode: str=None) -> NoReturn:
    await (chat_id := get_id_photo())
    await Bot.send_photo(Bot, chat_id, photo=open(f'{name}.{exp}', mode if mode is not None else 'r'))

async def get_id_photo(update=Update, context=ContextTypes.DEFAULT_TYPE):
    await update.message.id

async def start_download_photo(update=Update, context=ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Amm... Photo?')

async def united_downloader():
    # task1 = asyncio.create_task(get_id_photo())
    task2 = asyncio.create_task(downloader('new', 'jpg'))
    await task2

def main() -> None:
    application = Application.builder().token("7732354386:AAF2yzlF4fsmnul-YI21iiHO25SRxrU7qWE").build() # don't commit token!

    application.add_handler(CommandHandler('start', start_callback))
    application.add_handlers([CommandHandler('picture', start_download_photo), MessageHandler(filters.PHOTO, united_downloader())])

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
