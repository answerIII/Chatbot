import logging

from text_model import TextModel
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class TelegaAPI:
    def __init__(self, token, filepath, filename):

        self.token = token

        self.filepath = filepath
        self.filename = filename

        self.text_model = TextModel(self.filepath, self.filename, vectorizer='BOW')

    def main(self):
        updater = Updater(self.token, use_context=True)

        # Get the dispatcher to register handlers
        dp = updater.dispatcher

        # on different commands - answer in Telegram
        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(CommandHandler("help", help))

        # on noncommand i.e message - echo the message on Telegram
        dp.add_handler(MessageHandler(Filters.text, self.echo))

        # log all errors
        dp.add_error_handler(self.error)

        # Start the Bot
        updater.start_polling()
        updater.idle()

    def start(update, context):
        """Send a message when the command /start is issued."""
        update.message.reply_text('Hi!')

    def help(update, context):
        """Send a message when the command /help is issued."""
        update.message.reply_text('Help!')

    def echo(self, update, context):
        """Echo the user message."""
        answer = self.text_model.predict_answer([update.message.text])
        update.message.reply_text(text=''.join(answer[0]))

    def error(update, context):
        """Log Errors caused by Updates."""
        logger.warning('Update "%s" caused error "%s"', update, context.error)


if __name__ == '__main__':

    token = 'token'
    filepath = f'../data'
    filename = f'good_clean.tsv'

    api = TelegaAPI(token, filepath, filename)
    api.main()
