import logging
import sys

from lightning.assets.vista.plot_util.strings.prints import _MAIN_ANSI_SEQUENCES, print_red, print_yellow


# TODO - embed advanced logger version
# ----------------------------------------------------------------------------------------------------------------------
#                                                       logging Manips
# ----------------------------------------------------------------------------------------------------------------------

def duplicate_logging_stream_to_stdout():
    # Set logging to both the STDOUT and the File
    root_logger = logging.getLogger()
    logger_file_handlers = root_logger.handlers[0]
    new_format = logging.Formatter('[%(asctime)s] %(message)s')  # ,'%x %X.%f'
    logger_file_handlers.setFormatter(new_format)
    logger_file_handlers.stream = sys.stdout


# ----------------------------------------------------------------------------------------------------------------------
#                                                        logging Formats
# ----------------------------------------------------------------------------------------------------------------------


class LogColorer(logging.Formatter):
    # TODO - untested
    # Color messages depending on their levels
    def format(self, record):
        if record.levelno == logging.WARNING:
            record.msg = _MAIN_ANSI_SEQUENCES.YELLOW + record.msg + _MAIN_ANSI_SEQUENCES.RESET
        elif record.levelno == logging.ERROR:
            record.msg = _MAIN_ANSI_SEQUENCES.RED + record.msg + _MAIN_ANSI_SEQUENCES.RESET
        return logging.Formatter.format(self, record)


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Simple STDOUT logger
# ----------------------------------------------------------------------------------------------------------------------

class Logger:
    """
    # Usage 1
    @loggable
    def sum(a, b):
        return a+b
    sum(2, 4)

    # Usage 2
    Logger.warn('Computer might get destroyed')
    Logger.error('Computer destroyed')

    TODO  - see also advanced version at
     https://github.com/Pithikos/python-reusables/blob/master/logging/customgeneric_advanced.py
    """
    # Global configuration for Logger instances and class
    config = {
    }

    # Generic
    @classmethod
    def warn(cls, text):
        print_yellow(f'WARN:  {text}')

    @classmethod
    def error(cls, text):
        print_red(f'ERROR:  {text}')

    @classmethod
    def info(cls, text):
        print_red(f'INFO:  {text}')

    # Relaid
    @classmethod
    def precall_log(cls, fn, *args):
        print('PRECALL:  %s called with %s' % (fn.__name__, str(args)))

    @classmethod
    def postcall_log(cls, fn, retval):
        print('POSTCALL: %s returned %s' % (fn.__name__, retval))


# Decorator that relays to Logger
def loggable(fn):
    def wrapper(*args):
        Logger.precall_log(fn, args)
        return_value = fn(*args)
        Logger.postcall_log(fn, return_value)
        return return_value

    return wrapper
