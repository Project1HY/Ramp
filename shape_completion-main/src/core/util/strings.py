import logging
import os
import re
import sys
import warnings
from functools import wraps

import better_exceptions

better_exceptions.hook()

class TCHARS:  # Based on ANSI escape sequences - https://realpython.com/python-print/
    WHITE = '\033[1m\033[30m'  # \033 may be replaced with \e
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'  # not really green in pycharm
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESETER = '\033[0m'


def esc(code):
    return f'\033[{code}m'

# ----------------------------------------------------------------------------------------------------------------------
#                                                 String Manips
# ----------------------------------------------------------------------------------------------------------------------

def natural_keys(text):
    if isinstance(text, str):
        return [atoi(c) for c in extract_number(text)]
    else:
        return text


def extract_number(text):
    return re.split('(\d+)', text)


def atoi(text):
    return int(text) if text.isdigit() else text


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Pretty Prints
# ----------------------------------------------------------------------------------------------------------------------

def print_yellow(*args):
    # We do not dereference the TCHAR struct for a faster print
    sys.stdout.write('\033[93m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_red(*args):
    # We do not dereference the TCHAR struct for a faster print
    sys.stdout.write('\033[91m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_white(*args):
    # We do not dereference the TCHAR struct for a faster print
    sys.stdout.write('\033[30m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_purple(*args):
    # We do not dereference the TCHAR struct for a faster print
    sys.stdout.write('\033[95m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_blue(*args):
    # We do not dereference the TCHAR struct for a faster print
    sys.stdout.write('\033[94m')
    print(*args)
    sys.stdout.write('\033[0m')


def banner(text=None, ch='=', length=150):
    if text is not None:
        if len(text) + 2 >= length:
            spaced_text = text
        else:
            spaced_text = f' {text} '
    else:
        spaced_text = ''
    print(spaced_text.center(length, ch))


def purple_banner(text=None, ch='=', length=150):
    spaced_text = f' {text} ' if text is not None else ''
    print_purple(spaced_text.center(length, ch))


def white_banner(text=None, ch='=', length=150):
    spaced_text = f' {text} ' if text is not None else ''
    print_white(spaced_text.center(length, ch))


def green_banner(text=None, ch='=', length=150):
    spaced_text = f' {text} ' if text is not None else ''
    print_blue(spaced_text.center(length, ch))


def red_banner(text=None, ch='=', length=150):
    spaced_text = f' {text} ' if text is not None else ''
    print_red(spaced_text.center(length, ch))


def title(s):
    s = s.replace('_', ' ')
    s = s.replace('-', ' ')
    s = s.title()
    return s


def tutorial(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        banner(title(func.__name__))
        return func(*args, **kwargs)

    return wrapper


#
# def info(*args):
#     """
#     If configuration.config.VERBOSE_LEVEL >= 1:
#     Prints the sequence args with a prefixed timestamp and the string -I-, using bring yellow coloring,
#     """
#     from configuration.config import Config as Cfg
#     # Imported due to circular dependency. Python engine caches the import after the first time.
#     if Cfg.VERBOSE_LEVEL >= 1:
#         print_yellow(f'[{timestamp()}] -I-', *args)
#
#
# def comment(*args):
#     """
#     If configuration.config.VERBOSE_LEVEL >= 2:
#     Prints the sequence args with a prefixed timestamp and the string -C-, using standard print() coloring,
#     """
#     from configuration.config import Config as Cfg
#     # Imported due to circular dependency. Python engine caches the import after the first time.
#     if Cfg.VERBOSE_LEVEL >= 2:
#         print(f'[{timestamp()}] -C-', *args)


# ----------------------------------------------------------------------------------------------------------------------
#                                                      Custom Warn
# ----------------------------------------------------------------------------------------------------------------------


def _warn_format(message, category, filename, lineno, line=None):
    filename = os.path.basename(filename)
    return f'\033[93m{filename}:{lineno}:\nWARNING: {message}\n\033[0m'


warnings.formatwarning = _warn_format


def warn(s, stacklevel=1):
    warnings.warn(s, stacklevel=stacklevel + 1)


# warnings.simplefilter('always', DeprecationWarning)

def set_logging_to_stdout():
    # Set logging to both the STDOUT and the File
    root = logging.getLogger()
    hdlr = root.handlers[0]
    fmt = logging.Formatter('[%(asctime)s] %(message)s')  # ,'%x %X.%f'
    hdlr.setFormatter(fmt)
    hdlr.stream = sys.stdout


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


def __color_tester():
    i = 0
    for k, v in TCHARS.__dict__.items():
        if not k.startswith('__') and not k.startswith('RESETER'):
            i += 1
            print(f'{i}. {v}{k}{TCHARS.RESETER}')


if __name__ == '__main__':
    __color_tester()
