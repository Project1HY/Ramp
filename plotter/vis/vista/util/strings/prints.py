import os
import re
import sys
from functools import wraps


# TODO -
#  * Translate all Python2 to Python3 here.
#  * Find someway to capture colors with respect to color scheme
#  * Remove banner and prints once the ANSI codes are aligned to each other - This basically involves changing the
#    range from 30->38 and excluding the white color. See _ansi_compare_test
# ----------------------------------------------------------------------------------------------------------------------
#                                                Old ANSI Structs
# ----------------------------------------------------------------------------------------------------------------------
class _MAIN_ANSI_SEQUENCES:  # Based on ANSI escape sequences - https://realpython.com/python-print/
    WHITE = '\033[1m\033[30m'  # \033 may be replaced with \e
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'  # not really green in pycharm
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def esc(code):
    return f'\033[{code}m'


# ----------------------------------------------------------------------------------------------------------------------
#                                                New ANSI Structs
# ----------------------------------------------------------------------------------------------------------------------
_ANSI_ATTRIBUTES = dict(
    list(zip([
        'bold',
        'dark',
        '',
        'underline',
        'blink',
        '',
        'reverse',
        'concealed'
    ],
        list(range(1, 9))
    ))
)
del _ANSI_ATTRIBUTES['']

_ANSI_ATTRIBUTES_RE = '\033\[(?:%s)m' % '|'.join(['%d' % v for v in _ANSI_ATTRIBUTES.values()])
_ANSI_HIGHLIGHTS = dict(
    list(zip([
        'on_grey',
        'on_red',
        'on_green',
        'on_yellow',
        'on_blue',
        'on_purple',
        'on_cyan',
        'on_white'
    ],
        list(range(40, 48))
    ))
)

_ANSI_HIGHLIGHTS_RE = '\033\[(?:%s)m' % '|'.join(['%d' % v for v in _ANSI_HIGHLIGHTS.values()])
_ANSI_COLORS = dict(
    list(zip([
        'grey',
        'red',
        'green',
        'yellow',
        'blue',
        'purple',
        'cyan',
        'white',
    ],
        list(range(30, 38))
    ))
)

_ANSI_COLORS_RE = '\033\[(?:%s)m' % '|'.join(['%d' % v for v in _ANSI_COLORS.values()])
_ANSI_RESET = '\033[0m'
_ANSI_RESET_RE = '\033\[0m'


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Pretty Prints
# ----------------------------------------------------------------------------------------------------------------------

def colored(text, color=None, on_color=None, attrs=None):
    """Colorize text, while stripping nested ANSI color sequences.
    Available text colors:
        red, green, yellow, blue, purple, cyan, white.
    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_purple, on_cyan, on_white.
    Available attributes:
        bold, dark, underline, blink, reverse, concealed.
    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """
    if os.getenv('ANSI_COLORS_DISABLED') is None:
        fmt_str = '\033[%dm%s'
        if color is not None:
            text = re.sub(_ANSI_COLORS_RE + '(.*?)' + _ANSI_RESET_RE, r'\1', text)
            text = fmt_str % (_ANSI_COLORS[color], text)
        if on_color is not None:
            text = re.sub(_ANSI_HIGHLIGHTS_RE + '(.*?)' + _ANSI_RESET_RE, r'\1', text)
            text = fmt_str % (_ANSI_HIGHLIGHTS[on_color], text)
        if attrs is not None:
            text = re.sub(_ANSI_ATTRIBUTES_RE + '(.*?)' + _ANSI_RESET_RE, r'\1', text)
            for attr in attrs:
                text = fmt_str % (_ANSI_ATTRIBUTES[attr], text)
        return text + _ANSI_RESET
    else:
        return text


def cprint(text, color=None, on_color=None, attrs=None, **kwargs):
    """Print colorize text.
    It accepts arguments of print function.
    """

    print((colored(text, color, on_color, attrs)), **kwargs)


def banner_str(text, sep='=', length=150):
    if text is None:
        spaced_text = ''
    else:
        text = str(text)
        if len(text) + 2 >= length:
            spaced_text = text
        else:
            spaced_text = f' {text} '
    return spaced_text.center(length, sep)


def banner(text=None, color=None, sep='=', length=150):
    cprint(banner_str(text=text, sep=sep, length=length), color=color)


def tutorial(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        banner(title(func.__name__))
        return func(*args, **kwargs)

    return wrapper


def title(s):
    """
    Replaces underscores and hyphens with a space and then
    capitalizes the first letter in each word
    """
    s = s.replace('_', ' ')
    s = s.replace('-', ' ')
    s = s.title()
    return s


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Old Pretty Prints
# ----------------------------------------------------------------------------------------------------------------------


def print_white(*args):
    # We do not dereference the TCHAR struct for a faster print
    sys.stdout.write('\033[30m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_purple(*args):
    sys.stdout.write('\033[95m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_blue(*args):
    sys.stdout.write('\033[94m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_green(*args):
    sys.stdout.write('\033[92m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_yellow(*args):
    sys.stdout.write('\033[93m')
    print(*args)
    sys.stdout.write('\033[0m')


def print_red(*args):
    sys.stdout.write('\033[91m')
    print(*args)
    sys.stdout.write('\033[0m')


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Old Banners
# ----------------------------------------------------------------------------------------------------------------------


def white_banner(text=None, sep='=', length=150):
    print_white(banner_str(text=text, sep=sep, length=length))


def purple_banner(text=None, sep='=', length=150):
    print_purple(banner_str(text=text, sep=sep, length=length))


def blue_banner(text=None, sep='=', length=150):
    print_blue(banner_str(text=text, sep=sep, length=length))


def green_banner(text=None, sep='=', length=150):
    print_green(banner_str(text=text, sep=sep, length=length))


def yellow_banner(text=None, sep='=', length=150):
    print_yellow(banner_str(text=text, sep=sep, length=length))


def red_banner(text=None, sep='=', length=150):
    print_red(banner_str(text=text, sep=sep, length=length))


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Graveyard
# ----------------------------------------------------------------------------------------------------------------------

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
# def warn(s, stacklevel=1):
#     warnings.warn(s, stacklevel=stacklevel + 1)

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


def _old_color_test():
    i = 0
    for k, v in _MAIN_ANSI_SEQUENCES.__dict__.items():
        if not k.startswith('__') and not k.startswith('RESETER'):
            i += 1
            print(f'{i}. {v}{k}{_MAIN_ANSI_SEQUENCES.RESET}')


def _color_search_test():
    for i in range(1, 100):
        print(f'{i}. {esc(i)}Color{_MAIN_ANSI_SEQUENCES.RESET}')


def _color_test_2():
    print('Current terminal type: %s' % os.getenv('TERM'))
    print('Test basic colors:')
    cprint('Grey color', 'grey')
    cprint('Red color', 'red')
    cprint('Green color', 'green')
    cprint('Yellow color', 'yellow')
    cprint('Blue color', 'blue')
    cprint('purple color', 'purple')
    cprint('Cyan color', 'cyan')
    cprint('White color', 'white')
    print('-' * 78)

    print('Test highlights:')
    cprint('On grey color', on_color='on_grey')
    cprint('On red color', on_color='on_red')
    cprint('On green color', on_color='on_green')
    cprint('On yellow color', on_color='on_yellow')
    cprint('On blue color', on_color='on_blue')
    cprint('On purple color', on_color='on_purple')
    cprint('On cyan color', on_color='on_cyan')
    cprint('On white color', color='grey', on_color='on_white')
    print('-' * 78)

    print('Test attributes:')
    cprint('Bold grey color', 'grey', attrs=['bold'])
    cprint('Dark red color', 'red', attrs=['dark'])
    cprint('Underline green color', 'green', attrs=['underline'])
    cprint('Blink yellow color', 'yellow', attrs=['blink'])
    cprint('Reversed blue color', 'blue', attrs=['reverse'])
    cprint('Concealed purple color', 'purple', attrs=['concealed'])
    cprint('Bold underline reverse cyan color', 'cyan',
           attrs=['bold', 'underline', 'reverse'])
    cprint('Dark blink concealed white color', 'white',
           attrs=['dark', 'blink', 'concealed'])
    print('-' * 78)

    print('Test mixing:')
    cprint('Underline red on grey color', 'red', 'on_grey',
           ['underline'])
    cprint('Reversed green on red color', 'green', 'on_red', ['reverse'])


def _ansi_compare_test():
    banner('hello', color='grey')
    banner('hello', color='red')
    banner('hello', color='green')
    banner('hello', color='yellow')
    banner('hello', color='blue')
    banner('hello', color='purple')
    banner('hello', color='cyan')
    banner('hello', color='white')
    banner('hello')
    print_white('hello')
    print_purple('hello')
    print_blue('hello')
    print_green('hello')
    print_yellow('hello')
    print_red('hello')

    white_banner('hello')
    purple_banner('hello')
    blue_banner('hello')
    green_banner('hello')
    yellow_banner('hello')
    red_banner('hello')
    print(_ANSI_COLORS)

    _color_search_test()


# ----------------------------------------------------------------------------------------------------------------------
#                                                              Test Suite
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    _ansi_compare_test()
