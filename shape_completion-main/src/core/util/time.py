import sys
import threading
import time
import timeit
from datetime import datetime, timedelta
from functools import wraps
from itertools import cycle

from tqdm import tqdm


def busy_wait(dt):
    current_time = time.time()
    while time.time() < current_time + dt:
        pass


class Spinner:
    # Check out https://github.com/pavdmyt/yaspin for nicer examples
    def __init__(self, message="Computing ", delay=0.1):
        self.spinner = cycle(['-', '/', '|', '\\'])
        self.delay = delay
        self.busy = False
        self.spinner_visible = False
        sys.stdout.write(message)

    def write_next(self):
        with self._screen_lock:
            if not self.spinner_visible:
                sys.stdout.write(next(self.spinner))
                self.spinner_visible = True
                sys.stdout.flush()

    def remove_spinner(self, cleanup=False):
        with self._screen_lock:
            if self.spinner_visible:
                sys.stdout.write('\b')
                self.spinner_visible = False
                if cleanup:
                    sys.stdout.write(' ')  # overwrite spinner with blank
                    sys.stdout.write('\r')  # move to next line
                sys.stdout.flush()

    def spinner_task(self):
        while self.busy:
            self.write_next()
            time.sleep(self.delay)
            self.remove_spinner()

    def __enter__(self):
        # if sys.stdout.isatty():
        self._screen_lock = threading.Lock()
        self.busy = True
        self.thread = threading.Thread(target=self.spinner_task)
        self.thread.start()

    def __exit__(self, exception, value, tb):
        # if sys.stdout.isatty():
        self.busy = False
        self.remove_spinner(cleanup=True)
        # else:
        #     sys.stdout.write('\r')


def timestamp(for_fs=False):
    """
    Args:
        for_fs (bool):

    Returns:
        timestamp (str): The current time in Returns the current time in HH:MM:SS:FFF format,
        where H->hours,M->minutes,S->seconds,F->Milliseconds
    """
    if for_fs:
        return datetime.now().strftime('%H-%M-%S-%f')[:-3]
    else:
        return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def progress(iterable, **kwargs):
    """Wrapper on tqdm progress bar to give sensible defaults.

    Args:
        iterable: a data structure to be iterated with progress bar.
        kwargs: pass-through arguments to tqdm
    Returns:
        tqdm: progress bar with sensible defaults
    """

    defaults = {
        "bar_format": "{l_bar}{bar}|{n_fmt}/{total_fmt} {percentage:3.0f}%",
        "unit": "files",
        "desc": "Compute Progress",
        "dynamic_ncols": True,
        "file": sys.stdout
    }
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = defaults[k]
    return tqdm(iterable, **kwargs)


def time_me(func):
    @wraps(func)
    def timed(*args, **kw):
        ts = timeit.default_timer()
        result = func(*args, **kw)
        te = timeit.default_timer()
        # This snippet here allows extraction of the timing:
        # Snippet:
        # if 'log_time' in kw:
        #     name = kw.get('log_name', method.__name__.upper()) # Key defaults to method name
        #     kw['log_time'][name] = int((te - ts) * 1000)
        # Usage:
        # logtime_data = {}
        # ret_val = some_func_with_decorator(log_time=logtime_data)
        # else:
        print(f'{func.__name__} compute time :: {str(timedelta(seconds=te - ts))}')
        return result

    return timed


class timer:
    def __init__(self, code_name='Snippet'):
        self.name = code_name

    def __enter__(self):
        self.ts = timeit.default_timer()
        return self

    def __exit__(self, var_type, value, traceback):
        te = timeit.default_timer()
        print(f'{self.name} compute time :: {str(timedelta(seconds=te - self.ts))}')


if __name__ == '__main__':
    sys.stdout.write('\a')
    with Spinner():
        time.sleep(5)

    print('DONE')
