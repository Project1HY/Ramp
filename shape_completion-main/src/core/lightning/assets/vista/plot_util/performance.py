import sys
import threading
import time
import timeit
from datetime import timedelta
from functools import wraps
from itertools import cycle

import psutil
from tqdm import tqdm

from plot_util.func import parametrized


# ----------------------------------------------------------------------------------------------------------------------
#                                                         Time
# ----------------------------------------------------------------------------------------------------------------------
@parametrized
def time_me(func, num_reps=1):
    """
    Needs to be decorated as @time_me() or @time_me(5)
    # TODO - An implementation to handle nested timing:
        # https://github.com/leopd/timebudget/blob/master/timebudget/timebudget.py
    """

    @wraps(func)
    def timed(*args, **kw):
        total_time, result = 0, None
        assert num_reps > 0
        for i in range(num_reps):
            ts = timeit.default_timer()
            result = func(*args, **kw)  # result is computed multiple times.
            te = timeit.default_timer()
            total_time += te - ts
            # print(num_reps, i)
        avg_time = total_time / num_reps
        # This snippet here allows extraction of the timing:
        # Snippet:
        # if 'log_time' in kw:
        #     name = kw.get('log_name', method.__name__.upper()) # Key defaults to method name
        #     kw['log_time'][name] = int((te - ts) * 1000)
        # Usage:
        # logtime_data = {}
        # ret_val = some_func_with_decorator(log_time=logtime_data)
        # else:
        print(f'{func.__name__} compute time [x{num_reps} reps] :: {str(timedelta(seconds=avg_time))}')
        return result

    return timed


class timer:
    # Can be simplified with contextmanager:
    # https://blog.usejournal.com/how-to-create-your-own-timing-context-manager-in-python-a0e944b48cf8
    # TODO - An implementation to handle nested timing:
    #   https://github.com/leopd/timebudget/blob/master/timebudget/timebudget.py
    # Can timer be generalized to average the results? Answer is NO - not supported in Python
    def __init__(self, code_name='Snippet'):
        self.name = code_name

    def __enter__(self):
        self.ts = timeit.default_timer()
        return self

    def __exit__(self, var_type, value, traceback):
        self.te = timeit.default_timer()
        print(f'{self.name} compute time [x1 reps] :: {str(timedelta(seconds=self.te - self.ts))}')

    def __float__(self):
        return float(self.te - self.ts)

    def __coerce__(self, other):
        return float(self), other

    def __str__(self):
        return str(float(self))

    def __repr__(self):
        return str(float(self))


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Time Progression
# ----------------------------------------------------------------------------------------------------------------------
def progress(iterable, unit="iters", desc="Compute Progress", **kwargs):
    """Wrapper on tqdm progress bar to give sensible defaults.

    Args:
        iterable: a data structure to be iterated with progress bar.
        kwargs: pass-through arguments to tqdm
    Returns:
        tqdm: progress bar with sensible defaults
    """

    defaults = {
        # "bar_format": "{l_bar}{bar}|{n_fmt}/{total_fmt} {unit} {percentage:3.0f}%",
        "unit": unit,
        "desc": desc,
        "dynamic_ncols": True,
        "file": sys.stdout
    }
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = defaults[k]
    return tqdm(iterable, **kwargs)


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


def tic():
    global tic_toc_start_time
    tic_toc_start_time = time.time()


def silent_toc():
    if 'tic_toc_start_time' in globals():
        return time.time() - tic_toc_start_time
    else:
        raise AssertionError('Toc: start time not set')


def toc():
    if 'tic_toc_start_time' in globals():
        delta = time.time() - tic_toc_start_time
        print(f"Elapsed time is {delta} seconds.")
        return delta
    else:
        raise AssertionError('Toc: start time not set')


# ----------------------------------------------------------------------------------------------------------------------
#                                                     Memory
# ----------------------------------------------------------------------------------------------------------------------

def memory_now(pid):
    return psutil.Process(pid).memory_info()[0] / (1024 ** 2)


def memory_sampler(pid, samples_freq, samples_order):
    """Get memory usage samples for given pid
    Takes:
        samples_freq(dict): Counting the sample occurance in every interval
        samples_order(list): Keeps track of the samples' order
    """
    if not pid:
        print('No PID given.')
        exit(1)
    proc = psutil.Process(pid)
    while True:
        sample = proc.memory_info()[0] / (1024 ** 2)
        if sample not in samples_freq:
            samples_freq[sample] = 1
            samples_order.append(sample)
        else:
            samples_freq[sample] += 1


def memory_profile(fn, arg):
    from multiprocessing import Process, Manager
    import time
    """Measure memory consumption for given function
    The memory usage given back is the best possible we can that reflects
    the actual memory consumption. A small error is always there (for example due
    to the boilerplate of Python) but as long as you compare outputs relative to
    each other (function1 vs function2) there should be no problem.
    The way we accomplish the profiling is by (1) starting the function on its
    own process (in order the profiler not to interfere) and (2) starting a memory
    sampler before the function process even starts. That way we can get a realistic
    measurement of the actual memory consumption.
    Returns:
        total memory of the boilerplate (just before starting profiling)
        first memory peak for the function
        average memory usage for the function
        highest memory peak for the function
    """

    def sleep_and_start(func, *args):
        time.sleep(0.05)  # Give some time for to-be-sampled proc to start.
        return func(*args)

    manager = Manager()
    samples_freq = manager.dict()
    samples_order = manager.list()
    sampled_proc = Process(target=sleep_and_start, args=(fn, arg))

    sampled_proc.start()
    memsampler = Process(target=memory_sampler, args=(sampled_proc.pid, samples_freq, samples_order))
    memsampler.start()
    sampled_proc.join()
    memsampler.terminate()

    samples_freq = dict(samples_freq)
    samples_order = list(samples_order)
    mem_init = samples_order[0]

    # Since we start profiling before we actually run the function, we can find
    # roughly the memory size of the boilerplate and remove it from our samples.
    # We also want to remove any 'invalid' samples that occured before proper
    # initializations.
    if 0 in samples_freq and 0 in samples_order:
        del samples_freq[0]
        samples_order.remove(0)
    if len(samples_order) > 1:
        del samples_freq[samples_order[0]]
        samples_order = samples_order[1:]
    samples_order = list(map(lambda s: s - mem_init, samples_order))
    samples_freq = {(s - mem_init): v for s, v in samples_freq.items()}

    num_values = sum(samples_freq.values())
    sum_values = sum([k * v for k, v in samples_freq.items()])
    return mem_init, min(samples_order), sum_values / num_values, max(samples_order)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
@time_me(5)
def _time_sleep():
    from time import sleep
    sleep(0.5)


if __name__ == '__main__':
    _time_sleep()
    from time import sleep

    with timer():
        sleep(0.5)
