import time

def test_f_time(f)->float:
    """test_f_time.
    this funciton will test the time it take to preform a function f

    Args:
        f: a function to test

    Returns:
        float: the time it took to preform f

    Example:
           time_took=test_f_time(lambda: my_func(arg1,arg2,arg3)))
    """
    tic = time.perf_counter()
    f()
    toc = time.perf_counter()
    time_took=toc-tic
    return time_took
