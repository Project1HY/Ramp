import functools

"""
functools.lru_cache: when you use the @lru_cache decorator, subsequent calls with the same arguments to a function will 
fetch the value from the in-memory lru_cache instead of re-running the function. This is supremely useful if your 
function is computationally expensive to run from scratch each time. Be careful though, you can really chew up a lot of 
RAM if you @lru_cache indiscriminately, as you are holding the results of the functions in memory.
"""

def _memo(fn):
    """Helper decorator memoizes the given zero-argument function.
    Really helpful for memoizing properties so they don't have to be recomputed
    dozens of times. This is a more primitive version of lru_cache
    """
    @functools.wraps(fn)
    def memofn(self, *args, **kwargs):
        if id(fn) not in self._cache:
            self._cache[id(fn)] = fn(self)
        return self._cache[id(fn)]

    return memofn