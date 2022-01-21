import copy
import functools


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def copy_cache(maxsize=None, typed=False):
    # https://stackoverflow.com/questions/54909357/how-to-get-functools-lru-cache-to-return-new-instances
    def decorator(f):
        cached_func = functools.lru_cache(maxsize, typed)(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return copy.deepcopy(cached_func(*args, **kwargs))

        return wrapper

    return decorator


def cache(maxsize=None, typed=False):
    return functools.lru_cache(maxsize, typed)


# ---------------------------------------------------------------------------------------------------------------------#
#                                               TODO
# ---------------------------------------------------------------------------------------------------------------------#

def memoize(fn):
    """Helper decorator memoizes the given zero-argument function.
    Really helpful for memoizing properties so they don't have to be recomputed
    dozens of times. This is a more primitive version of lru_cache
    """

    @functools.wraps(fn)
    def memofn(self):
        if id(fn) not in self._cache:
            self._cache[id(fn)] = fn(self)
        return self._cache[id(fn)]

    return memofn


def cache_decorator(function):
    """
    # TODO - this is from trimesh. Check this out!
    A decorator for class methods, replaces @property
    but will store and retrieve function return values
    in object cache.
    Parameters
    ------------
    function : method
      This is used as a decorator:
      ```
      @cache_decorator
      def foo(self, things):
        return 'happy days'
      ```
    """

    # use wraps to preserve docstring
    @functools.wraps(function)
    def get_cached(*args, **kwargs):
        """
        Only execute the function if its value isn't stored
        in cache already.
        """
        self = args[0]
        # use function name as key in cache
        name = function.__name__
        # do the dump logic ourselves to avoid
        # verifying cache twice per call
        self._cache.verify()
        # access cache dict to avoid automatic validation
        # since we already called cache.verify manually
        if name in self._cache.cache:
            # already stored so return value
            return self._cache.cache[name]
        # value not in cache so execute the function
        value = function(*args, **kwargs)
        # store the value
        if self._cache.force_immutable and hasattr(
                value, 'flags') and len(value.shape) > 0:
            value.flags.writeable = False

        self._cache.cache[name] = value

        return value

    # all cached values are also properties
    # so they can be accessed like value attributes
    # rather than functions
    return property(get_cached)
