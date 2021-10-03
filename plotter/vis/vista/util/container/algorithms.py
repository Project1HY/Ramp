import numbers

from util.strings import try_str2int, try_str2num, inclusive_split_on_integer


# ----------------------------------------------------------------------------------------------------------------------
#                                                 Type Checking
# ----------------------------------------------------------------------------------------------------------------------
def is_integer(obj):
    return isinstance(obj, numbers.Integral)


def is_number(obj):
    return isinstance(obj, numbers.Number)


def string_is_integer(obj):
    return is_integer(try_str2int(obj))


def string_is_number(obj):
    return is_integer(try_str2num(obj))


def is_string(obj):
    return isinstance(obj, str)


# ----------------------------------------------------------------------------------------------------------------------
#                                                 Enum Handling
# ----------------------------------------------------------------------------------------------------------------------

def enum_eq(enum1, enum2):
    return enum1.__class__.__name__ == enum1.__class__.__name__ and enum1.value == enum2.value


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Sorting
# ----------------------------------------------------------------------------------------------------------------------
def alphanum_sort(lst):
    """
    Sort the given list in the way that humans expect.
    # https://nedbatchelder.com/blog/200712/human_sorting.html
    """
    lst.sort(key=alphanum_sort_key)
    return lst


def alphanum_sort_key(s):
    if isinstance(s, str):
        # TODO - Would not support all variations of floats - better suited for just integer
        # progressions. The number 3.56 would be sorted first by 3, and then by the 56, which
        # would turn out fine - but a number like 5e-4 would not be understood.
        return [try_str2int(c) for c in inclusive_split_on_integer(s)]
    else:
        return s


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Test Suite
# ----------------------------------------------------------------------------------------------------------------------
def _alphanum_sort_test():
    a = ['hello3', 'hello19', 'hello231', 'bibi_sucks312093', 'bibi_sucks1', 'bibi_sucks1.9']
    print(alphanum_sort(a))
    a = [100, 10, 1]
    print(alphanum_sort(a))
    a = [5.3, 5.2, 31203, 39]
    print(alphanum_sort(a))
    a = [1e-4, 1e-5, 1e-6, 1e-9]
    print(alphanum_sort(a))
    a = ['1e-4', '1e-5', '1e-6', '1e-9']
    print(alphanum_sort(a))


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    _alphanum_sort_test()
