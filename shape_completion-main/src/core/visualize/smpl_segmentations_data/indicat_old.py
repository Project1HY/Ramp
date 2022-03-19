from typing import Union

import numpy as np


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def indicator_to_index(indicator, flip=False):
    if flip:
        return np.where(indicator == 0)[0]  # Return index of all 0 elements
    else:
        return np.where(indicator != 0)[0]  # Return index of all non-zero elements


def flip_indicator(indicator):
    return ~indicator


def flip_index(n, index):
    return indicator_to_index(index_to_indicator(n, index, flip=True))


def index_to_indicator(n, index, flip=False, val: Union[bool, float, int] = True):
    assert val != 0
    if flip:
        indicator = np.full(n, val, dtype=type(val))
        indicator[index] = 0
    else:
        indicator = np.zeros(n, dtype=type(val))
        indicator[index] = val
    return indicator


def _indicator_test():
    print(index_to_indicator(n=10, index=[1, 2, 3]))
    print(index_to_indicator(n=10, index=[1, 2, 2]))
    print(index_to_indicator(n=10, index=[1, 2, 2], val=5))
    print(index_to_indicator(n=10, index=[1, 2, 3], flip=True))
    print(index_to_indicator(n=10, index=[1, 2, 2], flip=True))
    print(index_to_indicator(n=10, index=[1, 2, 2], val=5, flip=True))

    print('sep')

    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 3])))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 2])))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 2], val=5)))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 3], flip=True)))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 2], flip=True)))
    print(indicator_to_index(index_to_indicator(n=10, index=[1, 2, 2], val=5, flip=True)))
