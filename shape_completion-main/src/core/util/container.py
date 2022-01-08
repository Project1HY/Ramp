import random
from collections import OrderedDict, Sequence
from copy import deepcopy
from typing import MutableMapping
import numbers
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
#                                                     Type Checking
# ----------------------------------------------------------------------------------------------------------------------
from util.strings import natural_keys


def is_integer(x):
    return isinstance(x, numbers.Integral)


def is_number(x):
    return isinstance(x, numbers.Number)


# ----------------------------------------------------------------------------------------------------------------------
#                                                     Enum
# ----------------------------------------------------------------------------------------------------------------------

def enum_eq(enum1, enum2):
    return enum1.__class__.__name__ == enum1.__class__.__name__ and enum1.value == enum2.value


# ----------------------------------------------------------------------------------------------------------------------
#                                                     Lists
# ----------------------------------------------------------------------------------------------------------------------
def flatten(ll):
    """
    :param ll: A list of lists
    :return: A list composed of all the elements of the internal lists of ll, in the order the lists were supplied
    """
    return [item for sublist in ll for item in sublist]


def multishuffle(*args):
    # TODO - Add in option for different ordering per list?
    """
    Receive an arbitrary amount of lists and shuffles them by the same order
    :return: A list of the received lists, shuffled
    """
    length = len(args[0])
    order = np.random.permutation(length)
    shuffled_lists = []
    for l in args:
        shuffled_lists.append(l[order])

    return shuffled_lists


def list_dup(l, n):
    return [deepcopy(l) for _ in range(n)]


def grp_start_len(a):
    """Given a sorted 1D input array `a`, e.g., [0 0, 1, 2, 3, 4, 4, 4], this
    routine returns the indices where the blocks of equal integers start and
    how long the blocks are.
    """
    # https://stackoverflow.com/a/50394587/353337
    m = np.concatenate([[True], a[:-1] != a[1:], [True]])
    idx = np.flatnonzero(m)
    return idx[:-1], np.diff(idx)


def split_frac(l, fracs):
    # Accumulate the percentages
    splits = np.cumsum(fracs).astype(np.float)

    # Two cases: Percentage list is full or missing the last value
    if splits[-1] == 1:
        # Split doesn't need last percent, it will just take what is left
        splits = splits[:-1]
    elif splits[-1] > 1:
        raise ValueError("Sum of fracs are greater than one")
    # On < 1 -> Do Nothing

    # Turn values into indices
    splits *= len(l)

    # Turn double indices into integers.
    # CAUTION: numpy rounds to closest EVEN number when a number is halfway
    # between two integers. So 0.5 will become 0 and 1.5 will become 2!
    # If you want to round up in all those cases, do
    # splits += 0.5 instead of round() before casting to int
    splits += 0.5
    splits = splits.astype(np.int)
    l = np.random.permutation(l)
    return np.split(l, splits)


def first(iterable, condition=lambda x: True):
    """
    * Returns the first item in the `iterable` that satisfies the `condition`.
    * If the condition is not given, returns the first item of the iterable.
    * Raises `StopIteration` if no item satisfying the condition is found.
    """
    return next(x for x in iterable if condition(x))


def to_list(l, encapsulate_none=True):
    if not encapsulate_none and l is None:
        return []
    if isinstance(l, list):
        return l
    elif isinstance(l, Sequence):
        return list(l)
    else:  # Presuming Scalar
        return [l]


def all_unique(l):
    """
    :param l: A python iterable such as a list or tuple
    :return: bool: True if all elements are unique, or false otherwise
    """
    seen = set()
    return not any(i in seen or seen.add(i) for i in l)


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Dicts
# ----------------------------------------------------------------------------------------------------------------------

def max_dict_depth(dic, level=1):
    if not isinstance(dic, dict) or not dic:
        return level
    return max(max_dict_depth(dic[key], level + 1)
               for key in dic)


def min_dict_depth(dic, level=1):
    if not isinstance(dic, dict) or not dic:
        return level
    return min(min_dict_depth(dic[key], level + 1)
               for key in dic)


def deep_check_odict(d):
    if not isinstance(d, OrderedDict):
        return False
    else:
        kids_are_ordered = True
        for v in d.values():
            if isinstance(v, MutableMapping):
                kids_are_ordered &= deep_check_odict(v)
        return kids_are_ordered


def delete_keys_from_dict(dictionary, keys):
    keys_set = set(keys)  # Just an optimization for the "if key in keys" lookup.

    modified_dict = {}
    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, MutableMapping):
                modified_dict[key] = delete_keys_from_dict(value, keys_set)
            else:
                modified_dict[key] = value  # or copy.deepcopy(value) if a copy is desired for non-dicts.
    return modified_dict


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Test Suite
# ----------------------------------------------------------------------------------------------------------------------
def _split_frac_tester():
    import numpy as np
    import random
    for i in range(1, 10000):
        indices = list(range(i))
        n_split = random.randint(1, 20)
        n_dirac = random.randint(1, 1000 * 1000) / 1000
        splits = np.random.dirichlet(np.ones(n_split) * n_dirac, size=1) * 3 / 4
        # splits = [0.05,0.05,0.9]
        assert sum(map(len, split_frac(indices, splits))) == i, f"{i}"


# TODO - move this into structs directory - problematic due to the pickled hit dependencies 
class RandomDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        """ Create RandomDict object with contents specified by arguments.
        Any argument
        :param *args:       dictionaries whose contents get added to this dict
        :param **kwargs:    key, value pairs will be added to this dict
        """
        # mapping of keys to array positions
        self.keys = {}
        self.values = []
        self.last_index = -1

        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        if key in self.keys:
            i = self.keys[key]
        else:
            self.last_index += 1
            i = self.last_index

        self.values.append((key, val))
        self.keys[key] = i

    def __delitem__(self, key):
        if key not in self.keys:
            raise KeyError

        # index of item to delete is i
        i = self.keys[key]
        # last item in values array is
        move_key, move_val = self.values.pop()

        if i != self.last_index:
            # we move the last item into its location
            self.values[i] = (move_key, move_val)
            self.keys[move_key] = i
        # else it was the last item and we just throw
        # it away

        # shorten array of values
        self.last_index -= 1
        # remove deleted key
        del self.keys[key]

    def __getitem__(self, key):
        if key not in self.keys:
            raise KeyError

        i = self.keys[key]
        return self.values[i][1]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return self.last_index + 1

    def random_key(self):
        """ Return a random key from this dictionary in O(1) time """
        if len(self) == 0:
            raise KeyError("RandomDict is empty")

        i = random.randint(0, self.last_index)
        return self.values[i][0]

    def random_value(self):
        """ Return a random value from this dictionary in O(1) time """
        return self[self.random_key()]

    def random_item(self):
        """ Return a random key-value pair from this dictionary in O(1) time """
        k = self.random_key()
        return k, self[k]


def deep_dict_convert(d, type=RandomDict):
    rdict = type()

    for key in sorted(d.keys(), key=natural_keys):
        value = d[key]
        if isinstance(value, dict):
            rdict[key] = deep_dict_convert(value, type=type)
        else:
            rdict[key] = value

    return rdict


def deep_dict_change_leaves(d, new_val):
    for k, v in d.items():
        if isinstance(v, dict):
            deep_dict_change_leaves(v, new_val=new_val)
        else:  # Reached bottom
            d[k] = new_val
