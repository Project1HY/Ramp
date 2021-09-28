from collections.abc import Sequence,Iterable
from copy import deepcopy
from itertools import chain
import numpy as np

from util.strings import try_cast_string


# ----------------------------------------------------------------------------------------------------------------------
#                                                     List Query
# ----------------------------------------------------------------------------------------------------------------------
def n_elements_in_nested_list(nested_list):
    """ Get number of elements in a nested list"""
    count = 0
    # Iterate over the list
    for elem in nested_list:
        # Check if type of element is list
        if isinstance(elem, Sequence):
            # Again call this function to get the size of this element
            count += n_elements_in_nested_list(elem)
        else:
            count += 1
    return count


def linear_search(array, targets):
    """
    search the (first) occurrence of each item from the list of targets and return it's index, or -1 if it is not found
    # >>> linear_search([1, 2, 1, 4], [1, 4])
    array([0, 3])
    # >>> linear_search([1, 2, 1, 4], [5, 2, 1])
    array([-1, 1, 0])
    """
    array = np.asarray(array)
    results = np.full(len(targets), -1, np.int)
    for i, target_i in enumerate(targets):
        s = np.where(array == target_i)[0]
        if len(s) > 0:
            results[i] = s[0]
    return results


def first(iterable, condition=lambda x: True):
    """
    * Returns the first item in the `iterable` that satisfies the `condition`.
    * If the condition is not given, returns the first item of the iterable.
    * Raises `StopIteration` if no item satisfying the condition is found.
    """
    return next(x for x in iterable if condition(x))


def all_equal(elements):
    """
    :param elements: A collection of things
    :return: True if all things are equal, otherwise False.  (Note that an empty list of elements returns true, just as all([]) is True
    """
    element_iterator = iter(elements)
    try:
        first = next(element_iterator)  # Will throw exception
    except StopIteration:
        return True
    return all(a == first for a in element_iterator)


def all_equal_length(collection_if_collections):
    """
    :param collection_if_collections: A collection of collections
    :return: True if all collections have equal length, otherwise False
    """
    return all_equal([len(c) for c in collection_if_collections])


def all_unique(lst):
    """
    :param lst: A python iterable such as a list or tuple
    :return: bool: True if all elements are unique, or false otherwise
    """
    seen = set()
    return not any(i in seen or seen.add(i) for i in lst)


def unique(*lsts):
    return list(set(chain(*lsts)))


def find_majority(k):
    # http://stackoverflow.com/questions/20038011/trying-to-find-majority-element-in-a-list
    m = {}
    max_val = ('', 0)  # (occurring element, occurrences)
    for n in k:
        if n in m:
            m[n] += 1
        else:
            m[n] = 1
        if m[n] > max_val[1]:
            max_val = (n, m[n])
    return max_val[0]


# ---------------------------------------------------------------------------------------------------------------------#
#                                                     List Synthesis
# ---------------------------------------------------------------------------------------------------------------------#

def n_empty_lists(n):
    return [[] for _ in range(n)]


# ----------------------------------------------------------------------------------------------------------------------
#                                                     List Operators
# ----------------------------------------------------------------------------------------------------------------------

def stringify(lst):
    return [str(ele) for ele in lst]


def destringfy(lst):
    return [try_cast_string(ele) for ele in lst]


def flatten(lst_of_lsts):
    """
    :param lst_of_lsts: A list of lists
    :return: A list composed of all the elements of the internal lists of ll, in the order the lists were supplied
    """
    return [item for sublist in lst_of_lsts for item in sublist]


def list_dup(lst, n):
    return [deepcopy(lst) for _ in range(n)]


def split_frac(lst, fracs):
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
    splits *= len(lst)

    # Turn double indices into integers.
    # CAUTION: numpy rounds to closest EVEN number when a number is halfway
    # between two integers. So 0.5 will become 0 and 1.5 will become 2!
    # If you want to round up in all those cases, do
    # splits += 0.5 instead of round() before casting to int
    splits += 0.5
    splits = splits.astype(np.int)

    return np.split(lst, splits)


def to_list(lst, encapsulate_none=True):
    if not encapsulate_none and lst is None:
        return []
    if isinstance(lst, list):
        return lst
    elif isinstance(lst, Sequence):
        return list(lst)
    else:  # Presuming Scalar
        return [lst]


def multishuffle(*args):
    # TODO - Add in option for different ordering per list?
    """
    Receive an arbitrary amount of lists and shuffles them by the same order
    :return: A list of the received lists, shuffled
    """
    length = len(args[0])
    order = np.random.permutation(length)
    shuffled_lists = []
    for lst in args:
        shuffled_lists.append(lst[order])

    return shuffled_lists


def insert_at(list1, list2, indices):
    """
    Create a new list by insert elements from list 2 into list 1 at the given indices.
    (Note: this leaves list1 and list2 unchanged, unlike list.insert)
    :param list1: A list
    :param list2: Another list
    :param indices: The indices of list1 into which elements from list2 will be inserted.
    :return: A new list with len(list1)+len(list2) elements.
    """
    list3 = []
    assert len(list2) == len(
        indices), f'List 2 has {len(list2)} elements, but you provided {len(indices)} indices. ' \
                  f'They should have equal length'
    index_iterator = iter(sorted(indices))
    list_2_iter = iter(list2)
    if len(list2) == 0:
        return list3
    next_ix = next(index_iterator)

    iter_stopped = False
    for i in range(len(list1) + 1):
        while i == next_ix:
            list3.append(next(list_2_iter))
            try:
                next_ix = next(index_iterator)
            except StopIteration:
                next_ix = None
                iter_stopped = True
        if i < len(list1):
            list3.append(list1[i])

    assert iter_stopped, 'Not all elements from list 2 got used!'
    return list3


# ---------------------------------------------------------------------------------------------------------------------#
#                                                 Iterations, Chunking, Zipping
# ---------------------------------------------------------------------------------------------------------------------#

def chunks(n, lst):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def multichunck(n, *lsts):
    n_lists = len(lsts)
    for i in range(0, len(lsts[0]), n):
        chunck = []
        for j in range(n_lists):
            chunck.append(lsts[j][i:i + n])
        yield tuple(chunck)


def inique(iterable):
    """ Generate unique elements from `iterable`

    Parameters
    ----------
    iterable : iterable

    Returns
    -------
    gen : generator
       generator that yields unique elements from `iterable`

    Examples
    --------
    # >>> tuple(inique([0, 1, 2, 0, 2, 3]))
    (0, 1, 2, 3)
    """
    history = []
    for val in iterable:
        if val not in history:
            history.append(val)
            yield val


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Misc
# ----------------------------------------------------------------------------------------------------------------------

def list_cluster_ranges(lst):
    """Given a sorted 1D input array `lst`, e.g., [0 0, 1, 2, 3, 4, 4, 4], this
    routine returns the indices where the blocks of equal integers start and
    how long the blocks are.
    """
    # https://stackoverflow.com/a/50394587/353337
    m = np.concatenate([[True], lst[:-1] != lst[1:], [True]])
    idx = np.flatnonzero(m)
    return idx[:-1], np.diff(idx)


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Test Suite
# ----------------------------------------------------------------------------------------------------------------------
def _split_frac_test():
    import random
    for i in range(1, 10000):
        indices = list(range(i))
        n_split = random.randint(1, 20)
        n_dirac = random.randint(1, 1000 * 1000) / 1000
        splits = np.random.dirichlet(np.ones(n_split) * n_dirac, size=1) * 3 / 4
        # splits = [0.05,0.05,0.9]
        assert sum(map(len, split_frac(indices, splits))) == i, f"{i}"


def _unique_test():
    a = ['a', 'b', 'a']
    b = ['a', 'b', 'c']
    print(unique(a))
    print(unique(a, b))


if __name__ == '__main__':
    _unique_test()
