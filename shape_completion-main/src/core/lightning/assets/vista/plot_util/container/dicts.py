from collections import OrderedDict
from typing import MutableMapping

from lightning.assets.vista.plot_util.container.algorithms import alphanum_sort_key
from lightning.assets.vista.plot_util.design_patterns import RandomDict


# TODO - migrate the functions here to generic formulation from here:
#   https://github.com/Pithikos/python-reusables/tree/master/recursion
# ----------------------------------------------------------------------------------------------------------------------
#
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


def deep_dict_convert(d, cls=RandomDict):
    rdict = cls()

    for key in sorted(d.keys(), key=alphanum_sort_key):
        value = d[key]
        if isinstance(value, dict):
            rdict[key] = deep_dict_convert(value, cls=cls)
        else:
            rdict[key] = value

    return rdict


def deep_dict_change_leaves(d, new_val):
    for k, v in d.items():
        if isinstance(v, dict):
            deep_dict_change_leaves(v, new_val=new_val)
        else:  # Reached bottom
            d[k] = new_val
