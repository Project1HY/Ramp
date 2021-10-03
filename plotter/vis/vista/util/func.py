import functools
import inspect
import itertools as it
import os
import traceback
import warnings
from pathlib import Path
from types import FunctionType


# TODO -
#  * Find a smarter "handles_scalars" method
# ----------------------------------------------------------------------------------------------------------------------
#                                                       Func Tools
# ----------------------------------------------------------------------------------------------------------------------
def func_name():
    stack = traceback.extract_stack()
    # (filename, line, procname, text) = stack[-1]
    name = stack[-2][2]
    return name if name != '<module>' else f'<main>:{Path(stack[-2][0]).name}'


def caller_name(order=1):
    # order = 1 -> The caller
    # order = 2 -> The caller before that ..
    assert order >= 1
    idx = -1 * (2 + order)
    stack = traceback.extract_stack()
    # (filename, line, procname, text) = stack[-1]
    try:
        name = stack[idx][2]
        return name if name != '<module>' else f'<main>:{Path(stack[idx][0]).name}'

    except IndexError:
        return None


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Context Changes
# ----------------------------------------------------------------------------------------------------------------------

def handles_scalars(func, ids=(1,)):
    """
    # TODO - This is pretty shitty. Rewrite
    Presuming a single list output.
    Presuming no usage of kwargs
    For proper (and heavy in computations) implementation, see
    https://stackoverflow.com/questions/29318459/python-function-that-handles-scalar-or-arrays
    """

    @functools.wraps(func)
    def extend_to_scalar(*args):
        args = list(args)
        for idx in ids:
            if not (isinstance(args[idx], list) or isinstance(args[idx], tuple)):
                args[idx] = [args[idx]]
        res = func(*args)
        if len(res) == 1:
            return res[0]
        return res

    return extend_to_scalar


def parametrized(dec):
    # https://stackoverflow.com/questions/5929107/decorators-with-parameters
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


@parametrized
def types(f, *cls_types):
    """
    # Type-safe assertive decorator, as suggested in:
    # https://stackoverflow.com/questions/5929107/decorators-with-parameters
    Example:
        @cls_types(str, int)  # arg1 is str, arg2 is int
        def string_multiply(text, times):
            return text * times
    """

    def rep(*args):
        for a, t, n in zip(args, cls_types, it.count()):
            if type(a) is not t:
                raise TypeError('Value %d has not type %s. %s instead' %
                                (n, t, type(a))
                                )
        return f(*args)

    return rep


# ----------------------------------------------------------------------------------------------------------------------
#                                                     Class Query
# ----------------------------------------------------------------------------------------------------------------------
def is_function(obj):
    # Does not include callable like objects
    return isinstance(obj, (staticmethod, classmethod, FunctionType))


def list_class_declared_methods(o):
    # dynasty - parent = class_declared
    # narrow_class - parent_methods = class_declared
    # Only the new methods - not related to the parent class
    parent_methods = list_parent_class_methods(o)
    only_in_class_methods = list_narrow_class_methods(o)
    # Now remove the intersection
    return only_in_class_methods - parent_methods


def list_narrow_class_methods(o):
    # Class Only Methods
    if not inspect.isclass(o):
        o = o.__class__
    return set(x for x, y in o.__dict__.items() if isinstance(y, (FunctionType, classmethod, staticmethod)))


def list_dynasty_class_methods(o):
    # Class + Parent Class Methods
    if not inspect.isclass(o):
        o = o.__class__
    return {func for func in dir(o) if callable(getattr(o, func))}
    # # https://docs.python.org/3/library/inspect.html#inspect.isclass
    # TODO - Many objects inside the class are callable as well - this is a problem. Depends on definition.


def list_parent_class_methods(o):
    if not inspect.isclass(o):
        o = o.__class__

    parent_methods = set()
    for c in o.__bases__:
        parent_methods |= list_dynasty_class_methods(c)
        # parent_methods |= list_parent_class_methods(c) # Recursive Tree DFS - Removed
    return parent_methods


def all_subclasses(cls, non_inclusive=False):
    my_subs = set(cls.__subclasses__())
    if not my_subs:
        return my_subs
    child_subs = set()
    unwanted = set()
    for cls in my_subs:
        curr_child_subs = all_subclasses(cls, non_inclusive)
        if non_inclusive and curr_child_subs:  # If has further sons - don't include it
            unwanted.add(cls)
        child_subs |= curr_child_subs

    my_subs |= child_subs
    if non_inclusive:
        my_subs -= unwanted
    return my_subs


# ----------------------------------------------------------------------------------------------------------------------
#                                                        Module Query
# ----------------------------------------------------------------------------------------------------------------------
def all_variables_by_module_name(module_name):
    from importlib import import_module
    from types import ModuleType  # ,ClassType
    module = import_module(module_name)
    return {k: v for k, v in module.__dict__.items() if
            not (k.startswith('__') or k.startswith('_'))
            and not isinstance(v, ModuleType)}
    # and not isinstance(v,ClassType)}


def import_all(path, error=False):
    """
    Imports all modules in the given path

    Parameters
    ----------
    path : str
        the path where to import the modules from
    error : bool (optional)
        if True, whenever an import cannot be made, an exception will be raised

    Returns
    -------
    None

    Raises
    ------
    ImportError
        if error is set to True and a module cannot be imported
    """

    for module in os.listdir(path):
        if not (module.startswith('__') or module.startswith('.')) and module.endswith('.py'):
            module = module[:-3]
            if error:
                exec('from .{} import *'.format(module))
            else:
                try:
                    exec('from .{} import *'.format(module))
                except ImportError:
                    warnings.warn('Module {} could not be imported'.format(module))


# ----------------------------------------------------------------------------------------------------------------------
#                                                         Misc
# ----------------------------------------------------------------------------------------------------------------------

class ClassNamePrintMeta(type):
    # Good for clear printing of Class Types
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


# ----------------------------------------------------------------------------------------------------------------------
#                                                    Testing Suite
# ----------------------------------------------------------------------------------------------------------------------
#
# class Parent:
#     PARENT_STATIC = 1
#
#     def __init__(self):
#         self.father_inside = 5
#
#     def papa(self):
#         pass
#
#     def mama(self):
#         pass
#
#     @classmethod
#     def parent_class(cls):
#         pass
#
#     @staticmethod
#     def parent_static():
#         pass
#
#
# class Son(Parent):
#     SON_VAR = 1
#
#     def __init__(self):
#         super().__init__()
#         self.son_inside = 1
#
#     def papa(self):
#         pass
#
#     def child(self):
#         pass
#
#     @classmethod
#     def son_class(cls):
#         pass
#
#     @staticmethod
#     def son_static():
#         pass
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def _inner_function():
    print(f'_inner_function - My name is: {func_name()}')
    print(f'_inner_function caller was: {caller_name()}')


def _some_function():
    print(f'_some_function - My name is: {func_name()}')
    print(f'_some_function caller was: {caller_name()}')
    _inner_function()


if __name__ == '__main__':
    print(f'main - My name is: {func_name()}')
    print(f'main caller was: {caller_name()}')
    _some_function()
