import inspect
from types import FunctionType
from functools import wraps


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

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


def func_name():
    import traceback
    return traceback.extract_stack(None, 2)[0][2]


def all_variables_by_module_name(module_name):
    from importlib import import_module
    from types import ModuleType  # ,ClassType
    module = import_module(module_name)
    return {k: v for k, v in module.__dict__.items() if
            not (k.startswith('__') or k.startswith('_'))
            and not isinstance(v, ModuleType)}
    # and not isinstance(v,ClassType)}


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


def handles_scalars(func, ids=(1,)):
    """
    Presuming a single list output.
    Presuming no usage of kwargs
    For proper (and heavy in computations) implementation, see
    https://stackoverflow.com/questions/29318459/python-function-that-handles-scalar-or-arrays
    """

    @wraps(func)
    def extend_to_scalar(*args):
        args = list(args)
        for idx in ids:
            if not (isinstance(args[idx],list) or isinstance(args[idx],tuple)):
                args[idx] = [args[idx]]
        res = func(*args)
        if len(res) == 1:
            return res[0]
        return res

    return extend_to_scalar


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class ClassNamePrintMeta(type):
    # Good for clear printing of Class Types
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__

# ----------------------------------------------------------------------------------------------------------------------
#                                                  Testing Suite
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
