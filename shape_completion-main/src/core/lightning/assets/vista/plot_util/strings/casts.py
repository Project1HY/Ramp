import re


# ----------------------------------------------------------------------------------------------------------------------
#                                                      Casts
# ----------------------------------------------------------------------------------------------------------------------

def try_str2int(s):
    # Assuming s is a single
    try:
        return int(s)
    except ValueError:
        return s


def try_str2float(s):
    try:
        return float(s)
    except ValueError:
        return s


def try_str2num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def try_cast_string(s):
    import ast
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def inclusive_split_on_integer(s):
    """
        Splits the string on any integer sequence and then returns both the
        split in a list, including the integer sequences themselves
    """
    return re.split(r'(\d+)', s)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    L = ["hello", "3", "3.64", "-1", "True", "TRUE", "1e-4"]
    print([try_cast_string(x) for x in L])
