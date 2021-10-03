import re

_CONTAINS_A_NUMBER = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def grep(pattern, text, reverse=False):
    """
    Grep in text with regex or simple string

    Examples:
        assert grep('this', 'this is cool') == 'this is cool'
        assert grep('cool', 'this is cool') == 'this is cool'
        assert grep('is', 'this is cool') == 'this is cool'
        assert not grep(r'^is', 'this is cool') == 'this is cool'
        assert grep(r'^this', 'this is cool') == 'this is cool'
        assert not grep(r'$this', 'this is cool')
    """
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if reverse:
            conditional = lambda x: not x
        else:
            conditional = lambda x: x
        if conditional(re.match(pattern, line) or pattern in line):
            new_lines.append(line)
    return '\n'.join(new_lines)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def contains_a_number(s):
    if re.match(_CONTAINS_A_NUMBER, s):
        return True
    else:
        return False
