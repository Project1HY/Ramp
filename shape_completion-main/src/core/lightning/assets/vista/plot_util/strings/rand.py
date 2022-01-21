import random


def random_string(n):
    """Generates a random alphanumeric (lower case only) string of length n."""
    if n < 0:
        return ''
    return ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz')
                   for _ in range(n))
