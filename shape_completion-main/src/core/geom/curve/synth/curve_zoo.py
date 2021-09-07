import numpy as np


def helix(n=100):
    theta = np.linspace(-4 * np.pi, 4 * np.pi, n)
    z = np.linspace(-2, 2, n)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.column_stack((x, y, z))