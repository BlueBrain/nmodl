import math

import numpy as np
from neuron import gui
from neuron import h


def simulate():
    s = h.Section(name="soma")
    s.L = 10
    s.diam = 10
    s.insert("recursion")
    return s(0.5).recursion.myfactorial


def check_solution(f):
    for n in range(10):
        exact, expected = math.factorial(n), f(n)
        assert np.isclose(exact, expected), f"{expected} != {exact}"


if __name__ == "__main__":
    f = simulate()
    check_solution(f)
