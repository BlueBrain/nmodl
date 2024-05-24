import math

import numpy as np
from neuron import gui
from neuron import h


def naive_fib(n: int):
    if n in [0, 1]:
        return n
    return naive_fib(n - 1) + naive_fib(n - 2)


def simulate():
    s = h.Section(name="soma")
    s.L = 10
    s.diam = 10
    s.insert("recursion")
    fact = s(0.5).recursion.myfactorial
    fib = s(0.5).recursion.myfib

    return fact, fib


def check_solution(f, reference):
    for n in range(10):
        exact, expected = reference(n), f(n)
        assert np.isclose(exact, expected), f"{expected} != {exact}"


if __name__ == "__main__":
    fact, fib = simulate()
    check_solution(fact, math.factorial)
    check_solution(fib, naive_fib)
