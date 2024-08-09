import numpy as np

from neuron import h, gui


def test_default_values():
    s = h.Section()
    s.insert("default_values")
    mech = s(0.5).default_values

    h.stdinit()

    X = mech.X
    Y = mech.Y
    Z = mech.Z
    A = mech.A

    assert X == 2.0
    assert Y == 0.0
    assert Z == 3.0

    for i in range(3):
        assert A[i] == 4.0


if __name__ == "__main__":
    test_default_values()
