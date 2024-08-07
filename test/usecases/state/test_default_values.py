import numpy as np

from neuron import h, gui


def test_default_values():
    s = h.Section()
    s.insert("default_values")
    mech = s(0.5).default_values

    X_hoc = h.Vector().record(mech._ref_X)
    Y_hoc = h.Vector().record(mech._ref_Y)
    Z_hoc = h.Vector().record(mech._ref_Z)

    h.stdinit()

    X = np.array(X_hoc.as_numpy())
    Y = np.array(Y_hoc.as_numpy())
    Z = np.array(Z_hoc.as_numpy())

    assert X[0] == 2.0
    assert Y[0] == 0.0
    assert Z[0] == 3.0


if __name__ == "__main__":
    test_default_values()
