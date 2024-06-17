import numpy as np
from neuron import gui, h
from neuron.units import ms


def test_cacur():
    nseg = 1

    s = h.Section()
    s.insert("cacur")
    s.nseg = nseg

    v_hoc = h.Vector().record(s(0.5)._ref_v)
    t_hoc = h.Vector().record(h._ref_t)

    h.stdinit()
    h.tstop = 100.0 * ms
    h.run()

    v = np.array(v_hoc.as_numpy())
    t = np.array(t_hoc.as_numpy())

    return t, v


def check_solution(t, v):
    # solution is an affine function until t = 1 (ms), afterwards is a constant
    # (1000 - 65)
    solution = 1000 * t - 65
    solution[t > 1] = 1000 - 65
    assert np.allclose(solution, v)


def plot_solution(t, y):
    import matplotlib.pyplot as plt

    plt.plot(t, y)
    plt.xlim(0, 2)
    plt.show()


if __name__ == "__main__":
    t, v = test_cacur()
    check_solution(t, v)
