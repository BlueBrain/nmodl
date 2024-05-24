import numpy as np
from neuron import gui, h


def simulate():
    s = h.Section(name="soma")
    s.L = 10
    s.diam = 10
    s.insert("tbl")

    h.k_tbl = -0.1
    h.d_tbl = -40
    s(0.5).tbl.gmax = 0.001

    vvec = h.Vector().record(s(0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)

    # run without a table
    h.usetable_tbl = 0
    h.run()

    t = np.array(tvec.as_numpy())
    v_exact = np.array(vvec.as_numpy())

    # run with a table
    h.usetable_tbl = 1
    h.run()

    v_table = np.array(vvec.as_numpy())

    # run without a table, and changing params
    h.usetable_tbl = 0
    h.k_tbl = -0.05
    h.d_tbl = -45

    h.run()

    v_params = np.array(vvec.as_numpy())

    # run with a table (same params as above)
    h.usetable_tbl = 1
    h.run()

    v_params_table = np.array(vvec.as_numpy())

    return t, v_exact, v_table, v_params, v_params_table


def check_solution(v, v_table, rtol=1e-2):
    # the table should be accurate to 1%
    assert np.allclose(v, v_table, rtol=rtol), f"{v} != {v_table}, delta: {v - v_table}"


def plot_solution(t, y_exact, y):
    import matplotlib.pyplot as plt

    plt.plot(t, y_exact, label="exact")
    plt.plot(t, y, label="table", ls="--")
    plt.show()


if __name__ == "__main__":
    t, v_exact, v_table, v_params, v_params_table = simulate()
    check_solution(v_exact, v_table)
    check_solution(v_params, v_params_table)
