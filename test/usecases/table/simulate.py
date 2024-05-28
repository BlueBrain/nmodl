import numpy as np
from neuron import gui
from neuron import h


def setup_sim():
    section = h.Section()
    section.insert("tbl")

    return section


def check_table(y_no_table, y_table, rtol1=1e-2, rtol2=1e-8):
    assert np.allclose(y_no_table, y_table, rtol=rtol1), f"{y_no_table} != {y_table}"
    # if the table is not used, we should get identical results, but we don't,
    # hence the assert below
    assert not np.allclose(
        y_no_table, y_table, rtol=rtol2
    ), f"{y_no_table} != {y_table}"


def test_function():
    section = setup_sim()

    x = np.linspace(-3, 5, 500)

    func = section(0.5).tbl.example_function

    h.c1_tbl = 1
    h.c2_tbl = 2

    h.usetable_tbl = 0
    y_no_table = np.array([func(i) for i in x])

    h.usetable_tbl = 1
    y_table = np.array([func(i) for i in x])

    check_table(y_table, y_no_table, rtol1=1e-4)

    # verify that the table just "clips" the values outside of the range
    assert func(x[0] - 10) == y_table[0]
    assert func(x[-1] + 10) == y_table[-1]

    # change parameters and verify
    h.c1_tbl = 3
    h.c2_tbl = 4

    h.usetable_tbl = 0
    y_params_no_table = np.array([func(i) for i in x])

    h.usetable_tbl = 1
    y_params_table = np.array([func(i) for i in x])

    check_table(y_params_table, y_params_no_table, rtol1=1e-4)


def test_procedure():
    section = setup_sim()

    x = np.linspace(-4, 6, 300)

    proc = section(0.5).tbl.example_procedure

    def modified_args(arg):
        """
        Pass arg to the procedure and return new values
        """
        proc(arg)
        return section(0.5).tbl.v1, section(0.5).tbl.v2

    h.c1_tbl = 1
    h.c2_tbl = 2

    h.usetable_tbl = 0

    v1_no_table, v2_no_table = np.transpose(np.array([modified_args(i) for i in x]))

    h.usetable_tbl = 1
    v1_table, v2_table = np.transpose(np.array([modified_args(i) for i in x]))

    check_table(v1_table, v1_no_table, rtol1=1e-3)
    check_table(v2_table, v2_no_table, rtol1=1e-3)

    # verify that the table just "clips" the values outside of the range
    v1, v2 = modified_args(x[0] - 10)
    assert v1 == v1_table[0] and v2 == v2_table[0]
    v1, v2 = modified_args(x[-1] + 10)
    assert v1 == v1_table[-1] and v2 == v2_table[-1]

    # change params and verify
    # N.B. since this controls the frequency of the oscillations (as our
    # procedure is a trig function), a higher value of the frequency degrades
    # the accuracy of the interpolation
    h.c1_tbl = 0.1
    h.c2_tbl = 0.3

    h.usetable_tbl = 0
    v1_params_no_table, v2_params_no_table = np.transpose(
        np.array([modified_args(i) for i in x])
    )

    h.usetable_tbl = 1
    v1_params_table, v2_params_table = np.transpose(
        np.array([modified_args(i) for i in x])
    )

    check_table(v1_params_table, v1_params_no_table, rtol1=1e-3)
    check_table(v2_params_table, v2_params_no_table, rtol1=1e-3)


def simulate():
    s = setup_sim()

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
    test_function()
    test_procedure()
    t, v_exact, v_table, v_params, v_params_table = simulate()
    check_solution(v_exact, v_table)
    check_solution(v_params, v_params_table)
