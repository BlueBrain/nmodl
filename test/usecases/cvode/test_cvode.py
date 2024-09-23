import matplotlib.pyplot as plt
import numpy as np
from neuron import gui
from neuron import h
from neuron.units import ms


def simulate(rtol):
    nseg = 1
    mech = "scalar"

    s = h.Section()
    cvode = h.CVode()
    cvode.active(True)
    cvode.atol(1e-10)
    s.insert(mech)
    s.nseg = nseg

    t_hoc = h.Vector().record(h._ref_t)
    var1_hoc = h.Vector().record(getattr(s(0.5), f"_ref_var1_{mech}"))
    var2_hoc = h.Vector().record(getattr(s(0.5), f"_ref_var2_{mech}"))
    var3_hoc = h.Vector().record(getattr(s(0.5), f"_ref_var3_{mech}"))
    var4_hoc = h.Vector().record(getattr(s(0.5), f"_ref_var4_{mech}"))

    h.stdinit()
    h.tstop = 2.0 * ms
    h.run()

    t = np.array(t_hoc.as_numpy())
    var1 = np.array(var1_hoc.as_numpy())
    var2 = np.array(var2_hoc.as_numpy())
    var3 = np.array(var3_hoc.as_numpy())
    var4 = np.array(var4_hoc.as_numpy())

    var1_exact = (
        np.cos(t * getattr(h, f"freq_{mech}"))
        + getattr(h, f"v1_{mech}") * getattr(h, f"freq_{mech}")
        - 1
    ) / getattr(h, f"freq_{mech}")
    var2_exact = getattr(h, f"v2_{mech}") * np.exp(-t * getattr(h, f"v2_{mech}"))

    np.testing.assert_allclose(var1, var1_exact, rtol=rtol)
    np.testing.assert_allclose(var2, var2_exact, rtol=rtol)

    return t, var1, var2, var3, var4


if __name__ == "__main__":
    t, *x = simulate(rtol=1e-5)
    fig, ax = plt.subplots(nrows=len(x))
    for a, val in zip(ax, x):
        a.plot(t, val, ls="", marker="x", markersize=0.1)
    plt.show()
