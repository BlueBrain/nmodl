import numpy as np

from neuron import h, gui
from neuron.units import ms

nseg = 1

s = h.Section()
s.insert("cnexp_scalar")
s.nseg = nseg

x_hoc = h.Vector().record(s(0.5)._ref_x_cnexp_scalar)
t_hoc = h.Vector().record(h._ref_t)

h.stdinit()
h.tstop = 5.0 * ms
h.run()

x = np.array(x_hoc.as_numpy())
t = np.array(t_hoc.as_numpy())

x0 = 42.0
x_exact = 42.0 * np.exp(-t)
rel_err = np.abs(x - x_exact) / x_exact

assert np.all(rel_err < 1e-12)
