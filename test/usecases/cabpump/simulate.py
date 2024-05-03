import numpy as np

from neuron import h, gui
from neuron.units import ms

nseg = 1

s = h.Section()
s.insert("cadyn")
s.nseg = nseg

v_hoc = h.Vector().record(s(0.5)._ref_v)
t_hoc = h.Vector().record(h._ref_t)

h.stdinit()
h.tstop = 5.0 * ms
h.run()

v = np.array(v_hoc.as_numpy())
t = np.array(t_hoc.as_numpy())

assert np.allclose(v, -65)
