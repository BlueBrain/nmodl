import numpy as np

from neuron import gui, h
from neuron.units import ms

nseg = 1

s = h.Section()
s.insert("hodhux")
ic = h.IClamp(s(0.5))
ic.delay = 0
ic.dur = 1e9
ic.amp = 10
s.nseg = nseg

v_hoc = h.Vector().record(s(0.5)._ref_v)
t_hoc = h.Vector().record(h._ref_t)

h.stdinit()
h.tstop = 100.0 * ms
h.run()

v = np.array(v_hoc.as_numpy())
t = np.array(t_hoc.as_numpy())
