import numpy as np
from neuron import h, gui

s = h.Section(name="soma")
s.L = 10
s.diam = 10
s.insert("tbl")

h.k_tbl = -0.1
h.d_tbl = -40
s(0.5).tbl.gmax = .001

vvec = h.Vector().record(s(0.5)._ref_v)
tvec = h.Vector().record(h._ref_t)
gvec = h.Vector().record(s(0.5).tbl._ref_g)

h.usetable_tbl = 0
h.run()

t = np.array(tvec.as_numpy())
v = np.array(vvec.as_numpy())


std = (vvec.c(), gvec.c(), tvec.c())

h.usetable_tbl = 1
h.run()

t_table = np.array(tvec.as_numpy())
v_table = np.array(vvec.as_numpy())

# the table should be accurate to 1%
assert np.allclose(v, v_table, rtol=1e-2)
