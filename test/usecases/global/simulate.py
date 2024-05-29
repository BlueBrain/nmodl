import numpy as np

from neuron import h, gui
from neuron.units import ms

nseg = 1

s0 = h.Section()
s0.insert("shared_global")
s0.nseg = nseg

s1 = h.Section()
s1.insert("shared_global")
s1.nseg = nseg

pc = h.ParallelContext()
pc.nthread(2)
pc.set_maxstep(10.0 * ms)
pc.partition(0, h.SectionList([s0]))
pc.partition(1, h.SectionList([s1]))

pc.set_maxstep(10.0)
h.dt = 0.5


y0 = h.Vector().record(s0(0.5).shared_global._ref_y)
y1 = h.Vector().record(s1(0.5).shared_global._ref_y)

s0(0.5).shared_global.z = 3
s1(0.5).shared_global.z = 4

h.ggw_shared_global = 7
h.stdinit()
h.ggw_shared_global = 2
while h.t < 1.0:
    h.fadvance()

y0 = np.array(y0.as_numpy())
y1 = np.array(y1.as_numpy())
w = h.ggw_shared_global

print(y0)
print(y1)
print(w)
