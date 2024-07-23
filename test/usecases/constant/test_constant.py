import numpy as np

from neuron import h, gui
from neuron.units import ms

nseg = 1
s = h.Section()
s.insert("constant_mod")

expected = 2.3
assert s(0.5).constant_mod.foo() == expected

# CONSTANTs are not read-only:
expected = 42.0
s(0.5).constant_mod.set_a(42.0)

assert s(0.5).constant_mod.foo() == expected
