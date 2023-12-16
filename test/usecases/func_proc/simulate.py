from neuron import h

s = h.Section()

s.insert("test_func_proc")

s(0.5).test_func_proc.set_x_42_test_func_proc()

assert s(0.5).test_func_proc.x == 42

s(0.5).test_func_proc.set_x_a_test_func_proc(13.7)

assert s(0.5).test_func_proc.x == 13.7

assert s(0.5).test_func_proc.get_a_42_test_func_proc(42) == 84
