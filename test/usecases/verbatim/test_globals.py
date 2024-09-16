from neuron import h, gui

def test_globals():
    s = h.Section()
    s.insert("globals")

    h.gbl_globals = 654.0

    assert h.globals.get_gbl() == h.gbl


if __name__ == "__main__":
    test_globals()
