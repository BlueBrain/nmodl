from neuron import h, gui


def make_accessors(mech_name):
    get = getattr(h, f"get_gbl_{mech_name}")
    set = getattr(h, f"set_gbl_{mech_name}")

    return get, set


def check_write_read_cycle(mech_name):
    get, set = make_accessors(mech_name)

    expected = 278.045
    set(expected)

    actual = get()
    assert (
        actual == expected
    ), f"{actual = }, {expected = }, delta = {actual - expected}"


def test_top_local():
    check_write_read_cycle("top_local")


if __name__ == "__main__":
    test_top_local()
