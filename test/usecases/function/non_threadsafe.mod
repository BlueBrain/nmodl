NEURON {
    SUFFIX non_threadsafe
    RANGE x
    GLOBAL gbl
}

ASSIGNED {
    gbl
    v
    x
}

STATE {
  z
}

LINEAR lin {
  ~ z = 2
}

FUNCTION x_plus_a(a) {
    x_plus_a = x + a
}

FUNCTION v_plus_a(a) {
    v_plus_a = v + a
}

FUNCTION identity(v) {
    identity = v
}

INITIAL {
    x = 1.0
    gbl = 42.0
}
