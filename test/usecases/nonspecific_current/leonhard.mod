UNITS {
        (mA) = (milliamp)
}

NEURON {
        SUFFIX leonhard
        NONSPECIFIC_CURRENT il
        RANGE c
}

ASSIGNED {
        il (mA/cm2)
}

PARAMETER {
    c = 0.005
}

BREAKPOINT {
        func()
        func_with_v(v)
        func_with_other(c)
        il = c * (v - 1.5)
}

PROCEDURE func() {
}

PROCEDURE func_with_v(v) {
}

PROCEDURE func_with_other(q) {
}
