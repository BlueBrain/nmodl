UNITS {
        (mA) = (milliamp)
}

NEURON {
        SUFFIX leonhard2
        NONSPECIFIC_CURRENT il
}

ASSIGNED {
        il (mA/cm2)
}

BREAKPOINT {
        il = 0.5 * (v - 1.5)
}
