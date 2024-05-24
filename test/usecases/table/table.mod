NEURON {
    SUFFIX tbl
    NONSPECIFIC_CURRENT i
    RANGE e, g, gmax
    GLOBAL k, d
}

PARAMETER {
    e = 0
    gmax = 0
    k = .1
    d = -50
}

ASSIGNED {
    g
    i
    v
    sig
}

BREAKPOINT {
    sigmoid1(v)
    g = gmax * sig
    i = g*(v - e)
}

PROCEDURE sigmoid1(v) {
    TABLE sig DEPEND k, d FROM -127 TO 128 WITH 155
    sig = 1/(1 + exp(k*(v - d)))
}
