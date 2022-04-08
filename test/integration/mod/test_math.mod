NEURON {
    SUFFIX test
    RANGE x, y
}

ASSIGNED { x y }

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
}

DERIVATIVE states {
    m = exp(y) + x ^ 2
}
