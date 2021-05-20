NEURON {
    SUFFIX hh
    NONSPECIFIC_CURRENT il
    RANGE minf, mtau, gl, el
}

STATE {
    m
}

ASSIGNED {
    v (mV)
    minf
    mtau (ms)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    il = gl*(v - el)
}

DERIVATIVE states {
     m = exp(m) + exp(minf) + (minf-m)/mtau + m + minf * mtau
}
