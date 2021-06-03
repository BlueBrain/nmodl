NEURON {
    SUFFIX hh
    NONSPECIFIC_CURRENT il
    RANGE x, minf, mtau, gl, el
    USEION na WRITE nai
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
     m =  (minf-m)/mtau    
}
