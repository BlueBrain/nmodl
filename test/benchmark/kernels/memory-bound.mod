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
    il (mA/cm2)
}

PARAMETER {
    gl = .0003 (S/cm2)	<0,1e9>
    el = -54.3 (mV)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    il = gl*(v - el)
}

DERIVATIVE states {
     m =  (minf-m)/mtau    
}
