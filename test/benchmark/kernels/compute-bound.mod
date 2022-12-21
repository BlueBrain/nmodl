NEURON {
    SUFFIX compute_bound
    NONSPECIFIC_CURRENT il
    RANGE minf, mtau, gl, el
}

PARAMETER {
    gl = .0003 (S/cm2)	<0,1e9>
    el = -54.3 (mV)
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

BREAKPOINT {
    SOLVE states METHOD cnexp
    il = gl*(v - el)
}

DERIVATIVE states {
     m = exp(m) + exp(minf) + (minf-m)/mtau + m + minf * mtau
}