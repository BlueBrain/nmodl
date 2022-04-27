TITLE hh.mod   squid sodium, potassium, and leak channels
COMMENT
    This is the original Hodgkin-Huxley treatment for the set of sodium,
    potassium, and leakage channels found in the squid giant axon membrane.
    ("A quantitative description of membrane current and its application
    conduction and excitation in nerve" J.Physiol. (Lond.) 117:500-544 (1952).)
    Membrane voltage is in absolute mV and has been reversed in polarity
    from the original HH convention and shifted to reflect a resting potential
    of -65 mV.
    Remember to set celsius=6.3 (or whatever) in your HOC file.
    See squid.hoc for an example of a simulation using this model.
    SW Jaslove  6 March, 1992
ENDCOMMENT
UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}
NEURON {
    SUFFIX hh
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    NONSPECIFIC_CURRENT il
    RANGE gnabar, gkbar, gl, el, gna, gk
    RANGE minf, hinf, ninf, mtau, htau, ntau
    THREADSAFE
}
PARAMETER {
    gnabar = .12 (S/cm2) <0,1e9>
    gkbar = .036 (S/cm2) <0,1e9>
    gl = .0003 (S/cm2) <0,1e9>
    el = -54.3 (mV)
}
STATE {
    m
    h
    n
}
ASSIGNED {
    v (mV)
    celsius (degC)
    ena (mV)
    ek (mV)
    gna (S/cm2)
    gk (S/cm2)
    ina (mA/cm2)
    ik (mA/cm2)
    il (mA/cm2)
    minf
    hinf
    ninf
    mtau (ms)
    htau (ms)
    ntau (ms)
}
BREAKPOINT {
    SOLVE states METHOD cnexp
    gna = gnabar*m*m*m*h
    ina = gna*(v-ena)
    gk = gkbar*n*n*n*n
    ik = gk*(v-ek)
    il = gl*(v-el)
}
INITIAL {
    {
        : inlined rates
        LOCAL alpha, beta, sum, q10, vtrap_in_0, v_in_0
        v_in_0 = v
        q10 = 3*((celsius-6.3)/10)
        alpha = .07*exp(-(v_in_0+65)/20)
        beta = 1/(exp(-(v_in_0+35)/10)+1)
        sum = alpha+beta
        htau = 1/(q10*sum)
        hinf = alpha/sum
        {
            : inlined vtrap
            LOCAL x_in_0, y_in_0
            x_in_0 = alpha
            y_in_0 = alpha
            : no control flow
            vtrap_in_0 = y_in_0*(1-x_in_0/y_in_0/2)
        }
        hinf = vtrap_in_0
    }
    m = minf
    h = hinf
    n = ninf
}
DERIVATIVE states {
    {
        : inlined rates
        LOCAL alpha, beta, sum, q10, vtrap_in_0, v_in_1
        v_in_1 = v
        q10 = 3*((celsius-6.3)/10)
        alpha = .07*exp(-(v_in_1+65)/20)
        beta = 1/(exp(-(v_in_1+35)/10)+1)
        sum = alpha+beta
        htau = 1/(q10*sum)
        hinf = alpha/sum
        {
           : inlined vtrap
            LOCAL x_in_0, y_in_0
            x_in_0 = alpha
            y_in_0 = alpha
            : no control flow
            vtrap_in_0 = y_in_0*(1-x_in_0/y_in_0/2)  
        }
        hinf = vtrap_in_0
    }
    m = m+(1.0-exp(dt*((((-1.0)))/mtau)))*(-(((minf))/mtau)/((((-1.0)))/mtau)-m)
    h = h+(1.0-exp(dt*((((-1.0)))/htau)))*(-(((hinf))/htau)/((((-1.0)))/htau)-h)
    n = n+(1.0-exp(dt*((((-1.0)))/ntau)))*(-(((ninf))/ntau)/((((-1.0)))/ntau)-n)
}
UNITSON
