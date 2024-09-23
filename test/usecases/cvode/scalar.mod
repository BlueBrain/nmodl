NEURON {
    SUFFIX scalar
}

PARAMETER {
    freq = 10
    a = 5
    v1 = -1
    v2 = 5
    v3 = 15
    v4 = -4
}

STATE {var1 var2 var3 var4}

INITIAL {
    var1 = v1
    var2 = v2
    var3 = v3
    var4 = v4
}

BREAKPOINT {
  SOLVE equation METHOD derivimplicit
}


DERIVATIVE equation {
    : eq with a function on RHS
    var1' = -sin(freq * t)
    : simple ODE
    var2' = -var2 * a
    : not-so-simple system of ODEs
    var3' = var4 / var3
    var4' = -var3
}
