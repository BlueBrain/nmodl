NEURON {
    SUFFIX scalar
}

PARAMETER {
    freq = 10
    a = 5
    v1 = -1
    v2 = 5
    v3 = 15
    v4 = 0.8
    v5 = 0.3
    r = 3
    k = 0.2
    alpha = 1.2
    beta = 4.5
    gamma = 2.4
    delta = 7.5
}

STATE {var1 var2 var3 var4 var5}

INITIAL {
    var1 = v1
    var2 = v2
    var3 = v3
    var4 = v4
    var5 = v5
}

BREAKPOINT {
  SOLVE equation METHOD derivimplicit
}


DERIVATIVE equation {
    : eq with a function on RHS
    var1' = -sin(freq * t)
    : simple ODE (nonzero Jacobian)
    var2' = -var2 * a
    : logistic ODE
    var3' = r * var3 * (1 - var3 / k)
    : system of 2 ODEs (predator-prey model)
    var4' = alpha * var4 - beta * var4 * var5
    var5' = delta * var4 * var5 - gamma * var5
}
