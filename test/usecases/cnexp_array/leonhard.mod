NEURON {
    SUFFIX leonhard
}

STATE {
  x
  y[2]
}

INITIAL {
    x = 42.0
    y[0] = 0.1
    y[1] = -1.0
}

BREAKPOINT {
    SOLVE dX METHOD cnexp
}

DERIVATIVE dX {
    x' = (y[0] + y[1])*x
}
