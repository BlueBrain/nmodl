NEURON {
  SUFFIX default_values
  RANGE Z
  GLOBAL X0, Z0, A0
}

STATE {
    X
    Y
    Z
    A[3]
}

PARAMETER {
  X0 = 2.0
  Z0 = 3.0
  A0 = 4.0
}
