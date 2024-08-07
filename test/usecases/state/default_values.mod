NEURON {
  SUFFIX default_values
  RANGE Z
  GLOBAL X0, Z0
}

STATE {
    X
    Y
    Z
}

PARAMETER {
  X0 = 2.0
  Z0 = 3.0
}
