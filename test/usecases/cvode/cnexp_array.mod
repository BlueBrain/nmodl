NEURON {
    SUFFIX cnexp_array
    RANGE z
}

ASSIGNED {
  z[3]
}

STATE {
  x
  s[2]
}

INITIAL {
    x = 42.0
    s[0] = 0.1
    s[1] = -1.0
    z[0] = 0.7
    z[1] = 0.8
    z[2] = 0.9
}

BREAKPOINT {
    SOLVE dX METHOD cnexp
}

DERIVATIVE dX {
    :LOCAL s0_nmodl, s1_nmodl, z0_nmodl, z1_nmodl, z2_nmodl
    :s0_nmodl = s[0]
    :s1_nmodl = s[1]
    :z0_nmodl = z[0]
    :z1_nmodl = z[1]
    :z2_nmodl = z[2]
    x' = (s[0] + s[1])*(z[0]*z[1]*z[2])*x
    :x' = (s0_nmodl + s1_nmodl)*(z0_nmodl*z1_nmodl*z2_nmodl)*x
}
