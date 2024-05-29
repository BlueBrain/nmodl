NEURON {
  SUFFIX shared_global
  NONSPECIFIC_CURRENT il
  RANGE y, z
  GLOBAL ggw, ggro, ggp
  THREADSAFE
}

PARAMETER {
  ggp = 9
}

ASSIGNED {
  y
  z
  ggw
  ggro
  il
}

INITIAL {
  printf("INITIAL %g\n", z)

  ggw = 48.0
  y = 10.0
}

BREAKPOINT {
  if(t > 0.33) {
    ggw = ggp
  }

  if(t > 0.66) {
    ggw = z
  }
  y = ggw
  il = 0.0000001 * (v - 10.0)
}
