NEURON {
  SUFFIX constant_mod
}

CONSTANT {
  a = 2.3
}

PROCEDURE set_a(a_new) {
  a = a_new
}

FUNCTION foo() {
  foo = a
}
