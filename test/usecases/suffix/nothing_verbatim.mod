NEURON {
    SUFFIX nothing
}

VERBATIM
double forty_two_nothings() {
  return 42.0;
}
ENDVERBATIM

FUNCTION verbatim_forty_two() {
VERBATIM
    return forty_two_nothings();
ENDVERBATIM
}
