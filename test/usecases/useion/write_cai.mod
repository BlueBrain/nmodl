NEURON {
    SUFFIX style_ion
    USEION ca WRITE cai
}

ASSIGNED {
    cai
}

INITIAL {
    cai = 1124.0
}
