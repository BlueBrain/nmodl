NEURON {
    SUFFIX test_func_proc
    RANGE x
    GLOBAL y
    THREADSAFE
}

PARAMETER {
    y = 42
}

ASSIGNED {
    x
}

PROCEDURE set_x_42() {
    x = 42
}

PROCEDURE set_x_a(a) {
    x = a
}

FUNCTION get_a_42(a) {
    get_a_42 = a + 42
}

PROCEDURE set_y_42() {
    LOCAL a
    a = y
}
