NEURON {
    SUFFIX procedure_test
    THREADSAFE
    RANGE a, b
}

ASSIGNED {
    a
    b
}

PROCEDURE simple_sum(x, y) {
    LOCAL z
    z = x + y
}

PROCEDURE complex_sum(v) {
    LOCAL  alpha, beta, sum
    {
        alpha = .1 * exp(-(v+40))
        beta =  4 * exp(-(v+65)/18)
        sum = alpha + beta
    }
}

PROCEDURE loop_proc(v, t) {
    LOCAL i
    i = 0
    WHILE(i < 10) {
        i = i + 1
    }
}

FUNCTION square(x) {
    LOCAL res
    res = x * x
    square = res
}
