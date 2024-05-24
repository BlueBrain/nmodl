NEURON {
    SUFFIX recursion
}

: single recursive call
FUNCTION f_simple(n) {
    f_simple = f_simple(n)
}

: multiple recursive calls
FUNCTION fc(n) {
    fc = fc(fc(fc(fc(n))))
}

: multiple recursive calls with different args
FUNCTION fd(n) {
    fd = fd(fc(fd(n) + fc(fd(n))))
}

FUNCTION myfactorial(n) {
    if (n == 0 || n == 1) {
        myfactorial = 1
    } else {
        myfactorial = n * myfactorial(n - 1)
    }
} 

FUNCTION myfib(n) {
    if (n == 0 || n == 1){
        myfib = n
    } else {
        myfib = myfib(n - 1) + myfib(n - 2)
    }
}
