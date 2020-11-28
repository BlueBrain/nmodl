PROCEDURE state(x, y) {
    LOCAL z
    z = x + y
}

PROCEDURE rates(v) {
    LOCAL  alpha, beta, sum
    {
        alpha = .1 * exp(-(v+40))
        beta =  4 * exp(-(v+65)/18)
    }
    {
        sum = alpha + beta
    }
}
