: Collection of tricky variable names.

NEURON {
    SUFFIX variable_names
}

STATE {
    is
    be
    count
    as
    def
    del
    elif
    finally
    import
    in
    pass
    raise
    yield

    alpha
    beta
    gamma
    delta
    epsilon
    zeta
    eta
    theta
    iota
    kappa
    lambda
    mu
    nu
    xi
    omicron
    pi
    chi
    psi
    omega
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    ~ is <-> be (1, 1)
    ~ as <-> count (1, 1)
    ~ def <-> del (1, 1)
    ~ elif <-> finally (1, 1)
    ~ import <-> in (1, 1)
    ~ lambda <-> pass (1, 1)
    ~ raise <-> yield (1, 1)
    ~ alpha <-> beta (1, 1)
    ~ gamma <-> delta (1, 1)
    ~ epsilon <-> zeta (1, 1)
    ~ eta <-> theta (1, 1)
    ~ iota <-> kappa (1, 1)
    ~ lambda <-> mu (1, 1)
    ~ nu <-> xi (1, 1)
    ~ omicron <-> pi (1, 1)
    ~ chi <-> psi (1, 1)
    ~ omega <-> alpha (1, 1)
}
