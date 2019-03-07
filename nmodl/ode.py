# ***********************************************************************
# Copyright (C) 2018-2019 Blue Brain Project
#
# This file is part of NMODL distributed under the terms of the GNU
# Lesser General Public License. See top-level LICENSE file for details.
# ***********************************************************************

import sympy as sp

major, minor = (int(v) for v in sp.__version__.split(".")[:2])
if not ((major >= 1) and (minor >= 2)):
    raise ImportError(f"Requires SympPy version >= 1.2, found {major}.{minor}")


def jacobian_is_linear(jacobian, state_vars):
    for j in jacobian:
        for x in state_vars:
            if j.diff(x).simplify() != 0:
                return False
    return True


def make_unique_prefix(vars, default_prefix="tmp"):
    prefix = default_prefix
    # generate prefix that doesn't match first part
    # of any string in vars
    while True:
        for v in vars:
            # if v is long enough to match prefix
            # and first part of it matches prefix
            if ((len(v) >= len(prefix)) and (v[: len(prefix)] == prefix)):
                # append undescore to prefix, try again
                prefix += "_"
                break
        else:
            # for loop ended without finding possible clash
            return prefix


def solve_ode_system(diff_strings, t_var, dt_var, vars, do_cse=False):
    """Solve system of ODEs, return solution as C code.

    If system is linear, constructs the backwards Euler linear 
    system and solves analytically, optionally also
    with Common Subexpression Elimination if do_cse is true.

    Otherwise, constructs F(x) such that F(x)=0 is solution
    of backwards Euler equation, along with Jacobian of F,
    for use in a non-linear solver such as Newton.

    Args:
        diff_string: list of ODEs e.g. ["x' = a*x", "y' = 3"]
        t_var: name of time variable in NEURON
        dt_var: name of dt variable in NEURON
        vars: set of variables used in expression, e.g. {"x", "y", a"}
        do_cse: if True, do Common Subexpression Elimination

    Returns:
        List of strings containing analytic integral of derivative as C code
        List of strings containing new local variables

    Raises:
        ImportError: if SymPy version is too old (<1.2)
    """

    sympy_vars = {var: sp.symbols(var, real=True) for var in vars}

    # generate prefix for new local vars that avoids clashes
    prefix = make_unique_prefix(vars)

    old_state_vars = []
    for s in diff_strings:
        vstr = s.split("'")[0]
        old_state_var_name = f"{prefix}_{vstr}_old"
        var = sp.symbols(old_state_var_name, real=True)
        sympy_vars[old_state_var_name] = var
        old_state_vars.append(var)

    state_vars = [sp.sympify(s.split("'")[0], locals=sympy_vars) for s in diff_strings]
    diff_eqs = [sp.sympify(s.split("=", 1)[1], locals=sympy_vars) for s in diff_strings]

    t = sp.symbols(t_var, real=True)
    sympy_vars[t_var] = t

    jacobian = sp.Matrix(diff_eqs).jacobian(state_vars)

    dt = sp.symbols(dt_var, real=True)
    sympy_vars[dt_var] = dt

    code = []
    new_local_vars = []

    if jacobian_is_linear(jacobian, state_vars):
        # if linear system: construct implicit euler solution & solve by gaussian elimination
        eqs = []
        for x_new, x_old, dxdt in zip(state_vars, old_state_vars, diff_eqs):
            eqs.append(sp.Eq(x_new, x_old + dt * dxdt))
        for rhs in sp.linsolve(eqs, state_vars):
            for x in old_state_vars:
                new_local_vars.append(sp.ccode(x))
            for x, x_old in zip(state_vars, old_state_vars):
                code.append(f"{sp.ccode(x_old)} = {sp.ccode(x)}")
            if do_cse:
                my_symbols = sp.utilities.iterables.numbered_symbols(
                    prefix=prefix
                )
                sub_exprs, simplified_rhs = sp.cse(
                    rhs, symbols=my_symbols, optimizations="basic", order="canonical"
                )
                for v, _ in sub_exprs:
                    new_local_vars.append(sp.ccode(v))
                for v, e in sub_exprs:
                    code.append(f"{v} = {sp.ccode(e.evalf())}")
                rhs = simplified_rhs[0]
            for v, e in zip(state_vars, rhs):
                code.append(f"{sp.ccode(v)} = {sp.ccode(e.evalf())}")
    else:
        # otherwise: construct implicit euler solution in form F(x) = 0
        # also construct jacobian of this function dF/dx
        eqs = []
        for x_new, x_old, dxdt in zip(state_vars, old_state_vars, diff_eqs):
            eqs.append(x_new - dt * dxdt - x_old)
        for i, x in enumerate(state_vars):
            code.append(f"X[{i}] = {sp.ccode(x)}")
        for i, eq in enumerate(eqs):
            code.append(f"F[{i}] = {sp.ccode(eq.evalf().simplify())}")
        for i, jac in enumerate(sp.eye(jacobian.rows, jacobian.rows) - jacobian * dt):
            code.append(f"J{i//jacobian.rows}[{i%jacobian.rows}] = {sp.ccode(jac.evalf().simplify())}")
        new_local_vars.append("X")
        new_local_vars.append("F")
        for i in range(jacobian.rows):
            new_local_vars.append(f"J{i}")
    return code, new_local_vars


def integrate2c(diff_string, t_var, dt_var, vars, use_pade_approx=False):
    """Analytically integrate supplied derivative, return solution as C code.

    Derivative should be of the form "x' = f(x)",
    and vars should contain the set of all the variables
    referenced by f(x), for example:
    -integrate2c("x' = a*x", "a")
    -integrate2c("x' = a + b*x - sin(3.2)", {"a","b"})

    Args:
        diff_string: Derivative to be integrated e.g. "x' = a*x"
        t_var: name of time variable in NEURON
        dt_var: name of dt variable in NEURON
        vars: set of variables used in expression, e.g. {"x", "a"}
        use_pade_approx: if False:  return exact solution
                         if True:   return (1,1) Pade approx to solution
                                    correct to second order in dt_var

    Returns:
        String containing analytic integral of derivative as C code

    Raises:
        NotImplementedError: if ODE is too hard, or if fails to solve it.
        ImportError: if SymPy version is too old (<1.2)
    """

    # only try to solve ODEs that are not too hard
    ode_properties_require_all = {"separable"}
    ode_properties_require_one_of = {
        "1st_exact",
        "1st_linear",
        "almost_linear",
        "nth_linear_constant_coeff_homogeneous",
        "1st_exact_Integral",
        "1st_linear_Integral",
    }

    # every symbol (a.k.a variable) that SymPy
    # is going to manipulate needs to be declared
    # explicitly
    sympy_vars = {}
    t = sp.symbols(t_var, real=True, positive=True)
    vars = set(vars)
    vars.discard(t_var)
    # the dependent variable is a function of t
    # we use the python variable name x for this
    dependent_var = diff_string.split("=")[0].split("'")[0].strip()
    x = sp.Function(dependent_var, real=True)(t)
    vars.discard(dependent_var)
    # declare all other supplied variables
    sympy_vars = {var: sp.symbols(var) for var in vars}
    sympy_vars[dependent_var] = x
    sympy_vars[t_var] = t

    # parse string into SymPy equation
    diffeq = sp.Eq(
        x.diff(t), sp.sympify(diff_string.split("=", 1)[1], locals=sympy_vars)
    )

    # classify ODE, if it is too hard then exit
    ode_properties = set(sp.classify_ode(diffeq))
    if not ode_properties_require_all <= ode_properties:
        raise NotImplementedError("ODE too hard")
    if len(ode_properties_require_one_of & ode_properties) == 0:
        raise NotImplementedError("ODE too hard")

    # try to find analytic solution
    dt = sp.symbols(dt_var, real=True, positive=True)
    x_0 = sp.symbols(dependent_var, real=True)
    # note dsolve can return a list of solutions, in which case this fails:
    solution = sp.dsolve(diffeq, x, ics={x.subs({t: 0}): x_0}).subs({t: dt}).rhs

    if use_pade_approx:
        # (1,1) order pade approximant, correct to 2nd order in dt,
        # constructed from coefficients of 2nd order taylor expansion
        taylor_series = sp.Poly(sp.series(solution, dt, 0, 3).removeO(), dt)
        _a0 = taylor_series.nth(0)
        _a1 = taylor_series.nth(1)
        _a2 = taylor_series.nth(2)
        solution = (
            (_a0 * _a1 + (_a1 * _a1 - _a0 * _a2) * dt) / (_a1 - _a2 * dt)
        ).simplify()

    # return result as C code in NEURON format
    return f"{sp.ccode(x_0)} = {sp.ccode(solution)}"


def differentiate2c(expression, dependent_var, vars, prev_expressions=None):
    """Analytically differentiate supplied expression, return solution as C code.

    Expression should be of the form "f(x)", where "x" is
    the dependent variable, and the function returns df(x)/dx

    The set vars must contain all variables used in the expression.

    Furthermore, if any of these variables are themselves functions that should
    be substituted before differentiating, they can be supplied in the prev_expressions list.
    Before differentiating each of these expressions will be substituted into expressions,
    where possible, in reverse order - i.e. starting from the end of the list.

    If the result coincides with one of the vars, or the LHS of one of
    the prev_expressions, then it is simplified to this expression.

    Some simple examples of use:
    -differentiate2c("a*x", "x", {"a"}) == "a"
    -differentiate2c("cos(y) + b*y**2", "y", {"a","b"}) == "Dy = 2*b*y - sin(y)"

    Args:
        expression: expression to be differentiated e.g. "a*x + b"
        dependent_var: dependent variable, e.g. "x"
        vars: set of all other variables used in expression, e.g. {"a", "b", "c"}
        prev_expressions: time-ordered list of preceeding expressions
                          to evaluate & substitute, e.g. ["b = x + c", "a = 12*b"]

    Returns:
        String containing analytic derivative of expression (including any substitutions
        of variables from supplied prev_expressions) w.r.t dependent_var as C code.
    """
    prev_expressions = prev_expressions or []
    # every symbol (a.k.a variable) that SymPy
    # is going to manipulate needs to be declared
    # explicitly
    x = sp.symbols(dependent_var, real=True)
    vars = set(vars)
    vars.discard(dependent_var)
    # declare all other supplied variables
    sympy_vars = {var: sp.symbols(var, real=True) for var in vars}
    sympy_vars[dependent_var] = x

    # parse string into SymPy equation
    expr = sp.sympify(expression, locals=sympy_vars)

    # parse previous equations into (lhs, rhs) pairs & reverse order
    prev_eqs = [
        (
            sp.sympify(e.split("=", 1)[0], locals=sympy_vars),
            sp.sympify(e.split("=", 1)[1], locals=sympy_vars),
        )
        for e in prev_expressions
    ]
    prev_eqs.reverse()

    # substitute each prev equation in reverse order: latest first
    for eq in prev_eqs:
        expr = expr.subs(eq[0], eq[1])

    # differentiate w.r.t. x
    diff = expr.diff(x).simplify()

    # if expression is equal to one of the supplied vars, replace with this var
    for v in sympy_vars:
        if (diff - sympy_vars[v]).simplify() == 0:
            diff = sympy_vars[v]
    # or if equal to rhs of one of supplied equations, replace with lhs
    for i_eq, eq in enumerate(prev_eqs):
        # each supplied eq also needs recursive substitution of preceeding statements
        # here, before comparison with diff expression
        expr = eq[1]
        for sub_eq in prev_eqs[i_eq:]:
            expr = expr.subs(sub_eq[0], sub_eq[1])
        if (diff - expr).simplify() == 0:
            diff = eq[0]

    # return result as C code in NEURON format
    return sp.ccode(diff)


def forwards_euler2c(diff_string, dt_var, vars):
    """Return forwards euler solution of diff_string as C code.

    Derivative should be of the form "x' = f(x)",
    and vars should contain the set of all the variables
    referenced by f(x), for example:
    -forwards_euler2c("x' = a*x", "a")
    -forwards_euler2c("x' = a + b*x - sin(3.2)", {"a","b"})

    Args:
        diff_string: Derivative to be integrated e.g. "x' = a*x"
        dt_var: name of dt variable in NEURON
        vars: set of variables used in expression, e.g. {"x", "a"}

    Returns:
        String containing forwards Euler timestep as C code

    Raises:
        ImportError: if SymPy version is too old (<1.2)
    """

    # every symbol (a.k.a variable) that SymPy
    # is going to manipulate needs to be declared
    # explicitly
    sympy_vars = {}
    vars = set(vars)
    dependent_var = diff_string.split("=")[0].split("'")[0].strip()
    x = sp.symbols(dependent_var, real=True)
    vars.discard(dependent_var)
    # declare all other supplied variables
    sympy_vars = {var: sp.symbols(var, real=True) for var in vars}
    sympy_vars[dependent_var] = x

    # parse string into SymPy equation
    diffeq_rhs = sp.sympify(diff_string.split("=", 1)[1], locals=sympy_vars)

    # forwards Euler solution is x + dx/dt * dt
    dt = sp.symbols(dt_var, real=True, positive=True)
    solution = (x + diffeq_rhs * dt).simplify().evalf()

    # return result as C code in NEURON format
    return f"{sp.ccode(x)} = {sp.ccode(solution)}"
