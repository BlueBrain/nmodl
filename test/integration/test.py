import nmodl.dsl as nmodl


def run_sympy_solver(mod_string, pade=False):
    # parse NMDOL file (supplied as a string) into AST
    driver = nmodl.NmodlDriver()
    AST = driver.parse_string(mod_string)
    # run SymtabVisitor to generate Symbol Table
    nmodl.symtab.SymtabVisitor().visit_program(AST)
    # constant folding, inlining & local variable renaming passes
    nmodl.visitor.ConstantFolderVisitor().visit_program(AST)
    nmodl.visitor.InlineVisitor().visit_program(AST)
    nmodl.visitor.LocalVarRenameVisitor().visit_program(AST)
    # run SympySolver visitor
    nmodl.visitor.SympySolverVisitor(use_pade_approx=pade).visit_program(AST)
    # return contents of new DERIVATIVE block as list of strings
    return nmodl.to_nmodl(
        nmodl.visitor.AstLookupVisitor().lookup(
            AST, nmodl.ast.AstNodeType.DERIVATIVE_BLOCK
        )[0]
    ).splitlines()[1:-1]

ex1 = """
BREAKPOINT {
    SOLVE states METHOD cnexp
}
DERIVATIVE states {
    m' = 4
}
"""
print("exact solution:\t", run_sympy_solver(ex1, pade=False)[0])
print("pade approx:\t", run_sympy_solver(ex1, pade=True)[0])
