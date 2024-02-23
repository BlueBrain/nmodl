import nmodl.dsl as nmodl


def test_visitors():
    """
    Make sure all of the env variables are set when running the NMODL Python package
    """

    mod_string = """
    NEURON {
        USEION na READ ena WRITE ina
        RANGE gna
    }
    BREAKPOINT {
        ina = gna*(v - ena)
    }"""

    # parse NMDOL file (supplied as a string) into AST
    driver = nmodl.NmodlDriver()
    AST = driver.parse_string(mod_string)
    # run SymtabVisitor to generate Symbol Table
    nmodl.symtab.SymtabVisitor().visit_program(AST)
    # constant folding, inlining & local variable renaming passes
    nmodl.visitor.ConstantFolderVisitor().visit_program(AST)
    nmodl.visitor.InlineVisitor().visit_program(AST)
    nmodl.visitor.LocalVarRenameVisitor().visit_program(AST)
    # run CONDUCTANCE visitor
    nmodl.visitor.SympyConductanceVisitor().visit_program(AST)
