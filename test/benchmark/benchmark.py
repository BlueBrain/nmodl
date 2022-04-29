import sys

import nmodl.dsl as nmodl
from nmodl import ast, visitor

def main():
    driver = nmodl.NmodlDriver()
    lookup_visitor = visitor.AstLookupVisitor()

    cfg = nmodl.CodeGenConfig()
    cfg.llvm_vector_width = 4
    cfg.llvm_opt_level_ir = 2
    cfg.nmodl_ast = True
    fname = sys.argv[1]
    with open(fname) as f:
        hh = f.read()
        modast = driver.parse_string(hh)
        modname = lookup_visitor.lookup(modast, ast.AstNodeType.SUFFIX)[0].get_node_name()
        jit = nmodl.Jit(cfg)

        res = jit.run(modast, modname, 1000, 1000)
        print(res)


if __name__ == "__main__":
    main()
