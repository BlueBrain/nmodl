import numpy as np
import nmodl.dsl as nmodl
from nmodl import ast, visitor

# HH and expsyn mod files
models = ["hh.mod", "expsyn.mod"]

# parse NMODL mechanisms, create ASTs and find their SUFFIX
models_ast = []
models_name = []
driver = nmodl.NmodlDriver()
lookup_visitor = visitor.AstLookupVisitor()
for mod in models:
    models_ast.append(driver.parse_file(mod))
    models_name.append(lookup_visitor.lookup(models_ast[-1], ast.AstNodeType.SUFFIX)[0].get_node_name())

# code generation and JIT configuration
cfg = nmodl.CodeGenConfig()
cfg.llvm_vector_width = 4
cfg.llvm_opt_level_ir = 3
cfg.nmodl_ast = True
jit = nmodl.Jit(cfg)

# simulation configuration
dt = 0.025
tstop = 1000
instances_number = 10000

# representative simulator loop
for t in np.arange(0, tstop, dt):
    for i, modast in enumerate(models_ast):
        jit.run(modast, models_name[i], instances_number)
