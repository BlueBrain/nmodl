import sys
import os

import nmodl.dsl as nmodl
from nmodl import ast, visitor

def main():
    driver = nmodl.NmodlDriver()
    lookup_visitor = visitor.AstLookupVisitor()

    cfg = nmodl.CodeGenConfig()
    cfg.llvm_vector_width = 4
    cfg.llvm_opt_level_ir = 2
    fname = sys.argv[1]
    if len(sys.argv) > 2:  # GPU enabled
        cfg.llvm_math_library = "libdevice"
        cfg.llvm_gpu_name = "nvptx64"
        cfg.llvm_gpu_target_architecture = "sm_70"
        cfg.shared_lib_paths = [os.getenv("CUDA_HOME") + "/nvvm/libdevice/libdevice.10.bc"]
    with open(fname) as f:
        hh = f.read()
        modast = driver.parse_string(hh)
        modname = lookup_visitor.lookup(modast, ast.AstNodeType.SUFFIX)[0].get_node_name()
        jit = nmodl.Jit(cfg)

        res = jit.run(modast, modname, 1000, 1000)
        print(res)


if __name__ == "__main__":
    main()
