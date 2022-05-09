import argparse
import sys
import os

import nmodl.dsl as nmodl
from nmodl import ast, visitor

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark test script for NMODL.')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Enable GPU JIT execution')
    parser.add_argument('--vec', type=int, default=1,
                        help='Vector width for CPU execution')
    parser.add_argument('--file', type=str,
                        help='NMODL file to benchmark')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_arguments()

    driver = nmodl.NmodlDriver()
    lookup_visitor = visitor.AstLookupVisitor()

    cfg = nmodl.CodeGenConfig()
    cfg.llvm_vector_width = args.vec
    cfg.llvm_opt_level_ir = 2
    cfg.nmodl_ast = True
    fname = args.file
    if args.gpu:  # GPU enabled
        cfg.llvm_math_library = "libdevice"
        cfg.llvm_gpu_name = "nvptx64"
        cfg.llvm_gpu_target_architecture = "sm_70"
        if not os.environ.get("CUDA_HOME"):
            raise RuntimeError("CUDA_HOME environment variable not set")
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
