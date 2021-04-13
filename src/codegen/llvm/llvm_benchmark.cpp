/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "llvm_benchmark.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"

#include <chrono>


namespace nmodl {
namespace benchmark {

void LLVMBenchmark::benchmark(const ast::Program& node) {
    // Run the LLVM visitor first.
    auto llvm_visitor_start = std::chrono::high_resolution_clock::now();
    codegen::CodegenLLVMVisitor visitor(mod_filename,
                                        output_dir,
                                        llvm_info.opt_passes,
                                        llvm_info.use_single_precision,
                                        llvm_info.vector_width);
    visitor.visit_program(node);
    auto llvm_visitor_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> llvm_visitor_diff = llvm_visitor_end - llvm_visitor_start;

    // todo: add runtime benchmark

    if (output_dir != ".") {
        // If the output directory is specified, dump logs to the file.
        std::string filename = output_dir + "/" + mod_filename + ".log";
        std::freopen(filename.c_str(), "w", stdout);
    }

    std::cout << "Created LLVM IR module from NMODL AST in " << std::setprecision(9)
              << llvm_visitor_diff.count() << "\n";
}

}  // namespace benchmark
}  // namespace nmodl
