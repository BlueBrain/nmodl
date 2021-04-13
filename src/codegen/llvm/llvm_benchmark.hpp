/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "codegen/llvm/codegen_llvm_visitor.hpp"

#include<string>


namespace nmodl {
namespace benchmark {

/// A struct to hold LLVM visitor information.
struct LLVMInfo {
    int vector_width;
    bool opt_passes;
    bool use_single_precision;
};

/**
 * \class LLVMBenchmark
 * \brief A wrapper to execute MOD file kernels via LLVM IR backend, and
 * benchmark compile-time and runtime.
 */
class LLVMBenchmark {
  private:

    std::string mod_filename;

    std::string output_dir;

    int num_experiments;

    int instance_size;

    std::string backend;

    LLVMInfo llvm_info;

  public:
    LLVMBenchmark(const std::string& mod_filename,
                  const std::string& output_dir,
                  LLVMInfo info,
                  int num_experiments,
                  int instance_size,
                  const std::string& backend)
        : mod_filename(mod_filename)
        , output_dir(output_dir)
        , num_experiments(num_experiments)
        , instance_size(instance_size)
        , backend(backend)
        , llvm_info(info) {}

    /// Runs the benchmark.
    void benchmark(const std::shared_ptr<ast::Program>& node);
};


}  // namespace runner
}  // namespace nmodl
