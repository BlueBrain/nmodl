/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include<string>

#include "codegen/llvm/codegen_llvm_visitor.hpp"


namespace nmodl {
namespace benchmark {

/// A struct to hold LLVM visitor information.
struct LLVMBuildInfo {
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

    LLVMBuildInfo llvm_build_info;

    std::shared_ptr<std::ostream> log_stream;

    /// Disable the specified feature.
    void disable(const std::string& feature, std::vector<std::string>& host_features);

    /// Visits the AST to construct the LLVM IR module.
    void generate_llvm(codegen::CodegenLLVMVisitor& visitor,
                       const std::shared_ptr<ast::Program>& node);

    /// Get the host CPU features in the format:
    ///   +feature,+feature,-feature,+feature,...
    /// where `+` indicates that the feature is enabled.
    std::vector<std::string> get_cpu_features();

    /// Runs the main body of the benchmark, executing the compute kernels.
    void run_benchmark(codegen::CodegenLLVMVisitor& visitor,
                       const std::shared_ptr<ast::Program>& node);

    /// Sets the log output stream (file or console).
    void set_log_output();

  public:
    LLVMBenchmark(const std::string& mod_filename,
                  const std::string& output_dir,
                  LLVMBuildInfo info,
                  int num_experiments,
                  int instance_size,
                  const std::string& backend)
        : mod_filename(mod_filename)
        , output_dir(output_dir)
        , num_experiments(num_experiments)
        , instance_size(instance_size)
        , backend(backend)
        , llvm_build_info(info) {}

    /// Runs the benchmark.
    void benchmark(const std::shared_ptr<ast::Program>& node);
};


}  // namespace runner
}  // namespace nmodl
