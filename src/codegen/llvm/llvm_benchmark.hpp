/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "codegen/llvm/codegen_llvm_visitor.hpp"


namespace nmodl {
namespace benchmark {

/// A struct to hold LLVM visitor information.
struct LLVMBuildInfo {
    int vector_width;
    bool opt_passes;
    bool use_single_precision;
    std::string vec_lib;
};

/**
 * \class LLVMBenchmark
 * \brief A wrapper to execute MOD file kernels via LLVM IR backend, and
 * benchmark compile-time and runtime.
 */
class LLVMBenchmark {
  private:
    /// Source MOD file name.
    std::string mod_filename;

    /// The output directory for logs and other files.
    std::string output_dir;

    /// Paths to shared libraries.
    std::vector<std::string> shared_libs;

    /// The number of experiments to repeat.
    int num_experiments;

    /// The size of the instance struct for benchmarking.
    int instance_size;

    /// Benchmarking backend
    std::string backend;

    /// Optimisation level for LLVM IR transformations.
    int opt_level_ir;

    /// Optimisation level for machine code generation.
    int opt_level_codegen;

    /// LLVM visitor information.
    LLVMBuildInfo llvm_build_info;

    /// The log output stream (file or stdout).
    std::shared_ptr<std::ostream> log_stream;

    /// Filestream for dumping logs to the file.
    std::ofstream ofs;

  public:
    LLVMBenchmark(const std::string& mod_filename,
                  const std::string& output_dir,
                  std::vector<std::string> shared_libs,
                  LLVMBuildInfo info,
                  int num_experiments,
                  int instance_size,
                  const std::string& backend,
                  int opt_level_ir,
                  int opt_level_codegen)
        : mod_filename(mod_filename)
        , output_dir(output_dir)
        , shared_libs(shared_libs)
        , num_experiments(num_experiments)
        , instance_size(instance_size)
        , backend(backend)
        , llvm_build_info(info)
        , opt_level_ir(opt_level_ir)
        , opt_level_codegen(opt_level_codegen) {}

    /// Runs the benchmark.
    void run(const std::shared_ptr<ast::Program>& node);

  private:
    /// Disables the specified feature in the target.
    void disable(const std::string& feature, std::vector<std::string>& host_features);

    /// Visits the AST to construct the LLVM IR module.
    void generate_llvm(codegen::CodegenLLVMVisitor& visitor,
                       const std::shared_ptr<ast::Program>& node);

    /// Runs the main body of the benchmark, executing the compute kernels.
    void run_benchmark(codegen::CodegenLLVMVisitor& visitor,
                       const std::shared_ptr<ast::Program>& node);

    /// Sets the log output stream (file or console).
    void set_log_output();
};


}  // namespace benchmark
}  // namespace nmodl
