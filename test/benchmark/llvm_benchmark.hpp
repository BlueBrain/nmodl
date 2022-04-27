/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <fstream>
#include <string>

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "gpu_parameters.hpp"
#include "test/benchmark/jit_driver.hpp"
#include "utils/logger.hpp"

#ifdef NMODL_LLVM_CUDA_BACKEND
#include "test/benchmark/cuda_driver.hpp"
#endif

using nmodl::codegen::Platform;
using nmodl::cuda_details::GPUExecutionParameters;

namespace nmodl {
namespace benchmark {

/**
 * \class LLVMBenchmark
 * \brief A wrapper to execute MOD file kernels via LLVM IR backend, and
 * benchmark compile-time and runtime.
 */
class LLVMBenchmark {
  private:
    /// LLVM visitor.
    codegen::CodegenLLVMVisitor& llvm_visitor;

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

    /// Target platform for the code generation.
    Platform platform;

    /// The GPU execution parameters needed to configure the kernels' execution.
    GPUExecutionParameters gpu_execution_parameters;

    /// Optimisation level for IR generation.
    int opt_level_ir;

    /// Optimisation level for machine code generation.
    int opt_level_codegen;

    /// Filestream for dumping logs to the file.
    std::ofstream ofs;

    /// CPU benchmark runner
    std::unique_ptr<runner::BenchmarkRunner> cpu_runner;

#ifdef NMODL_LLVM_CUDA_BACKEND
    /// CUDA benchmark runner
    std::unique_ptr<runner::BenchmarkGPURunner> cuda_runner;
#endif

  public:
    LLVMBenchmark(codegen::CodegenLLVMVisitor& llvm_visitor,
                  const std::string& mod_filename,
                  const std::string& output_dir,
                  std::vector<std::string> shared_libs,
                  int num_experiments,
                  int instance_size,
                  const Platform& platform,
                  int opt_level_ir,
                  int opt_level_codegen)
        : llvm_visitor(llvm_visitor)
        , mod_filename(mod_filename)
        , output_dir(output_dir)
        , shared_libs(shared_libs)
        , num_experiments(num_experiments)
        , instance_size(instance_size)
        , platform(platform)
        , opt_level_ir(opt_level_ir)
        , opt_level_codegen(opt_level_codegen) {}
    LLVMBenchmark(codegen::CodegenLLVMVisitor& llvm_visitor,
                  const std::string& mod_filename,
                  const std::string& output_dir,
                  std::vector<std::string> shared_libs,
                  int num_experiments,
                  int instance_size,
                  const Platform& platform,
                  int opt_level_ir,
                  int opt_level_codegen,
                  const GPUExecutionParameters& gpu_exec_params)
        : llvm_visitor(llvm_visitor)
        , mod_filename(mod_filename)
        , output_dir(output_dir)
        , shared_libs(shared_libs)
        , num_experiments(num_experiments)
        , instance_size(instance_size)
        , platform(platform)
        , opt_level_ir(opt_level_ir)
        , opt_level_codegen(opt_level_codegen)
        , gpu_execution_parameters(gpu_exec_params) {}

    /// Runs the benchmark.
    void run(const std::shared_ptr<ast::Program>& node);

  private:
    /// Visits the AST to construct the LLVM IR module.
    void generate_llvm(const std::shared_ptr<ast::Program>& node);

    /// Runs the main body of the benchmark, executing the compute kernels on CPU or GPU.
    void run_benchmark(const std::shared_ptr<ast::Program>& node);

    /// Sets the log output stream (file or console).
    void set_log_output();
};


}  // namespace benchmark
}  // namespace nmodl
