/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <fstream>
#include <map>
#include <string>
#include <tuple>

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
 * map of {name: [avg, stdev, min, max]}
 */
using BenchmarkResults = std::map<std::string, std::tuple<double, double, double, double>>;

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

    /// Benchmark external kernel
    std::string external_kernel_library;

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
                  int opt_level_codegen,
                  std::string external_kernel_library)
        : llvm_visitor(llvm_visitor)
        , mod_filename(mod_filename)
        , output_dir(output_dir)
        , shared_libs(shared_libs)
        , num_experiments(num_experiments)
        , instance_size(instance_size)
        , platform(platform)
        , opt_level_ir(opt_level_ir)
        , opt_level_codegen(opt_level_codegen)
        , external_kernel_library(external_kernel_library) {}
    LLVMBenchmark(codegen::CodegenLLVMVisitor& llvm_visitor,
                  const std::string& mod_filename,
                  const std::string& output_dir,
                  std::vector<std::string> shared_libs,
                  int num_experiments,
                  int instance_size,
                  const Platform& platform,
                  int opt_level_ir,
                  int opt_level_codegen,
                  std::string external_kernel_library,
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
        , external_kernel_library(external_kernel_library)
        , gpu_execution_parameters(gpu_exec_params) {}

    /// Runs the main body of the benchmark, executing the compute kernels.
    BenchmarkResults run();

  private:

    /// Sets the log output stream (file or console).
    void set_log_output();
};


}  // namespace benchmark
}  // namespace nmodl
