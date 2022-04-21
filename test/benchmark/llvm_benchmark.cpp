/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <chrono>
#include <cmath>
#include <numeric>

#include "llvm_benchmark.hpp"
#include "llvm/Support/Host.h"
#include "test/benchmark/jit_driver.hpp"
#include "utils/logger.hpp"

#include "test/unit/codegen/codegen_data_helper.hpp"


namespace nmodl {
namespace benchmark {

BenchmarkResults LLVMBenchmark::run() {
    // create functions
    generate_llvm();
    // Finally, run the benchmark and log the measurements.
    return run_benchmark();
}

void LLVMBenchmark::generate_llvm() {
    // First, visit the AST to build the LLVM IR module and wrap the kernel function calls.
    auto start = std::chrono::steady_clock::now();
    llvm_visitor.wrap_kernel_functions();
    auto end = std::chrono::steady_clock::now();

    // Log the time taken to visit the AST and build LLVM IR.
    std::chrono::duration<double> diff = end - start;
    logger->info("Created LLVM IR module from NMODL AST in {} sec", diff.count());
}

BenchmarkResults LLVMBenchmark::run_benchmark() {
    // Set the codegen data helper and find the kernels.
    auto codegen_data = codegen::CodegenDataHelper(llvm_visitor.get_instance_struct_ptr());
    std::vector<std::string> kernel_names;
    llvm_visitor.find_kernel_names(kernel_names);

    // Get feature's string and turn them off depending on the cpu.
    std::string cpu_name = cpu == "default" ? llvm::sys::getHostCPUName().str() : cpu;
    logger->info("CPU: {}", cpu_name);

    std::unique_ptr<llvm::Module> m = llvm_visitor.get_module();

    // Create the benchmark runner and initialize it.
    std::string filename = "v" + std::to_string(llvm_visitor.get_vector_width()) + "_" +
                           mod_filename;
    runner::BenchmarkRunner runner(
        std::move(m), filename, output_dir, cpu_name, shared_libs, opt_level_ir, opt_level_codegen);
    runner.initialize_driver();

    BenchmarkResults results{};
    // Benchmark every kernel.
    for (const auto& kernel_name: kernel_names) {
        // For every kernel run the benchmark `num_experiments` times and collect runtimes.
        auto times = std::vector<double>(num_experiments, 0.0);
        for (int i = 0; i < num_experiments; ++i) {
            // Initialise the data.
            auto instance_data = codegen_data.create_data(instance_size, /*seed=*/1);

            // Log instance size once.
            if (i == 0) {
                double size_mbs = instance_data.num_bytes / (1024.0 * 1024.0);
                logger->info("Benchmarking kernel '{}' with {} MBs dataset", kernel_name, size_mbs);
            }

            // Record the execution time of the kernel.
            std::string wrapper_name = "__" + kernel_name + "_wrapper";
            auto start = std::chrono::steady_clock::now();
            runner.run_with_argument<int, void*>(kernel_name, instance_data.base_ptr);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;

            // Log the time taken for each run.
            logger->debug("Experiment {} compute time = {:.6f} sec", i, diff.count());

            times[i] = diff.count();
        }
        // Calculate statistics
        double time_mean = std::accumulate(times.begin(), times.end(), 0.0)/ num_experiments;
        double time_var = std::accumulate(times.begin(), times.end(), 0.0,
                                           [time_mean](const double& pres, const double& e) {
                                               return (e - time_mean)*(e - time_mean); }) / num_experiments;
        double time_stdev = std::sqrt(time_var);
        double time_min = *std::min_element(times.begin(), times.end());
        double time_max = *std::max_element(times.begin(), times.end());
        // Log the average time taken for the kernel.
        logger->info("Average compute time = {:.6f}", time_mean);
        logger->info("Compute time standard deviation = {:8f}", time_stdev);
        logger->info("Minimum compute time = {:.6f}", time_min);
        logger->info("Maximum compute time = {:.6f}\n", time_max);
        results[kernel_name] = {time_mean, time_stdev, time_min, time_max};
    }
    return results;
}

}  // namespace benchmark
}  // namespace nmodl
