/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <chrono>

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "llvm_benchmark.hpp"
#include "test/benchmark/jit_driver.hpp"
#include "llvm/Support/Host.h"

#include "test/unit/codegen/codegen_data_helper.hpp"


namespace nmodl {
namespace benchmark {

void LLVMBenchmark::run(const std::shared_ptr<ast::Program>& node) {
    // create functions
    generate_llvm(node);
    // Finally, run the benchmark and log the measurements.
    run_benchmark(node);
}

void LLVMBenchmark::generate_llvm(const std::shared_ptr<ast::Program>& node) {
    // First, visit the AST to build the LLVM IR module and wrap the kernel function calls.
    auto start = std::chrono::steady_clock::now();
    llvm_visitor.wrap_kernel_functions();
    auto end = std::chrono::steady_clock::now();

    // Log the time taken to visit the AST and build LLVM IR.
    std::chrono::duration<double> diff = end - start;
    logger->info("Created LLVM IR module from NMODL AST in {} sec", diff.count());
}

void LLVMBenchmark::run_benchmark(const std::shared_ptr<ast::Program>& node) {
    // Set the codegen data helper and find the kernels.
    auto codegen_data = codegen::CodegenDataHelper(node, llvm_visitor.get_instance_struct_ptr());
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

    // Benchmark every kernel.
    for (const auto& kernel_name: kernel_names) {
        // For every kernel run the benchmark `num_experiments` times.
        double time_min = std::numeric_limits<double>::max();
        double time_max = 0.0;
        double time_sum = 0.0;
        double time_squared_sum = 0.0;
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
            logger->info("Experiment {} compute time = {:.6f} sec", i, diff.count());

            // Update statistics.
            time_sum += diff.count();
            time_squared_sum += diff.count() * diff.count();
            time_min = std::min(time_min, diff.count());
            time_max = std::max(time_max, diff.count());
        }
        // Log the average time taken for the kernel.
        double time_mean = time_sum / num_experiments;
        logger->info("Average compute time = {:.6f}", time_mean);
        logger->info("Compute time variance = {:g}",
                     time_squared_sum / num_experiments - time_mean * time_mean);
        logger->info("Minimum compute time = {:.6f}", time_min);
        logger->info("Maximum compute time = {:.6f}\n", time_max);
    }
}

}  // namespace benchmark
}  // namespace nmodl
