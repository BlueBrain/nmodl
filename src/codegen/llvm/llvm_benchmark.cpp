/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <chrono>
#include <fstream>

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/jit_driver.hpp"
#include "llvm_benchmark.hpp"
#include "llvm/Support/Host.h"

#include "test/unit/codegen/codegen_data_helper.hpp"


namespace nmodl {
namespace benchmark {

/// Precision for the timing measurements.
static constexpr int PRECISION = 9;

/// Get the host CPU features in the format:
///   +feature,+feature,-feature,+feature,...
/// where `+` indicates that the feature is enabled.
static std::vector<std::string> get_cpu_features() {
    std::string cpu(llvm::sys::getHostCPUName());

    llvm::SubtargetFeatures features;
    llvm::StringMap<bool> host_features;
    if (llvm::sys::getHostCPUFeatures(host_features)) {
        for (auto& f: host_features)
            features.AddFeature(f.first(), f.second);
    }
    return features.getFeatures();
}


void LLVMBenchmark::disable(const std::string& feature, std::vector<std::string>& host_features) {
    for (auto& host_feature: host_features) {
        if (feature == host_feature.substr(1)) {
            host_feature[0] = '-';
            *log_stream << host_feature << "\n";
            return;
        }
    }
}

void LLVMBenchmark::run(const std::shared_ptr<ast::Program>& node) {
    // First, set the output stream for the logs.
    set_log_output();

    // Then, record the time taken for building the LLVM IR module.
    codegen::CodegenLLVMVisitor visitor(mod_filename,
                                        output_dir,
                                        llvm_build_info.opt_passes,
                                        llvm_build_info.use_single_precision,
                                        llvm_build_info.vector_width,
                                        llvm_build_info.vec_lib,
                                        /*add_debug_information=*/true);
    generate_llvm(visitor, node);

    // Finally, run the benchmark and log the measurements.
    run_benchmark(visitor, node);
}

void LLVMBenchmark::generate_llvm(codegen::CodegenLLVMVisitor& visitor,
                                  const std::shared_ptr<ast::Program>& node) {
    // First, visit the AST to build the LLVM IR module and wrap the kernel function calls.
    auto start = std::chrono::high_resolution_clock::now();
    visitor.visit_program(*node);
    visitor.wrap_kernel_functions();
    auto end = std::chrono::high_resolution_clock::now();

    // Log the time taken to visit the AST and build LLVM IR.
    std::chrono::duration<double> diff = end - start;
    *log_stream << "Created LLVM IR module from NMODL AST in " << std::setprecision(PRECISION)
                << diff.count() << "\n\n";
}

void LLVMBenchmark::run_benchmark(codegen::CodegenLLVMVisitor& visitor,
                                  const std::shared_ptr<ast::Program>& node) {
    // Set the codegen data helper and find the kernels.
    auto codegen_data = codegen::CodegenDataHelper(node, visitor.get_instance_struct_ptr());
    std::vector<std::string> kernel_names;
    visitor.find_kernel_names(kernel_names);

    // Get feature's string and turn them off depending on the backend.
    std::vector<std::string> features = get_cpu_features();
    *log_stream << "Backend: " << backend << "\n";
    if (backend == "avx2") {
        // Disable SSE.
        *log_stream << "Disabling features:\n";
        disable("sse", features);
        disable("sse2", features);
        disable("sse3", features);
        disable("sse4.1", features);
        disable("sse4.2", features);
    } else if (backend == "sse2") {
        // Disable AVX.
        *log_stream << "Disabling features:\n";
        disable("avx", features);
        disable("avx2", features);
    }

    std::string features_str = llvm::join(features.begin(), features.end(), ",");
    std::unique_ptr<llvm::Module> m = visitor.get_module();

    // Create the benchmark runner and initialize it.
    std::string filename = "v" + std::to_string(llvm_build_info.vector_width) + "_" + mod_filename;
    runner::BenchmarkRunner runner(std::move(m),
                                   filename,
                                   output_dir,
                                   features_str,
                                   shared_libs,
                                   opt_level_ir,
                                   opt_level_codegen);
    runner.initialize_driver();

    // Benchmark every kernel.
    for (const auto& kernel_name: kernel_names) {
        // Initialise the data.
        auto instance_data = codegen_data.create_data(instance_size, /*seed=*/1);

        double size_mbs = instance_data.num_bytes / (1024.0 * 1024.0);
        *log_stream << "Benchmarking kernel '" << kernel_name << ", with " << size_mbs << " MBs\n";

        // For every kernel run the benchmark `num_experiments` times.
        double time_sum = 0.0;
        for (int i = 0; i < num_experiments; ++i) {
            // Record the execution time of the kernel.
            std::string wrapper_name = "__" + kernel_name + "_wrapper";
            auto start = std::chrono::high_resolution_clock::now();
            runner.run_with_argument<int, void*>(kernel_name, instance_data.base_ptr);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;

            // Log the time taken for each run.
            *log_stream << "Experiment " << i << ": compute time = " << std::setprecision(9)
                        << diff.count() << "\n";

            time_sum += diff.count();
        }
        // Log the average time taken for the kernel.
        *log_stream << "Average compute time = " << std::setprecision(PRECISION)
                    << time_sum / num_experiments << "\n\n";
    }
}

void LLVMBenchmark::set_log_output() {
    // If the output directory is not specified, dump logs to the console.
    if (output_dir == ".") {
        log_stream = std::make_shared<std::ostream>(std::cout.rdbuf());
        return;
    }

    // Otherwise, dump logs to the specified file.
    std::string filename = output_dir + "/" + mod_filename + ".log";
    ofs.open(filename.c_str());

    if (ofs.fail())
        throw std::runtime_error("Error while opening a file '" + filename + "'");

    log_stream = std::make_shared<std::ostream>(ofs.rdbuf());
}

}  // namespace benchmark
}  // namespace nmodl
