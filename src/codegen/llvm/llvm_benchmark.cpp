/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "llvm_benchmark.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/jit_driver.hpp"

#include "../test/unit/codegen/codegen_data_helper.hpp"

#include <chrono>
#include <fstream>


namespace nmodl {
namespace benchmark {


/// Precision for the timing measurements.
static constexpr int PRECISION = 9;


void LLVMBenchmark::benchmark(const std::shared_ptr<ast::Program>& node) {
    // First, set the output stream for the logs.
    set_log_output();

    // Then, record the time taken for building the LLVM IR module.
    codegen::CodegenLLVMVisitor visitor(mod_filename,
                                        output_dir,
                                        llvm_build_info.opt_passes,
                                        llvm_build_info.use_single_precision,
                                        llvm_build_info.vector_width);
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

    // \todo: Here should be a switch statement on different backends.
    // Ideally, we want to pick the target triple (from command line?) and set the JIT accordingly.
    // For that, Runner must also take the target triple information? However, this is not strictly
    // necessary as we can just benchmark on different platforms and LLVM will pick up the
    // triple/data layout information automatically.
    std::unique_ptr<llvm::Module> m = visitor.get_module();
    runner::Runner runner(std::move(m));

    // Benchmark every kernel.
    for (const auto& kernel_name: kernel_names) {
        *log_stream << "Benchmarking kernel '" << kernel_name << "'\n";

        // For every kernel run the benchmark `num_experiments` times.
        double time_sum = 0.0;
        for (int i = 0; i < num_experiments; ++i) {
            // Initialise the data.
            auto instance_data = codegen_data.create_data(instance_size, /*seed=*/1);

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
    std::ofstream ofs;

    ofs.open(filename.c_str());

    if (ofs.fail())
        throw std::runtime_error("Error while opening a file '" + filename + "'");

    log_stream = std::make_shared<std::ostream>(ofs.rdbuf());
}

}  // namespace benchmark
}  // namespace nmodl
