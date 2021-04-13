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


namespace nmodl {
namespace benchmark {

void LLVMBenchmark::benchmark(const std::shared_ptr<ast::Program>& node) {
    // Run the LLVM visitor first.
    auto llvm_visitor_start = std::chrono::high_resolution_clock::now();
    codegen::CodegenLLVMVisitor visitor(mod_filename,
                                        output_dir,
                                        llvm_info.opt_passes,
                                        llvm_info.use_single_precision,
                                        llvm_info.vector_width);
    visitor.visit_program(*node);
    visitor.wrap_kernel_function("nrn_state_" + mod_filename);
    auto llvm_visitor_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> llvm_visitor_diff = llvm_visitor_end - llvm_visitor_start;

    if (output_dir != ".") {
        // If the output directory is specified, dump logs to the file.
        std::string filename = output_dir + "/" + mod_filename + ".log";
        std::freopen(filename.c_str(), "w", stdout);
    }

    std::cout << "Created LLVM IR module from NMODL AST in " << std::setprecision(9)
              << llvm_visitor_diff.count() << "\n";

    const auto& generated_instance_struct = visitor.get_instance_struct_ptr();
    auto codegen_data = codegen::CodegenDataHelper(node, generated_instance_struct);

    std::unique_ptr<llvm::Module> m = visitor.get_module();
    runner::Runner runner(std::move(m));

    // Todo: create a switch for the backend arch.

    // Todo: add information about target triple

    double runtime_sum = 0.0;
    for (int i = 0; i < num_experiments; ++i) {
        auto instance_data = codegen_data.create_data(instance_size, /*seed=*/1);

        auto runner_start = std::chrono::high_resolution_clock::now();
        runner.run_with_argument<int, void*>("__nrn_state_" + mod_filename + "_wrapper",
                                             instance_data.base_ptr);
        auto runner_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> runner_diff = runner_end - runner_start;

        std::cout << "Experiment " << i << ": runtime is " << std::setprecision(9)
                  << runner_diff.count() << "\n";

        runtime_sum += runner_diff.count();
    }
    std::cout << "The average runtime is " << std::setprecision(9) << runtime_sum / num_experiments
              << "\n";
}

}  // namespace benchmark
}  // namespace nmodl
