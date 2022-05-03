/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <chrono>

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "llvm_benchmark.hpp"
#include "test/benchmark/cuda_driver.hpp"
#include "test/benchmark/jit_driver.hpp"
#include "llvm/Support/Host.h"

#include "ext_kernel.hpp"
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

void checkCudaErrors(cudaError error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
                "CUDA Execution Error: {}\n"_format(cudaGetErrorString(error)));
    }
}

void* copy_instance_data_gpu(const codegen::CodegenInstanceData& data) {
    void* dev_base_ptr;
    const auto ptr_vars_size = data.num_ptr_members * sizeof(double*);
    auto scalar_vars_size = 0;
    const auto num_scalar_vars = data.members.size() - data.num_ptr_members;
    for (int i = 0; i < num_scalar_vars; i++) {
        scalar_vars_size += data.members_size[i+data.num_ptr_members];
    }
    checkCudaErrors(cudaMalloc(&dev_base_ptr, ptr_vars_size + scalar_vars_size));
    for (auto i = 0; i < data.num_ptr_members; i++) {
        // Allocate a vector with the correct size
        void* dev_member_ptr;
        auto size_of_var = data.members_size[i];
        checkCudaErrors(cudaMalloc(&dev_member_ptr, size_of_var*data.num_elements));
        checkCudaErrors(cudaMemcpy(dev_member_ptr, data.members[i], size_of_var*data.num_elements, cudaMemcpyHostToDevice));
        // Copy the pointer addresses to the struct
        auto offseted_place = (char*)dev_base_ptr+data.offsets[i];
        checkCudaErrors(cudaMemcpy(offseted_place, &dev_member_ptr, sizeof(double*), cudaMemcpyHostToDevice));
    }
    // memcpy the scalar values
    auto offseted_place_dev = (char*)dev_base_ptr+data.offsets[data.num_ptr_members];
    auto offseted_place_host = (char*)(data.base_ptr)+data.offsets[data.num_ptr_members];
    checkCudaErrors(cudaMemcpy(offseted_place_dev, offseted_place_host, scalar_vars_size, cudaMemcpyHostToDevice));
    return dev_base_ptr;
}

void copy_instance_data_host(codegen::CodegenInstanceData& data, void* dev_base_ptr) {
    const auto ptr_vars_size = data.num_ptr_members * sizeof(double*);
    auto scalar_vars_size = 0;
    const auto num_scalar_vars = data.members.size() - data.num_ptr_members;
    for (int i = 0; i < num_scalar_vars; i++) {
        scalar_vars_size += data.members_size[i+data.num_ptr_members];
    }
    const auto host_base_ptr = data.base_ptr;
    for (auto i = 0; i < data.num_ptr_members; i++) {
        auto size_of_var = data.members_size[i];
        void* offset_dev_ptr = (char*)dev_base_ptr+data.offsets[i];
        void* gpu_offset_addr;
        checkCudaErrors(cudaMemcpy(&gpu_offset_addr, offset_dev_ptr, sizeof(double*), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(data.members[i], gpu_offset_addr, size_of_var*data.num_elements, cudaMemcpyDeviceToHost));
    }
    // memcpy the scalar values
    void* offseted_place_dev = (char*)dev_base_ptr+data.offsets[data.num_ptr_members];
    void* offseted_place_host = (char*)(data.base_ptr)+data.offsets[data.num_ptr_members];
    checkCudaErrors(cudaMemcpy(offseted_place_host, offseted_place_dev, scalar_vars_size, cudaMemcpyDeviceToHost));
}

void LLVMBenchmark::run_benchmark(const std::shared_ptr<ast::Program>& node) {
    // Set the codegen data helper and find the kernels.
    auto codegen_data = codegen::CodegenDataHelper(node, llvm_visitor.get_instance_struct_ptr());
    std::vector<std::string> kernel_names;
    llvm_visitor.find_kernel_names(kernel_names);

    // Get feature's string and turn them off depending on the cpu.
    std::string backend_name;
#ifdef NMODL_LLVM_CUDA_BACKEND
    if (platform.is_CUDA_gpu()) {
        backend_name = platform.get_name();
    } else {
#endif
        backend_name = platform.get_name() == "default" ? llvm::sys::getHostCPUName().str()
                                                        : platform.get_name();
#ifdef NMODL_LLVM_CUDA_BACKEND
    }
#endif
    logger->info("Backend: {}", backend_name);

    std::unique_ptr<llvm::Module> m = llvm_visitor.get_module();

    // Create the benchmark runner and initialize it.
#ifdef NMODL_LLVM_CUDA_BACKEND
    if (platform.is_CUDA_gpu()) {
        std::string filename = "cuda_" + mod_filename;
        cuda_runner = std::make_unique<runner::BenchmarkGPURunner>(
            std::move(m), filename, output_dir, shared_libs, opt_level_ir, opt_level_codegen);
        cuda_runner->initialize_driver(platform);
    } else {
#endif
        std::string filename = "v" + std::to_string(llvm_visitor.get_vector_width()) + "_" +
                               mod_filename;
        cpu_runner = std::make_unique<runner::BenchmarkRunner>(std::move(m),
                                                               filename,
                                                               output_dir,
                                                               backend_name,
                                                               shared_libs,
                                                               opt_level_ir,
                                                               opt_level_codegen);
        cpu_runner->initialize_driver();
#ifdef NMODL_LLVM_CUDA_BACKEND
    }
#endif

    if (external_kernel) {
        // benchmark external kernel
        logger->info("Benchmarking external kernel");
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
                logger->info("Benchmarking kernel nrn_state_hh_ext with {} MBs dataset", size_mbs);
            }

            // Record the execution time of the kernel.
            auto start = std::chrono::steady_clock::now();
            nrn_state_hh_ext(instance_data.base_ptr);
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
    } else {
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
#ifdef NMODL_LLVM_CUDA_BACKEND
                void* dev_ptr;
                if (platform.is_CUDA_gpu()) {
                    dev_ptr = copy_instance_data_gpu(instance_data);
                }
#endif
                // Log instance size once.
                if (i == 0) {
                    double size_mbs = instance_data.num_bytes / (1024.0 * 1024.0);
                    logger->info("Benchmarking kernel '{}' with {} MBs dataset", kernel_name, size_mbs);
                }

                // Record the execution time of the kernel.
                std::string wrapper_name = "__" + kernel_name + "_wrapper";
                auto start = std::chrono::steady_clock::now();
#ifdef NMODL_LLVM_CUDA_BACKEND
                if (platform.is_CUDA_gpu()) {
                    cuda_runner->run_with_argument<void*>(wrapper_name,
                                                          dev_ptr,
                                                          gpu_execution_parameters);
                } else {
#endif
                    cpu_runner->run_with_argument<int, void*>(wrapper_name, instance_data.base_ptr);
#ifdef NMODL_LLVM_CUDA_BACKEND
                }
#endif
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> diff = end - start;
#ifdef NMODL_LLVM_CUDA_BACKEND
                if (platform.is_CUDA_gpu()) {
                    copy_instance_data_host(instance_data, dev_ptr);
                }
#endif
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
}

}  // namespace benchmark
}  // namespace nmodl
