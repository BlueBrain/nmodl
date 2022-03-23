/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \dir
 * \brief Implementation of CUDA and NVVM-based execution engine to run functions from MOD files
 *
 * \file
 * \brief \copybrief nmodl::runner::CUDADriver
 */

#ifdef NMODL_LLVM_CUDA_BACKEND

#include <memory>
#include <string>

#include "llvm/IR/Module.h"

#include "benchmark_info.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "gpu_parameters.hpp"
#include "nvvm.h"

using nmodl::cuda_details::GPUExecutionParameters;

namespace nmodl {
namespace runner {

struct DeviceInfo {
    int count;
    std::string name;
    int compute_version_major;
    int compute_version_minor;
};

/**
 * @brief Throw meaningful error in case CUDA API call fails
 *
 * Checks whether a call to the CUDA API was succsful and if not it throws a runntime_error with
 * the error message from CUDA.
 *
 * @param err Return value of the CUDA API call
 */
void checkCudaErrors(CUresult err);

/**
 * @brief Throw meaningful error in case NVVM API call fails
 *
 * Checks whether a call to the NVVM API was succsful and if not it throws a runntime_error with
 * the error message from NVVM.
 *
 * @param err Return value of the NVVM API call
 */
void checkNVVMErrors(nvvmResult err);

/**
 * \class CUDADriver
 * \brief Driver to execute a MOD file function via the CUDA and NVVM backend.
 */
class CUDADriver {
    /// LLVM IR module to execute.
    std::unique_ptr<llvm::Module> module;
    nvvmProgram prog;
    CUdevice device;
    CUmodule cudaModule;
    CUcontext context;
    CUfunction function;
    CUlinkState linker;
    DeviceInfo device_info;
    std::string ptx_compiled_module;

    void checkCudaErrors(CUresult err);
    void checkNVVMErrors(nvvmResult err);
    void load_libraries(BenchmarkInfo* benchmark_info);

  public:
    explicit CUDADriver(std::unique_ptr<llvm::Module> m)
        : module(std::move(m)) {}

    /// Initializes the CUDA GPU JIT driver.
    void init(const std::string& gpu, BenchmarkInfo* benchmark_info = nullptr);

    /// Lookups the entry-point without arguments in the CUDA module and executes it.
    void execute_without_arguments(const std::string& entry_point,
                                   const GPUExecutionParameters& gpu_execution_parameters) {
        // Get kernel function
        checkCudaErrors(cuModuleGetFunction(&function, cudaModule, entry_point.c_str()));

        // Kernel launch
        void* kernel_parameters[] = {};
        checkCudaErrors(cuLaunchKernel(function,
                                       gpu_execution_parameters.gridDimX,
                                       gpu_execution_parameters.gridDimY,
                                       gpu_execution_parameters.gridDimY,
                                       gpu_execution_parameters.blockDimX,
                                       gpu_execution_parameters.blockDimY,
                                       gpu_execution_parameters.blockDimY,
                                       gpu_execution_parameters.sharedMemBytes,
                                       nullptr,
                                       kernel_parameters,
                                       nullptr));
        cudaDeviceSynchronize();
    }

    /// Lookups the entry-point with arguments in the CUDA module and executes it.
    template <typename ArgType>
    void execute_with_arguments(const std::string& entry_point,
                                ArgType arg,
                                const GPUExecutionParameters& gpu_execution_parameters) {
        // Get kernel function
        logger->info("Executing kernel {}", entry_point);
        checkCudaErrors(cuModuleGetFunction(&function, cudaModule, entry_point.c_str()));

        // Kernel launch
        void* kernel_parameters[] = {&arg};
        checkCudaErrors(cuLaunchKernel(function,
                                       gpu_execution_parameters.gridDimX,
                                       gpu_execution_parameters.gridDimY,
                                       gpu_execution_parameters.gridDimY,
                                       gpu_execution_parameters.blockDimX,
                                       gpu_execution_parameters.blockDimY,
                                       gpu_execution_parameters.blockDimY,
                                       gpu_execution_parameters.sharedMemBytes,
                                       nullptr,
                                       kernel_parameters,
                                       nullptr));
        cudaDeviceSynchronize();
    }
};

/**
 * \class BaseGPURunner
 * \brief A base runner class that provides functionality to execute an
 * entry point in the CUDA module.
 */
class BaseGPURunner {
  protected:
    std::unique_ptr<CUDADriver> driver;

    explicit BaseGPURunner(std::unique_ptr<llvm::Module> m)
        : driver(std::make_unique<CUDADriver>(std::move(m))) {}

  public:
    /// Sets up the CUDA driver.
    virtual void initialize_driver() = 0;

    /// Runs the entry-point function without arguments.
    void run_without_arguments(const std::string& entry_point,
                               const GPUExecutionParameters& gpu_execution_parameters) {
        return driver->execute_without_arguments(entry_point, gpu_execution_parameters);
    }

    /// Runs the entry-point function with a pointer to the data as an argument.
    template <typename ArgType>
    void run_with_argument(const std::string& entry_point,
                           ArgType arg,
                           const GPUExecutionParameters& gpu_execution_parameters) {
        return driver->template execute_with_arguments(entry_point, arg, gpu_execution_parameters);
    }
};

/**
 * \class TestGPURunner
 * \brief A simple runner for testing purposes.
 */
class TestGPURunner: public BaseGPURunner {
    /// GPU backend to target.
    std::string backend;

  public:
    explicit TestGPURunner(std::unique_ptr<llvm::Module> m, std::string backend)
        : BaseGPURunner(std::move(m)) {}

    virtual void initialize_driver() {
        driver->init(backend);
    }
};

/**
 * \class BenchmarkGPURunner
 * \brief A runner with benchmarking functionality. It takes user-specified GPU
 * features into account, as well as it can link against shared libraries.
 */
class BenchmarkGPURunner: public BaseGPURunner {
  private:
    /// Benchmarking information passed to JIT driver.
    BenchmarkInfo benchmark_info;

    /// Beckend to target.
    std::string backend;

  public:
    BenchmarkGPURunner(std::unique_ptr<llvm::Module> m,
                       std::string filename,
                       std::string output_dir,
                       std::string backend,
                       std::vector<std::string> lib_paths = {},
                       int opt_level_ir = 0,
                       int opt_level_codegen = 0)
        : BaseGPURunner(std::move(m))
        , backend(backend)
        , benchmark_info{filename, output_dir, lib_paths, opt_level_ir, opt_level_codegen} {}

    virtual void initialize_driver() {
        driver->init(backend, &benchmark_info);
    }
};


}  // namespace runner
}  // namespace nmodl

#endif
