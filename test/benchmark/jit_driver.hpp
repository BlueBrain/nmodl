/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \dir
 * \brief Implementation of LLVM's JIT-based execution engine to run functions from MOD files
 *
 * \file
 * \brief \copybrief nmodl::runner::JITDriver
 */

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Host.h"

#ifdef NMODL_LLVM_CUDA_BACKEND
#include "cuda.h"
#include "nvvm.h"
#include "gpu_parameters.hpp"
#endif

using nmodl::cuda_details::GPUExecutionParameters;

namespace nmodl {
namespace runner {

/// A struct to hold the information for benchmarking.
struct BenchmarkInfo {
    /// Object filename to dump.
    std::string filename;

    /// Object file output directory.
    std::string output_dir;

    /// Shared libraries' paths to link against.
    std::vector<std::string> shared_lib_paths;

    /// Optimisation level for IT.
    int opt_level_ir;

    /// Optimisation level for machine code generation.
    int opt_level_codegen;
};

/**
 * \class JITDriver
 * \brief Driver to execute a MOD file function via LLVM IR backend.
 */
class JITDriver {
  protected:
    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();

    std::unique_ptr<llvm::orc::LLJIT> jit;

    /// LLVM IR module to execute.
    std::unique_ptr<llvm::Module> module;

    /// GDB event listener.
    llvm::JITEventListener* gdb_event_listener = nullptr;

    /// perf event listener.
    llvm::JITEventListener* perf_event_listener = nullptr;

    /// Intel event listener.
    llvm::JITEventListener* intel_event_listener = nullptr;

  public:
    explicit JITDriver(std::unique_ptr<llvm::Module> m)
        : module(std::move(m)) {}

    /// Initializes the JIT driver.
    void init(const std::string& cpu, BenchmarkInfo* benchmark_info = nullptr);

    /// Lookups the entry-point without arguments in the JIT and executes it, returning the result.
    template <typename ReturnType>
    ReturnType execute_without_arguments(const std::string& entry_point) {
        auto expected_symbol = jit->lookup(entry_point);
        if (!expected_symbol)
            throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

        auto (*res)() = (ReturnType(*)())(intptr_t) expected_symbol->getAddress();
        ReturnType result = res();
        return result;
    }

    /// Lookups the entry-point with an argument in the JIT and executes it, returning the result.
    template <typename ReturnType, typename ArgType>
    ReturnType execute_with_arguments(const std::string& entry_point, ArgType arg) {
        auto expected_symbol = jit->lookup(entry_point);
        if (!expected_symbol)
            throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

        auto (*res)(ArgType) = (ReturnType(*)(ArgType))(intptr_t) expected_symbol->getAddress();
        ReturnType result = res(arg);
        return result;
    }

    /// Lookups the entry-point with an argument in the JIT and executes it, returning the result.
    template <typename ReturnType, typename ArgType1, typename ArgType2>
    ReturnType execute_with_arguments(const std::string& entry_point, ArgType1 arg1, ArgType2 arg2) {
        auto expected_symbol = jit->lookup(entry_point);
        if (!expected_symbol)
            throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

        auto (*res)(ArgType1, ArgType2) = (ReturnType(*)(ArgType1, ArgType2))(intptr_t) expected_symbol->getAddress();
        ReturnType result = res(arg1, arg2);
        return result;
    }
};

#ifdef NMODL_LLVM_CUDA_BACKEND
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

class CUDADriver {
    /// LLVM IR module to execute.
    std::unique_ptr<llvm::Module> module;
    nvvmProgram prog;
    CUdevice    device;
    CUmodule    cudaModule;
    CUcontext   context;
    CUfunction  function;
    CUlinkState linker;
    DeviceInfo device_info;
    std::string ptx_compiled_module;

    public:
        explicit CUDADriver(std::unique_ptr<llvm::Module> m)
            : module(std::move(m)) {}
    
        /// Initializes the CUDA GPU JIT driver.
        void init(const std::string& gpu, BenchmarkInfo* benchmark_info = nullptr);

        /// Lookups the entry-point without arguments in the CUDA module and executes it.
        template <typename ReturnType, const GPUExecutionParameters&>
        ReturnType execute_with_arguments(const std::string& entry_point, const GPUExecutionParameters& gpu_execution_parameters) {
            // Get kernel function
            checkCudaErrors(cuModuleGetFunction(&function, cudaModule, entry_point.c_str()));

            // Kernel launch
            void *kernel_parameters[] = {};
            checkCudaErrors(cuLaunchKernel(function, gpu_execution_parameters.gridDimX, gpu_execution_parameters.gridDimY, gpu_execution_parameters.gridDimY,
                                    gpu_execution_parameters.blockDimX, gpu_execution_parameters.blockDimY, gpu_execution_parameters.blockDimY,
                                    gpu_execution_parameters.sharedMemBytes, nullptr, kernel_parameters, nullptr));
        }

        /// Lookups the entry-point with arguments in the CUDA module and executes it.
        template <typename ReturnType, typename ArgType1, typename ArgType2>
        ReturnType execute_with_arguments(const std::string& entry_point, ArgType1 arg1, ArgType2 gpu_execution_parameters) {
            // Get kernel function
            checkCudaErrors(cuModuleGetFunction(&function, cudaModule, entry_point.c_str()));

            // Kernel launch
            void *kernel_parameters[] = {&arg1};
            checkCudaErrors(cuLaunchKernel(function, gpu_execution_parameters.gridDimX, gpu_execution_parameters.gridDimY, gpu_execution_parameters.gridDimY,
                                    gpu_execution_parameters.blockDimX, gpu_execution_parameters.blockDimY, gpu_execution_parameters.blockDimY,
                                    gpu_execution_parameters.sharedMemBytes, nullptr, kernel_parameters, nullptr));
        }
};
#endif

/**
 * \class BaseRunner
 * \brief A base runner class that provides functionality to execute an
 * entry point in the LLVM IR module.
 */
template<typename DriverType = JITDriver>
class BaseRunner {
  protected:
    std::unique_ptr<DriverType> driver;

    explicit BaseRunner<DriverType>(std::unique_ptr<llvm::Module> m)
        : driver(std::make_unique<DriverType>(std::move(m))) {}

  public:
    /// Sets up the JIT driver.
    virtual void initialize_driver() = 0;

    /// Runs the entry-point function without arguments.
    template <typename ReturnType>
    ReturnType run_without_arguments(const std::string& entry_point) {
        return driver->template execute_without_arguments<ReturnType>(entry_point);
    }

    /// Runs the entry-point function with a pointer to the data as an argument.
    template <typename ReturnType, typename ArgType>
    ReturnType run_with_argument(const std::string& entry_point, ArgType arg) {
        return driver->template execute_with_arguments<ReturnType, ArgType>(entry_point, arg);
    }

    /// Runs the entry-point function with a pointer to the data as an argument.
    template <typename ReturnType, typename ArgType1, typename ArgType2>
    ReturnType run_with_argument(const std::string& entry_point, ArgType1 arg1, ArgType2 arg2) {
        return driver->template execute_with_arguments<ReturnType, ArgType1, ArgType2>(entry_point, arg1, arg2);
    }
};

/**
 * \class TestRunner
 * \brief A simple runner for testing purposes.
 */
template<typename DriverType = JITDriver>
class TestRunner: public BaseRunner<DriverType> {
  public:
    explicit TestRunner<DriverType>(std::unique_ptr<llvm::Module> m)
        : BaseRunner<DriverType>(std::move(m)) {}

    virtual void initialize_driver() {
        this->driver->init(llvm::sys::getHostCPUName().str());
    }
};

/**
 * \class BenchmarkRunner
 * \brief A runner with benchmarking functionality. It takes user-specified CPU
 * features into account, as well as it can link against shared libraries.
 */
template<typename DriverType = JITDriver>
class BenchmarkRunner: public BaseRunner<DriverType> {
  private:
    /// Benchmarking information passed to JIT driver.
    BenchmarkInfo benchmark_info;

    /// Beckend to target.
    std::string backend;

  public:
    BenchmarkRunner<DriverType>(std::unique_ptr<llvm::Module> m,
                    std::string filename,
                    std::string output_dir,
                    std::string backend,
                    std::vector<std::string> lib_paths = {},
                    int opt_level_ir = 0,
                    int opt_level_codegen = 0)
        : BaseRunner<DriverType>(std::move(m))
        , backend(backend)
        , benchmark_info{filename, output_dir, lib_paths, opt_level_ir, opt_level_codegen} {}

    virtual void initialize_driver() {
        this->driver->init(backend, &benchmark_info);
    }
};

}  // namespace runner
}  // namespace nmodl
