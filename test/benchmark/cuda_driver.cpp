/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#ifdef NMODL_LLVM_CUDA_BACKEND

#include "cuda_driver.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/llvm_utils.hpp"
#include "fmt/format.h"
#include "utils/common_utils.hpp"

using fmt::literals::operator""_format;

namespace nmodl {
namespace runner {

void checkCudaErrors(CUresult err) {
    if (err != CUDA_SUCCESS) {
        const char* ret = NULL;
        cuGetErrorName(err, &ret);
        throw std::runtime_error("CUDA error: " + std::string(ret));
    }
}

void checkNVVMErrors(nvvmResult err) {
    if (err != NVVM_SUCCESS) {
        throw std::runtime_error("NVVM Error: " + std::string(nvvmGetErrorString(err)));
    }
}
void CUDADriver::init(const std::string& gpu, BenchmarkInfo* benchmark_info) {
    // CUDA initialization
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGetCount(&device_info.count));
    checkCudaErrors(cuDeviceGet(&device, 0));

    char name[128];
    checkCudaErrors(cuDeviceGetName(name, 128, device));
    device_info.name = name;
    std::cout << "Using CUDA Device [0]: " << device_info.name << "\n";

    checkCudaErrors(cuDeviceGetAttribute(&device_info.compute_version_major,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                         device));
    checkCudaErrors(cuDeviceGetAttribute(&device_info.compute_version_minor,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                         device));
    std::cout << "Device Compute Capability: " << device_info.compute_version_major << "."
              << device_info.compute_version_minor << "\n";
    if (device_info.compute_version_major < 2) {
        throw std::runtime_error("ERROR: Device 0 is not SM 2.0 or greater");
    }

    // Save the LLVM IR module to string
    std::string kernel_llvm_ir;
    llvm::raw_string_ostream os(kernel_llvm_ir);
    os << *module;
    os.flush();

    // Create NVVM program object
    nvvmCreateProgram(&prog);

    // Add custom IR to program
    nvvmAddModuleToProgram(prog, kernel_llvm_ir.c_str(), kernel_llvm_ir.size(), "nmodl_llvm_ir");

    // Declare compile options
    const auto arch_option = "-arch=compute_{}0"_format(device_info.compute_version_major);
    const char* options[] = {arch_option.c_str()};

    // Compile the program
    nvvmCompileProgram(prog, 1, options);

    // Get compiled module
    char* compiled_module;
    size_t compiled_module_size;
    nvvmGetCompiledResultSize(prog, &compiled_module_size);
    compiled_module = (char*) malloc(compiled_module_size);
    nvvmGetCompiledResult(prog, compiled_module);
    ptx_compiled_module = std::string(compiled_module, compiled_module_size);

    // Create driver context
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    // Create module for object
    checkCudaErrors(cuModuleLoadDataEx(&cudaModule, compiled_module, 0, 0, 0));
}

}  // namespace runner
}  // namespace nmodl

#endif
