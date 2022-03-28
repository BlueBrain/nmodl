/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#ifdef NMODL_LLVM_CUDA_BACKEND

#include <fstream>
#include <regex>

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/llvm_utils.hpp"
#include "cuda_driver.hpp"
#include "fmt/format.h"
#include "utils/common_utils.hpp"

using fmt::literals::operator""_format;

namespace nmodl {
namespace runner {

void CUDADriver::checkCudaErrors(CUresult err) {
    if (err != CUDA_SUCCESS) {
        const char* ret = NULL;
        cuGetErrorName(err, &ret);
        throw std::runtime_error("CUDA error: " + std::string(ret));
    }
}

void CUDADriver::checkNVVMErrors(nvvmResult err) {
    if (err != NVVM_SUCCESS) {
        size_t program_log_size;
        nvvmGetProgramLogSize(prog, &program_log_size);
        auto program_log = (char*) malloc(program_log_size);
        nvvmGetProgramLog(prog, program_log);
        throw std::runtime_error(
            "Compilation Log:\n {}\nNVVM Error: {}\n"_format(program_log, nvvmGetErrorString(err)));
    }
}

std::string load_file_to_string(const std::string& filename) {
    std::ifstream t(filename);
    if (!t.is_open()) {
        throw std::runtime_error("File {} not found"_format(filename));
    }
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    return str;
}

void CUDADriver::load_libraries(BenchmarkInfo* benchmark_info) {
    for (const auto& lib_path: benchmark_info->shared_lib_paths) {
        const auto lib_name = lib_path.substr(lib_path.find_last_of("/\\") + 1);
        std::regex libdevice_bitcode_name{"libdevice.*.bc"};
        if (!std::regex_match(lib_name, libdevice_bitcode_name)) {
            throw std::runtime_error("Only libdevice is supported for now");
        }
        // Load libdevice module to the NVVM program
        const auto libdevice_module = load_file_to_string(lib_path);
        const auto libdevice_module_size = libdevice_module.size();
        checkNVVMErrors(nvvmAddModuleToProgram(
            prog, libdevice_module.c_str(), libdevice_module_size, "libdevice"));
    }
}

auto get_compilation_options(int compute_version_major, BenchmarkInfo* benchmark_info) {
    std::vector<std::string> compilation_options;
    // Set the correct architecture to generate the PTX for
    // Architectures should be based on the major compute capability of the GPU
    const std::string arch_option{"-arch=compute_{}0"_format(compute_version_major)};
    compilation_options.push_back(arch_option);
    // Set the correct optimization level
    const std::string optimization_option{"-opt={}"_format(benchmark_info->opt_level_codegen)};
    compilation_options.push_back(optimization_option);
    return compilation_options;
}

void print_ptx_to_file(const std::string& ptx_compiled_module, const std::string& filename) {
    std::ofstream ptx_file(filename);
    ptx_file << ptx_compiled_module;
    ptx_file.close();
}

void CUDADriver::init(const std::string& gpu, BenchmarkInfo* benchmark_info) {
    // CUDA initialization
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGetCount(&device_info.count));
    checkCudaErrors(cuDeviceGet(&device, 0));

    char name[128];
    checkCudaErrors(cuDeviceGetName(name, 128, device));
    device_info.name = name;
    logger->info("Using CUDA Device [0]: {}"_format(device_info.name));

    checkCudaErrors(cuDeviceGetAttribute(&device_info.compute_version_major,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                         device));
    checkCudaErrors(cuDeviceGetAttribute(&device_info.compute_version_minor,
                                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                         device));
    logger->info("Device Compute Capability: {}.{}"_format(device_info.compute_version_major,
                                                           device_info.compute_version_minor));
    if (device_info.compute_version_major < 2) {
        throw std::runtime_error("ERROR: Device 0 is not SM 2.0 or greater");
    }

    // Save the LLVM IR module to string
    std::string kernel_llvm_ir;
    llvm::raw_string_ostream os(kernel_llvm_ir);
    os << *module;
    os.flush();

    // Create NVVM program object
    checkNVVMErrors(nvvmCreateProgram(&prog));

    // Load the external libraries modules to the NVVM program
    // Currently only libdevice is supported
    load_libraries(benchmark_info);

    // Add custom IR to program
    checkNVVMErrors(nvvmAddModuleToProgram(
        prog, kernel_llvm_ir.c_str(), kernel_llvm_ir.size(), "nmodl_llvm_ir"));

    // Declare compile options
    auto compilation_options = get_compilation_options(device_info.compute_version_major,
                                                       benchmark_info);
    // transform compilation options to vector of const char*
    std::vector<const char*> compilation_options_c_str;
    for (const auto& option: compilation_options) {
        compilation_options_c_str.push_back(option.c_str());
    }
    // Compile the program
    checkNVVMErrors(nvvmCompileProgram(prog,
                                       compilation_options_c_str.size(),
                                       compilation_options_c_str.data()));

    // Get compiled module
    char* compiled_module;
    size_t compiled_module_size;
    nvvmGetCompiledResultSize(prog, &compiled_module_size);
    compiled_module = (char*) malloc(compiled_module_size);
    nvvmGetCompiledResult(prog, compiled_module);
    ptx_compiled_module = std::string(compiled_module);
    free(compiled_module);
    print_ptx_to_file(ptx_compiled_module,
                      benchmark_info->output_dir + "/" + benchmark_info->filename + ".ptx");

    // Create driver context
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    // Create module for object
    checkCudaErrors(cuModuleLoadDataEx(&cudaModule, ptx_compiled_module.c_str(), 0, 0, 0));
}

}  // namespace runner
}  // namespace nmodl

#endif
