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

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"

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

void CUDADriver::link_libraries(llvm::Module& module, BenchmarkInfo* benchmark_info) {
    llvm::Linker linker(module);
    for (const auto& lib_path: benchmark_info->shared_lib_paths) {
        const auto lib_name = lib_path.substr(lib_path.find_last_of("/\\") + 1);
        std::regex libdevice_bitcode_name{"libdevice.*.bc"};
        if (!std::regex_match(lib_name, libdevice_bitcode_name)) {
            throw std::runtime_error("Only libdevice is supported for now");
        }
        // Load libdevice module to the LLVM Module
        auto libdevice_file_memory_buffer = llvm::MemoryBuffer::getFile(lib_path);
        llvm::Expected<std::unique_ptr<llvm::Module>> libdevice_expected_module =
            parseBitcodeFile(libdevice_file_memory_buffer->get()->getMemBufferRef(),
                             module.getContext());
        if (std::error_code error = errorToErrorCode(libdevice_expected_module.takeError())) {
            throw std::runtime_error("Error reading bitcode: {}"_format(error.message()));
        }
        linker.linkInModule(std::move(libdevice_expected_module.get()),
                            llvm::Linker::LinkOnlyNeeded);
    }
}

void print_string_to_file(const std::string& ptx_compiled_module, const std::string& filename) {
    std::ofstream ptx_file(filename);
    ptx_file << ptx_compiled_module;
    ptx_file.close();
}

CUjit_target get_compute_architecture(const int compute_version_major,
                                      const int compute_version_minor) {
    auto compute_architecture = compute_version_major * 10 + compute_version_minor;
    switch (compute_architecture) {
    case 20:
        return CU_TARGET_COMPUTE_20;
    case 21:
        return CU_TARGET_COMPUTE_21;
    case 30:
        return CU_TARGET_COMPUTE_30;
    case 32:
        return CU_TARGET_COMPUTE_32;
    case 35:
        return CU_TARGET_COMPUTE_35;
    case 37:
        return CU_TARGET_COMPUTE_37;
    case 50:
        return CU_TARGET_COMPUTE_50;
    case 52:
        return CU_TARGET_COMPUTE_52;
    case 53:
        return CU_TARGET_COMPUTE_53;
    case 60:
        return CU_TARGET_COMPUTE_60;
    case 61:
        return CU_TARGET_COMPUTE_61;
    case 62:
        return CU_TARGET_COMPUTE_62;
    case 70:
        return CU_TARGET_COMPUTE_70;
    case 72:
        return CU_TARGET_COMPUTE_72;
    case 75:
        return CU_TARGET_COMPUTE_75;
    case 80:
        return CU_TARGET_COMPUTE_80;
    case 86:
        return CU_TARGET_COMPUTE_86;
    default:
        throw std::runtime_error("Unsupported compute architecture");
    }
}

void CUDADriver::init(const codegen::Platform& platform, BenchmarkInfo* benchmark_info) {
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

    // Load the external libraries modules to the NVVM program
    // Currently only libdevice is supported
    link_libraries(*module, benchmark_info);

    // Compile the program
    logger->info("Compiling the LLVM IR to PTX");

    // Optimize code for nvptx including the wrapper functions and generate PTX
    const auto opt_level_codegen = benchmark_info ? benchmark_info->opt_level_codegen : 0;
    utils::optimise_module_for_nvptx(platform, *module, opt_level_codegen, ptx_compiled_module);
    utils::save_ir_to_ll_file(*module,  benchmark_info->output_dir + "/" + benchmark_info->filename + "_benchmark");
    if (benchmark_info) {
        print_string_to_file(ptx_compiled_module,
                             benchmark_info->output_dir + "/" + benchmark_info->filename + ".ptx");
    }

    // Create driver context
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    // Create module for object
    logger->info("Loading PTX to CUDA module");
    const unsigned int jitNumOptions = 5;
    CUjit_option* jitOptions = new CUjit_option[jitNumOptions];
    void** jitOptVals = new void*[jitNumOptions];

    // set up size of compilation log buffer
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    size_t jitLogBufferSize = 1024 * 1024;
    jitOptVals[0] = (void*) jitLogBufferSize;

    // set up pointer to the compilation log buffer
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
    char* jitLogBuffer = new char[jitLogBufferSize];
    jitOptVals[1] = jitLogBuffer;

    // set up size of compilation error log buffer
    jitOptions[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    size_t jitErrorLogBufferSize = 1024 * 1024;
    jitOptVals[2] = (void*) jitErrorLogBufferSize;

    // set up pointer to the compilation error log buffer
    jitOptions[3] = CU_JIT_ERROR_LOG_BUFFER;
    char* jitErrorLogBuffer = new char[jitErrorLogBufferSize];
    jitOptVals[3] = jitErrorLogBuffer;

    jitOptions[4] = CU_JIT_TARGET;
    auto target_architecture = get_compute_architecture(device_info.compute_version_major,
                                                        device_info.compute_version_minor);
    jitOptVals[4] = (void*) target_architecture;

    auto cuda_jit_ret = cuModuleLoadDataEx(
        &cudaModule, ptx_compiled_module.c_str(), jitNumOptions, jitOptions, jitOptVals);
    if (!std::string(jitLogBuffer).empty()) {
        logger->info("CUDA JIT INFO LOG: {}"_format(std::string(jitLogBuffer)));
    }
    if (!std::string(jitErrorLogBuffer).empty()) {
        logger->info("CUDA JIT ERROR LOG: {}"_format(std::string(jitErrorLogBuffer)));
    }
    free(jitOptions);
    free(jitOptVals);
    free(jitLogBuffer);
    free(jitErrorLogBuffer);
    checkCudaErrors(cuda_jit_ret);
}

}  // namespace runner
}  // namespace nmodl

#endif
