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

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Linker/Linker.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"

using fmt::literals::operator""_format;

namespace nmodl {
namespace runner {

void CUDADriver::checkCudaErrors(CUresult err) {
    if (err != CUDA_SUCCESS) {
        const char* ret = NULL;
        cuGetErrorName(err, &ret);
        // throw std::runtime_error("CUDA error: " + std::string(ret));
        std::cout << "CUDA error: " << ret << std::endl;
    }
}

void CUDADriver::checkNVVMErrors(nvvmResult err) {
    if (err != NVVM_SUCCESS) {
        size_t program_log_size;
        nvvmGetProgramLogSize(prog, &program_log_size);
        std::string program_log(program_log_size, '\0');
        nvvmGetProgramLog(prog, &program_log.front());
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

void CUDADriver::link_libraries(llvm::Module& module, BenchmarkInfo* benchmark_info) {
    llvm::Linker linker(module);
    for (const auto& lib_path: benchmark_info->shared_lib_paths) {
        const auto lib_name = lib_path.substr(lib_path.find_last_of("/\\") + 1);
        std::regex libdevice_bitcode_name{"libdevice.*.bc"};
        if (!std::regex_match(lib_name, libdevice_bitcode_name)) {
            throw std::runtime_error("Only libdevice is supported for now");
        }
        // Load libdevice module to the NVVM program
        llvm::SMDiagnostic Error;

        llvm::errs() << lib_name << "\n";
        auto LibDeviceModule = parseIRFile(lib_name, Error, module.getContext());
        if (!LibDeviceModule) {
            throw std::runtime_error("Could not find or load libdevice\n");
        }
        linker.linkInModule(std::move(LibDeviceModule), llvm::Linker::LinkOnlyNeeded);
    }
}

std::string get_ptx_compiled_module(const llvm::Module& module) {
    std::string SPIRAssembly;
    llvm::raw_string_ostream IROstream(SPIRAssembly);
    IROstream << module;
    IROstream.flush();
    return SPIRAssembly;
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

void print_string_to_file(const std::string& ptx_compiled_module, const std::string& filename) {
    std::ofstream ptx_file(filename);
    ptx_file << ptx_compiled_module;
    ptx_file.close();
}

std::string print_bitcode_to_string(const llvm::Module& module) {
    std::string bitcode_string;
    llvm::raw_string_ostream os(bitcode_string);
    WriteBitcodeToFile(module, os);
    os.flush();
    return bitcode_string;
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

    // Save the LLVM module bitcode to string
    std::string kernel_bitcode = print_bitcode_to_string(*module);

    // Create NVVM program object
    // checkNVVMErrors(nvvmCreateProgram(&prog));

    // Load the external libraries modules to the NVVM program
    // Currently only libdevice is supported
    // link_libraries(*module, benchmark_info);

    // Add custom IR to program
    // checkNVVMErrors(nvvmAddModuleToProgram(
        // prog, kernel_bitcode.c_str(), kernel_bitcode.size(), "nmodl_kernel"));

    // Declare compile options
    auto compilation_options = get_compilation_options(device_info.compute_version_major,
                                                       benchmark_info);
    // transform compilation options to vector of const char*
    std::vector<const char*> compilation_options_c_str;
    for (const auto& option: compilation_options) {
        compilation_options_c_str.push_back(option.c_str());
    }
    // Compile the program
    logger->info("Compiling the LLVM IR to PTX");
    // checkNVVMErrors(nvvmCompileProgram(prog,
                                    //    compilation_options_c_str.size(),
                                    //    compilation_options_c_str.data()));

    // Get compiled module
    size_t compiled_module_size;
    // nvvmGetCompiledResultSize(prog, &compiled_module_size);
    // ptx_compiled_module.resize(compiled_module_size);
    // nvvmGetCompiledResult(prog, &ptx_compiled_module.front());
    // print_string_to_file(ptx_compiled_module,
                        //  benchmark_info->output_dir + "/" + benchmark_info->filename + ".ptx");

    // Create driver context
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    // Create target machine for CUDA GPU and generate PTX code
    // auto tm = utils::create_CUDA_target_machine(platform, *module);
    // ptx_compiled_module = utils::get_module_ptx(*tm, *module);
    const auto opt_level_codegen = benchmark_info ? benchmark_info->opt_level_codegen : 0;
    utils::optimise_module_for_nvptx(platform, *module, opt_level_codegen, ptx_compiled_module);
    if (benchmark_info) {
        print_string_to_file(ptx_compiled_module,
                            benchmark_info->output_dir + "/" + benchmark_info->filename + ".ptx");
    }

    // Create module for object
    logger->info("Loading PTX to CUDA module");
    // CUjit_option options[] = {CU_JIT_TARGET};
    // void** option_vals = new void*[1];
    // auto target_architecture = CU_TARGET_COMPUTE_86;
    // option_vals[0] = (void*)target_architecture;
    const unsigned int jitNumOptions = 6;
    CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
    void **jitOptVals = new void*[jitNumOptions];

    // set up size of compilation log buffer                                                                                                     
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    size_t jitLogBufferSize = 1024*1024;
    jitOptVals[0] = (void*)jitLogBufferSize;

    // set up pointer to the compilation log buffer
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
    char *jitLogBuffer = new char[jitLogBufferSize];
    jitOptVals[1] = jitLogBuffer;

    // set up size of compilation error log buffer
    jitOptions[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    size_t jitErrorLogBufferSize = 1024*1024;
    jitOptVals[2] = (void*)jitErrorLogBufferSize;

    // set up pointer to the compilation error log buffer
    jitOptions[3] = CU_JIT_ERROR_LOG_BUFFER;
    char *jitErrorLogBuffer = new char[jitErrorLogBufferSize];
    jitOptVals[3] = jitErrorLogBuffer;

    // set up wall clock time                                                                                                                    
    jitOptions[4] = CU_JIT_WALL_TIME;
    float jitTime = 0.0;

    jitOptions[5] = CU_JIT_TARGET;
    auto target_architecture = CU_TARGET_COMPUTE_86;
    jitOptVals[5] = (void*)target_architecture;
    checkCudaErrors(cuModuleLoadDataEx(&cudaModule, ptx_compiled_module.c_str(), jitNumOptions, jitOptions, jitOptVals));
    logger->info("CUDA JIT walltime: "_format((double)jitOptions[4]));
    logger->info("CUDA JIT INFO LOG: "_format(jitLogBuffer));
    logger->info("CUDA JIT ERROR LOG: "_format(jitErrorLogBuffer));
}

}  // namespace runner
}  // namespace nmodl

#endif
