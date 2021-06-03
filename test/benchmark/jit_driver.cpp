/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "jit_driver.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/llvm_utils.hpp"
#include "utils/common_utils.hpp"

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

namespace nmodl {
namespace runner {

/****************************************************************************************/
/*                            Utilities for JIT driver                                  */
/****************************************************************************************/

/// Get the host CPU features in the format:
///   +feature,+feature,-feature,+feature,...
/// where `+` indicates that the feature is enabled.
std::string get_cpu_features(const std::string& cpu) {
    llvm::SubtargetFeatures features;
    llvm::StringMap<bool> host_features;
    if (llvm::sys::getHostCPUFeatures(host_features)) {
        for (auto& f: host_features)
            features.AddFeature(f.first(), f.second);
    }
    return llvm::join(features.getFeatures().begin(), features.getFeatures().end(), ",");
}

/// Sets the target triple and the data layout of the module.
static void set_triple_and_data_layout(llvm::Module& module, const std::string& cpu) {
    // Get the default target triple for the host.
    auto target_triple = llvm::sys::getDefaultTargetTriple();
    std::string error_msg;
    auto* target = llvm::TargetRegistry::lookupTarget(target_triple, error_msg);
    if (!target)
        throw std::runtime_error("Error " + error_msg + "\n");

    // Set a target machine to create the data layout.
    std::string features = get_cpu_features(cpu);
    std::unique_ptr<llvm::TargetMachine> tm(
        target->createTargetMachine(target_triple, cpu, features, {}, {}));
    if (!tm)
        throw std::runtime_error("Error: could not create the target machine\n");

    // Set data layout and the target triple to the module.
    module.setDataLayout(tm->createDataLayout());
    module.setTargetTriple(target_triple);
}

/// Creates llvm::TargetMachine with for a specified CPU.
static std::unique_ptr<llvm::TargetMachine> create_target(
    llvm::orc::JITTargetMachineBuilder* tm_builder,
    const std::string& cpu,
    int opt_level) {
    // First, look up the target.
    std::string error_msg;
    auto target_triple = tm_builder->getTargetTriple().getTriple();
    auto* target = llvm::TargetRegistry::lookupTarget(target_triple, error_msg);
    if (!target)
        throw std::runtime_error("Error " + error_msg + "\n");

    // Create default target machine with provided features.
    auto tm = target->createTargetMachine(target_triple,
                                          cpu,
                                          get_cpu_features(cpu),
                                          tm_builder->getOptions(),
                                          tm_builder->getRelocationModel(),
                                          tm_builder->getCodeModel(),
                                          static_cast<llvm::CodeGenOpt::Level>(opt_level),
                                          /*JIT=*/true);
    if (!tm)
        throw std::runtime_error("Error: could not create the target machine\n");

    return std::unique_ptr<llvm::TargetMachine>(tm);
}

/****************************************************************************************/
/*                                      JIT driver                                      */
/****************************************************************************************/

void JITDriver::init(const std::string& cpu, BenchmarkInfo* benchmark_info) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    utils::initialise_optimisation_passes();

    // Set the target triple and the data layout for the module.
    set_triple_and_data_layout(*module, cpu);
    auto data_layout = module->getDataLayout();

    // If benchmarking, enable listeners to use GDB, perf or VTune. Note that LLVM should be built
    // with listeners on (e.g. -DLLVM_USE_PERF=ON).
    if (benchmark_info) {
        gdb_event_listener = llvm::JITEventListener::createGDBRegistrationListener();
#if defined(NMODL_HAVE_JIT_EVENT_LISTENERS)
        perf_event_listener = llvm::JITEventListener::createPerfJITEventListener();
        intel_event_listener = llvm::JITEventListener::createIntelJITEventListener();
#endif
    }

    // Create object linking function callback.
    auto object_linking_layer_creator = [&](llvm::orc::ExecutionSession& session,
                                            const llvm::Triple& triple) {
        // Create linking layer.
        auto layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, []() {
            return std::make_unique<llvm::SectionMemoryManager>();
        });

        // Register event listeners if they exist.
        if (gdb_event_listener)
            layer->registerJITEventListener(*gdb_event_listener);
        if (perf_event_listener)
            layer->registerJITEventListener(*perf_event_listener);
        if (intel_event_listener)
            layer->registerJITEventListener(*intel_event_listener);

        // If benchmarking, resolve shared libraries.
        if (benchmark_info) {
            for (const auto& lib_path: benchmark_info->shared_lib_paths) {
                // For every library path, create a corresponding memory buffer.
                auto memory_buffer = llvm::MemoryBuffer::getFile(lib_path);
                if (!memory_buffer)
                    throw std::runtime_error("Unable to create memory buffer for " + lib_path);

                // Create a new JIT library instance for this session and resolve symbols.
                auto& jd = session.createBareJITDylib(std::string(lib_path));
                auto loaded =
                    llvm::orc::DynamicLibrarySearchGenerator::Load(lib_path.data(),
                                                                   data_layout.getGlobalPrefix());

                if (!loaded)
                    throw std::runtime_error("Unable to load " + lib_path);
                jd.addGenerator(std::move(*loaded));
                cantFail(layer->add(jd, std::move(*memory_buffer)));
            }
        }
        return layer;
    };

    // Create IR compile function callback.
    auto compile_function_creator = [&](llvm::orc::JITTargetMachineBuilder tm_builder)
        -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
        // Create target machine with some features possibly turned off.
        int opt_level_codegen = benchmark_info ? benchmark_info->opt_level_codegen : 0;
        auto tm = create_target(&tm_builder, cpu, opt_level_codegen);

        // Optimise the LLVM IR module and save it to .ll file if benchmarking.
        if (benchmark_info) {
            utils::optimise_module(*module, benchmark_info->opt_level_ir, tm.get());
            const std::string filename = benchmark_info->output_dir + "/" +
                                         benchmark_info->filename + "_opt";
            utils::save_ir_to_ll_file(*module, filename);
        }

        return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(tm));
    };

    // Set the JIT instance.
    auto jit_instance = cantFail(llvm::orc::LLJITBuilder()
                                     .setCompileFunctionCreator(compile_function_creator)
                                     .setObjectLinkingLayerCreator(object_linking_layer_creator)
                                     .create());

    // Add a ThreadSafeModule to the driver.
    llvm::orc::ThreadSafeModule tsm(std::move(module), std::make_unique<llvm::LLVMContext>());
    cantFail(jit_instance->addIRModule(std::move(tsm)));
    jit = std::move(jit_instance);

    // Resolve symbols.
    llvm::orc::JITDylib& sym_tab = jit->getMainJITDylib();
    sym_tab.addGenerator(cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        data_layout.getGlobalPrefix())));

    // Optionally, dump the binary to the object file.
    if (benchmark_info) {
        std::string object_file = benchmark_info->filename + ".o";
        if (utils::file_exists(object_file)) {
            int status = remove(object_file.c_str());
            if (status) {
                throw std::runtime_error("Can not remove object file " + object_file);
            }
        }
        jit->getObjTransformLayer().setTransform(
            llvm::orc::DumpObjects(benchmark_info->output_dir, benchmark_info->filename));
    }
}
}  // namespace runner
}  // namespace nmodl
