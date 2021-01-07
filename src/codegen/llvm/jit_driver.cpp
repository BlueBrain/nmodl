/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "jit_driver.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

namespace nmodl {
namespace runner {

void JITDriver::init() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    set_target_triple(module.get());
    auto data_layout = module->getDataLayout();

    // Create IR compile function callback.
    auto compile_function_creator = [&](llvm::orc::JITTargetMachineBuilder tm_builder)
        -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
        auto tm = tm_builder.createTargetMachine();
        if (!tm)
            return tm.takeError();
        return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(*tm));
    };

    auto jit_instance = cantFail(
        llvm::orc::LLJITBuilder().setCompileFunctionCreator(compile_function_creator).create());

    // Add a ThreadSafeModule to the driver.
    llvm::orc::ThreadSafeModule tsm(std::move(module), std::make_unique<llvm::LLVMContext>());
    cantFail(jit_instance->addIRModule(std::move(tsm)));
    jit = std::move(jit_instance);

    // Resolve symbols.
    llvm::orc::JITDylib& sym_tab = jit->getMainJITDylib();
    sym_tab.addGenerator(cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        data_layout.getGlobalPrefix())));
}

void JITDriver::set_target_triple(llvm::Module* module) {
    auto target_triple = llvm::sys::getDefaultTargetTriple();
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(target_triple, error);
    if (!target)
        throw std::runtime_error("Error: " + error + "\n");

    std::string cpu(llvm::sys::getHostCPUName());
    llvm::SubtargetFeatures features;
    llvm::StringMap<bool> host_features;

    if (llvm::sys::getHostCPUFeatures(host_features)) {
        for (auto& f: host_features)
            features.AddFeature(f.first(), f.second);
    }

    std::unique_ptr<llvm::TargetMachine> machine(
        target->createTargetMachine(target_triple, cpu, features.getString(), {}, {}));
    if (!machine)
        throw std::runtime_error("Error: failed to create a target machine\n");

    module->setDataLayout(machine->createDataLayout());
    module->setTargetTriple(target_triple);
}

}  // namespace runner
}  // namespace nmodl
