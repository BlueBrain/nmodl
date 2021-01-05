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

namespace nmodl {
namespace runner {

void JITDriver::init() {
    set_target_triple(module.get());
    auto data_layout = module->getDataLayout();

    auto compileFunctionCreator = [&](llvm::orc::JITTargetMachineBuilder JTMB)
        -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
        auto TM = JTMB.createTargetMachine();
        if (!TM)
            return TM.takeError();
        return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(*TM));
    };

    auto JIT = cantFail(
        llvm::orc::LLJITBuilder().setCompileFunctionCreator(compileFunctionCreator).create());
    llvm::orc::ThreadSafeModule tsm(std::move(module), std::make_unique<llvm::LLVMContext>());
    cantFail(JIT->addIRModule(std::move(tsm)));
    jit = std::move(JIT);

    llvm::orc::JITDylib& mainJD = jit->getMainJITDylib();
    mainJD.addGenerator(cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        data_layout.getGlobalPrefix())));
}

// template<typename T>
void JITDriver::execute(std::string& entry_point) {
    auto expected_symbol = jit->lookup(entry_point);
    if (!expected_symbol)
        throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

    auto (*res)() = (double (*)())(intptr_t) expected_symbol->getAddress();
    fprintf(stderr, "Evaluated to %f\n", res());
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