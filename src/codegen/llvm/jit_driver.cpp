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

void JITDriver::init(std::string features) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Create IR compile function callback.
    auto compile_function_creator = [&](llvm::orc::JITTargetMachineBuilder tm_builder)
        -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
        // Create target machine with some features possibly turned off.
        auto tm = create_target(&tm_builder, features);

        // Set the target triple and the data layout for the module.
        module->setDataLayout(tm->createDataLayout());
        module->setTargetTriple(tm->getTargetTriple().getTriple());

        return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(tm));
    };

    // Set JIT instance and extract the data layout from the module.
    auto jit_instance = cantFail(
        llvm::orc::LLJITBuilder().setCompileFunctionCreator(compile_function_creator).create());
    auto data_layout = module->getDataLayout();

    // Add a ThreadSafeModule to the driver.
    llvm::orc::ThreadSafeModule tsm(std::move(module), std::make_unique<llvm::LLVMContext>());
    cantFail(jit_instance->addIRModule(std::move(tsm)));
    jit = std::move(jit_instance);

    // Resolve symbols.
    llvm::orc::JITDylib& sym_tab = jit->getMainJITDylib();
    sym_tab.addGenerator(cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
        data_layout.getGlobalPrefix())));
}

std::unique_ptr<llvm::TargetMachine> JITDriver::create_target(
    llvm::orc::JITTargetMachineBuilder* builder,
    const std::string& features) {
    // First, look up the target.
    std::string error_msg;
    auto target_triple = builder->getTargetTriple().getTriple();
    auto* target = llvm::TargetRegistry::lookupTarget(target_triple, error_msg);
    if (!target)
        throw std::runtime_error("Error " + error_msg + "\n");

    // Create default target machine with provided features.
    auto tm = target->createTargetMachine(target_triple,
                                          llvm::sys::getHostCPUName().str(),
                                          features,
                                          builder->getOptions(),
                                          builder->getRelocationModel(),
                                          builder->getCodeModel(),
                                          /*OL=*/llvm::CodeGenOpt::Default,
                                          /*JIT=*/true);
    if (!tm)
        throw std::runtime_error("Error: could not create the target machine\n");

    return std::unique_ptr<llvm::TargetMachine>(tm);
}

}  // namespace runner
}  // namespace nmodl
