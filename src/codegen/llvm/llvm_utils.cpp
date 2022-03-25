/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/llvm_utils.hpp"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace nmodl {
namespace utils {

/// Populates pass managers with passes for the given optimisation levels.
static void populate_pms(llvm::legacy::FunctionPassManager& func_pm,
                         llvm::legacy::PassManager& module_pm,
                         int opt_level,
                         int size_level,
                         llvm::TargetMachine* tm) {
    // First, set the pass manager builder with some basic optimisation information.
    llvm::PassManagerBuilder pm_builder;
    pm_builder.OptLevel = opt_level;
    pm_builder.SizeLevel = size_level;
    pm_builder.DisableUnrollLoops = opt_level == 0;

    // If target machine is defined, then initialise the TargetTransformInfo for the target.
    if (tm) {
        module_pm.add(createTargetTransformInfoWrapperPass(tm->getTargetIRAnalysis()));
        func_pm.add(createTargetTransformInfoWrapperPass(tm->getTargetIRAnalysis()));
    }

    // Populate pass managers.
    pm_builder.populateModulePassManager(module_pm);
    pm_builder.populateFunctionPassManager(func_pm);
}

/// Runs the function and module passes on the provided module.
static void run_optimisation_passes(llvm::Module& module,
                                    llvm::legacy::FunctionPassManager& func_pm,
                                    llvm::legacy::PassManager& module_pm) {
    func_pm.doInitialization();
    auto& functions = module.getFunctionList();
    for (auto& function: functions) {
        llvm::verifyFunction(function);
        func_pm.run(function);
    }
    func_pm.doFinalization();
    module_pm.run(module);
}

/****************************************************************************************/
/*                             Optimisation utils                                       */
/****************************************************************************************/

void initialise_nvptx_passes() {
    // Register targets.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXAsmPrinter();

    // Initialize passes.
    initialise_optimisation_passes();
}

void optimise_module_for_nvptx(codegen::Platform& platform,
                               llvm::Module& module,
                               int opt_level,
                               std::string& target_asm) {
    // CUDA target machine we generating code for.
    std::unique_ptr<llvm::TargetMachine> tm;
    std::string platform_name = platform.get_name();

    // Target and layout information.
    static const std::map<std::string, std::string> triple_str = {
            {"nvptx", "nvptx-nvidia-cuda"},
            {"nvptx64", "nvptx64-nvidia-cuda"}};
    static const std::map<std::string, std::string> data_layout_str = {
            {"nvptx", "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32"
                      "-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32"
                      "-v64:64:64-v128:128:128-n16:32:64"},
            {"nvptx64", "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32"
                        "-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32"
                        "-v64:64:64-v128:128:128-n16:32:64"}};

    // Set data layout and target triple information for the module.
    auto triple = triple_str.find(platform_name)->second;
    module.setDataLayout(data_layout_str.find(platform_name)->second);
    module.setTargetTriple(triple);

    std::string subtarget = platform.get_subtarget_name();
    std::string features = "+ptx70";

    // Find the specified target in registry.
    std::string error_msg;
    auto* target = llvm::TargetRegistry::lookupTarget(triple, error_msg);
    if (!target)
        throw std::runtime_error("Error: " + error_msg + "\n");

    tm.reset(target->createTargetMachine(triple, subtarget, features, {}, {}));
    if (!tm)
        throw std::runtime_error("Error: creating target machine failed! Aborting.");

    // Create pass managers.
    llvm::legacy::FunctionPassManager func_pm(&module);
    llvm::legacy::PassManager module_pm;
    llvm::PassManagerBuilder pm_builder;
    pm_builder.OptLevel = opt_level;
    pm_builder.SizeLevel = 0;
    pm_builder.Inliner = llvm::createFunctionInliningPass();

    // Do not vectorize!
    pm_builder.LoopVectorize = false;

    // Adjusting pass manager adds target-specific IR transformations, e.g.
    // inferring address spaces.
    tm->adjustPassManager(pm_builder);
    pm_builder.populateFunctionPassManager(func_pm);
    pm_builder.populateModulePassManager(module_pm);

    // This runs target-indepependent optimizations.
    run_optimisation_passes(module, func_pm, module_pm);

    // Now, we want to run target-specific (e.g. NVPTX) passes. In LLVM, this
    // is done via `addPassesToEmitFile`.
    llvm::raw_string_ostream stream(target_asm);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegen_pm;

    tm->addPassesToEmitFile(codegen_pm, pstream, nullptr, llvm::CGFT_AssemblyFile);
    codegen_pm.run(module);
}

void initialise_optimisation_passes() {
    auto& registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(registry);
    llvm::initializeTransformUtils(registry);
    llvm::initializeScalarOpts(registry);
    llvm::initializeIPO(registry);
    llvm::initializeInstCombine(registry);
    llvm::initializeAggressiveInstCombine(registry);
    llvm::initializeAnalysis(registry);
}

void optimise_module(llvm::Module& module, int opt_level, llvm::TargetMachine* tm) {
    llvm::legacy::FunctionPassManager func_pm(&module);
    llvm::legacy::PassManager module_pm;
    populate_pms(func_pm, module_pm, opt_level, /*size_level=*/0, tm);
    run_optimisation_passes(module, func_pm, module_pm);
}

/****************************************************************************************/
/*                                    File utils                                        */
/****************************************************************************************/

void save_ir_to_ll_file(llvm::Module& module, const std::string& filename) {
    std::error_code error_code;
    std::unique_ptr<llvm::ToolOutputFile> out = std::make_unique<llvm::ToolOutputFile>(
        filename + ".ll", error_code, llvm::sys::fs::OF_Text);
    if (error_code)
        throw std::runtime_error("Error: " + error_code.message());

    std::unique_ptr<llvm::AssemblyAnnotationWriter> annotator;
    module.print(out->os(), annotator.get());
    out->keep();
}
}  // namespace utils
}  // namespace nmodl
