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
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
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
