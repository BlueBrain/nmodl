/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "codegen/llvm/target_platform.hpp"

#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"

namespace nmodl {
namespace utils {

/// Initialises some LLVM optimisation passes.
void initialise_optimisation_passes();

/// Initialises NVPTX-specific optimisation passes.
void initialise_nvptx_passes();

//// Initializes a CUDA target machine
std::unique_ptr<llvm::TargetMachine> create_CUDA_target_machine(const codegen::Platform& platform,
                                                                llvm::Module& module);

/// Generate PTX code given a CUDA target machine and the module
std::string get_module_ptx(llvm::TargetMachine& tm, llvm::Module& module);

/// Optimises the given LLVM IR module for NVPTX targets.
void optimise_module_for_nvptx(const codegen::Platform& platform,
                               llvm::Module& module,
                               int opt_level,
                               std::string& target_asm);

/// Optimises the given LLVM IR module.
void optimise_module(llvm::Module& module, int opt_level, llvm::TargetMachine* tm = nullptr);

///
void save_ir_to_ll_file(llvm::Module& module, const std::string& filename);

}  // namespace utils
}  // namespace nmodl
