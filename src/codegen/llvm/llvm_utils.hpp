/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"

namespace nmodl {
namespace utils {

/// Initialises some LLVM optimisation passes.
void initialise_optimisation_passes();

/// Optimises the given LLVM IR module.
void optimise_module(llvm::Module& module, int opt_level, llvm::TargetMachine* tm = nullptr);

///
void save_ir_to_ll_file(llvm::Module& module, const std::string& filename);

}  // namespace utils
}  // namespace nmodl
