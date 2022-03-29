/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "codegen/llvm/target_platform.hpp"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Host.h"

using nmodl::codegen::Platform;

namespace llvm {

/**
 * \class ReplaceMathFunctions
 * \brief A module LLVM pass that replaces math intrinsics with
 * SIMD or libdevice library calls.
 */
class ReplaceMathFunctions : public ModulePass {
    static char ID;

  private:
    const Platform* platform;

  public:
    ReplaceMathFunctions(const Platform& platform)
        : ModulePass(ID)
        , platform(&platform) {}

    bool runOnModule(Module &module) override;

  private:

    /// Populates `tli` with vectorizable function definitions.
    void add_vectorizable_functions_from_vec_lib(TargetLibraryInfoImpl& tli,
                                                 Triple& triple);
};

}  // namespace llvm
