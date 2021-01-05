/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \dir
 * \brief Implementation of LLVM's JIT-based execution engine to run functions from MOD files
 *
 * \file
 * \brief \copybrief nmodl::runner::JITDriver
 */

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace nmodl {
namespace runner {

/**
 * \class JITDriver
 * \brief Driver to execute MOD file function via LLVM IR backend
 */
class JITDriver {
private:

    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();

    std::unique_ptr<llvm::orc::LLJIT> jit;

    std::unique_ptr<llvm::Module> module;

public:
    JITDriver(std::unique_ptr<llvm::Module> m): module(std::move(m)) {}

    void init();

    //template<typename T>
    void execute(std::string& entry_point);

    /// Set the target triple on the module.
    static void set_target_triple(llvm::Module* module);
};

}   // namespace runner
}   // namespace nmodl