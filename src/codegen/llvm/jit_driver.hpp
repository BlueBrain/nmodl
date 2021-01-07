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
    JITDriver(std::unique_ptr<llvm::Module> m)
            : module(std::move(m)) {}

    /// Initialize the JIT.
    void init();

    /// Lookup the entry-point in the JIT and execute it, returning the result.
    template<typename T>
    T execute(std::string& entry_point) {
        auto expected_symbol = jit->lookup(entry_point);
        if (!expected_symbol)
            throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

        auto (*res)() = (T (*)())(intptr_t) expected_symbol->getAddress();
        T result = res();
        return result;
    }

    /// Set the target triple on the module.
    static void set_target_triple(llvm::Module* module);
};

/**
 * \class Runner
 * \brief A wrapper around JITDriver to execute an entry point in the LLVM IR module.
 */
class Runner {
  private:

    std::unique_ptr<llvm::Module> module;

    std::unique_ptr<JITDriver> driver = std::make_unique<JITDriver>(std::move(module));

  public:
    Runner(std::unique_ptr<llvm::Module> m)
            : module(std::move(m)) {
        driver->init();
    }

    /// Run the entry-point function.
    template<typename T>
    double run(std::string& entry_point) {
        return driver->execute<T>(entry_point);
    }
};

}  // namespace runner
}  // namespace nmodl
