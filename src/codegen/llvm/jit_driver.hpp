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
    void init(std::string features, std::vector<std::string>& lib_paths);

    /// Lookup the entry-point without arguments in the JIT and execute it, returning the result.
    template <typename ReturnType>
    ReturnType execute_without_arguments(const std::string& entry_point) {
        auto expected_symbol = jit->lookup(entry_point);
        if (!expected_symbol)
            throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

        auto (*res)() = (ReturnType(*)())(intptr_t) expected_symbol->getAddress();
        ReturnType result = res();
        return result;
    }

    /// Lookup the entry-point with an argument in the JIT and execute it, returning the result.
    template <typename ReturnType, typename ArgType>
    ReturnType execute_with_arguments(const std::string& entry_point, ArgType arg) {
        auto expected_symbol = jit->lookup(entry_point);
        if (!expected_symbol)
            throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

        auto (*res)(ArgType) = (ReturnType(*)(ArgType))(intptr_t) expected_symbol->getAddress();
        ReturnType result = res(arg);
        return result;
    }

    /// A wrapper around llvm::createTargetMachine to turn on/off certain CPU features.
    std::unique_ptr<llvm::TargetMachine> create_target(llvm::orc::JITTargetMachineBuilder* builder,
                                                       const std::string& features);

    /// Sets the triple and the data layout for the module.
    void set_triple_and_data_layout(const std::string& features);
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
    Runner(std::unique_ptr<llvm::Module> m,
           std::string features = "",
           std::vector<std::string> lib_paths = {})
        : module(std::move(m)) {
        driver->init(features, lib_paths);
    }

    /// Run the entry-point function without arguments.
    template <typename ReturnType>
    ReturnType run_without_arguments(const std::string& entry_point) {
        return driver->template execute_without_arguments<ReturnType>(entry_point);
    }

    /// Run the entry-point function with a pointer to the data as an argument.
    template <typename ReturnType, typename ArgType>
    ReturnType run_with_argument(const std::string& entry_point, ArgType arg) {
        return driver->template execute_with_arguments<ReturnType, ArgType>(entry_point, arg);
    }
};

}  // namespace runner
}  // namespace nmodl
