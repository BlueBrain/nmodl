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

/// A struct to hold the information for dumping object file.
struct ObjDumpInfo {
    /// Object file name.
    std::string filename;

    /// Object file output directory.
    std::string output_dir;
};

/**
 * \class JITDriver
 * \brief Driver to execute a MOD file function via LLVM IR backend.
 */
class JITDriver {
  private:
    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();

    std::unique_ptr<llvm::orc::LLJIT> jit;

    std::unique_ptr<llvm::Module> module;

  public:
    explicit JITDriver(std::unique_ptr<llvm::Module> m)
        : module(std::move(m)) {}

    /// Initializes the JIT.
    void init(std::string features = "",
              std::vector<std::string> lib_paths = {},
              ObjDumpInfo* dump_info = nullptr);

    /// Lookups the entry-point without arguments in the JIT and executes it, returning the result.
    template <typename ReturnType>
    ReturnType execute_without_arguments(const std::string& entry_point) {
        auto expected_symbol = jit->lookup(entry_point);
        if (!expected_symbol)
            throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

        auto (*res)() = (ReturnType(*)())(intptr_t) expected_symbol->getAddress();
        ReturnType result = res();
        return result;
    }

    /// Lookups the entry-point with an argument in the JIT and executes it, returning the result.
    template <typename ReturnType, typename ArgType>
    ReturnType execute_with_arguments(const std::string& entry_point, ArgType arg) {
        auto expected_symbol = jit->lookup(entry_point);
        if (!expected_symbol)
            throw std::runtime_error("Error: entry-point symbol not found in JIT\n");

        auto (*res)(ArgType) = (ReturnType(*)(ArgType))(intptr_t) expected_symbol->getAddress();
        ReturnType result = res(arg);
        return result;
    }

  private:
    /// Creates llvm::TargetMachine with certain CPU features turned on/off.
    std::unique_ptr<llvm::TargetMachine> create_target(llvm::orc::JITTargetMachineBuilder* builder,
                                                       const std::string& features);

    /// Sets the triple and the data layout for the module.
    void set_triple_and_data_layout(const std::string& features);
};

/**
 * \class BaseRunner
 * \brief A base runner class that provides functionality to execute an
 * entry point in the LLVM IR module.
 */
class BaseRunner {
  protected:
    std::unique_ptr<JITDriver> driver;

    explicit BaseRunner(std::unique_ptr<llvm::Module> m)
        : driver(std::make_unique<JITDriver>(std::move(m))) { }

  public:
    /// Sets up the JIT driver.
    virtual void initialize_driver() = 0;

    /// Runs the entry-point function without arguments.
    template <typename ReturnType>
    ReturnType run_without_arguments(const std::string& entry_point) {
        return driver->template execute_without_arguments<ReturnType>(entry_point);
    }

    /// Runs the entry-point function with a pointer to the data as an argument.
    template <typename ReturnType, typename ArgType>
    ReturnType run_with_argument(const std::string& entry_point, ArgType arg) {
        return driver->template execute_with_arguments<ReturnType, ArgType>(entry_point, arg);
    }
};

/**
 * \class TestRunner
 * \brief A simple runner for testing purposes.
 */
class TestRunner: public BaseRunner {
  public:
    explicit TestRunner(std::unique_ptr<llvm::Module> m)
        : BaseRunner(std::move(m)) {}

    virtual void initialize_driver() {
        driver->init();
    }
};

/**
 * \class BenchmarkRunner
 * \brief A runner with benchmarking functionality. It takes user-specified CPU
 * features into account, as well as it can link against shared libraries.
 */
class BenchmarkRunner: public BaseRunner {
  private:
    /// Information on dumping object file generated from LLVM IR.
    ObjDumpInfo dump_info;

    /// CPU features specified by the user.
    std::string features;

    /// Shared libraries' paths to link against.
    std::vector<std::string> shared_lib_paths;

  public:
    BenchmarkRunner(std::unique_ptr<llvm::Module> m,
                    std::string filename,
                    std::string output_dir,
                    std::string features = "",
                    std::vector<std::string> lib_paths = {})
        : BaseRunner(std::move(m))
        , dump_info{filename, output_dir}
        , features(features)
        , shared_lib_paths(lib_paths) { }

    virtual void initialize_driver() {
        driver->init(features, shared_lib_paths, &dump_info);
    }
};

}  // namespace runner
}  // namespace nmodl
