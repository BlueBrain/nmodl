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

#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace nmodl {
namespace runner {

/// A struct to hold the information for benchmarking.
struct BenchmarkInfo {
    /// Object filename to dump.
    std::string filename;

    /// Object file output directory.
    std::string output_dir;

    /// Optimisation level for generated IR.
    int opt_level_ir;

    /// Optimisation level for machine code generation.
    int opt_level_codegen;
};

/**
 * \class JITDriver
 * \brief Driver to execute a MOD file function via LLVM IR backend.
 */
class JITDriver {
  private:
    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();

    std::unique_ptr<llvm::orc::LLJIT> jit;

    /// LLVM IR module to execute.
    std::unique_ptr<llvm::Module> module;

    /// GDB event listener.
    llvm::JITEventListener* gdb_event_listener = nullptr;

    /// perf event listener.
    llvm::JITEventListener* perf_event_listener = nullptr;

    /// Intel event listener.
    llvm::JITEventListener* intel_event_listener = nullptr;

  public:
    explicit JITDriver(std::unique_ptr<llvm::Module> m)
        : module(std::move(m)) {}

    /// Initializes the JIT driver.
    void init(std::string features = "",
              std::vector<std::string> lib_paths = {},
              BenchmarkInfo* benchmark_info = nullptr);

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
        : driver(std::make_unique<JITDriver>(std::move(m))) {}

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
    /// Benchmarking information passed to JIT driver.
    BenchmarkInfo benchmark_info;

    /// CPU features specified by the user.
    std::string features;

    /// Shared libraries' paths to link against.
    std::vector<std::string> shared_lib_paths;

  public:
    BenchmarkRunner(std::unique_ptr<llvm::Module> m,
                    std::string filename,
                    std::string output_dir,
                    std::string features = "",
                    std::vector<std::string> lib_paths = {},
                    int opt_level_ir = 0,
                    int opt_level_codegen = 0)
        : BaseRunner(std::move(m))
        , benchmark_info{filename, output_dir, opt_level_ir, opt_level_codegen}
        , features(features)
        , shared_lib_paths(lib_paths) {}

    virtual void initialize_driver() {
        driver->init(features, shared_lib_paths, &benchmark_info);
    }
};

}  // namespace runner
}  // namespace nmodl
