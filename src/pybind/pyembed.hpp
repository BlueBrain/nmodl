/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <pybind11/embed.h>

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace nmodl {
namespace pybind_wrappers {


void initialize_interpreter_func();
void finalize_interpreter_func();


std::tuple<std::vector<std::string>, std::vector<std::string>, std::string>
call_solve_linear_system(const std::vector<std::string>& eq_system,
                         const std::vector<std::string>& state_vars,
                         const std::set<std::string>& vars,
                         bool small_system,
                         bool elimination,
                         const std::string& tmp_unique_prefix,
                         const std::set<std::string>& function_calls);

std::tuple<std::vector<std::string>, std::string> call_solve_nonlinear_system(
    const std::vector<std::string>& eq_system,
    const std::vector<std::string>& state_vars,
    const std::set<std::string>& vars,
    const std::set<std::string>& function_calls);

std::tuple<std::string, std::string> call_diffeq_solver(const std::string& node_as_nmodl,
                                                        const std::string& dt_var,
                                                        const std::set<std::string>& vars,
                                                        bool use_pade_approx,
                                                        const std::set<std::string>& function_calls,
                                                        const std::string& method);

std::tuple<std::string, std::string> call_analytic_diff(
    const std::vector<std::string>& expressions,
    const std::set<std::string>& used_names_in_block);

struct pybind_wrap_api {
    decltype(&initialize_interpreter_func) initialize_interpreter;
    decltype(&finalize_interpreter_func) finalize_interpreter;
    decltype(&call_solve_nonlinear_system) solve_nonlinear_system;
    decltype(&call_solve_linear_system) solve_linear_system;
    decltype(&call_diffeq_solver) diffeq_solver;
    decltype(&call_analytic_diff) analytic_diff;
};


/**
 * A singleton class handling access to the pybind_wrap_api struct
 *
 * This class manages the runtime loading of the libpython so/dylib file and the python binding
 * wrapper library and provides access to the API wrapper struct that can be used to access the
 * pybind11 embedded python functionality.
 */
class EmbeddedPythonLoader {
  public:
    /**
     * Construct (if not already done) and get the only instance of this class
     *
     * @return the EmbeddedPythonLoader singleton instance
     */
    static EmbeddedPythonLoader& get_instance() {
        static EmbeddedPythonLoader instance;

        return instance;
    }

    EmbeddedPythonLoader(const EmbeddedPythonLoader&) = delete;
    void operator=(const EmbeddedPythonLoader&) = delete;


    /**
     * Get a pointer to the pybind_wrap_api struct
     *
     * Get access to the container struct for the pointers to the functions in the wrapper library.
     * @return a pybind_wrap_api pointer
     */
    const pybind_wrap_api& api();

    ~EmbeddedPythonLoader() {
        unload();
    }

  private:
    pybind_wrap_api wrappers;

    void* pylib_handle = nullptr;
    void* pybind_wrapper_handle = nullptr;

    bool have_wrappers();
    void load_libraries();
    void populate_symbols();
    void unload();

    EmbeddedPythonLoader() {
        if (!have_wrappers()) {
            load_libraries();
            populate_symbols();
        }
    }
};


extern "C" {
pybind_wrap_api nmodl_init_pybind_wrapper_api() noexcept;
}

}  // namespace pybind_wrappers
}  // namespace nmodl
