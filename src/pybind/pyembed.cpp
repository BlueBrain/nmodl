/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <cstdlib>
#include <dlfcn.h>

#include "utils/logger.hpp"
#include "pybind/pyembed.hpp"

namespace nmodl {

    namespace pybind_wrappers {
        pybind_wrap_api* wrappers;


        void* load_libraries() {
            const auto pylib = std::getenv("NMODL_PYLIB");
            if (!pylib) {
                logger->critical("Trying to load libpython dynamically but did not find it set in NMODL_PYLIB environment variable");
                std::exit(EXIT_FAILURE);
            }
            const auto dlopen_opts = RTLD_NOW|RTLD_GLOBAL;
            const auto pylib_handle = dlopen(pylib, dlopen_opts);
            if (!pylib_handle) {
                logger->critical("Tried to but failed to load {}", pylib);
                std::exit(EXIT_FAILURE);
            }
            const char* pybind_wrap_lib = "./lib/python/nmodl/libpywrapper.dylib";
            const auto pybind_wrap_handle = dlopen(pybind_wrap_lib, dlopen_opts);
            if (!pybind_wrap_handle) {
                logger->critical("Tried to but failed to load {}", pybind_wrap_lib);
                std::exit(EXIT_FAILURE);
            }

            return pybind_wrap_handle;
        }

        void populate_symbols(void* pybind_wrap_handle) {
            wrappers = static_cast<pybind_wrap_api *>(dlsym(pybind_wrap_handle, "wrapper_api"));
        }

        void loader() {
            auto pybind_wrap_handle = load_libraries();
            populate_symbols(pybind_wrap_handle);
        }


    } // pybind_wrappers

} // nmodl
