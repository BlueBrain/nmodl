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

        bool EmbeddedPythonLoader::have_wrappers() {
            wrappers = static_cast<pybind_wrap_api *>(dlsym(RTLD_NEXT, "wrapper_api"));
            return wrappers != nullptr;
        }

        void EmbeddedPythonLoader::load_libraries() {
            const auto pylib = std::getenv("NMODL_PYLIB");
            if (!pylib) {
                logger->critical("NMODL_PYLIB environment variable must be set to load embedded python");
                throw std::runtime_error("NMODL_PYLIB not set");
            }
            const auto dlopen_opts = RTLD_NOW|RTLD_GLOBAL;
            pylib_handle = dlopen(pylib, dlopen_opts);
            if (!pylib_handle) {
                const auto errstr = dlerror();
                logger->critical("Tried but failed to load {}", pylib);
                logger->critical(errstr);
                throw std::runtime_error("Failed to dlopen");
            }
#if defined(__APPLE__)
            const char* pybind_wrap_lib = "./lib/python/nmodl/libpywrapper.dylib";
#else
            const char* pybind_wrap_lib = "./lib/python/nmodl/libpywrapper.so";
#endif
            pybind_wrapper_handle = dlopen(pybind_wrap_lib, dlopen_opts);
            if (!pybind_wrapper_handle) {
                const auto errstr = dlerror();
                logger->critical("Tried but failed to load {}", pybind_wrap_lib);
                logger->critical(errstr);
                throw std::runtime_error("Failed to dlopen");
            }

        }

        void EmbeddedPythonLoader::populate_symbols() {
            wrappers = static_cast<pybind_wrap_api *>(dlsym(pybind_wrapper_handle, "wrapper_api"));
            if (!wrappers) {
                const auto errstr = dlerror();
                logger->critical("Tried but failed to load pybind wrapper symbols");
                logger->critical(errstr);
                throw std::runtime_error("Failed to dlsym");
            }
        }

        void EmbeddedPythonLoader::unload() {
            dlclose(pybind_wrapper_handle);
            dlclose(pylib_handle);
        }

        const pybind_wrap_api* EmbeddedPythonLoader::api() {
            return wrappers;
        }


    } // pybind_wrappers

} // nmodl
