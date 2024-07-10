/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \dir
 * \brief Global project configurations
 *
 * \file
 * \brief Version information and units file path
 */

#include <string>
#include <vector>

#include "utils/common_utils.hpp"

namespace nmodl {

/**
 * \brief Project version information
 */
struct Version {
    /// git revision id
    static const std::string GIT_REVISION;

    /// project tagged version in the cmake
    static const std::string NMODL_VERSION;

    /// return version string (version + git id) as a string
    static std::string to_string() {
        return NMODL_VERSION + " " + GIT_REVISION;
    }
};

/**
 * \brief Information of units database i.e. `nrnunits.lib`
 */
class PathHelper {
    /// pre-defined paths to search for files
    const static std::vector<std::string> BASE_SEARCH_PATHS;

    /// suffix to use when looking for libraries
    const static std::string SHARED_LIBRARY_SUFFIX;

    /// base directory of the NMODL installation
    static std::string nmodl_home;

    /**
     * Search for a given relative file path
     */
    static std::string get_path(const std::string& what, bool add_library_suffix=false);

  public:
    /**
     * Set the NMODL base installation directory from the executable if not defined in the
     * environment
     */
    static void setup(const std::string& executable);

    /**
     * Return the NMODL base installation directory
     */
    static std::string get_home() {
        return nmodl_home;
    }

    /**
     * Return path of units database file
     */
    static std::string get_units_path() {
        return get_path("share/nmodl/nrnunits.lib");
    };

    /**
     * Return path of the python wrapper library
     */
    static std::string get_wrapper_path() {
        return get_path("lib/libpywrapper", true);
    };
};

}  // namespace nmodl
