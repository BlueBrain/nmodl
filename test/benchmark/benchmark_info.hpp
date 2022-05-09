/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>
#include <vector>

/// A struct to hold the information for benchmarking.
struct BenchmarkInfo {
    /// Object or PTX filename to dump.
    std::string filename;

    /// Object file output directory.
    std::string output_dir;

    /// Shared libraries' paths to link against.
    std::vector<std::string> shared_lib_paths;

    /// Optimisation level for IT.
    int opt_level_ir;

    /// Optimisation level for machine code generation.
    int opt_level_codegen;
};
