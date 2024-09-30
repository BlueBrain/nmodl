/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#define CATCH_CONFIG_RUNNER

#include "test/unit/utils/catch2.hpp"
#if NMODL_CATCH2_VERSION_MAJOR == 3
#include <catch2/catch_session.hpp>
#endif

#include "pybind/pyembed.hpp"
#include "utils/logger.hpp"

using namespace nmodl;

int main(int argc, char* argv[]) {
    // initialize python interpreter once for entire catch executable
    nmodl::pybind_wrappers::EmbeddedPythonLoader::get_instance().api().initialize_interpreter();
    // enable verbose logger output
    logger->set_level(spdlog::level::debug);
    // run all catch tests
    int result = Catch::Session().run(argc, argv);
    nmodl::pybind_wrappers::EmbeddedPythonLoader::get_instance().api().finalize_interpreter();
    return result;
}
