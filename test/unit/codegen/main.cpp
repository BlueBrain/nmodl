/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pybind/pyembed.hpp"
#include "utils/logger.hpp"

using namespace nmodl;

int main(int argc, char* argv[]) {
    // initialize python interpreter once for entire catch executable
    nmodl::pybind_wrappers::EmbeddedPythonLoader::get_instance().api()->initialize_interpreter();
    // enable verbose logger output
    logger->set_level(spdlog::level::debug);
    // run all catch tests
    int result = Catch::Session().run(argc, argv);
    nmodl::pybind_wrappers::EmbeddedPythonLoader::get_instance().api()->finalize_interpreter();
    return result;
}
