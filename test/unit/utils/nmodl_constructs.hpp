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

#pragma once

#include <map>
#include <string>
#include <vector>

namespace nmodl {
namespace test_utils {

/// represent nmodl test construct
struct NmodlTestCase {
    /// name of the test
    std::string name;

    /// input nmodl construct
    std::string input;

    /// expected nmodl output
    std::string output;

    /// \todo : add associated json (to use in visitor test)

    NmodlTestCase() = delete;

    NmodlTestCase(std::string name, std::string input)
        : name(name)
        , input(input)
        , output(input) {}

    NmodlTestCase(std::string name, std::string input, std::string output)
        : name(name)
        , input(input)
        , output(output) {}
};

/// represent differential equation test construct
struct DiffEqTestCase {
    /// name of the mod file
    std::string name;

    /// differential equation to solve
    std::string equation;

    /// expected solution
    std::string solution;

    /// solve method
    std::string method;
};

extern std::map<std::string, NmodlTestCase> const nmodl_invalid_constructs;
extern std::map<std::string, NmodlTestCase> const nmodl_valid_constructs;
extern std::vector<DiffEqTestCase> const diff_eq_constructs;

}  // namespace test_utils
}  // namespace nmodl
