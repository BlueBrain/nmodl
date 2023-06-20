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

#include <catch2/catch_test_macros.hpp>

#include "codegen/codegen_cpp_visitor.hpp"
#include "codegen/codegen_utils.hpp"

using namespace nmodl;
using namespace visitor;
using namespace codegen;

using input_result_map = std::unordered_map<std::string, std::string>;

SCENARIO("C codegen utility functions", "[codegen][util][c]") {
    GIVEN("Double constant as string") {
        std::string double_constant = "0.012345678901234567";

        THEN("Codegen C Visitor prints double with same precision") {
            auto nmodl_constant_result = codegen::utils::format_double_string<CodegenCVisitor>(
                double_constant);
            REQUIRE(nmodl_constant_result == double_constant);
        }
    }

    GIVEN("Integer constant as string") {
        std::string double_constant = "1";

        std::string codegen_output = "1.0";

        THEN("Codegen C Visitor prints integer as double number") {
            auto nmodl_constant_result = codegen::utils::format_double_string<CodegenCVisitor>(
                double_constant);
            REQUIRE(nmodl_constant_result == codegen_output);
        }
    }

    GIVEN("Double constants in scientific notation as strings") {
        input_result_map tests({{"1e+18", "1e+18"}, {"1e-18", "1e-18"}, {"1E18", "1E18"}});

        THEN("Codegen C Visitor prints doubles with scientific notation") {
            for (const auto& test: tests) {
                REQUIRE(codegen::utils::format_double_string<CodegenCVisitor>(test.first) ==
                        test.second);
            }
        }
    }

    GIVEN("Float constant as string") {
        std::string float_constant = "0.01234567";

        THEN("Codegen C Visitor prints float with same precision") {
            auto nmodl_constant_result = codegen::utils::format_float_string<CodegenCVisitor>(
                float_constant);
            REQUIRE(nmodl_constant_result == float_constant);
        }
    }

    GIVEN("Float constant as string") {
        std::string float_constant = "1";

        std::string codegen_output = "1.0";

        THEN("Codegen C Visitor prints integer as double number") {
            auto nmodl_constant_result = codegen::utils::format_float_string<CodegenCVisitor>(
                float_constant);
            REQUIRE(nmodl_constant_result == codegen_output);
        }
    }

    GIVEN("Float constants in scientific notation as strings") {
        input_result_map tests({{"1e+18", "1e+18"}, {"1e-18", "1e-18"}, {"1E18", "1E18"}});

        THEN("Codegen C Visitor prints doubles with scientific notation") {
            for (const auto& test: tests) {
                REQUIRE(codegen::utils::format_float_string<CodegenCVisitor>(test.first) ==
                        test.second);
            }
        }
    }
}
