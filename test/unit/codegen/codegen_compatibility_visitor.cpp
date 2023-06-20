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
#include <catch2/matchers/catch_matchers_string.hpp>

#include "ast/program.hpp"
#include "codegen/codegen_compatibility_visitor.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/test_utils.hpp"
#include "visitors/perf_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using Catch::Matchers::ContainsSubstring;

using namespace nmodl;
using namespace visitor;
using namespace codegen;

using nmodl::parser::NmodlDriver;

/// Return true if it failed and false otherwise
bool runCompatibilityVisitor(const std::string& nmodl_text) {
    const auto& ast = NmodlDriver().parse_string(nmodl_text);
    SymtabVisitor().visit_program(*ast);
    PerfVisitor().visit_program(*ast);
    return CodegenCompatibilityVisitor().find_unhandled_ast_nodes(*ast);
}

SCENARIO("Uncompatible constructs should failed", "[codegen][compatibility_visitor]") {
    GIVEN("A mod file containing an EXTERNAL construct") {
        std::string const nmodl_text = R"(
            NEURON {
                EXTERNAL apc_metap
            }
        )";

        THEN("should failed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(failed);
        }
    }
    GIVEN("A mod file containing a written GLOBAL var") {
        std::string const nmodl_text = R"(
            NEURON {
                GLOBAL foo
            }

            PROCEDURE bar() {
                foo = 1
            }
        )";

        THEN("should failed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(failed);
        }
    }
    GIVEN("A mod file containing a written un-writtable var") {
        std::string const nmodl_text = R"(
            PARAMETER {
                foo = 1
            }

            PROCEDURE bar() {
                foo = 1
            }
        )";

        THEN("should failed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(failed);
        }
    }
    GIVEN("A mod file with BBCOREPOINTER without bbcore_read / bbcore_write") {
        std::string const nmodl_text = R"(
            NEURON {
                BBCOREPOINTER rng
            }
        )";

        THEN("should failed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(failed);
        }
    }
    GIVEN("A mod file with BBCOREPOINTER without bbcore_write") {
        std::string const nmodl_text = R"(
            NEURON {
                BBCOREPOINTER rng
            }

            VERBATIM
            static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
            }
            ENDVERBATIM
        )";

        THEN("should failed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(failed);
        }
    }
    GIVEN("A mod file with BBCOREPOINTER without bbcore_read") {
        std::string const nmodl_text = R"(
            NEURON {
                BBCOREPOINTER rng
            }

            VERBATIM
            static void bbcore_write(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
            }
            ENDVERBATIM
        )";

        THEN("should failed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(failed);
        }
    }
    GIVEN("A mod file with BBCOREPOINTER with bbcore_read / bbcore_write") {
        std::string const nmodl_text = R"(
            NEURON {
                BBCOREPOINTER rng
            }

            VERBATIM
            static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
            }
            static void bbcore_write(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
            }
            ENDVERBATIM
        )";

        THEN("should succeed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(!failed);
        }
    }
    GIVEN("A mod file with a no SOLVE method") {
        std::string const nmodl_text = R"(
            BREAKPOINT {
                SOLVE state
            }
        )";

        THEN("should succeed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(!failed);
        }
    }
    GIVEN("A mod file with a invalid SOLVE method") {
        std::string const nmodl_text = R"(
            BREAKPOINT {
                SOLVE state METHOD runge
            }
        )";

        THEN("should failed") {
            bool failed = runCompatibilityVisitor(nmodl_text);
            REQUIRE(failed);
        }
    }
}
