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

#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/test_utils.hpp"
#include "visitors/implicit_argument_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/visitor_utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace nmodl;
using nmodl::test_utils::reindent_text;

using Catch::Matchers::ContainsSubstring;  // ContainsSubstring in newer Catch2

//=============================================================================
// Implicit visitor tests
//=============================================================================

std::string generate_mod_after_implicit_argument_visitor(std::string const& text) {
    parser::NmodlDriver driver{};
    auto const ast = driver.parse_string(text);
    visitor::SymtabVisitor{}.visit_program(*ast);
    visitor::ImplicitArgumentVisitor{}.visit_program(*ast);
    return to_nmodl(*ast);
}

SCENARIO("Check insertion of implicit arguments", "[codegen][implicit_arguments]") {
    GIVEN("A mod file that calls functions that may need implicit arguments") {
        auto const nmodl_text = R"(
            NEURON {
                SUFFIX ImplicitTest
            }
            INITIAL {
                at_time(foo)
                at_time(a+b)
                at_time(nt, flop)
                nrn_ghk(-50(mV), .001(mM), 10(mM), 2)
                nrn_ghk(-50(mV), .001(mM), 10(mM), 2, -273.15)
            }
        )";
        auto const modified_nmodl = generate_mod_after_implicit_argument_visitor(nmodl_text);
        THEN("at_time should have nt as its first argument") {
            REQUIRE_THAT(modified_nmodl, ContainsSubstring("at_time(nt, foo)"));
            REQUIRE_THAT(modified_nmodl, ContainsSubstring("at_time(nt, a+b)"));
            REQUIRE_THAT(modified_nmodl, ContainsSubstring("at_time(nt, flop)"));
        }
        THEN("nrn_ghk should have a temperature as its last argument") {
            REQUIRE_THAT(modified_nmodl,
                         ContainsSubstring("nrn_ghk(-50(mV), .001(mM), 10(mM), 2, celsius)"));
            REQUIRE_THAT(modified_nmodl,
                         ContainsSubstring("nrn_ghk(-50(mV), .001(mM), 10(mM), 2, -273.15)"));
        }
    }
}
