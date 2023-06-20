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

#include "ast/program.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/nmodl_constructs.hpp"
#include "visitors/global_var_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/perf_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test_utils;

using nmodl::parser::NmodlDriver;
using symtab::syminfo::NmodlType;

//=============================================================================
// GlobalToRange visitor tests
//=============================================================================

std::shared_ptr<ast::Program> run_global_to_var_visitor(const std::string& text) {
    NmodlDriver driver;
    auto ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    PerfVisitor().visit_program(*ast);
    GlobalToRangeVisitor(*ast).visit_program(*ast);
    SymtabVisitor().visit_program(*ast);
    return ast;
}

SCENARIO("GLOBAL to RANGE variable transformer", "[visitor][globaltorange]") {
    GIVEN("mod file with GLOBAL variables that are written") {
        std::string input_nmodl = R"(
            NEURON {
                SUFFIX test
                RANGE a, b
                GLOBAL x, y
            }
            ASSIGNED {
                x
            }
            BREAKPOINT {
                x = y
            }
        )";
        auto ast = run_global_to_var_visitor(input_nmodl);
        auto symtab = ast->get_symbol_table();
        THEN("GLOBAL variables that are written are turned to RANGE") {
            /// check for all RANGE variables : old ones + newly converted ones
            auto vars = symtab->get_variables_with_properties(NmodlType::range_var);
            REQUIRE(vars.size() == 3);

            /// x should be converted from GLOBAL to RANGE
            auto x = symtab->lookup("x");
            REQUIRE(x != nullptr);
            REQUIRE(x->has_any_property(NmodlType::range_var) == true);
            REQUIRE(x->has_any_property(NmodlType::global_var) == false);
        }
        THEN("GLOBAL variables that are read only remain GLOBAL") {
            auto vars = symtab->get_variables_with_properties(NmodlType::global_var);
            REQUIRE(vars.size() == 1);
            REQUIRE(vars[0]->get_name() == "y");
        }
    }
}
