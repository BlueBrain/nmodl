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
#include "visitors/local_to_assigned_visitor.hpp"
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

std::shared_ptr<ast::Program> run_local_to_assigned_visitor(const std::string& text) {
    std::map<std::string, std::string> rval;
    NmodlDriver driver;
    auto ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    PerfVisitor().visit_program(*ast);
    LocalToAssignedVisitor().visit_program(*ast);
    SymtabVisitor().visit_program(*ast);
    return ast;
}

SCENARIO("LOCAL to ASSIGNED variable transformer", "[visitor][localtoassigned]") {
    GIVEN("mod file with LOCAL variables that are written") {
        std::string input_nmodl = R"(
            NEURON {
                SUFFIX test
            }

            LOCAL x, y, z

            INITIAL {
                x = 1
            }

            BREAKPOINT {
                z = 2
            }
        )";

        auto ast = run_local_to_assigned_visitor(input_nmodl);
        auto symtab = ast->get_symbol_table();

        THEN("LOCAL variables that are written are turned to ASSIGNED") {
            /// check for all ASSIGNED variables : old ones + newly converted ones
            auto vars = symtab->get_variables_with_properties(NmodlType::assigned_definition);
            REQUIRE(vars.size() == 2);

            /// x and z should be converted from LOCAL to ASSIGNED
            auto x = symtab->lookup("x");
            REQUIRE(x != nullptr);
            REQUIRE(x->has_any_property(NmodlType::assigned_definition) == true);
            REQUIRE(x->has_any_property(NmodlType::local_var) == false);

            auto z = symtab->lookup("z");
            REQUIRE(z != nullptr);
            REQUIRE(z->has_any_property(NmodlType::assigned_definition) == true);
            REQUIRE(z->has_any_property(NmodlType::local_var) == false);
        }

        THEN("LOCAL variables that are read only remain LOCAL") {
            auto vars = symtab->get_variables_with_properties(NmodlType::local_var);
            REQUIRE(vars.size() == 1);
            REQUIRE(vars[0]->get_name() == "y");
        }
    }
}
