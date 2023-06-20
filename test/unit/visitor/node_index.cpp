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
#include "test/unit/utils/test_utils.hpp"
#include "visitors/indexedname_visitor.hpp"
#include "visitors/visitor_utils.hpp"

using namespace nmodl;
using namespace visitor;

//=============================================================================
// Get the indexed node name and the dependencies of differential equations
//=============================================================================
std::pair<std::string, std::pair<std::string, std::unordered_set<std::string>>>
get_indexedname_dependencies(ast::Program& node) {
    IndexedNameVisitor testvisitor;
    testvisitor.visit_program(node);
    return std::make_pair(testvisitor.get_indexed_name(), testvisitor.get_dependencies());
}

SCENARIO("Get node name with index TestVisitor", "[visitor][node_index]") {
    auto to_ast = [](const std::string& text) {
        parser::NmodlDriver driver;
        return driver.parse_string(text);
    };

    GIVEN("A simple NMODL block") {
        std::string nmodl_text_a = R"(
            STATE {
                m[1]
            }
            BREAKPOINT  {
                SOLVE states METHOD euler
            }
            DERIVATIVE states {
                m'[0] = mInf/mTau
            }
        )";
        std::string nmodl_text_b = R"(
            BREAKPOINT  {
                SOLVE states STEADYSTATE sparse
            }
            DERIVATIVE states {
                m' = m + h
            }
        )";

        WHEN("Get node name with index") {
            THEN("Get node name with index") {
                auto ast = to_ast(nmodl_text_a);
                std::unordered_set<std::string> vars{"mInf", "mTau"};
                std::string var("m[0]");
                auto expect = std::make_pair(var, vars);
                auto result_name = get_indexedname_dependencies(*ast).first;
                auto result_dependencies = get_indexedname_dependencies(*ast).second;
                REQUIRE(result_name == var);
                REQUIRE(result_dependencies.first == expect.first);
                REQUIRE(result_dependencies.second == expect.second);
            }
            THEN("Get dependencies") {
                auto ast = to_ast(nmodl_text_b);
                std::unordered_set<std::string> vars{"m", "h"};
                std::string var("m");
                auto expect = std::make_pair(var, vars);
                auto result_name = get_indexedname_dependencies(*ast).first;
                auto result_dependencies = get_indexedname_dependencies(*ast).second;
                REQUIRE(result_dependencies.first == expect.first);
                REQUIRE(result_dependencies.second == expect.second);
            }
        }
    }
}
