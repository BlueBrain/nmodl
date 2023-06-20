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
#include "visitors/visitor_utils.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test_utils;

using ast::AstNodeType;
using nmodl::parser::NmodlDriver;

//=============================================================================
// Ast lookup visitor tests
//=============================================================================

SCENARIO("Searching for ast nodes using AstLookupVisitor", "[visitor][lookup]") {
    auto to_ast = [](const std::string& text) {
        NmodlDriver driver;
        return driver.parse_string(text);
    };

    GIVEN("A mod file with nodes of type NEURON, RANGE, BinaryExpression") {
        std::string nmodl_text = R"(
            NEURON {
                RANGE tau, h
            }

            DERIVATIVE states {
                tau = 11.1
                exp(tau)
                h' = h + 2
            }

            : My comment here
        )";

        auto ast = to_ast(nmodl_text);

        WHEN("Looking for existing nodes") {
            THEN("Can find RANGE variables") {
                const auto& result = collect_nodes(*ast, {AstNodeType::RANGE_VAR});
                REQUIRE(result.size() == 2);
                REQUIRE(to_nmodl(result[0]) == "tau");
                REQUIRE(to_nmodl(result[1]) == "h");
            }

            THEN("Can find NEURON block") {
                const auto& nodes = collect_nodes(*ast, {AstNodeType::NEURON_BLOCK});
                REQUIRE(nodes.size() == 1);

                const std::string neuron_block = R"(
                    NEURON {
                        RANGE tau, h
                    })";
                const auto& result = reindent_text(to_nmodl(nodes[0]));
                const auto& expected = reindent_text(neuron_block);
                REQUIRE(result == expected);
            }

            THEN("Can find Binary Expressions and function call") {
                const auto& result =
                    collect_nodes(*ast,
                                  {AstNodeType::BINARY_EXPRESSION, AstNodeType::FUNCTION_CALL});
                REQUIRE(result.size() == 4);
            }
        }

        WHEN("Looking for missing nodes") {
            THEN("Can not find BREAKPOINT block") {
                const auto& result = collect_nodes(*ast, {AstNodeType::BREAKPOINT_BLOCK});
                REQUIRE(result.empty());
            }
        }
    }
}
