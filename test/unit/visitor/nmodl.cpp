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
#include "test/unit/utils/test_utils.hpp"
#include "utils/common_utils.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test;
using namespace test_utils;

using nmodl::parser::NmodlDriver;


//=============================================================================
// AST to NMODL printer tests
//=============================================================================

std::string run_nmodl_visitor(const std::string& text) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    std::stringstream stream;
    NmodlPrintVisitor(stream).visit_program(*ast);

    // check that, after visitor rearrangement, parents are still up-to-date
    CheckParentVisitor().check_ast(*ast);

    return stream.str();
}

SCENARIO("Convert AST back to NMODL form", "[visitor][nmodl]") {
    TempFile unit("Unit.inc", nmodl_valid_constructs.at("unit_statement_1").input);
    for (const auto& construct: nmodl_valid_constructs) {
        auto test_case = construct.second;
        const std::string& input_nmodl_text = reindent_text(test_case.input);
        const std::string& output_nmodl_text = reindent_text(test_case.output);
        GIVEN(test_case.name) {
            THEN("Visitor successfully returns : " + input_nmodl_text) {
                auto result = run_nmodl_visitor(input_nmodl_text);
                REQUIRE(result == output_nmodl_text);
            }
        }
    }
}
