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
#include "visitors/checkparent_visitor.hpp"
#include "visitors/verbatim_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test;
using nmodl::parser::NmodlDriver;


//=============================================================================
// Verbatim visitor tests
//=============================================================================

std::vector<std::string> run_verbatim_visitor(const std::string& text) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    VerbatimVisitor v;
    v.visit_program(*ast);

    // check that, after visitor rearrangement, parents are still up-to-date
    CheckParentVisitor().check_ast(*ast);

    return v.verbatim_blocks();
}

TEST_CASE("Parse VERBATIM block using Verbatim Visitor") {
    SECTION("Single Block") {
        const std::string text = "VERBATIM int a; ENDVERBATIM";
        auto blocks = run_verbatim_visitor(text);

        REQUIRE(blocks.size() == 1);
        REQUIRE(blocks.front() == " int a; ");
    }

    SECTION("Multiple Blocks") {
        std::string text = "VERBATIM int a; ENDVERBATIM VERBATIM float b; ENDVERBATIM";
        auto blocks = run_verbatim_visitor(text);

        REQUIRE(blocks.size() == 2);
        REQUIRE(blocks[0] == " int a; ");
        REQUIRE(blocks[1] == " float b; ");
    }
}
