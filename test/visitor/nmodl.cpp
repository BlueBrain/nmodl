/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "catch/catch.hpp"

#include "parser/nmodl_driver.hpp"
#include "test/utils/nmodl_constructs.hpp"
#include "test/utils/test_utils.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test_utils;

using nmodl::parser::NmodlDriver;


//=============================================================================
// AST to NMODL printer tests
//=============================================================================

std::string run_nmodl_visitor(const std::string& text) {
    NmodlDriver driver;
    auto ast = driver.parse_string(text);

    std::stringstream stream;
    NmodlPrintVisitor(stream).visit_program(ast.get());

    // check that, after visitor rearrangement, parents are still up-to-date
    CheckParentVisitor().visit_program(ast.get());

    return stream.str();
}

SCENARIO("Convert AST back to NMODL form", "[visitor][nmodl]") {
    for (const auto& construct: nmodl_valid_constructs) {
        auto test_case = construct.second;
        std::string input_nmodl_text = reindent_text(test_case.input);
        std::string output_nmodl_text = reindent_text(test_case.output);
        GIVEN(test_case.name) {
            THEN("Visitor successfully returns : " + input_nmodl_text) {
                auto result = run_nmodl_visitor(input_nmodl_text);
                REQUIRE(result == output_nmodl_text);
            }
        }
    }
}
