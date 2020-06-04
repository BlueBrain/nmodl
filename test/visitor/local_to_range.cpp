/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "catch/catch.hpp"

#include "ast/program.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/utils/nmodl_constructs.hpp"
#include "visitors/local_var_visitor.hpp"
#include "visitors/lookup_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/perf_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test_utils;

using ast::AstNodeType;
using nmodl::parser::NmodlDriver;
using symtab::syminfo::NmodlType;

//=============================================================================
// GlobalToRange visitor tests
//=============================================================================

std::shared_ptr<ast::Program> run_local_to_var_visitor(const std::string& text) {
    std::map<std::string, std::string> rval;
    NmodlDriver driver;
    auto ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    PerfVisitor().visit_program(*ast);
    LocalToRangeVisitor().visit_program(*ast);
    SymtabVisitor().visit_program(*ast);
    return ast;
}

SCENARIO("LOCAL to RANGE variable transformer", "[visitor][localtorange]") {
    GIVEN("mod file with LOCAL variables that are written") {
        std::string input_nmodl = R"(
            NEURON {
                SUFFIX test
            }
            LOCAL x, y
            BREAKPOINT {
                x = 1
            }
        )";
        auto ast = run_local_to_var_visitor(input_nmodl);
        auto symtab = ast->get_symbol_table();
        THEN("LOCAL variables that are written are turned to RANGE") {
            /// check for all RANGE variables : old ones + newly converted ones
            auto vars = symtab->get_variables_with_properties(NmodlType::range_var);
            REQUIRE(vars.size() == 1);

            /// x should be converted from LOCAL to RANGE
            auto x = symtab->lookup("x");
            REQUIRE(x != nullptr);
            REQUIRE(x->has_any_property(NmodlType::range_var) == true);
            REQUIRE(x->has_any_property(NmodlType::local_var) == false);
        }
        THEN("LOCAL variables that are read only remain LOCAL") {
            auto vars = symtab->get_variables_with_properties(NmodlType::local_var);
            REQUIRE(vars.size() == 1);
            REQUIRE(vars[0]->get_name() == "y");
        }
    }
}
