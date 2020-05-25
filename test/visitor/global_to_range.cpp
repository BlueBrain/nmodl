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
#include "visitors/global_var_visitor.hpp"
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

std::map<std::string, std::string> run_global_to_var_visitor(const std::string& text) {
    std::map<std::string, std::string> rval;
    NmodlDriver driver;
    auto ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    PerfVisitor().visit_program(*ast);
    GlobalToRangeVisitor(ast).visit_program(*ast);
    SymtabVisitor().visit_program(*ast);
    std::stringstream ss;
    NmodlPrintVisitor(ss).visit_program(*ast);
    rval["nmodl"] = ss.str();
    ss.str("");
    auto variables = ast->get_symbol_table()->get_variables_with_properties(NmodlType::range_var);
    for (const auto& variable: variables) {
        ss << variable->get_name() << std::endl;
    }
    rval["symtab_range"] = ss.str();
    return rval;
}

SCENARIO("GLOBAL to RANGE variable transformer", "[visitor][globaltorange]") {
    GIVEN("mod file with GLOBAL variables that are written") {
        std::string input_nmodl = R"(
            NEURON {
                SUFFIX test
                GLOBAL x, y
            }
            ASSIGNED {
                x
            }
            BREAKPOINT {
                x = y
            }
        )";
        std::string output_range = "RANGE x";
        std::string output_global = "GLOBAL y";
        std::string symtab_range_vars = "x\n";
        THEN("GLOBAL variables that are written are turned to RANGE") {
            auto result = run_global_to_var_visitor(input_nmodl);
            REQUIRE(result["nmodl"].find(output_range) != std::string::npos);
            REQUIRE(result["nmodl"].find(output_global) != std::string::npos);
            REQUIRE(result["symtab_range"] == symtab_range_vars);
        }
    }
}
