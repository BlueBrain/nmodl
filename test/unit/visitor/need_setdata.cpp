/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include "ast/program.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/nmodl_constructs.hpp"
#include "visitors/needsetdata_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test_utils;

using nmodl::parser::NmodlDriver;
using symtab::syminfo::NmodlType;

//=============================================================================
// NeedSetData visitor tests
//=============================================================================

std::shared_ptr<ast::Program> run_NeedSetData_visitor(const std::string& text) {
    NmodlDriver driver;
    auto ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    NeedSetDataVisitor().visit_program(*ast);
    return ast;
}

SCENARIO("Check whether PROCEDURE and FUNCTION need setdata call", "[visitor][needsetdata]") {
    GIVEN("mod file with GLOBAL and RANGE variables used in FUNC and PROC") {
        std::string input_nmodl = R"(
            NEURON {
                SUFFIX test
                RANGE x
                GLOBAL s
            }
            PARAMETER {
                s = 2
            }
            ASSIGNED {
                x
            }
            PROCEDURE a() {
                x = get_42()
            }
            FUNCTION b() {
                a()
            }
            FUNCTION get_42() {
                get_42 = 42
            }
        )";
        auto ast = run_NeedSetData_visitor(input_nmodl);
        auto symtab = ast->get_symbol_table();
        THEN("need_setdata property is added to needed FUNC and PROC") {
            auto need_setdata_funcs = symtab->get_variables_with_properties(NmodlType::need_setdata);
            REQUIRE(need_setdata_funcs.size() == 2);
            const auto a = symtab->lookup("a");
            REQUIRE(a->has_any_property(NmodlType::need_setdata));
            const auto b = symtab->lookup("b");
            REQUIRE(b->has_any_property(NmodlType::need_setdata));
            const auto get_42 = symtab->lookup("get_42");
            REQUIRE(!get_42->has_any_property(NmodlType::need_setdata));
        }
    }
}
