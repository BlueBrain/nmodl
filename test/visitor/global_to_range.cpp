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

std::string run_global_to_var_visitor(const std::string& text) {
    NmodlDriver driver;
    auto ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    PerfVisitor().visit_program(*ast);
    GlobalToRangeVisitor(ast).visit_program(*ast);
    SymtabVisitor().visit_program(*ast);
    std::stringstream ss;
    NmodlPrintVisitor(ss).visit_program(*ast);
    return ss.str();
}

SCENARIO("GLOBAL to RANGE variable transformer", "[visitor][globaltorange]") {
    GIVEN("mod file with GLOBAL variables that are written") {
        std::string input_nmodl = R"(
            UNITS {
                (mV) = (millivolt)
            }

            NEURON {
                GLOBAL minf, hinf, ninf, mtau, htau, ntau
            }

            STATE {
                m h n
            }

            ASSIGNED {
                minf ninf
                mtau (ms) htau (ms) ntau (ms)
            }

            INITIAL {
                rates(v)
                m = minf
                h = hinf
                n = ninf
            }

            DERIVATIVE states {
                rates(v)
                m' =  (minf-m)/mtau
                h' = (hinf-h)/htau
                n' = (ninf-n)/ntau
            }

            PROCEDURE rates(v(mV)) {
            UNITSOFF
                mtau = mtau+1
                minf = minf+1
                htau = htau+1
                ntau = ntau+1
                ninf = ninf+1
            }
            UNITSON
        )";
        std::string output_range = "RANGE minf, ninf, mtau, htau, ntau";
        std::string output_global = "GLOBAL hinf";
        THEN("GLOBAL variables that are written are turned to RANGE") {
            auto result = run_global_to_var_visitor(input_nmodl);
            REQUIRE(result.find(output_range) != std::string::npos);
            REQUIRE(result.find(output_global) != std::string::npos);
        }
    }
}
