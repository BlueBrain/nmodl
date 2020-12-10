/*************************************************************************
 * Copyright (C) 2019-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#define CATCH_CONFIG_MAIN

#include <catch/catch.hpp>

#include "ast/program.hpp"
#include "codegen/codegen_ispc_visitor.hpp"
#include "config/config.h"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/test_utils.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/units_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace codegen;
using namespace test_utils;

using nmodl::NrnUnitsLib;
using nmodl::parser::NmodlDriver;

//=============================================================================
// Helper for codegen related visitor
//=============================================================================
std::string print_ispc_nmodl_constants(const std::string& text) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    /// directory where units lib file is located
    std::string units_dir(NrnUnitsLib::get_path());
    /// parse units of text
    UnitsVisitor(units_dir).visit_program(*ast);

    /// construct symbol table
    SymtabVisitor().visit_program(*ast);

    /// initialize CodegenIspcVisitor
    std::stringbuf strbuf;
    std::ostream oss(&strbuf);
    CodegenIspcVisitor visitor("unit_test", oss, codegen::LayoutType::soa, "double", false);
    visitor.setup(*ast);

    /// print nmodl constants
    visitor.print_nmodl_constants();

    return strbuf.str();
}

std::string print_ispc_compute_functions(const std::string& text) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    /// directory where units lib file is located
    std::string units_dir(NrnUnitsLib::get_path());
    /// parse units of text
    UnitsVisitor(units_dir).visit_program(*ast);

    /// construct symbol table
    SymtabVisitor().visit_program(*ast);

    /// initialize CodegenIspcVisitor
    std::stringbuf strbuf;
    std::ostream oss(&strbuf);
    CodegenIspcVisitor visitor("unit_test", oss, codegen::LayoutType::soa, "double", false);
    visitor.setup(*ast);

    /// print compute functions
    visitor.print_compute_functions();

    return strbuf.str();
}


SCENARIO("ISPC codegen", "[codegen][ispc]") {
    GIVEN("Mod file that has multiple double, float and in constants") {
        std::string nmodl_text = R"(
            TITLE UnitTest
            NEURON {
                SUFFIX unit_test
                RANGE a, b
            }
            UNITS {
                FARADAY = (faraday) (coulomb)
            }
            INITIAL {
                a = 0.
                b = .0
            }
            BREAKPOINT {
                LOCAL x, y
                x = 1E-18 + FARADAY * 1.2345
                y = 1e+18 + FARADAY * .1234
                a = x * 1.012345678901234567 + y
                b = a + 1 + 2.0
            }
        )";

        std::string nmodl_constants_declaration = R"(
            /** constants used in nmodl */
            static const uniform double FARADAY = 96485.3321233100141d;
        )";

        std::string nrn_init_state_block = R"(
            /** initialize channel */
            export void nrn_init_unit_test(uniform unit_test_Instance* uniform inst, uniform NrnThread* uniform nt, uniform Memb_list* uniform ml, uniform int type) {
                uniform int nodecount = ml->nodecount;
                uniform int pnodecount = ml->_nodecount_padded;
                const int* uniform node_index = ml->nodeindices;
                double* uniform data = ml->data;
                const double* uniform voltage = nt->_actual_v;
                Datum* uniform indexes = ml->pdata;
                ThreadDatum* uniform thread = ml->_thread;

                int uniform start = 0;
                int uniform end = nodecount;
                foreach (id = start ... end) {
                    int node_id = node_index[id];
                    double v = voltage[node_id];
                    a = 0.d;
                    b = 0.0d;
                }
            }


            /** update state */
            export void nrn_state_unit_test(uniform unit_test_Instance* uniform inst, uniform NrnThread* uniform nt, uniform Memb_list* uniform ml, uniform int type) {
                uniform int nodecount = ml->nodecount;
                uniform int pnodecount = ml->_nodecount_padded;
                const int* uniform node_index = ml->nodeindices;
                double* uniform data = ml->data;
                const double* uniform voltage = nt->_actual_v;
                Datum* uniform indexes = ml->pdata;
                ThreadDatum* uniform thread = ml->_thread;

                int uniform start = 0;
                int uniform end = nodecount;
                foreach (id = start ... end) {
                    int node_id = node_index[id];
                    double v = voltage[node_id];
                    
                    double x, y;
                    x = 1d-18 + FARADAY * 1.2345d;
                    y = 1d+18 + FARADAY * 0.1234d;
                    a = x * 1.012345678901234567d + y;
                    b = a + 1.0d + 2.0d;
                }
            }
        )";
        THEN("Check that the nmodl constants and computer functions are printed correctly") {
            auto nmodl_constants_result = reindent_text(print_ispc_nmodl_constants(nmodl_text));
            REQUIRE(nmodl_constants_result == reindent_text(nmodl_constants_declaration));
            auto nmodl_init_cur_state = reindent_text(print_ispc_compute_functions(nmodl_text));
            REQUIRE(nmodl_init_cur_state == reindent_text(nrn_init_state_block));
        }
    }
}
