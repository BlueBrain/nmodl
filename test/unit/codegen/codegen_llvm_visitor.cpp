/*************************************************************************
 * Copyright (C) 2019-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>

#include "ast/program.hpp"
#include "codegen/codegen_helper_visitor.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "config/config.h"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/test_utils.hpp"
#include "visitors/neuron_solve_visitor.hpp"
#include "visitors/solve_block_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/units_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace codegen;

using nmodl::NrnUnitsLib;
using nmodl::parser::NmodlDriver;
using nmodl::test_utils::reindent_text;

/// Run LLVM codegen visitor and get instance struct declaration and setup of C++ wrapper
std::string get_wrapper_instance_struct(const std::string& nmodl_text) {
    const auto& ast = NmodlDriver().parse_string(nmodl_text);
    std::stringbuf strbuf;
    std::ostream oss(&strbuf);
    /// directory where units lib file is located
    std::string units_dir(NrnUnitsLib::get_path());
    /// parse units of text
    UnitsVisitor(units_dir).visit_program(*ast);
    SymtabVisitor().visit_program(*ast);
    NeuronSolveVisitor().visit_program(*ast);
    SolveBlockVisitor().visit_program(*ast);

    /// create LLVM and C++ wrapper code generation visitor
    codegen::Platform cpu_platform(/*use_single_precision=*/false, /*instruction_width=*/1);
    codegen::CodegenLLVMVisitor llvm_visitor("hh.mod", oss, cpu_platform, 0);
    llvm_visitor.visit_program(*ast);
    strbuf.str("");
    llvm_visitor.print_mechanism_range_var_structure();
    llvm_visitor.print_instance_variable_setup();
    return strbuf.str();
}

SCENARIO("Check instance struct declaration and setup in wrapper",
         "[codegen][llvm][instance_struct]") {
    GIVEN("hh: simple mod file") {
        std::string nmodl_text = R"(
            TITLE hh.mod   squid sodium, potassium, and leak channels

            UNITS {
                (mA) = (milliamp)
                (mV) = (millivolt)
                (S) = (siemens)
            }

            NEURON {
                SUFFIX hh
                USEION na READ ena WRITE ina
                USEION k READ ek WRITE ik
                NONSPECIFIC_CURRENT il
                RANGE gnabar, gkbar, gl, el, gna, gk
                RANGE minf, hinf, ninf, mtau, htau, ntau
                THREADSAFE : assigned GLOBALs will be per thread
            }

            PARAMETER {
                gnabar = .12 (S/cm2)    <0,1e9>
                gkbar = .036 (S/cm2)    <0,1e9>
                gl = .0003 (S/cm2)    <0,1e9>
                el = -54.3 (mV)
            }

            STATE {
                m h n
            }

            ASSIGNED {
                v (mV)
                celsius (degC)
                ena (mV)
                ek (mV)
                gna (S/cm2)
                gk (S/cm2)
                ina (mA/cm2)
                ik (mA/cm2)
                il (mA/cm2)
                minf hinf ninf
                mtau (ms) htau (ms) ntau (ms)
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
                gna = gnabar*m*m*m*h
                ina = gna*(v - ena)
                gk = gkbar*n*n*n*n
                ik = gk*(v - ek)
                il = gl*(v - el)
            }

            DERIVATIVE states {
                m' =  (minf-m)/mtau
                h' = (hinf-h)/htau
                n' = (ninf-n)/ntau
            }
        )";

        std::string generated_instance_struct_declaration = R"(
            struct hh__instance_var__type  {
                const double* __restrict__ gnabar;
                const double* __restrict__ gkbar;
                const double* __restrict__ gl;
                const double* __restrict__ el;
                double* __restrict__ gna;
                double* __restrict__ gk;
                double* __restrict__ il;
                double* __restrict__ minf;
                double* __restrict__ hinf;
                double* __restrict__ ninf;
                double* __restrict__ mtau;
                double* __restrict__ htau;
                double* __restrict__ ntau;
                double* __restrict__ m;
                double* __restrict__ h;
                double* __restrict__ n;
                double* __restrict__ Dm;
                double* __restrict__ Dh;
                double* __restrict__ Dn;
                double* __restrict__ ena;
                double* __restrict__ ek;
                double* __restrict__ ina;
                double* __restrict__ ik;
                double* __restrict__ v_unused;
                double* __restrict__ g_unused;
                const double* __restrict__ ion_ena;
                double* __restrict__ ion_ina;
                double* __restrict__ ion_dinadv;
                const double* __restrict__ ion_ek;
                double* __restrict__ ion_ik;
                double* __restrict__ ion_dikdv;
                int* __restrict__ ion_ena_index;
                int* __restrict__ ion_ina_index;
                int* __restrict__ ion_dinadv_index;
                int* __restrict__ ion_ek_index;
                int* __restrict__ ion_ik_index;
                int* __restrict__ ion_dikdv_index;
                double* __restrict__ voltage;
                int* __restrict__ node_index;
                double t;
                double dt;
                double celsius;
                int secondorder;
                int node_count;
                double* __restrict__ vec_rhs;
                double* __restrict__ vec_d;
                double* __restrict__ _shadow_rhs;
                double* __restrict__ _shadow_d;
            };
        )";
        std::string generated_instance_struct_setup = R"(
            static inline void setup_instance(NrnThread* nt, Memb_list* ml)  {
                hh__instance_var__type* inst = (hh__instance_var__type*) mem_alloc(1, sizeof(hh__instance_var__type));
                int pnodecount = ml->_nodecount_padded;
                Datum* indexes = ml->pdata;
                inst->gnabar = ml->data+0*pnodecount;
                inst->gkbar = ml->data+1*pnodecount;
                inst->gl = ml->data+2*pnodecount;
                inst->el = ml->data+3*pnodecount;
                inst->gna = ml->data+4*pnodecount;
                inst->gk = ml->data+5*pnodecount;
                inst->il = ml->data+6*pnodecount;
                inst->minf = ml->data+7*pnodecount;
                inst->hinf = ml->data+8*pnodecount;
                inst->ninf = ml->data+9*pnodecount;
                inst->mtau = ml->data+10*pnodecount;
                inst->htau = ml->data+11*pnodecount;
                inst->ntau = ml->data+12*pnodecount;
                inst->m = ml->data+13*pnodecount;
                inst->h = ml->data+14*pnodecount;
                inst->n = ml->data+15*pnodecount;
                inst->Dm = ml->data+16*pnodecount;
                inst->Dh = ml->data+17*pnodecount;
                inst->Dn = ml->data+18*pnodecount;
                inst->ena = ml->data+19*pnodecount;
                inst->ek = ml->data+20*pnodecount;
                inst->ina = ml->data+21*pnodecount;
                inst->ik = ml->data+22*pnodecount;
                inst->v_unused = ml->data+23*pnodecount;
                inst->g_unused = ml->data+24*pnodecount;
                inst->ion_ena = nt->_data;
                inst->ion_ina = nt->_data;
                inst->ion_dinadv = nt->_data;
                inst->ion_ek = nt->_data;
                inst->ion_ik = nt->_data;
                inst->ion_dikdv = nt->_data;
                inst->ion_ena_index = indexes+0*pnodecount;
                inst->ion_ina_index = indexes+1*pnodecount;
                inst->ion_dinadv_index = indexes+2*pnodecount;
                inst->ion_ek_index = indexes+3*pnodecount;
                inst->ion_ik_index = indexes+4*pnodecount;
                inst->ion_dikdv_index = indexes+5*pnodecount;
                inst->voltage = nt->_actual_v;
                inst->node_index = ml->nodeindices;
                inst->t = nt->t;
                inst->dt = nt->dt;
                inst->celsius = celsius;
                inst->secondorder = secondorder;
                inst->node_count = ml->nodecount;
                ml->instance = inst;
            }
        )";

        THEN("index and nt variables") {
            auto result_instance_struct_declaration_setup = reindent_text(
                get_wrapper_instance_struct(nmodl_text));
            std::cout << "Result\n" << result_instance_struct_declaration_setup << std::endl;

            auto expected_instance_struct_declaration = reindent_text(
                generated_instance_struct_declaration);
            auto expected_instance_struct_setup = reindent_text(generated_instance_struct_setup);

            REQUIRE(result_instance_struct_declaration_setup.find(
                        expected_instance_struct_declaration) != std::string::npos);
            REQUIRE(result_instance_struct_declaration_setup.find(expected_instance_struct_setup) !=
                    std::string::npos);
        }
    }
}
