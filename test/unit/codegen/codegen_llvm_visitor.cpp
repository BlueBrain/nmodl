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
#include "visitors/inline_visitor.hpp"
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

// Run LLVM codegen helper visitor with given platform as target
static std::vector<std::shared_ptr<ast::Ast>> run_llvm_visitor_helper(
    const std::string& text,
    codegen::Platform& platform,
    const std::vector<ast::AstNodeType>& nodes_to_collect) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    InlineVisitor().visit_program(*ast);
    NeuronSolveVisitor().visit_program(*ast);
    SolveBlockVisitor().visit_program(*ast);
    CodegenLLVMHelperVisitor(platform).visit_program(*ast);

    return collect_nodes(*ast, nodes_to_collect);
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
                double* __restrict__ vec_rhs;
                double* __restrict__ vec_d;
                double* __restrict__ _shadow_rhs;
                double* __restrict__ _shadow_d;
                double t;
                double dt;
                double celsius;
                int secondorder;
                int node_count;
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

        THEN("index and nt variables created correctly") {
            auto result_instance_struct_declaration_setup = reindent_text(
                get_wrapper_instance_struct(nmodl_text));

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


SCENARIO("Channel: Derivative and breakpoint block llvm transformations",
         "[visitor][llvm_helper][channel]") {
    GIVEN("A hh.mod file with derivative and breakpoint block") {
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
                THREADSAFE
            }

            PARAMETER {
                gnabar = .12 (S/cm2) <0,1e9>
                gkbar = .036 (S/cm2) <0,1e9>
                gl = .0003 (S/cm2) <0,1e9>
                el = -54.3 (mV)
            }

            STATE {
                m
                h
                n
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
                minf
                hinf
                ninf
                mtau (ms)
                htau (ms)
                ntau (ms)
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
                gna = gnabar*m*m*m*h
                ina = gna*(v-ena)
                gk = gkbar*n*n*n*n
                ik = gk*(v-ek)
                il = gl*(v-el)
            }

            DERIVATIVE states {
                rates(v)
                m' = (minf-m)/mtau
                h' = (hinf-h)/htau
                n' = (ninf-n)/ntau
            }

            PROCEDURE rates(v(mV)) {
                LOCAL alpha, beta, sum, q10
                UNITSOFF
                q10 = 3^((celsius-6.3)/10)
                alpha = .1*vtrap(-(v+40), 10)
                beta = 4*exp(-(v+65)/18)
                sum = alpha+beta
                mtau = 1/(q10*sum)
                minf = alpha/sum
                alpha = .07*exp(-(v+65)/20)
                beta = 1/(exp(-(v+35)/10)+1)
                sum = alpha+beta
                htau = 1/(q10*sum)
                hinf = alpha/sum
                alpha = .01*vtrap(-(v+55), 10)
                beta = .125*exp(-(v+65)/80)
                sum = alpha+beta
                ntau = 1/(q10*sum)
                ninf = alpha/sum
            }

            FUNCTION vtrap(x, y) {
                IF (fabs(x/y)<1e-6) {
                    vtrap = y*(1-x/y/2)
                } ELSE {
                    vtrap = x/(exp(x/y)-1)
                }
            }
        )";

        std::string expected_state_function = R"(
            VOID nrn_state_hh(INSTANCE_STRUCT *mech){
                INTEGER id
                INTEGER node_id, ena_id, ek_id
                DOUBLE v
                for(id = 0; id<mech->node_count; id = id+1) {
                    node_id = mech->node_index[id]
                    ena_id = mech->ion_ena_index[id]
                    ek_id = mech->ion_ek_index[id]
                    v = mech->voltage[node_id]
                    mech->ena[id] = mech->ion_ena[ena_id]
                    mech->ek[id] = mech->ion_ek[ek_id]
                    {
                        DOUBLE alpha, beta, sum, q10, vtrap_in_0, vtrap_in_1, v_in_0
                        v_in_0 = v
                        UNITSOFF
                        q10 = 3^((mech->celsius-6.3)/10)
                        {
                            DOUBLE x_in_0, y_in_0
                            x_in_0 = -(v_in_0+40)
                            y_in_0 = 10
                            IF (fabs(x_in_0/y_in_0)<1e-6) {
                                vtrap_in_0 = y_in_0*(1-x_in_0/y_in_0/2)
                            } ELSE {
                                vtrap_in_0 = x_in_0/(exp(x_in_0/y_in_0)-1)
                            }
                        }
                        alpha = .1*vtrap_in_0
                        beta = 4*exp(-(v_in_0+65)/18)
                        sum = alpha+beta
                        mech->mtau[id] = 1/(q10*sum)
                        mech->minf[id] = alpha/sum
                        alpha = .07*exp(-(v_in_0+65)/20)
                        beta = 1/(exp(-(v_in_0+35)/10)+1)
                        sum = alpha+beta
                        mech->htau[id] = 1/(q10*sum)
                        mech->hinf[id] = alpha/sum
                        {
                            DOUBLE x_in_1, y_in_1
                            x_in_1 = -(v_in_0+55)
                            y_in_1 = 10
                            IF (fabs(x_in_1/y_in_1)<1e-6) {
                                vtrap_in_1 = y_in_1*(1-x_in_1/y_in_1/2)
                            } ELSE {
                                vtrap_in_1 = x_in_1/(exp(x_in_1/y_in_1)-1)
                            }
                        }
                        alpha = .01*vtrap_in_1
                        beta = .125*exp(-(v_in_0+65)/80)
                        sum = alpha+beta
                        mech->ntau[id] = 1/(q10*sum)
                        mech->ninf[id] = alpha/sum
                    }
                    mech->m[id] = mech->m[id]+(1.0-exp(mech->dt*((((-1.0)))/mech->mtau[id])))*(-(((mech->minf[id]))/mech->mtau[id])/((((-1.0)))/mech->mtau[id])-mech->m[id])
                    mech->h[id] = mech->h[id]+(1.0-exp(mech->dt*((((-1.0)))/mech->htau[id])))*(-(((mech->hinf[id]))/mech->htau[id])/((((-1.0)))/mech->htau[id])-mech->h[id])
                    mech->n[id] = mech->n[id]+(1.0-exp(mech->dt*((((-1.0)))/mech->ntau[id])))*(-(((mech->ninf[id]))/mech->ntau[id])/((((-1.0)))/mech->ntau[id])-mech->n[id])
                }
            })";

        std::string expected_cur_function = R"(
            VOID nrn_cur_hh(INSTANCE_STRUCT *mech){
                INTEGER id
                INTEGER node_id, ena_id, ek_id, ion_dinadv_id, ion_dikdv_id, ion_ina_id, ion_ik_id
                DOUBLE v, g, rhs, v_org, current, dina, dik
                for(id = 0; id<mech->node_count; id = id+1) {
                    node_id = mech->node_index[id]
                    ena_id = mech->ion_ena_index[id]
                    ek_id = mech->ion_ek_index[id]
                    ion_dinadv_id = mech->ion_dinadv_index[id]
                    ion_dikdv_id = mech->ion_dikdv_index[id]
                    ion_ina_id = mech->ion_ina_index[id]
                    ion_ik_id = mech->ion_ik_index[id]
                    v = mech->voltage[node_id]
                    mech->ena[id] = mech->ion_ena[ena_id]
                    mech->ek[id] = mech->ion_ek[ek_id]
                    v_org = v
                    v = v+0.001
                    {
                        current = 0
                        mech->gna[id] = mech->gnabar[id]*mech->m[id]*mech->m[id]*mech->m[id]*mech->h[id]
                        mech->ina[id] = mech->gna[id]*(v-mech->ena[id])
                        mech->gk[id] = mech->gkbar[id]*mech->n[id]*mech->n[id]*mech->n[id]*mech->n[id]
                        mech->ik[id] = mech->gk[id]*(v-mech->ek[id])
                        mech->il[id] = mech->gl[id]*(v-mech->el[id])
                        current = current+mech->il[id]
                        current = current+mech->ina[id]
                        current = current+mech->ik[id]
                        g = current
                    }
                    dina = mech->ina[id]
                    dik = mech->ik[id]
                    v = v_org
                    {
                        current = 0
                        mech->gna[id] = mech->gnabar[id]*mech->m[id]*mech->m[id]*mech->m[id]*mech->h[id]
                        mech->ina[id] = mech->gna[id]*(v-mech->ena[id])
                        mech->gk[id] = mech->gkbar[id]*mech->n[id]*mech->n[id]*mech->n[id]*mech->n[id]
                        mech->ik[id] = mech->gk[id]*(v-mech->ek[id])
                        mech->il[id] = mech->gl[id]*(v-mech->el[id])
                        current = current+mech->il[id]
                        current = current+mech->ina[id]
                        current = current+mech->ik[id]
                        rhs = current
                    }
                    g = (g-rhs)/0.001
                    mech->ion_dinadv[ion_dinadv_id] = mech->ion_dinadv[ion_dinadv_id]+(dina-mech->ina[id])/0.001
                    mech->ion_dikdv[ion_dikdv_id] = mech->ion_dikdv[ion_dikdv_id]+(dik-mech->ik[id])/0.001
                    mech->ion_ina[ion_ina_id] = mech->ion_ina[ion_ina_id]+mech->ina[id]
                    mech->ion_ik[ion_ik_id] = mech->ion_ik[ion_ik_id]+mech->ik[id]
                    mech->vec_rhs[node_id] = mech->vec_rhs[node_id]-rhs
                    mech->vec_d[node_id] = mech->vec_d[node_id]+g
                }
            })";

        THEN("codegen functions are constructed correctly for density channel") {
            codegen::Platform simd_platform(/*use_single_precision=*/false,
                                            /*instruction_width=*/1);
            auto result = run_llvm_visitor_helper(nmodl_text,
                                                  simd_platform,
                                                  {ast::AstNodeType::CODEGEN_FUNCTION});
            REQUIRE(result.size() == 2);

            auto cur_function = reindent_text(to_nmodl(result[0]));
            REQUIRE(cur_function == reindent_text(expected_cur_function));

            auto state_function = reindent_text(to_nmodl(result[1]));
            REQUIRE(state_function == reindent_text(expected_state_function));
        }
    }
}

SCENARIO("Synapse: Derivative and breakpoint block llvm transformations",
         "[visitor][llvm_helper][derivative]") {
    GIVEN("A exp2syn.mod file with derivative and breakpoint block") {
        // note that USEION statement is added just for better code coverage (ionic current)
        std::string nmodl_text = R"(
            NEURON {
                POINT_PROCESS Exp2Syn
                USEION na READ ena WRITE ina
                RANGE tau1, tau2, e, i
                NONSPECIFIC_CURRENT i
                RANGE g, gna
            }

            UNITS {
                (nA) = (nanoamp)
                (mV) = (millivolt)
                (uS) = (microsiemens)
            }

            PARAMETER {
                tau1 = 0.1 (ms) <1e-9,1e9>
                tau2 = 10 (ms) <1e-9,1e9>
                e = 0 (mV)
            }

            ASSIGNED {
                v (mV)
                i (nA)
                g (uS)
                gna (S/cm2)
                factor
            }

            STATE {
                A (uS)
                B (uS)
            }

            INITIAL {
                LOCAL tp
                IF (tau1/tau2>0.9999) {
                    tau1 = 0.9999*tau2
                }
                IF (tau1/tau2<1e-9) {
                    tau1 = tau2*1e-9
                }
                A = 0
                B = 0
                tp = (tau1*tau2)/(tau2-tau1)*log(tau2/tau1)
                factor = -exp(-tp/tau1)+exp(-tp/tau2)
                factor = 1/factor
            }

            BREAKPOINT {
                SOLVE state METHOD cnexp
                ina = gna*(v-ena)
                g = B-A
                i = g*(v-e)
            }

            DERIVATIVE state {
                A' = -A/tau1
                B' = -B/tau2
            }

            NET_RECEIVE (weight(uS)) {
                A = A+weight*factor
                B = B+weight*factor
            })";

        std::string expected_cur_function = R"(
            VOID nrn_cur_exp2syn(INSTANCE_STRUCT *mech){
                INTEGER id
                INTEGER node_id, ena_id, node_area_id, ion_dinadv_id, ion_ina_id
                DOUBLE v, g, rhs, v_org, current, dina, mfactor
                for(id = 0; id<mech->node_count; id = id+1) {
                    node_id = mech->node_index[id]
                    ena_id = mech->ion_ena_index[id]
                    node_area_id = mech->node_area_index[id]
                    ion_dinadv_id = mech->ion_dinadv_index[id]
                    ion_ina_id = mech->ion_ina_index[id]
                    v = mech->voltage[node_id]
                    mech->ena[id] = mech->ion_ena[ena_id]
                    v_org = v
                    v = v+0.001
                    {
                        current = 0
                        mech->ina[id] = mech->gna[id]*(v-mech->ena[id])
                        mech->g[id] = mech->B[id]-mech->A[id]
                        mech->i[id] = mech->g[id]*(v-mech->e[id])
                        current = current+mech->i[id]
                        current = current+mech->ina[id]
                        mech->g[id] = current
                    }
                    dina = mech->ina[id]
                    v = v_org
                    {
                        current = 0
                        mech->ina[id] = mech->gna[id]*(v-mech->ena[id])
                        mech->g[id] = mech->B[id]-mech->A[id]
                        mech->i[id] = mech->g[id]*(v-mech->e[id])
                        current = current+mech->i[id]
                        current = current+mech->ina[id]
                        rhs = current
                    }
                    mech->g[id] = (mech->g[id]-rhs)/0.001
                    mech->ion_dinadv[ion_dinadv_id] = mech->ion_dinadv[ion_dinadv_id]+(dina-mech->ina[id])/0.001*1.e2/mech->node_area[node_area_id]
                    mech->ion_ina[ion_ina_id] = mech->ion_ina[ion_ina_id]+mech->ina[id]*(1.e2/mech->node_area[node_area_id])
                    mfactor = 1.e2/mech->node_area[node_area_id]
                    mech->g[id] = mech->g[id]*mfactor
                    rhs = rhs*mfactor
                    mech->vec_rhs[node_id] = mech->vec_rhs[node_id]-rhs
                    mech->vec_d[node_id] = mech->vec_d[node_id]+mech->g[id]
                }
            })";

        std::string expected_state_function = R"(
            VOID nrn_state_exp2syn(INSTANCE_STRUCT *mech){
                INTEGER id
                INTEGER node_id, ena_id
                DOUBLE v
                for(id = 0; id<mech->node_count; id = id+1) {
                    node_id = mech->node_index[id]
                    ena_id = mech->ion_ena_index[id]
                    v = mech->voltage[node_id]
                    mech->ena[id] = mech->ion_ena[ena_id]
                    mech->A[id] = mech->A[id]+(1.0-exp(mech->dt*((-1.0)/mech->tau1[id])))*(-(0.0)/((-1.0)/mech->tau1[id])-mech->A[id])
                    mech->B[id] = mech->B[id]+(1.0-exp(mech->dt*((-1.0)/mech->tau2[id])))*(-(0.0)/((-1.0)/mech->tau2[id])-mech->B[id])
                }
            })";

        THEN("codegen functions are constructed correctly for synapse") {
            codegen::Platform simd_platform(/*use_single_precision=*/false,
                                            /*instruction_width=*/1);
            auto result = run_llvm_visitor_helper(nmodl_text,
                                                  simd_platform,
                                                  {ast::AstNodeType::CODEGEN_FUNCTION});
            REQUIRE(result.size() == 2);

            auto cur_function = reindent_text(to_nmodl(result[0]));
            REQUIRE(cur_function == reindent_text(expected_cur_function));

            auto state_function = reindent_text(to_nmodl(result[1]));
            REQUIRE(state_function == reindent_text(expected_state_function));
        }
    }
}
