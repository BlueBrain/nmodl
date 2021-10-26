/*************************************************************************
 * Copyright (C) 2019-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>

#include "ast/program.hpp"
#include "codegen/codegen_c_visitor.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/test_utils.hpp"
#include "visitors/kinetic_block_visitor.hpp"
#include "visitors/steadystate_visitor.hpp"
#include "visitors/sympy_solver_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace codegen;

using nmodl::parser::NmodlDriver;
using nmodl::test_utils::reindent_text;

/// Helper for creating C codegen visitor
std::shared_ptr<CodegenCVisitor> create_c_visitor(const std::string& text,
                                                  std::stringstream& ss,
                                                  bool apply_extra_visitors = false) {
    /// parse mod file and create AST
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    /// construct symbol table
    SymtabVisitor().visit_program(*ast);

    /// more visitors are required for the SH_na8st.mod test below to work.
    if (apply_extra_visitors) {
        KineticBlockVisitor{}.visit_program(*ast);
        SymtabVisitor{}.visit_program(*ast);
        SteadystateVisitor{}.visit_program(*ast);
        SymtabVisitor{}.visit_program(*ast);
        SympySolverVisitor{}.visit_program(*ast);
        SymtabVisitor{true}.visit_program(*ast);
    }

    /// create C code generation visitor
    auto cv = std::make_shared<CodegenCVisitor>("temp.mod", ss, "double", false);
    cv->setup(*ast);
    return cv;
}

/// print instance structure for testing purpose
std::string get_instance_var_setup_function(std::string& nmodl_text) {
    std::stringstream ss;
    auto cvisitor = create_c_visitor(nmodl_text, ss);
    cvisitor->print_instance_variable_setup();
    return reindent_text(ss.str());
}

std::string get_global_variables_setup_function(std::string const& nmodl_text) {
    std::stringstream ss;
    auto cvisitor = create_c_visitor(nmodl_text, ss, true);
    // This is needed for `print_global_variable_setup()` to print the correct suffixes.
    cvisitor->print_mechanism_global_var_structure();
    cvisitor->print_global_variable_setup();
    return reindent_text(ss.str());
}

SCENARIO("Check instance variable definition order", "[codegen][var_order]") {
    GIVEN("cal_mig.mod: USEION variables declared as RANGE") {
        // In the below mod file, the ion variables cai and cao are also
        // declared as RANGE variables. The ordering issue was fixed in #443.
        std::string nmodl_text = R"(
            PARAMETER {
              gcalbar=.003 (mho/cm2)
              ki=.001 (mM)
              cai = 50.e-6 (mM)
              cao = 2 (mM)
            }
            NEURON {
              SUFFIX cal
              USEION ca READ cai,cao WRITE ica
              RANGE gcalbar, cai, ica, gcal, ggk
              RANGE minf, tau
            }
            STATE {
              m
            }
            ASSIGNED {
              ica (mA/cm2)
              gcal (mho/cm2)
              minf
              tau   (ms)
              ggk
            }
        )";

        THEN("ionic current variable declared as RANGE appears first") {
            std::string generated_code = R"(
                static inline void setup_instance(NrnThread* nt, Memb_list* ml)  {
                    cal_Instance* inst = (cal_Instance*) mem_alloc(1, sizeof(cal_Instance));
                    int pnodecount = ml->_nodecount_padded;
                    Datum* indexes = ml->pdata;
                    inst->gcalbar = ml->data+0*pnodecount;
                    inst->ica = ml->data+1*pnodecount;
                    inst->gcal = ml->data+2*pnodecount;
                    inst->minf = ml->data+3*pnodecount;
                    inst->tau = ml->data+4*pnodecount;
                    inst->ggk = ml->data+5*pnodecount;
                    inst->m = ml->data+6*pnodecount;
                    inst->cai = ml->data+7*pnodecount;
                    inst->cao = ml->data+8*pnodecount;
                    inst->Dm = ml->data+9*pnodecount;
                    inst->v_unused = ml->data+10*pnodecount;
                    inst->ion_cai = nt->_data;
                    inst->ion_cao = nt->_data;
                    inst->ion_ica = nt->_data;
                    inst->ion_dicadv = nt->_data;
                    ml->instance = (void*) inst;
                }
            )";
            auto expected = reindent_text(generated_code);
            auto result = get_instance_var_setup_function(nmodl_text);
            REQUIRE(result.find(expected) != std::string::npos);
        }
    }

    // In the below mod file, the `cao` is defined first in the PARAMETER
    // block but it appears after cai in the USEION statement. As per NEURON
    // implementation, variables should appear in the order of USEION
    // statements i.e. ion_cai should come before ion_cao. This was a bug
    // and it has been fixed in #697.
    GIVEN("LcaMig.mod: mod file from reduced_dentate model") {
        std::string nmodl_text = R"(
            PARAMETER {
              ki = .001(mM)
              cao(mM)
              tfa = 1
            }
            NEURON {
              SUFFIX lca
              USEION ca READ cai, cao VALENCE 2
              RANGE cai, ilca, elca
            }
            STATE {
              m
            }
        )";

        THEN("Ion variables are defined in the order of USEION") {
            std::string generated_code = R"(
                static inline void setup_instance(NrnThread* nt, Memb_list* ml)  {
                    lca_Instance* inst = (lca_Instance*) mem_alloc(1, sizeof(lca_Instance));
                    int pnodecount = ml->_nodecount_padded;
                    Datum* indexes = ml->pdata;
                    inst->m = ml->data+0*pnodecount;
                    inst->cai = ml->data+1*pnodecount;
                    inst->cao = ml->data+2*pnodecount;
                    inst->Dm = ml->data+3*pnodecount;
                    inst->v_unused = ml->data+4*pnodecount;
                    inst->ion_cai = nt->_data;
                    inst->ion_cao = nt->_data;
                    ml->instance = (void*) inst;
                }
            )";

            auto expected = reindent_text(generated_code);
            auto result = get_instance_var_setup_function(nmodl_text);
            REQUIRE(result.find(expected) != std::string::npos);
        }
    }

    // In the below mod file, ion variables ncai and lcai are declared
    // as state variables as well as range variables. The issue about
    // this mod file ordering was fixed in #443.
    GIVEN("ccanl.mod: mod file from reduced_dentate model") {
        std::string nmodl_text = R"(
            NEURON {
              SUFFIX ccanl
              USEION nca READ ncai, inca, enca WRITE enca, ncai VALENCE 2
              USEION lca READ lcai, ilca, elca WRITE elca, lcai VALENCE 2
              RANGE caiinf, catau, cai, ncai, lcai, eca, elca, enca
            }
            UNITS {
              FARADAY = 96520(coul)
              R = 8.3134(joule / degC)
            }
            PARAMETER {
              depth = 200(nm): assume volume = area * depth
              catau = 9(ms)
              caiinf = 50.e-6(mM)
              cao = 2(mM)
            }
            ASSIGNED {
              celsius(degC)
              ica(mA / cm2)
              inca(mA / cm2)
              ilca(mA / cm2)
              cai(mM)
              enca(mV)
              elca(mV)
              eca(mV)
            }
            STATE {
              ncai(mM)
              lcai(mM)
            }
        )";

        THEN("Ion variables are defined in the order of USEION") {
            std::string generated_code = R"(
                static inline void setup_instance(NrnThread* nt, Memb_list* ml)  {
                    ccanl_Instance* inst = (ccanl_Instance*) mem_alloc(1, sizeof(ccanl_Instance));
                    int pnodecount = ml->_nodecount_padded;
                    Datum* indexes = ml->pdata;
                    inst->catau = ml->data+0*pnodecount;
                    inst->caiinf = ml->data+1*pnodecount;
                    inst->cai = ml->data+2*pnodecount;
                    inst->eca = ml->data+3*pnodecount;
                    inst->ica = ml->data+4*pnodecount;
                    inst->inca = ml->data+5*pnodecount;
                    inst->ilca = ml->data+6*pnodecount;
                    inst->enca = ml->data+7*pnodecount;
                    inst->elca = ml->data+8*pnodecount;
                    inst->ncai = ml->data+9*pnodecount;
                    inst->Dncai = ml->data+10*pnodecount;
                    inst->lcai = ml->data+11*pnodecount;
                    inst->Dlcai = ml->data+12*pnodecount;
                    inst->v_unused = ml->data+13*pnodecount;
                    inst->ion_ncai = nt->_data;
                    inst->ion_inca = nt->_data;
                    inst->ion_enca = nt->_data;
                    inst->style_nca = ml->pdata;
                    inst->ion_lcai = nt->_data;
                    inst->ion_ilca = nt->_data;
                    inst->ion_elca = nt->_data;
                    inst->style_lca = ml->pdata;
                    ml->instance = (void*) inst;
                }
            )";

            auto expected = reindent_text(generated_code);
            auto result = get_instance_var_setup_function(nmodl_text);
            REQUIRE(result.find(expected) != std::string::npos);
        }
    }
}

SCENARIO("Check global variable setup", "[codegen][global_variables]") {
    GIVEN("SH_na8st.mod: modfile from reduced_dentate model") {
        std::string const nmodl_text{R"(
            NEURON {
                SUFFIX na8st
            }
            STATE { c1 c2 }
            BREAKPOINT {
                SOLVE kin METHOD sparse
            }
            INITIAL {
                SOLVE kin STEADYSTATE sparse
            }
            KINETIC kin {
                ~ c1 <-> c2 (a1, b1)
            }
        )"};
        THEN("Correct number of global variables are initialised") {
            auto const expected = reindent_text(R"(
                static inline void setup_global_variables()  {
                    static int setup_done = 0;
                    if (setup_done) {
                        return;
                    }
                    na8st_global.slist1 = (int*) mem_alloc(2, sizeof(int));
                    na8st_global.dlist1 = (int*) mem_alloc(2, sizeof(int));
                    na8st_global.slist1[0] = 0;
                    na8st_global.dlist1[0] = 2;
                    na8st_global.slist1[1] = 1;
                    na8st_global.dlist1[1] = 3;
                    na8st_global.c10 = 0.0;
                    na8st_global.c20 = 0.0;

                    setup_done = 1;
                }
            )");
            auto const result = get_global_variables_setup_function(nmodl_text);
            REQUIRE_THAT(result, Catch::Matchers::Contains(expected));
        }
    }
}
