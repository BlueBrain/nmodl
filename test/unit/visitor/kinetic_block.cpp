/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch2/catch.hpp>

#include "ast/program.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/test_utils.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/constant_folder_visitor.hpp"
#include "visitors/kinetic_block_visitor.hpp"
#include "visitors/loop_unroll_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/visitor_utils.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test;
using namespace test_utils;

using ast::AstNodeType;
using nmodl::parser::NmodlDriver;

//=============================================================================
// KineticBlock visitor tests
//=============================================================================

std::vector<std::string> run_kinetic_block_visitor(const std::string& text) {
    std::vector<std::string> results;

    // construct AST from text including KINETIC block(s)
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    // construct symbol table from AST
    SymtabVisitor().visit_program(*ast);

    // unroll loops and fold constants
    ConstantFolderVisitor().visit_program(*ast);
    LoopUnrollVisitor().visit_program(*ast);
    ConstantFolderVisitor().visit_program(*ast);
    SymtabVisitor().visit_program(*ast);

    // run KineticBlock visitor on AST
    KineticBlockVisitor().visit_program(*ast);

    // run lookup visitor to extract DERIVATIVE block(s) from AST
    const auto& blocks = collect_nodes(*ast, {AstNodeType::DERIVATIVE_BLOCK});
    results.reserve(blocks.size());
    for (const auto& block: blocks) {
        results.push_back(to_nmodl(block));
    }


    // check that, after visitor rearrangement, parents are still up-to-date
    CheckParentVisitor().check_ast(*ast);

    return results;
}

SCENARIO("Convert KINETIC to DERIVATIVE using KineticBlock visitor", "[kinetic][visitor]") {
    GIVEN("KINETIC block with << reaction statement, 1 state var") {
        static const std::string input_nmodl_text = R"(
            STATE {
                x
            }
            KINETIC states {
                ~ x << (a*c/3.2)
            })";
        static const std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x' = (a*c/3.2)
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with << reaction statement, 1 array state var") {
        std::string input_nmodl_text = R"(
            STATE {
                x[1]
            }
            KINETIC states {
                ~ x[0] << (a*c/3.2)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x'[0] = (a*c/3.2)
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with << reaction statement, 1 array state var, flux vars") {
        std::string input_nmodl_text = R"(
            STATE {
                x[1]
            }
            KINETIC states {
                ~ x[0] << (a*c/3.2)
                f0 = f_flux*2
                f1 = b_flux + f_flux
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                f0 = 0*2
                f1 = 0+0
                x'[0] = (a*c/3.2)
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with invalid << reaction statement with 2 state vars") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                ~ x + y << (2*z)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
            })";
        THEN("Emit warning & do not process statement") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with -> reaction statement, 1 state var, flux vars") {
        std::string input_nmodl_text = R"(
            STATE {
                x
            }
            KINETIC states {
                ~ x -> (a)
                zf = f_flux
                zb = b_flux
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                zf = a*x
                zb = 0
                x' = (-1*(a*x))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with -> reaction statement, 2 state vars") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                ~ x + y -> (f(v))
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x' = (-1*(f(v)*x*y))
                y' = (-1*(f(v)*x*y))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with -> reaction statement, 2 state vars, CONSERVE statement") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                ~ x + y -> (f(v))
                CONSERVE x + y = 1
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                CONSERVE y = 1-x
                x' = (-1*(f(v)*x*y))
                y' = (-1*(f(v)*x*y))
            })";
        THEN("Convert to equivalent DERIVATIVE block, rewrite CONSERVE statement") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with -> reaction statement, 2 state vars, CONSERVE & COMPARTMENT") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                COMPARTMENT a { x }
                COMPARTMENT b { y }
                ~ x + y -> (f(v))
                CONSERVE x + y = 1
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                CONSERVE y = (1-(a*1*x))/(b*1)
                x' = ((-1*(f(v)*x*y)))/(a)
                y' = ((-1*(f(v)*x*y)))/(b)
            })";
        THEN(
            "Convert to equivalent DERIVATIVE block, rewrite CONSERVE statement inc COMPARTMENT "
            "factors") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with -> reaction statement, array of 2 state var") {
        std::string input_nmodl_text = R"(
            STATE {
                x[2]
            }
            KINETIC states {
                ~ x[0] + x[1] -> (f(v))
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x'[0] = (-1*(f(v)*x[0]*x[1]))
                x'[1] = (-1*(f(v)*x[0]*x[1]))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with one reaction statement, 1 state var, 1 non-state var, flux vars") {
        // Here c is NOT a state variable
        // see 9.9.2.1 of NEURON book
        // c should be treated as a constant, i.e.
        // -the diff. eq. for x should include the contribution from c
        // -no diff. eq. should be generated for c itself
        std::string input_nmodl_text = R"(
            STATE {
                x
            }
            KINETIC states {
                ~ x <-> c (r, r)
                c1 = f_flux - b_flux
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                c1 = r*x-r*c
                x' = (-1*(r*x-r*c))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with one reaction statement, 2 state vars") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                ~ x <-> y (a, b)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x' = (-1*(a*x-b*y))
                y' = (1*(a*x-b*y))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with one reaction statement, 2 state vars, CONSERVE statement") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                ~ x <-> y (a, b)
                CONSERVE x + y = 0
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                CONSERVE y = 0-x
                x' = (-1*(a*x-b*y))
                y' = (1*(a*x-b*y))
            })";
        THEN("Convert to equivalent DERIVATIVE block, rewrite CONSERVE statement") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    // array vars in CONSERVE statements are implicit sums over elements
    // see p34 of http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.7812&rep=rep1&type=pdf
    GIVEN("KINETIC block with array state vars, CONSERVE statement") {
        std::string input_nmodl_text = R"(
            STATE {
                x[3] y
            }
            KINETIC states {
                ~ x[0] <-> x[1] (a, b)
                ~ x[2] <-> y (c, d)
                CONSERVE y + x = 1
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                CONSERVE x[2] = 1-y-x[0]-x[1]
                x'[0] = (-1*(a*x[0]-b*x[1]))
                x'[1] = (1*(a*x[0]-b*x[1]))
                x'[2] = (-1*(c*x[2]-d*y))
                y' = (1*(c*x[2]-d*y))
            })";
        THEN(
            "Convert to equivalent DERIVATIVE block, rewrite CONSERVE statement after summing over "
            "array elements, with last state var on LHS") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    // array vars in CONSERVE statements are implicit sums over elements
    // see p34 of http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.7812&rep=rep1&type=pdf
    GIVEN("KINETIC block with array state vars, re-ordered CONSERVE statement") {
        std::string input_nmodl_text = R"(
            STATE {
                x[3] y
            }
            KINETIC states {
                ~ x[0] <-> x[1] (a, b)
                ~ x[2] <-> y (c, d)
                CONSERVE x + y = 1
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                CONSERVE y = 1-x[0]-x[1]-x[2]
                x'[0] = (-1*(a*x[0]-b*x[1]))
                x'[1] = (1*(a*x[0]-b*x[1]))
                x'[2] = (-1*(c*x[2]-d*y))
                y' = (1*(c*x[2]-d*y))
            })";
        THEN(
            "Convert to equivalent DERIVATIVE block, rewrite CONSERVE statement after summing over "
            "array elements, with last state var on LHS") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with one reaction statement & 1 COMPARTMENT statement") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                COMPARTMENT c-d {x y}
                ~ x <-> y (a, b)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x' = ((-1*(a*x-b*y)))/(c-d)
                y' = ((1*(a*x-b*y)))/(c-d)
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with two CONSERVE statements") {
        std::string input_nmodl_text = R"(
            STATE {
                c1 o1 o2 p0 p1
            }
            KINETIC ihkin {
                evaluate_fct(v, cai)
                ~ c1 <-> o1 (alpha, beta)
                ~ p0 <-> p1 (k1ca, k2)
                ~ o1 <-> o2 (k3p, k4)
                CONSERVE p0+p1 = 1
                CONSERVE c1+o1+o2 = 1
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE ihkin {
                evaluate_fct(v, cai)
                CONSERVE p1 = 1-p0
                CONSERVE o2 = 1-c1-o1
                c1' = (-1*(alpha*c1-beta*o1))
                o1' = (1*(alpha*c1-beta*o1))+(-1*(k3p*o1-k4*o2))
                o2' = (1*(k3p*o1-k4*o2))
                p0' = (-1*(k1ca*p0-k2*p1))
                p1' = (1*(k1ca*p0-k2*p1))
            })";
        THEN("Convert to equivalent DERIVATIVE block, re-order both CONSERVE statements") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with one reaction statement & 2 COMPARTMENT statements") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                COMPARTMENT cx {x}
                COMPARTMENT cy {y}
                ~ x <-> y (a, b)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x' = ((-1*(a*x-b*y)))/(cx)
                y' = ((1*(a*x-b*y)))/(cy)
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with two independent reaction statements") {
        std::string input_nmodl_text = R"(
            STATE {
                w x y z
            }
            KINETIC states {
                ~ x <-> y (a, b)
                ~ w <-> z (c, d)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                w' = (-1*(c*w-d*z))
                x' = (-1*(a*x-b*y))
                y' = (1*(a*x-b*y))
                z' = (1*(c*w-d*z))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with two dependent reaction statements") {
        std::string input_nmodl_text = R"(
            STATE {
                x y z
            }
            KINETIC states {
                ~ x <-> y (a, b)
                ~ y <-> z (c, d)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x' = (-1*(a*x-b*y))
                y' = (1*(a*x-b*y))+(-1*(c*y-d*z))
                z' = (1*(c*y-d*z))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with two dependent reaction statements, flux vars") {
        std::string input_nmodl_text = R"(
            STATE {
                x y z
            }
            KINETIC states {
                c0 = f_flux
                ~ x <-> y (a, b)
                c1 = f_flux + b_flux
                ~ y <-> z (c, d)
                c2 = f_flux - 2*b_flux
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                c0 = 0
                c1 = a*x+b*y
                c2 = c*y-2*d*z
                x' = (-1*(a*x-b*y))
                y' = (1*(a*x-b*y))+(-1*(c*y-d*z))
                z' = (1*(c*y-d*z))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with a stoch coeff of 2") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                ~ 2x <-> y (a, b)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x' = (-2*(a*x*x-b*y))
                y' = (1*(a*x*x-b*y))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with duplicate state vars") {
        std::string input_nmodl_text = R"(
            STATE {
                x y
            }
            KINETIC states {
                ~ x + x <-> y (a, b)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                x' = (-2*(a*x*x-b*y))
                y' = (1*(a*x*x-b*y))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with functions for reaction rates") {
        // Example from sec 9.8, p238 of NEURON book
        std::string input_nmodl_text = R"(
            STATE {
                mc m
            }
            KINETIC states {
                ~ mc <-> m (a(v), b(v))
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                mc' = (-1*(a(v)*mc-b(v)*m))
                m' = (1*(a(v)*mc-b(v)*m))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with stoch coeff 2, coupled pair of statements") {
        // Example from sec 9.8, p239 of NEURON book
        std::string input_nmodl_text = R"(
            STATE {
                A B C D
            }
            KINETIC states {
                ~ 2A + B <-> C (k1, k2)
                ~ C + D <-> A + 2B (k3, k4)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE states {
                A' = (-2*(k1*A*A*B-k2*C))+(1*(k3*C*D-k4*A*B*B))
                B' = (-1*(k1*A*A*B-k2*C))+(2*(k3*C*D-k4*A*B*B))
                C' = (1*(k1*A*A*B-k2*C))+(-1*(k3*C*D-k4*A*B*B))
                D' = (-1*(k3*C*D-k4*A*B*B))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("KINETIC block with loop over array variable") {
        std::string input_nmodl_text = R"(
            DEFINE N 5
            ASSIGNED {
                a
                b[N]
                c[N]
                d
            }
            STATE {
                x[N]
            }
            KINETIC kin {
                ~ x[0] << (a)
                FROM i=0 TO N-2 {
                    ~ x[i] <-> x[i+1] (b[i], c[i])
                }
                ~ x[N-1] -> (d)
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE kin {
                {
                }
                x'[0] = (a)+(-1*(b[0]*x[0]-c[0]*x[1]))
                x'[1] = (1*(b[0]*x[0]-c[0]*x[1]))+(-1*(b[1]*x[1]-c[1]*x[2]))
                x'[2] = (1*(b[1]*x[1]-c[1]*x[2]))+(-1*(b[2]*x[2]-c[2]*x[3]))
                x'[3] = (1*(b[2]*x[2]-c[2]*x[3]))+(-1*(b[3]*x[3]-c[3]*x[4]))
                x'[4] = (1*(b[3]*x[3]-c[3]*x[4]))+(-1*(d*x[4]))
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
    GIVEN("Complicated KINETIC block from DBBS") {
        std::string input_nmodl_text = R"(
            NEURON {
            SUFFIX glia__dbbs_mod_collection__cdp5__CR
            USEION ca READ cao, cai, ica WRITE cai
            RANGE ica_pmp
            RANGE Nannuli, Buffnull2, rf3, rf4, vrat, cainull, CR, CR_1C_0N, CR_2C_2N, CR_1V, CRnull
            RANGE TotalPump
            RANGE dsq, dsqvol

            }


            UNITS {
                (mol)   = (1)
                (molar) = (1/liter)
                (mM)    = (millimolar)
                (um)    = (micron)
                (mA)    = (milliamp)
                FARADAY = (faraday)  (10000 coulomb)
                PI      = (pi)       (1)
            }

            PARAMETER {
                Nannuli = 10.9495 (1)
                celsius (degC)
                    
                cainull = 45e-6 (mM)
                    mginull =.59    (mM)

            :	values for a buffer compensating the diffusion

                Buffnull1 = 0	(mM)
                rf1 = 0.0134329	(/ms mM)
                rf2 = 0.0397469	(/ms)

                Buffnull2 = 60.9091	(mM)
                rf3 = 0.1435	(/ms mM)
                rf4 = 0.0014	(/ms)

            :	values for benzothiazole coumarin (BTC)
                BTCnull = 0	(mM)
                b1 = 5.33	(/ms mM)
                b2 = 0.08	(/ms)

            :	values for caged compound DMNPE-4
                DMNPEnull = 0	(mM)
                c1 = 5.63	(/ms mM)
                c2 = 0.107e-3	(/ms)

            :       values for Calretinin (6 sites but only 5 active) (2*2 cooperative sites and 1 single indipendent site)

                    CRnull =	0.9             (mM):0.7-1.2
                    nT1   = 1.8            (/ms mM)
                    nT2   = 0.053        (/ms)
                    nR1   = 310           (/ms mM)
                    nR2   = 0.02        (/ms)
                    
                nV1   = 7.3            (/ms mM)
                    nV2   = 0.24        (/ms)
                    
                    :pumps

                kpmp1    = 3e-3       (/mM-ms)
                kpmp2    = 1.75e-5   (/ms)
                kpmp3    = 7.255e-5  (/ms)
                TotalPump = 1e-9	(mol/cm2)	

            }

            ASSIGNED {
                diam      (um)
                ica       (mA/cm2)
                ica_pmp   (mA/cm2)
                parea     (um)     : pump area per unit length
                parea2	  (um)
                cai       (mM)
                cao       (mM)
                mgi	(mM)
                vrat	(1)
                dsq
                dsqvol	
            }

            STATE {
                : ca[0] is equivalent to cai
                : ca[] are very small, so specify absolute tolerance
                : let it be ~1.5 - 2 orders of magnitude smaller than baseline level

                ca		(mM)    <1e-3>
                mg		(mM)	<1e-6>
                
                Buff1		(mM)	
                Buff1_ca	(mM)

                Buff2		(mM)
                Buff2_ca	(mM)

                BTC		(mM)
                BTC_ca		(mM)

                DMNPE		(mM)
                DMNPE_ca	(mM)	
                    
                    :calretinin
                    
                CR		(mM)
                
                    CR_1C_0N	(mM)
                CR_2C_0N	(mM)  
                CR_2C_1N	(mM)
                
                CR_1C_1N	(mM)

                CR_0C_1N	(mM)
                CR_0C_2N	(mM)
                CR_1C_2N	(mM)
                
                CR_2C_2N	(mM)
                
                CR_1V 		(mM)

                

                :pumps
                
                pump		(mol/cm2) <1e-15>
                pumpca		(mol/cm2) <1e-15>

            }

            BREAKPOINT {
                SOLVE state METHOD sparse
            }

            KINETIC state {
            COMPARTMENT diam*diam*vrat {ca mg Buff1 Buff1_ca Buff2 Buff2_ca BTC BTC_ca DMNPE DMNPE_ca CR CR_1C_0N CR_2C_0N CR_2C_1N CR_0C_1N CR_0C_2N CR_1C_2N CR_1C_1N CR_2C_1N CR_1C_2N CR_2C_2N}
            COMPARTMENT (1e10)*parea {pump pumpca}


                :pump
                ~ ca + pump <-> pumpca  (kpmp1*parea*(1e10), kpmp2*parea*(1e10))
                ~ pumpca <-> pump   (kpmp3*parea*(1e10), 0)
                CONSERVE pump + pumpca = TotalPump * parea * (1e10)
                
                ica_pmp = 2*FARADAY*(f_flux - b_flux)/parea	
                : all currents except pump
                : ica is Ca efflux
                ~ ca << (-ica*PI*diam/(2*FARADAY))

                :RADIAL DIFFUSION OF ca, mg and mobile buffers

                dsq = diam*diam
                    dsqvol = dsq*vrat
                    ~ ca + Buff1 <-> Buff1_ca (rf1*dsqvol, rf2*dsqvol)
                    ~ ca + Buff2 <-> Buff2_ca (rf3*dsqvol, rf4*dsqvol)
                    ~ ca + BTC <-> BTC_ca (b1*dsqvol, b2*dsqvol)
                    ~ ca + DMNPE <-> DMNPE_ca (c1*dsqvol, c2*dsqvol)
                        
                        :Calretinin
                        :Slow state
                    ~ ca + CR <-> CR_1C_0N (nT1*dsqvol, nT2*dsqvol)
                        ~ ca + CR_1C_0N <-> CR_2C_0N (nR1*dsqvol, nR2*dsqvol)
                        ~ ca + CR_2C_0N <-> CR_2C_1N (nT1*dsqvol, nT2*dsqvol)
                        
                        :fast state
                    ~ ca + CR <-> CR_0C_1N (nT1*dsqvol, nT2*dsqvol)
                    ~ ca + CR_0C_1N <-> CR_0C_2N (nR1*dsqvol, nR2*dsqvol)
                    ~ ca + CR_0C_2N <-> CR_1C_2N (nT1*dsqvol, nT2*dsqvol)
                    
                        :complete
                        ~ ca + CR_2C_1N <-> CR_2C_2N (nR1*dsqvol, nR2*dsqvol)
                        ~ ca + CR_1C_2N <-> CR_2C_2N (nR1*dsqvol, nR2*dsqvol)
                        
                        :mixed
                        ~ ca + CR_1C_0N <-> CR_1C_1N (nT1*dsqvol, nT2*dsqvol)   
                    ~ ca + CR_0C_1N <-> CR_1C_1N (nT1*dsqvol, nT2*dsqvol) 
                    
                    ~ ca + CR_1C_1N <-> CR_2C_1N (nR1*dsqvol, nR2*dsqvol)   
                    ~ ca + CR_1C_1N <-> CR_1C_2N (nR1*dsqvol, nR2*dsqvol) 
                        
                        :Fith site
                        ~ ca + CR  <-> CR_1V	     (nV1*dsqvol, nV2*dsqvol)
                        

                        
                cai = ca
                mgi = mg
            }

            FUNCTION ssBuff1() (mM) {
                ssBuff1 = Buffnull1/(1+((rf1/rf2)*cainull))
            }
            FUNCTION ssBuff1ca() (mM) {
                ssBuff1ca = Buffnull1/(1+(rf2/(rf1*cainull)))
            }
            FUNCTION ssBuff2() (mM) {
                    ssBuff2 = Buffnull2/(1+((rf3/rf4)*cainull))
            }
            FUNCTION ssBuff2ca() (mM) {
                    ssBuff2ca = Buffnull2/(1+(rf4/(rf3*cainull)))
            }

            FUNCTION ssBTC() (mM) {
                ssBTC = BTCnull/(1+((b1/b2)*cainull))
            }

            FUNCTION ssBTCca() (mM) {
                ssBTCca = BTCnull/(1+(b2/(b1*cainull)))
            }

            FUNCTION ssDMNPE() (mM) {
                ssDMNPE = DMNPEnull/(1+((c1/c2)*cainull))
            }

            FUNCTION ssDMNPEca() (mM) {
                ssDMNPEca = DMNPEnull/(1+(c2/(c1*cainull)))
            })";
        std::string output_nmodl_text = R"(
            DERIVATIVE state {
                CONSERVE pumpca = (TotalPump*parea*1e10-(1e10*parea*1*pump))/(1e10*parea*1)
                ica_pmp = 2*FARADAY*(kpmp3*parea*1e10*pumpca-0*pump)/parea
                dsq = diam*diam
                dsqvol = dsq*vrat
                cai = ca
                mgi = mg
                ca' = ((-ica*PI*diam/(2*FARADAY))+(-1*(kpmp1*parea*1e10*ca*pump-kpmp2*parea*1e10*pumpca))+(-1*(rf1*dsqvol*ca*Buff1-rf2*dsqvol*Buff1_ca))+(-1*(rf3*dsqvol*ca*Buff2-rf4*dsqvol*Buff2_ca))+(-1*(b1*dsqvol*ca*BTC-b2*dsqvol*BTC_ca))+(-1*(c1*dsqvol*ca*DMNPE-c2*dsqvol*DMNPE_ca))+(-1*(nT1*dsqvol*ca*CR-nT2*dsqvol*CR_1C_0N))+(-1*(nR1*dsqvol*ca*CR_1C_0N-nR2*dsqvol*CR_2C_0N))+(-1*(nT1*dsqvol*ca*CR_2C_0N-nT2*dsqvol*CR_2C_1N))+(-1*(nT1*dsqvol*ca*CR-nT2*dsqvol*CR_0C_1N))+(-1*(nR1*dsqvol*ca*CR_0C_1N-nR2*dsqvol*CR_0C_2N))+(-1*(nT1*dsqvol*ca*CR_0C_2N-nT2*dsqvol*CR_1C_2N))+(-1*(nR1*dsqvol*ca*CR_2C_1N-nR2*dsqvol*CR_2C_2N))+(-1*(nR1*dsqvol*ca*CR_1C_2N-nR2*dsqvol*CR_2C_2N))+(-1*(nT1*dsqvol*ca*CR_1C_0N-nT2*dsqvol*CR_1C_1N))+(-1*(nT1*dsqvol*ca*CR_0C_1N-nT2*dsqvol*CR_1C_1N))+(-1*(nR1*dsqvol*ca*CR_1C_1N-nR2*dsqvol*CR_2C_1N))+(-1*(nR1*dsqvol*ca*CR_1C_1N-nR2*dsqvol*CR_1C_2N))+(-1*(nV1*dsqvol*ca*CR-nV2*dsqvol*CR_1V)))/(diam*diam*vrat)
                CR' = ((-1*(nT1*dsqvol*ca*CR-nT2*dsqvol*CR_1C_0N))+(-1*(nT1*dsqvol*ca*CR-nT2*dsqvol*CR_0C_1N))+(-1*(nV1*dsqvol*ca*CR-nV2*dsqvol*CR_1V)))/(diam*diam*vrat)
                CR_1C_0N' = ((1*(nT1*dsqvol*ca*CR-nT2*dsqvol*CR_1C_0N))+(-1*(nR1*dsqvol*ca*CR_1C_0N-nR2*dsqvol*CR_2C_0N))+(-1*(nT1*dsqvol*ca*CR_1C_0N-nT2*dsqvol*CR_1C_1N)))/(diam*diam*vrat)
                CR_2C_2N' = ((1*(nR1*dsqvol*ca*CR_2C_1N-nR2*dsqvol*CR_2C_2N))+(1*(nR1*dsqvol*ca*CR_1C_2N-nR2*dsqvol*CR_2C_2N)))/(diam*diam*vrat)
                CR_1V' = (1*(nV1*dsqvol*ca*CR-nV2*dsqvol*CR_1V))
                mg' = (1)/(diam*diam*vrat)
                Buff1' = ((-1*(rf1*dsqvol*ca*Buff1-rf2*dsqvol*Buff1_ca)))/(diam*diam*vrat)
                Buff1_ca' = ((1*(rf1*dsqvol*ca*Buff1-rf2*dsqvol*Buff1_ca)))/(diam*diam*vrat)
                Buff2' = ((-1*(rf3*dsqvol*ca*Buff2-rf4*dsqvol*Buff2_ca)))/(diam*diam*vrat)
                Buff2_ca' = ((1*(rf3*dsqvol*ca*Buff2-rf4*dsqvol*Buff2_ca)))/(diam*diam*vrat)
                BTC' = ((-1*(b1*dsqvol*ca*BTC-b2*dsqvol*BTC_ca)))/(diam*diam*vrat)
                BTC_ca' = ((1*(b1*dsqvol*ca*BTC-b2*dsqvol*BTC_ca)))/(diam*diam*vrat)
                DMNPE' = ((-1*(c1*dsqvol*ca*DMNPE-c2*dsqvol*DMNPE_ca)))/(diam*diam*vrat)
                DMNPE_ca' = ((1*(c1*dsqvol*ca*DMNPE-c2*dsqvol*DMNPE_ca)))/(diam*diam*vrat)
                CR_2C_0N' = ((1*(nR1*dsqvol*ca*CR_1C_0N-nR2*dsqvol*CR_2C_0N))+(-1*(nT1*dsqvol*ca*CR_2C_0N-nT2*dsqvol*CR_2C_1N)))/(diam*diam*vrat)
                CR_2C_1N' = ((1*(nT1*dsqvol*ca*CR_2C_0N-nT2*dsqvol*CR_2C_1N))+(-1*(nR1*dsqvol*ca*CR_2C_1N-nR2*dsqvol*CR_2C_2N))+(1*(nR1*dsqvol*ca*CR_1C_1N-nR2*dsqvol*CR_2C_1N)))/(diam*diam*vrat)
                CR_1C_1N' = ((1*(nT1*dsqvol*ca*CR_1C_0N-nT2*dsqvol*CR_1C_1N))+(1*(nT1*dsqvol*ca*CR_0C_1N-nT2*dsqvol*CR_1C_1N))+(-1*(nR1*dsqvol*ca*CR_1C_1N-nR2*dsqvol*CR_2C_1N))+(-1*(nR1*dsqvol*ca*CR_1C_1N-nR2*dsqvol*CR_1C_2N)))/(diam*diam*vrat)
                CR_0C_1N' = ((1*(nT1*dsqvol*ca*CR-nT2*dsqvol*CR_0C_1N))+(-1*(nR1*dsqvol*ca*CR_0C_1N-nR2*dsqvol*CR_0C_2N))+(-1*(nT1*dsqvol*ca*CR_0C_1N-nT2*dsqvol*CR_1C_1N)))/(diam*diam*vrat)
                CR_0C_2N' = ((1*(nR1*dsqvol*ca*CR_0C_1N-nR2*dsqvol*CR_0C_2N))+(-1*(nT1*dsqvol*ca*CR_0C_2N-nT2*dsqvol*CR_1C_2N)))/(diam*diam*vrat)
                CR_1C_2N' = ((1*(nT1*dsqvol*ca*CR_0C_2N-nT2*dsqvol*CR_1C_2N))+(-1*(nR1*dsqvol*ca*CR_1C_2N-nR2*dsqvol*CR_2C_2N))+(1*(nR1*dsqvol*ca*CR_1C_1N-nR2*dsqvol*CR_1C_2N)))/(diam*diam*vrat)
                pump' = ((-1*(kpmp1*parea*1e10*ca*pump-kpmp2*parea*1e10*pumpca))+(1*(kpmp3*parea*1e10*pumpca-0*pump)))/(1e10*parea)
                pumpca' = ((1*(kpmp1*parea*1e10*ca*pump-kpmp2*parea*1e10*pumpca))+(-1*(kpmp3*parea*1e10*pumpca-0*pump)))/(1e10*parea)
            })";
        THEN("Convert to equivalent DERIVATIVE block") {
            auto result = run_kinetic_block_visitor(input_nmodl_text);
            CAPTURE(input_nmodl_text);
            REQUIRE(result[0] == reindent_text(output_nmodl_text));
        }
    }
}
