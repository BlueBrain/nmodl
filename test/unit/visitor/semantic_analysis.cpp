/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <catch2/catch_test_macros.hpp>

#include "ast/program.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/test_utils.hpp"
#include "visitors/semantic_analysis_visitor.hpp"
#include "visitors/symtab_visitor.hpp"


using namespace nmodl;
using namespace visitor;
using namespace test_utils;

using nmodl::parser::NmodlDriver;

//=============================================================================
// Procedure/Function inlining tests
//=============================================================================

bool run_semantic_analysis_visitor(const std::string& text) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);
    SymtabVisitor().visit_program(*ast);
    return SemanticAnalysisVisitor{}.check(*ast);
}

SCENARIO("TABLE stmt", "[visitor][semantic_analysis]") {
    GIVEN("Procedure with more than one argument") {
        std::string nmodl_text = R"(
            PROCEDURE rates_1(a, b) {
                TABLE ainf FROM 0 TO 1 WITH 1
                ainf = 1
            }
        )";
        THEN("fail") {
            REQUIRE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
    GIVEN("Procedure with exactly one argument") {
        std::string nmodl_text = R"(
            PROCEDURE rates_1(a) {
                TABLE ainf FROM 0 TO 1 WITH 1
                ainf = 1
            }
        )";
        THEN("pass") {
            REQUIRE_FALSE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
    GIVEN("Procedure with less than one argument") {
        std::string nmodl_text = R"(
            PROCEDURE rates_1() {
                TABLE ainf FROM 0 TO 1 WITH 1
                ainf = 1
            }
        )";
        THEN("fail") {
            REQUIRE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
}

SCENARIO("Destructor block", "[visitor][semantic_analysis]") {
    GIVEN("A point-process mod file, with a destructor") {
        std::string nmodl_text = R"(
            DESTRUCTOR { : Destructor is before
            }

            NEURON {
                POINT_PROCESS test
            }
        )";
        THEN("pass") {
            REQUIRE_FALSE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
    GIVEN("A artifial-cell mod file, with a destructor") {
        std::string nmodl_text = R"(
            NEURON {
                ARTIFICIAL_CELL test
            }

            DESTRUCTOR {
            }
        )";
        THEN("pass") {
            REQUIRE_FALSE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
    GIVEN("A non point-process mod file, with a destructor") {
        std::string nmodl_text = R"(
            NEURON {
            }

            DESTRUCTOR {
            }
        )";
        THEN("fail") {
            REQUIRE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
}

SCENARIO("Ion variable in CONSTANT block", "[visitor][semantic_analysis]") {
    GIVEN("A mod file with ion variable redeclared in a CONSTANT block") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX cdp4Nsp
                USEION ca READ cao, cai, ica WRITE cai
            }
            CONSTANT { cao = 2  (mM) }
        )";
        THEN("Semantic analysis fails") {
            REQUIRE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
}

SCENARIO("INDEPENDENT block", "[visitor][semantic_analysis]") {
    GIVEN("A mod file with Independent block with only t") {
        std::string nmodl_text = R"(
            INDEPENDENT {
                t FROM 0 TO 1 WITH 100
            }
        )";
        THEN("Semantic analysis succeed") {
            REQUIRE_FALSE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
    GIVEN("A mod file with Independent block with something else than t") {
        std::string nmodl_text = R"(
            INDEPENDENT {
                t FROM 0 TO 1 WITH 100
                u FROM 0 TO 1 WITH 100
            }
        )";
        THEN("Semantic analysis fails") {
            REQUIRE_FALSE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
}

SCENARIO("FUNCTION_TABLE block", "[visitor][semantic_analysis]") {
    GIVEN("A mod file with FUNCTION_TABLE without argument") {
        std::string nmodl_text = R"(
            FUNCTION_TABLE ttt()
        )";
        THEN("Semantic analysis should fail") {
            REQUIRE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
    GIVEN("A mod file with FUNCTION_TABLE with at least one argument") {
        std::string nmodl_text = R"(
            FUNCTION_TABLE ttt(w (mV))
        )";
        THEN("Semantic analysis should success") {
            REQUIRE_FALSE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
}


SCENARIO("At most one DERIVATIVE block", "[visitor][semantic_analysis]") {
    GIVEN("Only one DERIVATIVE block") {
        std::string nmodl_text = R"(
            DERIVATIVE states {
                m' = m/mTau
            }
        )";
        THEN("Semantic analysis should success") {
            REQUIRE_FALSE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
    GIVEN("2 DERIVATIVE blocks") {
        std::string nmodl_text = R"(
            DERIVATIVE states1 {
                m' = m/mTau
            }
            DERIVATIVE states2 {
                h' = h/hTau
            }
        )";
        THEN("Semantic analysis should failed") {
            REQUIRE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
}

SCENARIO("Recursive function calls", "[visitor][semantic_analysis]") {
    GIVEN("A file with recursive function calls") {
        std::string nmodl_text = R"(
            FUNCTION a() {
                b()
                a = 42
            }
            PROCEDURE b() {
                LOCAL x
                x = a()
            }
        )";
        THEN("Semantic analysis should fail") {
            REQUIRE(run_semantic_analysis_visitor(nmodl_text));
        }
    }
}
