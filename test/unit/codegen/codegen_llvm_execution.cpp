/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>
#include <regex>

#include "ast/program.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/jit_driver.hpp"
#include "parser/nmodl_driver.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace runner;
using namespace visitor;
using nmodl::parser::NmodlDriver;

static double EPSILON = 1e-15;

//=============================================================================
// No optimisations
//=============================================================================

SCENARIO("Arithmetic expression", "[llvm][runner]") {
    GIVEN("Functions with some arithmetic expressions") {
        std::string nmodl_text = R"(
            FUNCTION exponential() {
                LOCAL i
                i = 1
                exponential = exp(i)
            }

            FUNCTION constant() {
                constant = 10
            }

            FUNCTION arithmetic() {
                LOCAL x, y
                x = 3
                y = 7
                arithmetic = x * y / (x + y)
            }

            FUNCTION bar() {
                LOCAL i, j
                i = 2
                j = i + 2
                bar = 2 * 3 + j
            }

            FUNCTION function_call() {
                foo()
                function_call = bar() / constant()
            }

            PROCEDURE foo() {}
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        SymtabVisitor().visit_program(*ast);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 /*opt_passes=*/false);
        llvm_visitor.visit_program(*ast);

        std::unique_ptr<llvm::Module> m = llvm_visitor.get_module();
        Runner runner(std::move(m));

        THEN("functions are evaluated correctly") {
            auto exp_result = runner.run<double>("exponential");
            REQUIRE(fabs(exp_result - 2.718281828459045) < EPSILON);

            auto constant_result = runner.run<double>("constant");
            REQUIRE(fabs(constant_result - 10.0) < EPSILON);

            auto arithmetic_result = runner.run<double>("arithmetic");
            REQUIRE(fabs(arithmetic_result - 2.1) < EPSILON);

            auto function_call_result = runner.run<double>("function_call");
            REQUIRE(fabs(function_call_result - 1.0) < EPSILON);
        }
    }
}

//=============================================================================
// With optimisations
//=============================================================================

SCENARIO("Optimised arithmetic expression", "[llvm][runner]") {
    GIVEN("Functions with some arithmetic expressions") {
        std::string nmodl_text = R"(
            FUNCTION exponential() {
                LOCAL i
                i = 1
                exponential = exp(i)
            }

            FUNCTION constant() {
                constant = 10 * 2 - 100 / 50 * 5
            }

            FUNCTION arithmetic() {
                LOCAL x, y
                x = 3
                y = 7
                arithmetic = x * y / (x + y)
            }

            FUNCTION conditionals() {
                LOCAL x, y, z
                x = 100
                y = -100
                z = 0
                if (x == 200) {
                    conditionals = 1
                } else if (x == 400) {
                    conditionals = 2
                } else if (x == 100) {
                    if (y == -100 && z != 0) {
                        conditionals = 3
                    } else {
                        if (y < -99 && z == 0) {
                          conditionals = 4
                        } else {
                            conditionals = 5
                        }
                    }
                } else {
                    conditionals = 6
                }
            }

            FUNCTION bar() {
                LOCAL i, j
                i = 2
                j = i + 2
                bar = 2 * 3 + j
            }

            FUNCTION function_call() {
                foo()
                function_call = bar() / constant()
            }

            PROCEDURE foo() {}

            FUNCTION loop() {
                LOCAL i, j, sum, result
                    result = 0
                    WHILE (j < 2) {
                        i = 0
                        sum = 0
                        WHILE (i < 10) {
                            sum = sum + i
                            i = i + 1
                        }
                        j = j + 1
                        result = result + sum
                    }
                loop = result
            }
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        SymtabVisitor().visit_program(*ast);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 /*opt_passes=*/true);
        llvm_visitor.visit_program(*ast);

        std::unique_ptr<llvm::Module> m = llvm_visitor.get_module();
        Runner runner(std::move(m));

        THEN("optimizations preserve function results") {
            // Check exponential is turned into a constant.
            auto exp_result = runner.run<double>("exponential");
            REQUIRE(fabs(exp_result - 2.718281828459045) < EPSILON);

            // Check constant folding.
            auto constant_result = runner.run<double>("constant");
            REQUIRE(fabs(constant_result - 10.0) < EPSILON);

            // Check nested conditionals
            auto conditionals_result = runner.run<double>("conditionals");
            REQUIRE(fabs(conditionals_result - 4.0) < EPSILON);

            // Check constant folding.
            auto arithmetic_result = runner.run<double>("arithmetic");
            REQUIRE(fabs(arithmetic_result - 2.1) < EPSILON);

            auto function_call_result = runner.run<double>("function_call");
            REQUIRE(fabs(function_call_result - 1.0) < EPSILON);

            // Check nested while statement produces correct result.
            auto loop_result = runner.run<double>("loop");
            REQUIRE(fabs(loop_result - 90.0) < EPSILON);
        }
    }
}
