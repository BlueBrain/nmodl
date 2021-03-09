/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>

#include "ast/program.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/jit_driver.hpp"
#include "parser/nmodl_driver.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/neuron_solve_visitor.hpp"
#include "visitors/solve_block_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace runner;
using namespace visitor;
using nmodl::parser::NmodlDriver;

static double EPSILON = 1e-15;

//=============================================================================
// Simple functions: no optimisations
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

            FUNCTION with_argument(x) {
                with_argument = x
            }

            FUNCTION loop() {
                LOCAL i, j, sum, result
                result = 0
                j = 0
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
                                                 /*opt_passes=*/false);
        llvm_visitor.visit_program(*ast);

        std::unique_ptr<llvm::Module> m = llvm_visitor.get_module();
        Runner runner(std::move(m));

        THEN("functions are evaluated correctly") {
            auto exp_result = runner.run_without_arguments<double>("exponential");
            REQUIRE(fabs(exp_result - 2.718281828459045) < EPSILON);

            auto constant_result = runner.run_without_arguments<double>("constant");
            REQUIRE(fabs(constant_result - 10.0) < EPSILON);

            auto arithmetic_result = runner.run_without_arguments<double>("arithmetic");
            REQUIRE(fabs(arithmetic_result - 2.1) < EPSILON);

            auto function_call_result = runner.run_without_arguments<double>("function_call");
            REQUIRE(fabs(function_call_result - 1.0) < EPSILON);

            double data = 10.0;
            auto with_argument_result = runner.run_with_argument<double, double>("with_argument",
                                                                                 data);
            REQUIRE(fabs(with_argument_result - 10.0) < EPSILON);

            auto loop_result = runner.run_without_arguments<double>("loop");
            REQUIRE(fabs(loop_result - 90.0) < EPSILON);
        }
    }
}

//=============================================================================
// Simple functions: with optimisations
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
            auto exp_result = runner.run_without_arguments<double>("exponential");
            REQUIRE(fabs(exp_result - 2.718281828459045) < EPSILON);

            // Check constant folding.
            auto constant_result = runner.run_without_arguments<double>("constant");
            REQUIRE(fabs(constant_result - 10.0) < EPSILON);

            // Check nested conditionals
            auto conditionals_result = runner.run_without_arguments<double>("conditionals");
            REQUIRE(fabs(conditionals_result - 4.0) < EPSILON);

            // Check constant folding.
            auto arithmetic_result = runner.run_without_arguments<double>("arithmetic");
            REQUIRE(fabs(arithmetic_result - 2.1) < EPSILON);

            auto function_call_result = runner.run_without_arguments<double>("function_call");
            REQUIRE(fabs(function_call_result - 1.0) < EPSILON);
        }
    }
}

//=============================================================================
// State kernel.
//=============================================================================

SCENARIO("Simple scalar kernel", "[llvm][runner]") {
    GIVEN("Simple MOD file with a state update") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX hh
                NONSPECIFIC_CURRENT il
                RANGE minf, mtau, gl, el
            }

            STATE {
                m
            }

            ASSIGNED {
                v (mV)
                minf
                mtau (ms)
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
                il = gl * (v - el)
            }

            DERIVATIVE states {
                m = (minf - m) / mtau
            }
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        // Run passes on the AST to generate LLVM.
        SymtabVisitor().visit_program(*ast);
        NeuronSolveVisitor().visit_program(*ast);
        SolveBlockVisitor().visit_program(*ast);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 /*opt_passes=*/false,
                                                 /*use_single_precision=*/false,
                                                 /*vector_width=*/1);
        llvm_visitor.visit_program(*ast);
        llvm_visitor.wrap_kernel_function("nrn_state_hh");

        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        Runner runner(std::move(module));

        // Create a struct that represents the instance data;
        // \todo: This is a placeholder and will substituted by CodegenInstanceData!
        struct InstanceType {
            double* minf;
            double* mtau;
            double* m;
            double* Dm;
            double* v_unused;
            double* g_unused;
            double* voltage;
            int* node_index;
            double t;
            double dt;
            double celsious;
            int secondorder;
            int node_count;
        };

        double minf[] = {10.0, 10.0};
        double mtau[] = {1.0, 1.0};
        double m[] = {5.0, 2.0};
        double Dm[] = {0.0, 0.0};
        double v_unused[] = {0.0, 0.0};
        double g_unused[] = {0.0, 0.0};
        double volatge[] = {0.0, 0.0};
        int node_index[] = {0, 1};

        InstanceType s;
        s.minf = minf;
        s.mtau = mtau;
        s.m = m;
        s.Dm = Dm;
        s.v_unused = v_unused;
        s.g_unused = g_unused;
        s.voltage = volatge;
        s.node_index = node_index;
        s.t = 0.0;
        s.dt = 1.0;
        s.celsious = 0.0;
        s.secondorder = 0;
        s.node_count = 2;

        void* ptr;
        ptr = &s;

        THEN("Values in struct have changed!") {
            // 10 and 10
            printf("Before: %.1f and %.1f\n", s.m[0], s.m[1]);
            runner.run_with_argument<int, void*>("__nrn_state_hh_wrapper", ptr);
            // (10 - 5) / 1 and (10 - 2) / 1
            // 5            and 8
            printf("After: %.1f and %.1f\n", s.m[0], s.m[1]);
        }
    }
}
