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
#include "codegen_data_helper.hpp"
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

        // Set up the JIT runner.
        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        Runner runner(std::move(module));

        // \todo: change code below
        // 1. Create a helper function to initialise the data based on the vector.
        //    (Possibly variable name (like "m") as well)
        // 2. Create comparison helper for comparing doubles.

        // Create the instance struct data.
        int num_elements = 4;
        const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
        auto codegen_data = codegen::CodegenDataHelper(ast, generated_instance_struct);
        auto instance_data = codegen_data.create_data(num_elements, /*seed=*/1);

        // Initialise the kernel variables to some predefined data.
        int m_index = llvm_visitor.get_instance_var_helper().get_variable_index("m");
        int minf_index = llvm_visitor.get_instance_var_helper().get_variable_index("minf");
        int mtau_index = llvm_visitor.get_instance_var_helper().get_variable_index("mtau");

        std::cout << m_index << " " << minf_index << " " << mtau_index << "\n";

        std::vector<double> m = {1.0, 2.0, 3.0, 4.0};
        std::vector<double> minf = {5.0, 5.0, 5.0, 5.0};
        std::vector<double> mtau = {1.0, 1.0, 1.0, 1.0};

        std::vector<double> m_expected = {4.0, 3.0, 2.0, 1.0};

        double* m_start = static_cast<double*>(instance_data.members[m_index]);
        double* minf_start = static_cast<double*>(instance_data.members[minf_index]);
        double* mtau_start = static_cast<double*>(instance_data.members[mtau_index]);
        for (int i = 0; i < num_elements; ++i) {
            *(m_start + i) = m[i];
            *(minf_start + i) = minf[i];
            *(mtau_start + i) = mtau[i];
        }

        THEN("Values in struct have changed!") {
            runner.run_with_argument<int, void*>("__nrn_state_hh_wrapper", instance_data.base_ptr);
            std::vector<double> m_actual;
            m_actual.assign(static_cast<double*>(instance_data.members[m_index]),
                            static_cast<double*>(instance_data.members[m_index]) + num_elements);
            for (auto res: m_actual)
                std::cout << res << " ";
        }
    }
}
