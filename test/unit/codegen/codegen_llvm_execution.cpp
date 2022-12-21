/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>

#include "ast/program.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen_data_helper.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/benchmark/jit_driver.hpp"
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
// Utilities for testing.
//=============================================================================

struct InstanceTestInfo {
    codegen::CodegenInstanceData* instance;
    codegen::InstanceVarHelper helper;
    int num_elements;
};

template <typename T>
bool check_instance_variable(InstanceTestInfo& instance_info,
                             std::vector<T>& expected,
                             const std::string& variable_name) {
    std::vector<T> actual;
    int variable_index = instance_info.helper.get_variable_index(variable_name);
    actual.assign(static_cast<T*>(instance_info.instance->members[variable_index]),
                  static_cast<T*>(instance_info.instance->members[variable_index]) +
                      instance_info.num_elements);

    // While we are comparing double types as well, for simplicity the test cases are hand-crafted
    // so that no floating-point arithmetic is really involved.
    return actual == expected;
}

template <typename T>
void initialise_instance_variable(InstanceTestInfo& instance_info,
                                  std::vector<T>& data,
                                  const std::string& variable_name) {
    int variable_index = instance_info.helper.get_variable_index(variable_name);
    T* data_start = static_cast<T*>(instance_info.instance->members[variable_index]);
    for (int i = 0; i < instance_info.num_elements; ++i)
        *(data_start + i) = data[i];
}

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

        codegen::Platform cpu_platform(/*use_single_precision=*/false,
                                       /*instruction_width=*/1);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 cpu_platform,
                                                 /*opt_level_ir=*/0);
        llvm_visitor.visit_program(*ast);

        std::unique_ptr<llvm::Module> m = llvm_visitor.get_module();
        TestRunner runner(std::move(m));
        runner.initialize_driver();

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

        codegen::Platform cpu_platform(/*use_single_precision=*/false,
                                       /*instruction_width=*/1);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 cpu_platform,
                                                 /*opt_level_ir=*/3);
        llvm_visitor.visit_program(*ast);

        std::unique_ptr<llvm::Module> m = llvm_visitor.get_module();
        TestRunner runner(std::move(m));
        runner.initialize_driver();

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
// State scalar kernel.
//=============================================================================

SCENARIO("Simple scalar kernel", "[llvm][runner]") {
    GIVEN("Simple MOD file with a state update") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                NONSPECIFIC_CURRENT i
                RANGE x0, x1
            }

            STATE {
                x
            }

            ASSIGNED {
                v
                x0
                x1
                i (mA/cm2)
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
                i = 0
            }

            DERIVATIVE states {
                x = (x0 - x) / x1
            }
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        // Run passes on the AST to generate LLVM.
        SymtabVisitor().visit_program(*ast);
        NeuronSolveVisitor().visit_program(*ast);
        SolveBlockVisitor().visit_program(*ast);

        codegen::Platform cpu_platform(/*use_single_precision=*/false,
                                       /*instruction_width=*/1);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 cpu_platform,
                                                 /*opt_level_ir=*/0);
        llvm_visitor.visit_program(*ast);
        llvm_visitor.wrap_kernel_functions();

        // Create the instance struct data.
        int num_elements = 4;
        const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
        auto codegen_data = codegen::CodegenDataHelper(generated_instance_struct);
        auto instance_data = codegen_data.create_data(num_elements, /*seed=*/1);

        // Fill the instance struct data with some values.
        std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
        std::vector<double> x0 = {5.0, 5.0, 5.0, 5.0};
        std::vector<double> x1 = {1.0, 1.0, 1.0, 1.0};

        InstanceTestInfo instance_info{&instance_data,
                                       llvm_visitor.get_instance_var_helper(),
                                       num_elements};
        initialise_instance_variable(instance_info, x, "x");
        initialise_instance_variable(instance_info, x0, "x0");
        initialise_instance_variable(instance_info, x1, "x1");

        // Set up the JIT runner.
        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        TestRunner runner(std::move(module));
        runner.initialize_driver();

        THEN("Values in struct have changed according to the formula") {
            runner.run_with_argument<int, void*>("__nrn_state_test_wrapper",
                                                 instance_data.base_ptr);
            std::vector<double> x_expected = {4.0, 3.0, 2.0, 1.0};
            REQUIRE(check_instance_variable(instance_info, x_expected, "x"));
        }
    }
}

//=============================================================================
// State vectorised kernel with optimisations on.
//=============================================================================

SCENARIO("Simple vectorised kernel", "[llvm][runner]") {
    GIVEN("Simple MOD file with a state update") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                NONSPECIFIC_CURRENT i
                RANGE x0, x1
            }

            STATE {
                x y
            }

            ASSIGNED {
                v
                x0
                x1
                i (mA/cm2)
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
                i = 0
            }

            DERIVATIVE states {
                x = (x0 - x) / x1
                y = v
            }
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        // Run passes on the AST to generate LLVM.
        SymtabVisitor().visit_program(*ast);
        NeuronSolveVisitor().visit_program(*ast);
        SolveBlockVisitor().visit_program(*ast);

        codegen::Platform simd_cpu_platform(/*use_single_precision=*/true,
                                            /*instruction_width=*/4);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 simd_cpu_platform,
                                                 /*opt_level_ir=*/3);
        llvm_visitor.visit_program(*ast);
        llvm_visitor.wrap_kernel_functions();

        // Create the instance struct data.
        int num_elements = 10;
        const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
        auto codegen_data = codegen::CodegenDataHelper(generated_instance_struct);
        auto instance_data = codegen_data.create_data(num_elements, /*seed=*/1);

        // Fill the instance struct data with some values for unit testing.
        std::vector<float> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        std::vector<float> x0 = {11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0};
        std::vector<float> x1 = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        std::vector<float> voltage = {3.0, 4.0, 7.0, 1.0, 2.0, 5.0, 8.0, 6.0, 10.0, 9.0};
        std::vector<int> node_index = {3, 4, 0, 1, 5, 7, 2, 6, 9, 8};

        InstanceTestInfo instance_info{&instance_data,
                                       llvm_visitor.get_instance_var_helper(),
                                       num_elements};
        initialise_instance_variable<float>(instance_info, x, "x");
        initialise_instance_variable<float>(instance_info, x0, "x0");
        initialise_instance_variable<float>(instance_info, x1, "x1");

        initialise_instance_variable<float>(instance_info, voltage, "voltage");
        initialise_instance_variable<int>(instance_info, node_index, "node_index");

        // Set up the JIT runner.
        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        TestRunner runner(std::move(module));
        runner.initialize_driver();

        THEN("Values in struct have changed according to the formula") {
            runner.run_with_argument<int, void*>("__nrn_state_test_wrapper",
                                                 instance_data.base_ptr);
            // Check that the main and remainder loops correctly change the data stored in x.
            std::vector<float> x_expected = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
            REQUIRE(check_instance_variable<float>(instance_info, x_expected, "x"));

            // Check that the gather load produces correct results in y:
            //   y[id] = voltage[node_index[id]]
            std::vector<float> y_expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
            REQUIRE(check_instance_variable<float>(instance_info, y_expected, "y"));
        }
    }
}

//=============================================================================
// Vectorised kernel with ion writes.
//=============================================================================

SCENARIO("Vectorised kernel with scatter instruction", "[llvm][runner]") {
    GIVEN("Simple MOD file with ion writes") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                USEION ca WRITE cai
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
                : increment cai to test scatter
                cai = cai + 1
            }
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        // Run passes on the AST to generate LLVM.
        SymtabVisitor().visit_program(*ast);
        NeuronSolveVisitor().visit_program(*ast);
        SolveBlockVisitor().visit_program(*ast);

        codegen::Platform simd_cpu_platform(/*use_single_precision=*/false,
                                            /*instruction_width=*/2);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 simd_cpu_platform,
                                                 /*opt_level_ir=*/0);
        llvm_visitor.visit_program(*ast);
        llvm_visitor.wrap_kernel_functions();

        // Create the instance struct data.
        int num_elements = 5;
        const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
        auto codegen_data = codegen::CodegenDataHelper(generated_instance_struct);
        auto instance_data = codegen_data.create_data(num_elements, /*seed=*/1);

        // Fill the instance struct data with some values.
        std::vector<double> cai = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> ion_cai = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<int> ion_cai_index = {4, 2, 3, 0, 1};

        InstanceTestInfo instance_info{&instance_data,
                                       llvm_visitor.get_instance_var_helper(),
                                       num_elements};
        initialise_instance_variable(instance_info, cai, "cai");
        initialise_instance_variable(instance_info, ion_cai, "ion_cai");
        initialise_instance_variable(instance_info, ion_cai_index, "ion_cai_index");

        // Set up the JIT runner.
        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        TestRunner runner(std::move(module));
        runner.initialize_driver();

        THEN("Ion values in struct have been updated correctly") {
            runner.run_with_argument<int, void*>("__nrn_state_test_wrapper",
                                                 instance_data.base_ptr);
            // cai[id] = ion_cai[ion_cai_index[id]]
            // cai[id] += 1
            std::vector<double> cai_expected = {6.0, 4.0, 5.0, 2.0, 3.0};
            REQUIRE(check_instance_variable(instance_info, cai_expected, "cai"));

            // ion_cai[ion_cai_index[id]] = cai[id]
            std::vector<double> ion_cai_expected = {2.0, 3.0, 4.0, 5.0, 6.0};
            REQUIRE(check_instance_variable(instance_info, ion_cai_expected, "ion_cai"));
        }
    }
}

//=============================================================================
// Vectorised kernel with control flow.
//=============================================================================

SCENARIO("Vectorised kernel with simple control flow", "[llvm][runner]") {
    GIVEN("Simple MOD file with if statement") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
            }

            STATE {
                w x y z
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
                IF (v > 0) {
                    w = v * w
                }

                IF (x < 0) {
                    x = 7
                }

                IF (0 <= y && y < 10 || z == 0) {
                    y = 2 * y
                } ELSE {
                    z = z - y
                }

            }
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        // Run passes on the AST to generate LLVM.
        SymtabVisitor().visit_program(*ast);
        NeuronSolveVisitor().visit_program(*ast);
        SolveBlockVisitor().visit_program(*ast);

        codegen::Platform simd_cpu_platform(/*use_single_precision=*/false,
                                            /*instruction_width=*/2);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 simd_cpu_platform,
                                                 /*opt_level_ir=*/0);
        llvm_visitor.visit_program(*ast);
        llvm_visitor.wrap_kernel_functions();

        // Create the instance struct data.
        int num_elements = 5;
        const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
        auto codegen_data = codegen::CodegenDataHelper(generated_instance_struct);
        auto instance_data = codegen_data.create_data(num_elements, /*seed=*/1);

        // Fill the instance struct data with some values.
        std::vector<double> x = {-1.0, 2.0, -3.0, 4.0, -5.0};
        std::vector<double> y = {11.0, 2.0, -3.0, 4.0, 100.0};
        std::vector<double> z = {0.0, 1.0, 20.0, 0.0, 40.0};

        std::vector<double> w = {10.0, 20.0, 30.0, 40.0, 50.0};
        std::vector<double> voltage = {-1.0, 2.0, -1.0, 2.0, -1.0};
        std::vector<int> node_index = {1, 2, 3, 4, 0};

        InstanceTestInfo instance_info{&instance_data,
                                       llvm_visitor.get_instance_var_helper(),
                                       num_elements};
        initialise_instance_variable(instance_info, w, "w");
        initialise_instance_variable(instance_info, voltage, "voltage");
        initialise_instance_variable(instance_info, node_index, "node_index");

        initialise_instance_variable(instance_info, x, "x");
        initialise_instance_variable(instance_info, y, "y");
        initialise_instance_variable(instance_info, z, "z");

        // Set up the JIT runner.
        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        TestRunner runner(std::move(module));
        runner.initialize_driver();

        THEN("Masked instructions are generated") {
            runner.run_with_argument<int, void*>("__nrn_state_test_wrapper",
                                                 instance_data.base_ptr);
            std::vector<double> w_expected = {20.0, 20.0, 60.0, 40.0, 50.0};
            REQUIRE(check_instance_variable(instance_info, w_expected, "w"));

            std::vector<double> x_expected = {7.0, 2.0, 7.0, 4.0, 7.0};
            REQUIRE(check_instance_variable(instance_info, x_expected, "x"));

            std::vector<double> y_expected = {22.0, 4.0, -3.0, 8.0, 100.0};
            std::vector<double> z_expected = {0.0, 1.0, 23.0, 0.0, -60.0};
            REQUIRE(check_instance_variable(instance_info, y_expected, "y"));
            REQUIRE(check_instance_variable(instance_info, z_expected, "z"));
        }
    }
}

//=============================================================================
// Kernel with atomic updates.
//=============================================================================

SCENARIO("Kernel with atomic updates", "[llvm][runner]") {
    GIVEN("An atomic update") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                USEION na READ ena WRITE ina
                USEION ka READ eka WRITE ika
            }

            STATE { }

            ASSIGNED {
                v (mV)
                ena (mV)
                ina (mA/cm2)
            }

            BREAKPOINT { }

            DERIVATIVE states { }

            : The atomic update that we want to check is:
            :
            :     ion_ina_id = mech->ion_ina_index[id]
            :     ion_ika_id = mech->ion_ika_index[id]
            :     mech->ion_ina[ion_ina_id] += mech->ina[id]
            :     mech->ion_ika[ion_ika_id] += mech->ika[id]
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        // Run passes on the AST to generate LLVM.
        SymtabVisitor().visit_program(*ast);
        NeuronSolveVisitor().visit_program(*ast);
        SolveBlockVisitor().visit_program(*ast);

        codegen::Platform simd_cpu_platform(/*use_single_precision=*/false,
                                            /*instruction_width=*/4);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 simd_cpu_platform,
                                                 /*opt_level_ir=*/3);
        llvm_visitor.visit_program(*ast);
        llvm_visitor.wrap_kernel_functions();

        // Create the instance struct data.
        int num_elements = 5;
        const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
        auto codegen_data = codegen::CodegenDataHelper(generated_instance_struct);
        auto instance_data = codegen_data.create_data(num_elements, /*seed=*/1);

        // With these indices ion_ina[1] = 1 + 2 + 3 + 4 + 5 = 15.
        std::vector<int> ion_ina_index = {1, 1, 1, 1, 1};
        std::vector<double> ion_ina = {0.0, 0.0, 0.0, 0.0, 0.0};
        std::vector<double> ina = {1.0, 2.0, 3.0, 4.0, 5.0};

        // With these indices:
        //     ion_ika[1] = 3 + 4 = 7.
        //     ion_ika[2] = 1 + 20 = 21.
        //     ion_ika[3] = -5 + 5 = 0.
        std::vector<int> ion_ika_index = {2, 2, 1, 1, 3};
        std::vector<double> ion_ika = {0.0, 0.0, 0.0, -5.0, 0.0};
        std::vector<double> ika = {1.0, 20.0, 3.0, 4.0, 5.0};

        InstanceTestInfo instance_info{&instance_data,
                                       llvm_visitor.get_instance_var_helper(),
                                       num_elements};

        initialise_instance_variable(instance_info, ion_ina_index, "ion_ina_index");
        initialise_instance_variable(instance_info, ion_ina, "ion_ina");
        initialise_instance_variable(instance_info, ina, "ina");
        initialise_instance_variable(instance_info, ion_ika_index, "ion_ika_index");
        initialise_instance_variable(instance_info, ion_ika, "ion_ika");
        initialise_instance_variable(instance_info, ika, "ika");

        // Set up the JIT runner.
        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        TestRunner runner(std::move(module));
        runner.initialize_driver();

        THEN("updates are commputed correctly with vector instructions and optimizations on") {
            runner.run_with_argument<int, void*>("__nrn_cur_test_wrapper", instance_data.base_ptr);
            // Recall:
            //     ion_ina_id = mech->ion_ina_index[id]
            //     ion_ika_id = mech->ion_ika_index[id]
            //     mech->ion_ina[ion_ina_id] += mech->ina[id]
            //     mech->ion_ika[ion_ika_id] += mech->ika[id]
            std::vector<double> ion_ina_expected = {0.0, 15.0, 0.0, 0.0, 0.0};
            REQUIRE(check_instance_variable(instance_info, ion_ina_expected, "ion_ina"));

            std::vector<double> ion_ika_expected = {0.0, 7.0, 21.0, 0.0, 0.0};
            REQUIRE(check_instance_variable(instance_info, ion_ika_expected, "ion_ika"));
        }
    }

    GIVEN("Another atomic update") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                USEION na READ ena WRITE ina
                USEION ka READ eka WRITE ika
            }

            STATE { }

            ASSIGNED {
                v (mV)
                ena (mV)
                ina (mA/cm2)
            }

            BREAKPOINT { }

            DERIVATIVE states { }

            : The atomic update that we want to check is again:
            :
            :     ion_ina_id = mech->ion_ina_index[id]
            :     ion_ika_id = mech->ion_ika_index[id]
            :     mech->ion_ina[ion_ina_id] += mech->ina[id]
            :     mech->ion_ika[ion_ika_id] += mech->ika[id]
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        // Run passes on the AST to generate LLVM.
        SymtabVisitor().visit_program(*ast);
        NeuronSolveVisitor().visit_program(*ast);
        SolveBlockVisitor().visit_program(*ast);

        codegen::Platform simd_cpu_platform(/*use_single_precision=*/false,
                                            /*instruction_width=*/2);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 simd_cpu_platform,
                                                 /*opt_level_ir=*/0);
        llvm_visitor.visit_program(*ast);
        llvm_visitor.wrap_kernel_functions();

        // Create the instance struct data.
        int num_elements = 6;
        const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
        auto codegen_data = codegen::CodegenDataHelper(generated_instance_struct);
        auto instance_data = codegen_data.create_data(num_elements, /*seed=*/1);

        // With these indices ion_ina[1] = 1 + 3 + 5 = 9.
        // With these indices ion_ina[4] = 2 + 4 + 6 = 12.
        std::vector<int> ion_ina_index = {1, 4, 1, 4, 1, 4};
        std::vector<double> ion_ina = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        std::vector<double> ina = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

        // With these indices:
        //     ion_ika[1] = 3 + 4 + 5 = 12.
        //     ion_ika[2] = 1 + 20 + 6 = 27.
        std::vector<int> ion_ika_index = {2, 2, 1, 1, 1, 2};
        std::vector<double> ion_ika = {0.0, 0.0, 0.0, -5.0, 0.0, 0.0};
        std::vector<double> ika = {1.0, 20.0, 3.0, 4.0, 5.0, 6.0};

        InstanceTestInfo instance_info{&instance_data,
                                       llvm_visitor.get_instance_var_helper(),
                                       num_elements};

        initialise_instance_variable(instance_info, ion_ina_index, "ion_ina_index");
        initialise_instance_variable(instance_info, ion_ina, "ion_ina");
        initialise_instance_variable(instance_info, ina, "ina");
        initialise_instance_variable(instance_info, ion_ika_index, "ion_ika_index");
        initialise_instance_variable(instance_info, ion_ika, "ion_ika");
        initialise_instance_variable(instance_info, ika, "ika");

        // Set up the JIT runner.
        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        TestRunner runner(std::move(module));
        runner.initialize_driver();

        THEN("Atomic updates are correct without optimizations") {
            runner.run_with_argument<int, void*>("__nrn_cur_test_wrapper", instance_data.base_ptr);
            // Recall:
            //     ion_ina_id = mech->ion_ina_index[id]
            //     ion_ika_id = mech->ion_ika_index[id]
            //     mech->ion_ina[ion_ina_id] += mech->ina[id]
            //     mech->ion_ika[ion_ika_id] += mech->ika[id]
            std::vector<double> ion_ina_expected = {0.0, 9.0, 0.0, 0.0, 12.0, 0.0};
            REQUIRE(check_instance_variable(instance_info, ion_ina_expected, "ion_ina"));

            std::vector<double> ion_ika_expected = {0.0, 12.0, 27.0, -5.0, 0.0, 0.0};
            REQUIRE(check_instance_variable(instance_info, ion_ika_expected, "ion_ika"));
        }
    }

    GIVEN("Atomic updates of rhs and d") {
        std::string nmodl_text = R"(
            NEURON {
                POINT_PROCESS test
                USEION na READ ena WRITE ina
                USEION ka READ eka WRITE ika
            }

            STATE { }

            ASSIGNED {
                v (mV)
                ena (mV)
                ina (mA/cm2)
            }

            BREAKPOINT { }

            DERIVATIVE states { }

            : The atomic update that we want to check is again:
            :
            :     node_id = mech->node_index[id]
            :     mech->vec_rhs[node_id] -= rhs
            :     mech->vec_d[node_id] -= g
        )";


        NmodlDriver driver;
        const auto& ast = driver.parse_string(nmodl_text);

        // Run passes on the AST to generate LLVM.
        SymtabVisitor().visit_program(*ast);
        NeuronSolveVisitor().visit_program(*ast);
        SolveBlockVisitor().visit_program(*ast);

        codegen::Platform simd_cpu_platform(/*use_single_precision=*/false,
                                            /*instruction_width=*/2);
        codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                 /*output_dir=*/".",
                                                 simd_cpu_platform,
                                                 /*opt_level_ir=*/0);
        llvm_visitor.visit_program(*ast);
        llvm_visitor.wrap_kernel_functions();

        // Create the instance struct data.
        int num_elements = 6;
        const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
        auto codegen_data = codegen::CodegenDataHelper(generated_instance_struct);
        auto instance_data = codegen_data.create_data(num_elements, /*seed=*/1);

        // With these indices vec_rhs[1] = -0.2-1.e2/1.5*2-1.e2/3.4*6-1.e2/5.2*10 =
        // -502.3116138763197.
        // With these indices vec_rhs[4] =
        // -0.54-1.e2/2.3*22.0-1.e2/4.1*8.0-1.e2/6.0*12.0 = -1351.103690349947.
        // vec_d remains the same because the contribution of g each time is 0.0.
        std::vector<int> node_index = {1, 4, 1, 4, 1, 4};
        std::vector<double> ina = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<double> ika = {1.0, 20.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<double> vec_rhs = {0.64, -0.2, 1.1, 0.42, 0.54, -0.36};
        std::vector<double> vec_d = {1.6, 2.5, 3.4, 4.3, 5.2, 6.1};
        std::vector<int> node_area_index = {0, 1, 2, 3, 4, 5};
        std::vector<double> node_area = {1.5, 2.3, 3.4, 4.1, 5.2, 6.0};

        InstanceTestInfo instance_info{&instance_data,
                                       llvm_visitor.get_instance_var_helper(),
                                       num_elements};

        initialise_instance_variable(instance_info, node_index, "node_index");
        initialise_instance_variable(instance_info, ina, "ina");
        initialise_instance_variable(instance_info, ika, "ika");
        initialise_instance_variable(instance_info, vec_rhs, "vec_rhs");
        initialise_instance_variable(instance_info, vec_d, "vec_d");
        initialise_instance_variable(instance_info, node_area_index, "node_area_index");
        initialise_instance_variable(instance_info, node_area, "node_area");

        // Set up the JIT runner.
        std::unique_ptr<llvm::Module> module = llvm_visitor.get_module();
        TestRunner runner(std::move(module));
        runner.initialize_driver();

        THEN("Atomic updates are correct") {
            runner.run_with_argument<int, void*>("__nrn_cur_test_wrapper", instance_data.base_ptr);
            // Recall:
            //     node_id = mech->node_index[id]
            //     mech->vec_rhs[node_id] -= rhs
            //     mech->vec_d[node_id] -= g
            std::vector<double> vec_rhs_expected = {
                0.64, -502.3116138763197, 1.1, 0.42, -1351.103690349947, -0.36};
            REQUIRE(check_instance_variable(instance_info, vec_rhs_expected, "vec_rhs"));

            std::vector<double> vec_d_expected = {1.6, 2.5, 3.4, 4.3, 5.2, 6.1};
            REQUIRE(check_instance_variable(instance_info, vec_d_expected, "vec_d"));
        }
    }
}