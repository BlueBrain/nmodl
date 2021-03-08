/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>

#include "ast/all.hpp"
#include "ast/program.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen_data_helper.hpp"
#include "parser/nmodl_driver.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/neuron_solve_visitor.hpp"
#include "visitors/perf_visitor.hpp"
#include "visitors/solve_block_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace codegen;
using namespace visitor;
using nmodl::parser::NmodlDriver;

//=============================================================================
// Utility to get initialized Struct Instance data
//=============================================================================

codegen::CodegenInstanceData generate_instance_data(const std::string& text,
                                                    bool opt = false,
                                                    bool use_single_precision = false,
                                                    const size_t num_elements = 100) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    // Generate full AST and solve the BREAKPOINT block to be able to generate the Instance Struct
    AstVisitor().visit_program(*ast);
    SymtabVisitor(true).visit_program(*ast);
    NeuronSolveVisitor().visit_program(*ast);
    SolveBlockVisitor().visit_program(*ast);
    SymtabVisitor(true).visit_program(*ast);

    codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"test",
                                             /*output_dir=*/".",
                                             opt,
                                             use_single_precision);
    llvm_visitor.visit_program(*ast);
    llvm_visitor.print_module();
    const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
    auto codegen_data = codegen::CodegenDataHelper(ast, generated_instance_struct);
    auto instance_data = codegen_data.create_data(num_elements, 42);
    return instance_data;
}

template <typename T>
bool compare_vectors(T* instance_struct_data_ptr, const std::vector<T>& generated_data) {
    std::vector<T> instance_struct_vector;
    std::cout << "Generated data size: " << generated_data.size() << std::endl;
    instance_struct_vector.assign(static_cast<T*>(instance_struct_data_ptr),
                                  static_cast<T*>(instance_struct_data_ptr) +
                                      generated_data.size());
    for (auto value: instance_struct_vector) {
        std::cout << value << std::endl;
    }
    return instance_struct_vector == generated_data;
}

//=============================================================================
// Simple Instance Struct creation
//=============================================================================

SCENARIO("Instance Struct creation", "[visitor][llvm][instance_struct]") {
    GIVEN("Instantiate simple Instance Struct") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                USEION na READ ena
                RANGE minf, mtau
                THREADSAFE : assigned GLOBALs will be per thread
            }

            STATE {
                m
            }

            ASSIGNED {
                v (mV)
                celsius (degC)
                ena (mV)
                minf
                mtau
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
                m' =  (minf-m)/mtau
            }
        )";


        THEN("instance struct elements are properly initialized") {
            const size_t num_elements = 10;
            auto instance_data = generate_instance_data(nmodl_text,
                                                        /*opt=*/false,
                                                        /*use_single_precision=*/true,
                                                        num_elements);
            size_t minf_index = 0;
            size_t mtau_index = 1;
            size_t m_index = 2;
            size_t Dm_index = 3;
            size_t ena_index = 4;
            size_t v_unused_index = 5;
            size_t g_unused_index = 6;
            size_t ion_ena_index = 7;
            size_t ion_ena_index_index = 8;
            size_t voltage_index = 9;
            size_t node_index_index = 10;
            size_t t_index = 11;
            size_t dt_index = 12;
            size_t celsius_index = 13;
            size_t secondorder_index = 14;
            size_t node_count_index = 15;
            // Check if the various instance struct fields are properly initialized
            REQUIRE(compare_vectors((double*) instance_data.members[minf_index],
                                    generate_double_data(minf_index, num_elements)));
            REQUIRE(compare_vectors((double*) instance_data.members[ena_index],
                                    generate_double_data(ena_index, num_elements)));
            REQUIRE(compare_vectors((double*) instance_data.members[ion_ena_index],
                                    generate_double_data(ion_ena_index, num_elements)));
            REQUIRE(compare_vectors((int*) instance_data.members[node_index_index],
                                    generate_int_data(node_index_index, num_elements)));
            REQUIRE(*static_cast<double*>(instance_data.members[t_index]) ==
                    default_nthread_t_value);
            REQUIRE(*static_cast<int*>(instance_data.members[node_count_index]) == num_elements);

            // Hard code test__instance_var__type struct
            struct test__instance_var__type {
                double* minf;
                double* mtau;
                double* m;
                double* Dm;
                double* ena;
                double* v_unused;
                double* g_unused;
                double* ion_ena;
                int* ion_ena_index;
                double* voltage;
                int* node_index;
                double t;
                double dt;
                double celsius;
                int secondorder;
                int node_count;
            };
            // Test if test__instance_var__type struct is properly initialized
            test__instance_var__type* test_instance_struct =
                (test__instance_var__type*) instance_data.base_ptr;
            REQUIRE(compare_vectors(test_instance_struct->minf,
                                    generate_double_data(minf_index, num_elements)));
            REQUIRE(compare_vectors(test_instance_struct->ena,
                                    generate_double_data(ena_index, num_elements)));
            REQUIRE(compare_vectors(test_instance_struct->ion_ena,
                                    generate_double_data(ion_ena_index, num_elements)));
            REQUIRE(compare_vectors(test_instance_struct->node_index,
                                    generate_int_data(node_index_index, num_elements)));
            REQUIRE(test_instance_struct->t == default_nthread_t_value);
            REQUIRE(test_instance_struct->celsius == default_celsius_value);
        }
    }
}
