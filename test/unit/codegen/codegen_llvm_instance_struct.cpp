/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>

#include "ast/program.hpp"
#include "ast/all.hpp"
#include "codegen_data_helper.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "parser/nmodl_driver.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

#include "visitors/after_cvode_to_cnexp_visitor.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/constant_folder_visitor.hpp"
#include "visitors/global_var_visitor.hpp"
#include "visitors/inline_visitor.hpp"
#include "visitors/ispc_rename_visitor.hpp"
#include "visitors/json_visitor.hpp"
#include "visitors/kinetic_block_visitor.hpp"
#include "visitors/local_to_assigned_visitor.hpp"
#include "visitors/local_var_rename_visitor.hpp"
#include "visitors/localize_visitor.hpp"
#include "visitors/loop_unroll_visitor.hpp"
#include "visitors/neuron_solve_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/perf_visitor.hpp"
#include "visitors/solve_block_visitor.hpp"
#include "visitors/steadystate_visitor.hpp"
#include "visitors/sympy_conductance_visitor.hpp"
#include "visitors/sympy_solver_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/units_visitor.hpp"
#include "visitors/verbatim_var_rename_visitor.hpp"
#include "visitors/verbatim_visitor.hpp"
#include "config/config.h"


using namespace nmodl;
using namespace codegen;
using namespace visitor;
using nmodl::parser::NmodlDriver;

//=============================================================================
// Utility to get LLVM module as a string
//=============================================================================

codegen::CodegenInstanceData generate_instance_data(const std::string& text,
                             bool opt = false,
                             bool use_single_precision = false) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    AstVisitor().visit_program(*ast);

    SymtabVisitor(true).visit_program(*ast);
    PerfVisitor().visit_program(*ast);
    NeuronSolveVisitor().visit_program(*ast);
    SolveBlockVisitor().visit_program(*ast);
    SymtabVisitor(true).visit_program(*ast);

    codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                                /*output_dir=*/".",
                                                opt,
                                                use_single_precision);
    llvm_visitor.visit_program(*ast);
    llvm_visitor.print_module();
    const auto& generated_instance_struct = llvm_visitor.get_instance_struct_ptr();
    auto codegen_data = codegen::CodegenDataHelper(ast, generated_instance_struct);
    auto instance_data = codegen_data.create_data(100, 42);
    return instance_data;
}


//=============================================================================
// Simple Instance Struct creation
//=============================================================================

SCENARIO("Instance Struct creation", "[visitor][llvm][instance_struct]") {
    GIVEN("Instantiate simple Instance Struct") {
        std::string nmodl_text = R"(
            TITLE hh.mod   squid sodium, potassium, and leak channels

            NEURON {
                SUFFIX hh
                RANGE minf, hinf, ninf, mtau, htau, ntau
                THREADSAFE : assigned GLOBALs will be per thread
            }

            STATE {
                m
            }

            ASSIGNED {
                v (mV)
                celsius (degC)
                minf hinf ninf
                mtau (ms) htau (ms) ntau (ms)
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
                m' =  (minf-m)/mtau
            }
        )";

        THEN("instance struct elements are properly initialized") {
            auto instance_data =
                generate_instance_data(nmodl_text, /*opt=*/false, /*use_single_precision=*/true);
            
            const auto minf_index = 0;
            const auto mtau_index = 1;
            const auto gl_index = 2;
            const auto el_index = 3;
            const auto v_index = 4;


            // Check the float values are loaded correctly and added
            REQUIRE(*(double*)(instance_data.members[12]) == 34);
        }
    }

}
