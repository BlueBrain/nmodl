/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <fmt/format.h>

#include "codegen/codegen_ispc_visitor.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/string_utils.hpp"

using namespace fmt::literals;
using namespace symtab;
using namespace syminfo;


/****************************************************************************************/
/*                      Routines must be overloaded in backend                          */
/****************************************************************************************/


std::string CodegenIspcVisitor::compute_method_name(BlockType type) {
    /*if (type == BlockType::Initial) { @todo : init method could be added later into ispc
        return method_name("nrn_init");
    }*/
    if (type == BlockType::State) {
        return method_name("ispc_nrn_state");
    }
    if (type == BlockType::Equation) {
        return method_name("ispc_nrn_cur");
    }
    throw std::runtime_error("compute_method_name not implemented");
}


void CodegenIspcVisitor::print_backend_includes() {
    printer->add_line("#include <cuda.h>");
}


std::string CodegenIspcVisitor::backend_name() {
    return "ispc (api-compatibility)";
}


/*
 * Depending on the backend, print condition/loop for iterating over channels
 *
 * Use ispc foreach loop
 */
void CodegenIspcVisitor::print_channel_iteration_block_begin() {
    printer->start_block("foreach (int id = start ... end) ");
}


void CodegenIspcVisitor::print_channel_iteration_block_end() {
    printer->end_block();
    printer->add_newline();
}

/* @todo : check why in CUDA those are off?
void CodegenIspcVisitor::print_nrn_cur_matrix_shadow_reduction() {
    // do nothing
}


void CodegenIspcVisitor::print_rhs_d_shadow_variables() {
    // do nothing
}

bool CodegenIspcVisitor::nrn_cur_reduction_loop_required() {
    return false;
}
*/


void CodegenIspcVisitor::print_backend_namespace_start() {
    printer->add_newline(1);
    printer->start_block("namespace ispc");
}


void CodegenIspcVisitor::print_backend_namespace_stop() {
    printer->end_block();
    printer->add_newline();
}


void CodegenIspcVisitor::print_compute_functions() {
    print_top_verbatim_blocks();
    print_function_prototypes();

    for (const auto& procedure: info.procedures) {
        print_procedure(procedure);
    }

    for (const auto& function: info.functions) {
        print_function(function);
    }

    print_net_send_buffering();
    print_net_receive();
    print_net_receive_buffering();
    print_nrn_cur();
    print_nrn_state();
}

void CodegenIspcVisitor::print_ispc_includes() {
    printer->add_newline();
    printer->add_line("#include \"fast_math.ispc\"");
}

/****************************************************************************************/
/*                    Main code printing entry points and wrappers                      */
/****************************************************************************************/

void CodegenIspcVisitor::print_headers_include() {
    print_ispc_includes();
}

void CodegenIspcVisitor::print_wrapper_headers_include() {
    print_standard_includes();
    print_backend_includes();
    print_coreneuron_includes();
}

void CodegenIspcVisitor::print_wrapper_routine(std::string wraper_function, BlockType type) {
    auto args = "NrnThread* nt, Memb_list* ml, int type";
    wraper_function = method_name(wraper_function);
    auto compute_function = compute_method_name(type);

    printer->add_newline(2);
    printer->start_block("void {}({})"_format(wraper_function, args));
    printer->add_line("int nodecount = ml->nodecount;");
    printer->add_line("int nthread = 256;");
    printer->add_line("int nblock = (nodecount+nthread-1)/nthread;");
    printer->add_line("{}<<<nblock, nthread>>>(nt, ml, type);"_format(compute_function));
    printer->add_line("cudaDeviceSynchronize();");
    printer->end_block();
    printer->add_newline();
}


void CodegenIspcVisitor::codegen_wrapper_routines() {
    print_wrapper_routine("nrn_cur", BlockType::Equation);
    print_wrapper_routine("nrn_state", BlockType::State);
}


void CodegenIspcVisitor::print_codegen_routines() {
    codegen = true;
    print_backend_info();
    print_headers_include();

    print_data_structures();
    print_common_getters();

    print_compute_functions();

    codegen_wrapper_routines();
    print_codegen_wrapper_routines();
}

void CodegenIspcVisitor::print_codegen_wrapper_routines() {
    printer = wrapper_printer;
    print_backend_info();
    print_headers_include();
    print_namespace_begin();

    print_data_structures();
    print_common_getters();

    print_compute_functions();

    codegen_wrapper_routines();

    print_namespace_end();
}
