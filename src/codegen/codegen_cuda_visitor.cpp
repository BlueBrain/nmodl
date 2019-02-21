/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <fmt/format.h>

#include "codegen/codegen_cuda_visitor.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/string_utils.hpp"

using namespace fmt::literals;
using namespace symtab;
using namespace syminfo;


/****************************************************************************************/
/*                      Routines must be overloaded in backend                          */
/****************************************************************************************/


/**
 * As initial block is/can be executed on c/cpu backend, gpu/cuda
 * backend can mark the parameter as constant even if they have
 * write count > 0 (typically due to initial block).
 */
bool CodegenCudaVisitor::is_constant_variable(std::string name) {
    auto symbol = program_symtab->lookup_in_scope(name);
    bool is_constant = false;
    if (symbol != nullptr) {
        if (symbol->has_properties(NmodlType::read_ion_var)) {
            is_constant = true;
        }
        if (symbol->has_properties(NmodlType::param_assign)) {
            is_constant = true;
        }
    }
    return is_constant;
}


std::string CodegenCudaVisitor::compute_method_name(BlockType type) {
    if (type == BlockType::Initial) {
        return method_name("nrn_init");
    }
    if (type == BlockType::State) {
        return method_name("cuda_nrn_state");
    }
    if (type == BlockType::Equation) {
        return method_name("cuda_nrn_cur");
    }
    throw std::runtime_error("compute_method_name not implemented");
}


void CodegenCudaVisitor::print_atomic_op(const std::string& lhs,
                                         const std::string& op,
                                         const std::string& rhs) {
    std::string function;
    if (op == "+") {
        function = "atomicAdd";
    } else if (op == "-") {
        function = "atomicSub";
    } else {
        throw std::runtime_error("CUDA backend error : {} not supported"_format(op));
    }
    printer->add_line("{}(&{}, {});"_format(function, lhs, rhs));
}


void CodegenCudaVisitor::print_backend_includes() {
    printer->add_line("#include <cuda.h>");
}


std::string CodegenCudaVisitor::backend_name() {
    return "C-CUDA (api-compatibility)";
}


void CodegenCudaVisitor::print_global_method_annotation() {
    printer->add_line("__global__");
}


void CodegenCudaVisitor::print_device_method_annotation() {
    printer->add_line("__device__");
}


void CodegenCudaVisitor::print_nrn_cur_matrix_shadow_update() {
    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    stringutils::remove_character(rhs_op, '=');
    stringutils::remove_character(d_op, '=');
    print_atomic_op("vec_rhs[node_id]", rhs_op, "rhs");
    print_atomic_op("vec_d[node_id]", d_op, "g");
}


/*
 * Depending on the backend, print condition/loop for iterating over channels
 *
 * For GPU backend its thread id less than total channel instances. Below we
 * assume we launch 1-d grid.
 */
void CodegenCudaVisitor::print_channel_iteration_block_begin() {
    printer->add_line("int id = blockIdx.x * blockDim.x + threadIdx.x;");
    printer->start_block("if (id < end) ");
}


void CodegenCudaVisitor::print_channel_iteration_block_end() {
    printer->end_block();
    printer->add_newline();
}


void CodegenCudaVisitor::print_nrn_cur_matrix_shadow_reduction() {
    // do nothing
}


void CodegenCudaVisitor::print_rhs_d_shadow_variables() {
    // do nothing
}


bool CodegenCudaVisitor::nrn_cur_reduction_loop_required() {
    return false;
}


void CodegenCudaVisitor::print_backend_namespace_start() {
    printer->add_newline(1);
    printer->start_block("namespace cuda");
}


void CodegenCudaVisitor::print_backend_namespace_stop() {
    printer->end_block();
    printer->add_newline();
}


void CodegenCudaVisitor::print_compute_functions() {
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


void CodegenCudaVisitor::print_wrapper_routine(std::string wraper_function, BlockType type) {
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


void CodegenCudaVisitor::codegen_wrapper_routines() {
    print_wrapper_routine("nrn_cur", BlockType::Equation);
    print_wrapper_routine("nrn_state", BlockType::State);
}


void CodegenCudaVisitor::print_codegen_routines() {
    codegen = true;
    print_backend_info();
    print_headers_include();
    print_namespace_begin();

    print_data_structures();
    print_common_getters();

    print_compute_functions();

    codegen_wrapper_routines();

    print_namespace_end();
}
