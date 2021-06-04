/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/codegen_cuda_visitor.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/string_utils.hpp"

#include "ast/eigen_linear_solver_block.hpp"
#include "ast/integer.hpp"

using namespace fmt::literals;

namespace nmodl {
namespace codegen {

using symtab::syminfo::NmodlType;

/****************************************************************************************/
/*                      Routines must be overloaded in backend                          */
/****************************************************************************************/

/**
 * As initial block is/can be executed on c/cpu backend, gpu/cuda
 * backend can mark the parameter as constant even if they have
 * write count > 0 (typically due to initial block).
 */
bool CodegenCudaVisitor::is_constant_variable(const std::string& name) const {
    auto symbol = program_symtab->lookup_in_scope(name);
    bool is_constant = false;
    if (symbol != nullptr) {
        if (symbol->has_any_property(NmodlType::read_ion_var)) {
            is_constant = true;
        }
        if (symbol->has_any_property(NmodlType::param_assign)) {
            is_constant = true;
        }
    }
    return is_constant;
}


std::string CodegenCudaVisitor::compute_method_name(BlockType type) const {
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
                                         const std::string& rhs) const {
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

    if (info.crout_solver_exist) {
        printer->add_line("#include <crout/crout.hpp>");
    }
}


std::string CodegenCudaVisitor::backend_name() const {
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

void CodegenCudaVisitor::print_fast_imem_calculation() {
    if (!info.electrode_current) {
        return;
    }

    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    stringutils::remove_character(rhs_op, '=');
    stringutils::remove_character(d_op, '=');
    printer->start_block("if (nt->nrn_fast_imem)");
    print_atomic_reduction_pragma();
    print_atomic_op("nt->nrn_fast_imem->nrn_sav_rhs[node_id]", rhs_op, "rhs");
    print_atomic_reduction_pragma();
    print_atomic_op("nt->nrn_fast_imem->nrn_sav_d[node_id]", d_op, "g");
    printer->end_block(1);
}

/*
 * Depending on the backend, print condition/loop for iterating over channels
 *
 * For GPU backend its thread id less than total channel instances. Below we
 * assume we launch 1-d grid.
 */
void CodegenCudaVisitor::print_channel_iteration_block_begin(BlockType type) {
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
        print_procedure(*procedure);
    }

    for (const auto& function: info.functions) {
        print_function(*function);
    }

    print_net_send_buffering();
    print_net_receive_kernel();
    print_net_receive_buffering();
    print_nrn_cur();
    print_nrn_state();
}


void CodegenCudaVisitor::print_wrapper_routine(std::string wrapper_function, BlockType type) {
    static const auto args = "NrnThread* nt, Memb_list* ml, int type";
    wrapper_function = method_name(wrapper_function);
    auto compute_function = compute_method_name(type);

    printer->add_newline(2);
    printer->start_block("void {}({})"_format(wrapper_function, args));
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


void CodegenCudaVisitor::visit_eigen_linear_solver_block(const ast::EigenLinearSolverBlock& node) {
    printer->add_newline();

    // Check if there is a variable defined in the mod file as X, J, Jm or F and if yes
    // try to use a different string for the matrices created by sympy in the form
    // X_<random_number>, J_<random_number>, Jm_<random_number> and F_<random_number>
    std::string X = find_var_unique_name("X");
    std::string J = find_var_unique_name("J");
    std::string Jm = find_var_unique_name("Jm");
    std::string F = find_var_unique_name("F");

    const std::string float_type = default_float_data_type();
    int N = node.get_n_state_vars()->get_value();
    printer->add_line("Eigen::Matrix<{0}, {1}, 1> {2}, {3};"_format(float_type, N, X, F));
    if (N <= 4) {
        printer->add_line("Eigen::Matrix<{0}, {1}, {1}> {2};"_format(float_type, N, Jm));
    } else {
        // Eigen::RowMajor needed for Crout implementation
        printer->add_line(
            "Eigen::Matrix<{0}, {1}, {1}, Eigen::RowMajor> {2};"_format(float_type, N, Jm));
    }
    printer->add_line("{}* {} = {}.data();"_format(float_type, J, Jm));
    print_statement_block(*node.get_variable_block(), false, false);
    print_statement_block(*node.get_initialize_block(), false, false);
    print_statement_block(*node.get_setup_x_block(), false, false);

    printer->add_newline();
    // The Eigen::PartialPivLU is not compatible with GPUs (no __device__ tokens).
    // For matrices up to 4x4, the Eigen inverse() has template specializations decorated with
    // __host__ & __device__ tokens. Therefore, we use the inverse method instead of the
    // PartialPivLU (requires an invertible matrix) which supports both CPUs & GPUs.
    //
    // For matrices 5x5 and above, Eigen does not provide GPU-enabled methods to solve small linear
    // systems. For this reason, we use the Crout LU decomposition (Legacy code :
    // coreneuron/sim/scopmath/crout_thread.cpp).
    if (N <= 4) {
        printer->add_line("{0} = {1}.inverse()*{2};"_format(X, Jm, F));
    } else {
        // In-place LU-Decomposition (Crout Algo) : Jm is replaced by its LU-decomposition
        printer->add_line(
            "nmodl::crout::Crout<{0}>({1}, {2}.data(), {2}.data());"_format(float_type, N, Jm));
        // Solve the linear system : Forward/Backward substitution part
        printer->add_line(
            "nmodl::crout::solveCrout<{0}>({1}, {2}.data(), {3}.data(), {4}.data());"_format(
                float_type, N, Jm, F, X));
    }

    print_statement_block(*node.get_update_states_block(), false, false);
    print_statement_block(*node.get_finalize_block(), false, false);
}


}  // namespace codegen
}  // namespace nmodl
