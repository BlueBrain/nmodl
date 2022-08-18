/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/codegen_acc_visitor.hpp"

#include "ast/eigen_linear_solver_block.hpp"
#include "ast/integer.hpp"


namespace nmodl {
namespace codegen {

/****************************************************************************************/
/*                      Routines must be overloaded in backend                          */
/****************************************************************************************/


/**
 * Depending programming model and compiler, we print compiler hint
 * for parallelization. For example:
 *
 *      #pragma ivdep
 *      for(int id=0; id<nodecount; id++) {
 *
 *      #pragma acc parallel loop
 *      for(int id=0; id<nodecount; id++) {
 *
 */
void CodegenAccVisitor::print_channel_iteration_block_parallel_hint(BlockType type) {
    if (info.artificial_cell) {
        return;
    }

    std::ostringstream present_clause;
    present_clause << "present(inst";

    if (type == BlockType::NetReceive) {
        present_clause << ", nrb";
    } else {
        present_clause << ", node_index, data, voltage, indexes, thread";
        if (type == BlockType::Equation) {
            present_clause << ", vec_rhs, vec_d";
        }
    }
    present_clause << ')';
    printer->fmt_line("nrn_pragma_acc(parallel loop {} async(nt->stream_id) if(nt->compute_gpu))",
                      present_clause.str());
    printer->add_line("nrn_pragma_omp(target teams distribute parallel for if(nt->compute_gpu))");
}


void CodegenAccVisitor::print_atomic_reduction_pragma() {
    printer->add_line("nrn_pragma_acc(atomic update)");
    printer->add_line("nrn_pragma_omp(atomic update)");
}


void CodegenAccVisitor::print_backend_includes() {
    /**
     * Artificial cells are executed on CPU. As Random123 is allocated on GPU by default,
     * we have to disable GPU allocations using `DISABLE_OPENACC` macro.
     */
    if (info.artificial_cell) {
        printer->add_line("#undef DISABLE_OPENACC");
        printer->add_line("#define DISABLE_OPENACC");
    } else {
        printer->add_line("#include <coreneuron/utils/offload.hpp>");
        printer->add_line("#include <cuda_runtime_api.h>");
    }

    if (info.eigen_linear_solver_exist && std::accumulate(info.state_vars.begin(),
                                                          info.state_vars.end(),
                                                          0,
                                                          [](int l, const SymbolType& variable) {
                                                              return l += variable->get_length();
                                                          }) > 4) {
        printer->add_line("#include <partial_piv_lu/partial_piv_lu.h>");
    }
}


std::string CodegenAccVisitor::backend_name() const {
    return "C-OpenAcc (api-compatibility)";
}


void CodegenAccVisitor::print_memory_allocation_routine() const {
    // memory for artificial cells should be allocated on CPU
    if (info.artificial_cell) {
        CodegenCVisitor::print_memory_allocation_routine();
        return;
    }
    printer->add_newline(2);
    auto args = "size_t num, size_t size, size_t alignment = 16";
    printer->add_line(fmt::format("static inline void* mem_alloc({}) {}", args, "{"));
    printer->add_line("    void* ptr;");
    printer->add_line("    cudaMallocManaged(&ptr, num*size);");
    printer->add_line("    cudaMemset(ptr, 0, num*size);");
    printer->add_line("    return ptr;");
    printer->add_line("}");

    printer->add_newline(2);
    printer->add_line("static inline void mem_free(void* ptr) {");
    printer->add_line("    cudaFree(ptr);");
    printer->add_line("}");
}

/**
 * OpenACC kernels running on GPU doesn't support `abort()`. CUDA/OpenACC supports
 * `assert()` in device kernel that can be used for similar purpose. Also, `printf`
 * is supported on device.
 *
 * @todo : we need to implement proper error handling mechanism to propogate errors
 *         from GPU to CPU. For example, error code can be returned like original
 *         neuron implementation. For now we use `assert(0==1)` pattern which is
 *         used for OpenACC/CUDA.
 */
void CodegenAccVisitor::print_abort_routine() const {
    printer->add_newline(2);
    printer->add_line("static inline void coreneuron_abort() {");
    printer->add_line("    printf(\"Error : Issue while running OpenACC kernel \\n\");");
    printer->add_line("    assert(0==1);");
    printer->add_line("}");
}

void CodegenAccVisitor::print_net_send_buffering_grow() {
    // can not grow buffer during gpu execution
}

void CodegenAccVisitor::print_eigen_linear_solver(const std::string& /* float_type */, int N) {
    if (N <= 4) {
        printer->add_line("nmodl_eigen_xm = nmodl_eigen_jm.inverse()*nmodl_eigen_fm;");
    } else {
        printer->add_line(
            fmt::format("nmodl_eigen_xm = partialPivLu<{}>nmodl_eigen_jm, nmodl_eigen_fm);", N));
    }
}

/**
 * Each kernel like nrn_init, nrn_state and nrn_cur could be offloaded
 * to accelerator. In this case, at very top level, we print pragma
 * for data present. For example:
 *
 * \code{.cpp}
 *  void nrn_state(...) {
 *      #pragma acc data present (nt, ml...)
 *      {
 *
 *      }
 *  }
 * \endcode
 */
void CodegenAccVisitor::print_kernel_data_present_annotation_block_begin() {
    if (!info.artificial_cell) {
        printer->add_line("nrn_pragma_acc(data present(nt, ml) if(nt->compute_gpu))");
        printer->add_line("{");
        printer->increase_indent();
    }
}

/**
 * `INITIAL` block from `NET_RECEIVE` generates `net_init` function. The `net_init`
 * function pointer is registered with the coreneuron and called from the CPU.
 * As the data is on GPU, we need to launch `net_init` on the GPU.
 *
 * \todo: With the current code structure for NMODL and MOD2C, we use `serial`
 *        construct to launch serial kernels. This is during initialization
 *        but still inefficient. This should be improved when we drop MOD2C.
 */
void CodegenAccVisitor::print_net_init_acc_serial_annotation_block_begin() {
    if (!info.artificial_cell) {
        printer->add_line("#pragma acc serial present(inst, indexes, weights) if(nt->compute_gpu)");
        printer->add_line("{");
        printer->increase_indent();
    }
}

void CodegenAccVisitor::print_net_init_acc_serial_annotation_block_end() {
    if (!info.artificial_cell) {
        printer->end_block(1);
    }
}

void CodegenAccVisitor::print_nrn_cur_matrix_shadow_update() {
    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    if (info.point_process) {
        print_atomic_reduction_pragma();
    }
    printer->add_line(fmt::format("vec_rhs[node_id] {} rhs;", rhs_op));
    if (info.point_process) {
        print_atomic_reduction_pragma();
    }
    printer->add_line(fmt::format("vec_d[node_id] {} g;", d_op));
}

void CodegenAccVisitor::print_fast_imem_calculation() {
    if (!info.electrode_current) {
        return;
    }

    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    printer->start_block("if (nt->nrn_fast_imem)");
    if (info.point_process) {
        print_atomic_reduction_pragma();
    }
    printer->add_line(fmt::format("nt->nrn_fast_imem->nrn_sav_rhs[node_id] {} rhs;", rhs_op));
    if (info.point_process) {
        print_atomic_reduction_pragma();
    }
    printer->add_line(fmt::format("nt->nrn_fast_imem->nrn_sav_d[node_id] {} g;", d_op));
    printer->end_block(1);
}

void CodegenAccVisitor::print_nrn_cur_matrix_shadow_reduction() {
    // do nothing
}


/**
 * End of print_kernel_enter_data_begin
 */
void CodegenAccVisitor::print_kernel_data_present_annotation_block_end() {
    if (!info.artificial_cell) {
        printer->end_block(1);
    }
}


void CodegenAccVisitor::print_rhs_d_shadow_variables() {
    // do nothing
}


bool CodegenAccVisitor::nrn_cur_reduction_loop_required() {
    return false;
}

void CodegenAccVisitor::print_global_variable_device_update_annotation() {
    if (!info.artificial_cell) {
        printer->add_line(
            "// Update the instance struct copies of 'global' variables from the global struct");
        printer->fmt_line("inst->global = {}_global;", info.mod_suffix);
        // The actual host -> device synchronisation is done separately, when
        // the whole instance struct is updated on the device.
    }
}


std::string CodegenAccVisitor::get_variable_device_pointer(const std::string& variable,
                                                           const std::string& /* type */) const {
    if (info.artificial_cell) {
        return variable;
    }
    return fmt::format("nt->compute_gpu ? cnrn_target_deviceptr({}) : {}", variable, variable);
}


void CodegenAccVisitor::print_newtonspace_transfer_to_device() const {
    int list_num = info.derivimplicit_list_num;
    printer->start_block("if(nt->compute_gpu)");
    printer->add_line("double* device_vec = cnrn_target_copyin(vec, vec_size / sizeof(double));");
    printer->add_line("void* device_ns = cnrn_target_deviceptr(*ns);");
    printer->add_line("ThreadDatum* device_thread = cnrn_target_deviceptr(thread);");
    printer->add_line(
        fmt::format("cnrn_target_memcpy_to_device(&(device_thread[{}]._pvoid), &device_ns);",
                    info.thread_data_index - 1));
    printer->add_line(
        fmt::format("cnrn_target_memcpy_to_device(&(device_thread[dith{}()].pval), &device_vec);",
                    list_num));
    printer->end_block(1);
}


void CodegenAccVisitor::print_instance_variable_transfer_to_device() const {
    if (!info.artificial_cell) {
        printer->start_block("if(nt->compute_gpu)");
        printer->add_line("cnrn_target_update_on_device(inst);");
        printer->end_block(1);
    }
}


void CodegenAccVisitor::print_deriv_advance_flag_transfer_to_device() const {
    printer->add_line("nrn_pragma_acc(update device (deriv_advance_flag) if(nt->compute_gpu))");
    printer->add_line("nrn_pragma_omp(target update to(deriv_advance_flag) if(nt->compute_gpu))");
}


void CodegenAccVisitor::print_device_atomic_capture_annotation() const {
    printer->add_line("nrn_pragma_acc(atomic capture)");
    printer->add_line("nrn_pragma_omp(atomic capture)");
}


void CodegenAccVisitor::print_device_stream_wait() const {
    printer->start_block("if(nt->compute_gpu)");
    printer->add_line("nrn_pragma_acc(wait(nt->stream_id))");
    printer->end_block(1);
}


void CodegenAccVisitor::print_net_send_buf_count_update_to_host() const {
    printer->add_line("nrn_pragma_acc(update self(nsb->_cnt) if(nt->compute_gpu))");
    printer->add_line("nrn_pragma_omp(target update from(nsb->_cnt) if(nt->compute_gpu))");
}


void CodegenAccVisitor::print_net_send_buf_update_to_host() const {
    print_device_stream_wait();
    printer->start_block("if (nsb)");
    print_net_send_buf_count_update_to_host();
    printer->add_line("update_net_send_buffer_on_host(nt, nsb);");
    printer->end_block(1);
}


void CodegenAccVisitor::print_net_send_buf_count_update_to_device() const {
    printer->add_line("nrn_pragma_acc(update device(nsb->_cnt) if(nt->compute_gpu))");
    printer->add_line("nrn_pragma_omp(target update to(nsb->_cnt) if(nt->compute_gpu))");
}


void CodegenAccVisitor::print_dt_update_to_device() const {
    printer->add_line(fmt::format("#pragma acc update device({}) if (nt->compute_gpu)",
                                  get_variable_name(naming::NTHREAD_DT_VARIABLE)));
}

}  // namespace codegen
}  // namespace nmodl
