#include "codegen/codegen_acc_visitor.hpp"
#include <fmt/format.h>


using namespace fmt::literals;


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
void CodegenAccVisitor::print_channel_iteration_block_parallel_hint() {
    printer->add_line("#pragma acc parallel loop");
}


void CodegenAccVisitor::print_atomic_reduction_pragma() {
    printer->add_line("#pragma acc atomic update");
}


void CodegenAccVisitor::print_backend_includes() {
    printer->add_line("#include <cuda.h>");
    printer->add_line("#include <openacc.h>");
}


std::string CodegenAccVisitor::backend_name() {
    return "C-OpenAcc (api-compatibility)";
}


void CodegenAccVisitor::print_memory_allocation_routine() {
    printer->add_newline(2);
    auto args = "size_t num, size_t size, size_t alignment = 16";
    printer->add_line("static inline void* mem_alloc({}) {}"_format(args, "{"));
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
 * Each kernel like nrn_init, nrn_state and nrn_cur could be offloaded
 * to accelerator. In this case, at very top level, we print pragma
 * for data present. For example:
 *
 *  void nrn_state(...) {
 *      #pragma acc data present (nt, ml...)
 *      {
 *
 *      }
 *  }
 */
void CodegenAccVisitor::print_kernel_data_present_annotation_block_begin() {
    auto global_variable = "{}_global"_format(info.mod_suffix);
    printer->add_line("#pragma acc data present(nt, ml, {})"_format(global_variable));
    printer->add_line("{");
    printer->increase_indent();
}


void CodegenAccVisitor::print_nrn_cur_matrix_shadow_update() {
    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    print_atomic_reduction_pragma();
    printer->add_line("vec_rhs[node_id] {} rhs;"_format(rhs_op));
    print_atomic_reduction_pragma();
    printer->add_line("vec_d[node_id] {} g;"_format(d_op));
}


void CodegenAccVisitor::print_nrn_cur_matrix_shadow_reduction() {
    // do nothing
}


/**
 * End of print_kernel_enter_data_begin
 */
void CodegenAccVisitor::print_kernel_data_present_annotation_block_end() {
    printer->decrease_indent();
    printer->add_line("}");
}


void CodegenAccVisitor::print_rhs_d_shadow_variables() {
    // do nothing
}


bool CodegenAccVisitor::nrn_cur_reduction_loop_required() {
    return false;
}
