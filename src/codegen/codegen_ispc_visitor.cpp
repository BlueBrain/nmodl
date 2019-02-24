/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <fmt/format.h>

#include "codegen/codegen_ispc_visitor.hpp"
#include "codegen/codegen_naming.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/string_utils.hpp"

using namespace fmt::literals;

namespace nmodl {
namespace codegen {


using symtab::syminfo::Status;

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
    printer->add_line("#include \"fast_math.ispc\"");
}


std::string CodegenIspcVisitor::backend_name() {
    return "ispc (api-compatibility)";
}


void CodegenIspcVisitor::print_atomic_op(const std::string& lhs,
                                         const std::string& op,
                                         const std::string& rhs) {
    std::string function;
    if (op == "+") {
        function = "atomic_add_local";
    } else if (op == "-") {
        function = "atomic_subtract_local";
    } else {
        throw std::runtime_error("ISPC backend error : {} not supported"_format(op));
    }
    printer->add_line("{}(&{}, {});"_format(function, lhs, rhs));
}


void CodegenIspcVisitor::print_nrn_cur_matrix_shadow_update() {
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
 * Use ispc foreach loop
 */
void CodegenIspcVisitor::print_channel_iteration_block_begin() {
    printer->start_block("foreach (int id = start ... end) ");
}


void CodegenIspcVisitor::print_channel_iteration_block_end() {
    printer->end_block();
    printer->add_newline();
}

void CodegenIspcVisitor::print_nrn_cur_matrix_shadow_reduction() {
    // do nothing
}


void CodegenIspcVisitor::print_rhs_d_shadow_variables() {
    // do nothing
}

bool CodegenIspcVisitor::nrn_cur_reduction_loop_required() {
    return false;
}

std::string CodegenIspcVisitor::ptr_type_qualifier() {
    return "uniform ";  // @note: extra space needed to separate qualifier from var name.
}

void CodegenIspcVisitor::print_backend_namespace_start() {
    printer->add_newline(1);
    printer->start_block("namespace ispc");
}


void CodegenIspcVisitor::print_backend_namespace_stop() {
    printer->end_block();
    printer->add_newline();
}

// @todo : use base visitor function with provision to override specific qualifiers
//         hh_ is hardcoded
void CodegenIspcVisitor::print_global_function_common_code(BlockType type) {
    std::string method = compute_method_name(type);
    auto args = "{0} {1}* {0} {2}, {0} {3}* {0} {4}, {0} {5}* {0} {6}, {7} {8}"_format(
        ptr_type_qualifier(), "hh_Instance", "inst", "hh_thread", "nt", "hh_memb_list", "ml", "int",
        "type");
    print_global_method_annotation();
    printer->start_block("void {}({})"_format(method, args));
    print_kernel_data_present_annotation_block_begin();
    printer->add_line("int nodecount = ml->nodecount;");
    printer->add_line("int pnodecount = ml->_nodecount_padded;");
    printer->add_line(
        "{}int* {}node_index = ml->nodeindices;"_format(k_const(), ptr_type_qualifier()));
    printer->add_line("double* {}data = ml->data;"_format(ptr_type_qualifier()));
    printer->add_line(
        "{}double* {}voltage = nt->_actual_v;"_format(k_const(), ptr_type_qualifier()));

    if (type == BlockType::Equation) {
        printer->add_line("double* {} vec_rhs = nt->_actual_rhs;"_format(ptr_type_qualifier()));
        printer->add_line("double* {} vec_d = nt->_actual_d;"_format(ptr_type_qualifier()));
        print_rhs_d_shadow_variables();
    }
    printer->add_line("Datum* {}indexes = ml->pdata;"_format(ptr_type_qualifier()));
    printer->add_line("ThreadDatum* {}thread = ml->_thread;"_format(ptr_type_qualifier()));

    if (type == BlockType::Initial) {
        printer->add_newline();
        printer->add_line("setup_instance(nt, ml);");
    }
    // clang-format off
    printer->add_line("{0}* {1}inst = ({0}*) ml->instance;"_format(instance_struct(), ptr_type_qualifier()));
    // clang-format on
    printer->add_newline(1);
}


void CodegenIspcVisitor::print_compute_functions() {
    // print_top_verbatim_blocks(); @todo: see where to add this

    for (const auto& function: info.functions) {
        if (!program_symtab->lookup(function->get_node_name())
                 .get()
                 ->has_all_status(Status::inlined)) {
            print_function(function);
        }
    }
    //print_net_receive_kernel();
    //print_net_receive_buffering();
    print_nrn_cur();
    print_nrn_state();
}

// @todo : use base visitor function with provision to override specific qualifiers
void CodegenIspcVisitor::print_mechanism_global_var_structure() {
    auto float_type = default_float_data_type();
    printer->add_newline(2);
    printer->add_line("/** all global variables */");
    printer->add_line("struct {} {}"_format(global_struct(), "{"));
    printer->increase_indent();

    if (!info.ions.empty()) {
        for (const auto& ion: info.ions) {
            auto name = "{}_type"_format(ion.name);
            printer->add_line("int {};"_format(name));
            codegen_global_variables.push_back(make_symbol(name));
        }
    }

    if (info.point_process) {
        printer->add_line("int point_type;");
        codegen_global_variables.push_back(make_symbol("point_type"));
    }

    if (!info.state_vars.empty()) {
        for (const auto& var: info.state_vars) {
            auto name = var->get_name() + "0";
            auto symbol = program_symtab->lookup(name);
            if (symbol == nullptr) {
                printer->add_line("{} {};"_format(float_type, name));
                codegen_global_variables.push_back(make_symbol(name));
            }
        }
    }

    if (!info.vectorize) {
        printer->add_line("{} v;"_format(float_type));
        codegen_global_variables.push_back(make_symbol("v"));
    }

    auto& top_locals = info.top_local_variables;
    if (!info.vectorize && !top_locals.empty()) {
        for (const auto& var: top_locals) {
            auto name = var->get_name();
            auto length = var->get_length();
            if (var->is_array()) {
                printer->add_line("{} {}[{}];"_format(float_type, name, length));
            } else {
                printer->add_line("{} {};"_format(float_type, name));
            }
            codegen_global_variables.push_back(var);
        }
    }

    if (!info.thread_variables.empty()) {
        printer->add_line("int thread_data_in_use;");
        printer->add_line("{} thread_data[{}];"_format(float_type, info.thread_var_data_size));
        codegen_global_variables.push_back(make_symbol("thread_data_in_use"));
        auto symbol = make_symbol("thread_data");
        symbol->set_as_array(info.thread_var_data_size);
        codegen_global_variables.push_back(symbol);
    }

    printer->add_line("int reset;");
    codegen_global_variables.push_back(make_symbol("reset"));

    printer->add_line("int mech_type;");
    codegen_global_variables.push_back(make_symbol("mech_type"));

    auto& globals = info.global_variables;
    auto& constants = info.constant_variables;

    if (!globals.empty()) {
        for (const auto& var: globals) {
            auto name = var->get_name();
            auto length = var->get_length();
            if (var->is_array()) {
                printer->add_line("{} {}[{}];"_format(float_type, name, length));
            } else {
                printer->add_line("{} {};"_format(float_type, name));
            }
            codegen_global_variables.push_back(var);
        }
    }

    if (!constants.empty()) {
        for (const auto& var: constants) {
            auto name = var->get_name();
            auto value_ptr = var->get_value();
            printer->add_line("{} {};"_format(float_type, name));
            codegen_global_variables.push_back(var);
        }
    }

    if (info.primes_size != 0) {
        printer->add_line("int* uniform slist1;");
        printer->add_line("int* uniform dlist1;");
        codegen_global_variables.push_back(make_symbol("slist1"));
        codegen_global_variables.push_back(make_symbol("dlist1"));
        if (info.derivimplicit_used) {
            printer->add_line("int* uniform slist2;");
            codegen_global_variables.push_back(make_symbol("slist2"));
        }
    }

    if (info.table_count > 0) {
        printer->add_line("double usetable;");
        codegen_global_variables.push_back(make_symbol(naming::USE_TABLE_VARIABLE));

        for (const auto& block: info.functions_with_table) {
            auto name = block->get_node_name();
            printer->add_line("{} tmin_{};"_format(float_type, name));
            printer->add_line("{} mfac_{};"_format(float_type, name));
            codegen_global_variables.push_back(make_symbol("tmin_" + name));
            codegen_global_variables.push_back(make_symbol("mfac_" + name));
        }

        for (const auto& variable: info.table_statement_variables) {
            auto name = "t_" + variable->get_name();
            printer->add_line("{}* {};"_format(float_type, name));
            codegen_global_variables.push_back(make_symbol(name));
        }
    }

    if (info.vectorize) {
        printer->add_line("ThreadDatum* {}ext_call_thread;"_format(ptr_type_qualifier()));
        codegen_global_variables.push_back(make_symbol("ext_call_thread"));
    }

    printer->decrease_indent();
    printer->add_line("};");

    printer->add_newline(1);
    printer->add_line("/** holds object of global variable */");
    printer->add_line("{} {}_global;"_format(global_struct(), info.mod_suffix));
}


// @todo : use base visitor function with provision to override specific qualifiers
void CodegenIspcVisitor::print_mechanism_range_var_structure() {
    auto float_type = default_float_data_type();
    auto int_type = default_int_data_type();
    printer->add_newline(2);
    printer->add_line("/** all mechanism instance variables */");
    printer->start_block("struct {} "_format(instance_struct()));
    for (auto& var: codegen_float_variables) {
        auto name = var->get_name();
        auto type = get_range_var_float_type(var);
        auto qualifier = is_constant_variable(name) ? k_const() : "";
        printer->add_line("{}{}* {}{};"_format(qualifier, type, ptr_type_qualifier(), name));
    }
    for (auto& var: codegen_int_variables) {
        auto name = var.symbol->get_name();
        if (var.is_index || var.is_integer) {
            auto qualifier = var.is_constant ? k_const() : "";
            printer->add_line(
                "{}{}* {}{};"_format(qualifier, int_type, ptr_type_qualifier(), name));
        } else {
            auto qualifier = var.is_constant ? k_const() : "";
            auto type = var.is_vdata ? "void*" : default_float_data_type();
            printer->add_line("{}{}* {}{};"_format(qualifier, type, ptr_type_qualifier(), name));
        }
    }
    if (channel_task_dependency_enabled()) {
        for (auto& var: codegen_shadow_variables) {
            auto name = var->get_name();
            printer->add_line("{}* {}{};"_format(float_type, ptr_type_qualifier(), name));
        }
    }
    printer->end_block();
    printer->add_text(";");
    printer->add_newline();
}

void CodegenIspcVisitor::print_ispc_helper_ds() {
    auto float_type = default_float_data_type();
    auto int_type = default_int_data_type();
    auto qualifier = "uniform ";
    printer->add_newline(2);
    printer->add_line("/** helper structures for ispc */");
    printer->start_block("struct {} "_format("hh_memb_list"));
    printer->add_line("{}{} {};"_format(qualifier, int_type, "nodecount"));
    printer->add_line("{}{} {};"_format(qualifier, int_type, "_nodecount_padded"));
    printer->add_line(
        "{}{}* {}{};"_format("const uniform ", int_type, ptr_type_qualifier(), "node_index"));
    printer->add_line("{}* {}{};"_format(float_type, ptr_type_qualifier(), "data"));
    printer->add_line(
        "{}* {}{};"_format(int_type, ptr_type_qualifier(), "indexes"));  // @todo: Datum, not int
    // ThreadDatum* uniform thread;
    printer->end_block();
    printer->add_text(";");
    printer->add_newline();

    printer->add_newline(2);
    printer->start_block("struct {} "_format("hh_thread"));
    printer->add_line("{}{} {};"_format(qualifier, float_type, "_dt"));
    printer->add_line("{}{} {};"_format(qualifier, float_type, "celsius"));
    printer->add_line("{}{}* {}{};"_format("const ", float_type, ptr_type_qualifier(), "voltage"));
    printer->add_line("{}* {}{};"_format(float_type, ptr_type_qualifier(), "vec_rhs"));
    printer->add_line("{}* {}{};"_format(float_type, ptr_type_qualifier(), "vec_d"));
    printer->end_block();
    printer->add_text(";");
    printer->add_newline();
}

void CodegenIspcVisitor::print_data_structures() {
    print_mechanism_global_var_structure();
    print_mechanism_range_var_structure();
    print_ion_var_structure();
    print_ispc_helper_ds();
}

void CodegenIspcVisitor::print_wrapper_data_structures() {
    print_mechanism_global_var_structure();
    print_mechanism_range_var_structure();
    print_ion_var_structure();
}

/****************************************************************************************/
/*                    Main code printing entry points and wrappers                      */
/****************************************************************************************/

void CodegenIspcVisitor::print_headers_include() {
    print_backend_includes();
}

void CodegenIspcVisitor::print_wrapper_headers_include() {
    print_standard_includes();
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

    print_compute_functions();

    // now print the ispc wrapper code
    print_codegen_wrapper_routines();
}

void CodegenIspcVisitor::print_codegen_wrapper_routines() {
    printer = wrapper_printer;
    wrapper_codegen = true;
    print_backend_info();
    print_wrapper_headers_include();
    print_namespace_begin();

    print_nmodl_constant();
    print_mechanism_info();
    print_wrapper_data_structures();
    print_global_variables_for_hoc();
    print_common_getters();

    print_thread_memory_callbacks();
    print_memory_allocation_routine();
    print_global_variable_setup();
    print_instance_variable_setup();
    print_nrn_alloc();
    print_check_table_thread_function();

    for (const auto& procedure: info.procedures) {
        print_procedure(procedure);
    }

    print_net_send_buffering();
    print_net_receive();
    print_net_receive_buffering();

    codegen_wrapper_routines();

    print_mechanism_register();

    print_namespace_end();
}

}  // namespace codegen
}  // namespace nmodl