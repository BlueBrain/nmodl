/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "codegen/codegen_coreneuron_cpp_visitor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <regex>

#include "ast/all.hpp"
#include "codegen/codegen_helper_visitor.hpp"
#include "codegen/codegen_naming.hpp"
#include "codegen/codegen_utils.hpp"
#include "config/config.h"
#include "lexer/token_mapping.hpp"
#include "parser/c11_driver.hpp"
#include "utils/logger.hpp"
#include "utils/string_utils.hpp"
#include "visitors/defuse_analyze_visitor.hpp"
#include "visitors/rename_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/var_usage_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace codegen {

using namespace ast;

using visitor::DefUseAnalyzeVisitor;
using visitor::DUState;
using visitor::RenameVisitor;
using visitor::SymtabVisitor;
using visitor::VarUsageVisitor;

using symtab::syminfo::NmodlType;

extern const std::regex regex_special_chars;

/****************************************************************************************/
/*                     Common helper routines accross codegen functions                 */
/****************************************************************************************/


int CodegenCoreneuronCppVisitor::position_of_float_var(const std::string& name) const {
    int index = 0;
    for (const auto& var: codegen_float_variables) {
        if (var->get_name() == name) {
            return index;
        }
        index += var->get_length();
    }
    throw std::logic_error(name + " variable not found");
}


int CodegenCoreneuronCppVisitor::position_of_int_var(const std::string& name) const {
    int index = 0;
    for (const auto& var: codegen_int_variables) {
        if (var.symbol->get_name() == name) {
            return index;
        }
        index += var.symbol->get_length();
    }
    throw std::logic_error(name + " variable not found");
}


/**
 * \details Current variable used in breakpoint block could be local variable.
 * In this case, neuron has already renamed the variable name by prepending
 * "_l". In our implementation, the variable could have been renamed by
 * one of the pass. And hence, we search all local variables and check if
 * the variable is renamed. Note that we have to look into the symbol table
 * of statement block and not breakpoint.
 */
std::string CodegenCoreneuronCppVisitor::breakpoint_current(std::string current) const {
    auto breakpoint = info.breakpoint_node;
    if (breakpoint == nullptr) {
        return current;
    }
    auto symtab = breakpoint->get_statement_block()->get_symbol_table();
    auto variables = symtab->get_variables_with_properties(NmodlType::local_var);
    for (const auto& var: variables) {
        auto renamed_name = var->get_name();
        auto original_name = var->get_original_name();
        if (current == original_name) {
            current = renamed_name;
            break;
        }
    }
    return current;
}


int CodegenCoreneuronCppVisitor::float_variables_size() const {
    return codegen_float_variables.size();
}


int CodegenCoreneuronCppVisitor::int_variables_size() const {
    const auto count_semantics = [](int sum, const IndexSemantics& sem) { return sum += sem.size; };
    return std::accumulate(info.semantics.begin(), info.semantics.end(), 0, count_semantics);
}


/**
 * \details Depending upon the block type, we have to print read/write ion variables
 * during code generation. Depending on block/procedure being printed, this
 * method return statements as vector. As different code backends could have
 * different variable names, we rely on backend-specific read_ion_variable_name
 * and write_ion_variable_name method which will be overloaded.
 */
std::vector<std::string> CodegenCoreneuronCppVisitor::ion_read_statements(BlockType type) const {
    if (optimize_ion_variable_copies()) {
        return ion_read_statements_optimized(type);
    }
    std::vector<std::string> statements;
    for (const auto& ion: info.ions) {
        auto name = ion.name;
        for (const auto& var: ion.reads) {
            auto const iter = std::find(ion.implicit_reads.begin(), ion.implicit_reads.end(), var);
            if (iter != ion.implicit_reads.end()) {
                continue;
            }
            auto variable_names = read_ion_variable_name(var);
            auto first = get_variable_name(variable_names.first);
            auto second = get_variable_name(variable_names.second);
            statements.push_back(fmt::format("{} = {};", first, second));
        }
        for (const auto& var: ion.writes) {
            if (ion.is_ionic_conc(var)) {
                auto variables = read_ion_variable_name(var);
                auto first = get_variable_name(variables.first);
                auto second = get_variable_name(variables.second);
                statements.push_back(fmt::format("{} = {};", first, second));
            }
        }
    }
    return statements;
}


std::vector<std::string> CodegenCoreneuronCppVisitor::ion_read_statements_optimized(
    BlockType type) const {
    std::vector<std::string> statements;
    for (const auto& ion: info.ions) {
        for (const auto& var: ion.writes) {
            if (ion.is_ionic_conc(var)) {
                auto variables = read_ion_variable_name(var);
                auto first = "ionvar." + variables.first;
                const auto& second = get_variable_name(variables.second);
                statements.push_back(fmt::format("{} = {};", first, second));
            }
        }
    }
    return statements;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::vector<ShadowUseStatement> CodegenCoreneuronCppVisitor::ion_write_statements(BlockType type) {
    std::vector<ShadowUseStatement> statements;
    for (const auto& ion: info.ions) {
        std::string concentration;
        auto name = ion.name;
        for (const auto& var: ion.writes) {
            auto variable_names = write_ion_variable_name(var);
            if (ion.is_ionic_current(var)) {
                if (type == BlockType::Equation) {
                    auto current = breakpoint_current(var);
                    auto lhs = variable_names.first;
                    auto op = "+=";
                    auto rhs = get_variable_name(current);
                    if (info.point_process) {
                        auto area = get_variable_name(naming::NODE_AREA_VARIABLE);
                        rhs += fmt::format("*(1.e2/{})", area);
                    }
                    statements.push_back(ShadowUseStatement{lhs, op, rhs});
                }
            } else {
                if (!ion.is_rev_potential(var)) {
                    concentration = var;
                }
                auto lhs = variable_names.first;
                auto op = "=";
                auto rhs = get_variable_name(variable_names.second);
                statements.push_back(ShadowUseStatement{lhs, op, rhs});
            }
        }

        if (type == BlockType::Initial && !concentration.empty()) {
            int index = 0;
            if (ion.is_intra_cell_conc(concentration)) {
                index = 1;
            } else if (ion.is_extra_cell_conc(concentration)) {
                index = 2;
            } else {
                /// \todo Unhandled case in neuron implementation
                throw std::logic_error(fmt::format("codegen error for {} ion", ion.name));
            }
            auto ion_type_name = fmt::format("{}_type", ion.name);
            auto lhs = fmt::format("int {}", ion_type_name);
            auto op = "=";
            auto rhs = get_variable_name(ion_type_name);
            statements.push_back(ShadowUseStatement{lhs, op, rhs});
            auto statement = conc_write_statement(ion.name, concentration, index);
            statements.push_back(ShadowUseStatement{statement, "", ""});
        }
    }
    return statements;
}


/**
 * \details Often top level verbatim blocks use variables with old names.
 * Here we process if we are processing verbatim block at global scope.
 */
std::string CodegenCoreneuronCppVisitor::process_verbatim_token(const std::string& token) {
    const std::string& name = token;

    /*
     * If given token is procedure name and if it's defined
     * in the current mod file then it must be replaced
     */
    if (program_symtab->is_method_defined(token)) {
        return method_name(token);
    }

    /*
     * Check if token is commongly used variable name in
     * verbatim block like nt, \c \_threadargs etc. If so, replace
     * it and return.
     */
    auto new_name = replace_if_verbatim_variable(name);
    if (new_name != name) {
        return get_variable_name(new_name, false);
    }

    /*
     * For top level verbatim blocks we shouldn't replace variable
     * names with Instance because arguments are provided from coreneuron
     * and they are missing inst.
     */
    auto use_instance = !printing_top_verbatim_blocks;
    return get_variable_name(token, use_instance);
}


bool CodegenCoreneuronCppVisitor::ion_variable_struct_required() const {
    return optimize_ion_variable_copies() && info.ion_has_write_variable();
}


/**
 * \details This can be override in the backend. For example, parameters can be constant
 * except in INITIAL block where they are set to 0. As initial block is/can be
 * executed on c++/cpu backend, gpu backend can mark the parameter as constant.
 */
bool CodegenCoreneuronCppVisitor::is_constant_variable(const std::string& name) const {
    auto symbol = program_symtab->lookup_in_scope(name);
    bool is_constant = false;
    if (symbol != nullptr) {
        // per mechanism ion variables needs to be updated from neuron/coreneuron values
        if (info.is_ion_variable(name)) {
            is_constant = false;
        }
        // for parameter variable to be const, make sure it's write count is 0
        // and it's not used in the verbatim block
        else if (symbol->has_any_property(NmodlType::param_assign) &&
                 info.variables_in_verbatim.find(name) == info.variables_in_verbatim.end() &&
                 symbol->get_write_count() == 0) {
            is_constant = true;
        }
    }
    return is_constant;
}



/****************************************************************************************/
/*                      Routines must be overloaded in backend                          */
/****************************************************************************************/

std::string CodegenCoreneuronCppVisitor::get_parameter_str(const ParamVector& params) {
    std::string str;
    bool is_first = true;
    for (const auto& param: params) {
        if (is_first) {
            is_first = false;
        } else {
            str += ", ";
        }
        str += fmt::format("{}{} {}{}",
                           std::get<0>(param),
                           std::get<1>(param),
                           std::get<2>(param),
                           std::get<3>(param));
    }
    return str;
}


void CodegenCoreneuronCppVisitor::print_deriv_advance_flag_transfer_to_device() const {
    // backend specific, do nothing
}

void CodegenCoreneuronCppVisitor::print_device_atomic_capture_annotation() const {
    // backend specific, do nothing
}

void CodegenCoreneuronCppVisitor::print_net_send_buf_count_update_to_host() const {
    // backend specific, do nothing
}

void CodegenCoreneuronCppVisitor::print_net_send_buf_update_to_host() const {
    // backend specific, do nothing
}

void CodegenCoreneuronCppVisitor::print_net_send_buf_count_update_to_device() const {
    // backend specific, do nothing
}

void CodegenCoreneuronCppVisitor::print_dt_update_to_device() const {
    // backend specific, do nothing
}

void CodegenCoreneuronCppVisitor::print_device_stream_wait() const {
    // backend specific, do nothing
}

/**
 * \details Each kernel such as \c nrn\_init, \c nrn\_state and \c nrn\_cur could be offloaded
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
 *  \endcode
 */
void CodegenCoreneuronCppVisitor::print_kernel_data_present_annotation_block_begin() {
    // backend specific, do nothing
}


void CodegenCoreneuronCppVisitor::print_kernel_data_present_annotation_block_end() {
    // backend specific, do nothing
}

void CodegenCoreneuronCppVisitor::print_net_init_acc_serial_annotation_block_begin() {
    // backend specific, do nothing
}

void CodegenCoreneuronCppVisitor::print_net_init_acc_serial_annotation_block_end() {
    // backend specific, do nothing
}

/**
 * \details Depending programming model and compiler, we print compiler hint
 * for parallelization. For example:
 *
 * \code
 *      #pragma omp simd
 *      for(int id = 0; id < nodecount; id++) {
 *
 *      #pragma acc parallel loop
 *      for(int id = 0; id < nodecount; id++) {
 * \endcode
 */
void CodegenCoreneuronCppVisitor::print_channel_iteration_block_parallel_hint(
    BlockType /* type */,
    const ast::Block* block) {
    // ivdep allows SIMD parallelisation of a block/loop but doesn't provide
    // a standard mechanism for atomics. Also, even with openmp 5.0, openmp
    // atomics do not enable vectorisation under "omp simd" (gives compiler
    // error with gcc < 9 if atomic and simd pragmas are nested). So, emit
    // ivdep/simd pragma when no MUTEXLOCK/MUTEXUNLOCK/PROTECT statements
    // are used in the given block.
    std::vector<std::shared_ptr<const ast::Ast>> nodes;
    if (block) {
        nodes = collect_nodes(*block,
                              {ast::AstNodeType::PROTECT_STATEMENT,
                               ast::AstNodeType::MUTEX_LOCK,
                               ast::AstNodeType::MUTEX_UNLOCK});
    }
    if (nodes.empty()) {
        printer->add_line("#pragma omp simd");
        printer->add_line("#pragma ivdep");
    }
}


bool CodegenCoreneuronCppVisitor::nrn_cur_reduction_loop_required() {
    return info.point_process;
}


void CodegenCoreneuronCppVisitor::print_rhs_d_shadow_variables() {
    if (info.point_process) {
        printer->fmt_line("double* shadow_rhs = nt->{};", naming::NTHREAD_RHS_SHADOW);
        printer->fmt_line("double* shadow_d = nt->{};", naming::NTHREAD_D_SHADOW);
    }
}


void CodegenCoreneuronCppVisitor::print_nrn_cur_matrix_shadow_update() {
    if (info.point_process) {
        printer->add_line("shadow_rhs[id] = rhs;");
        printer->add_line("shadow_d[id] = g;");
    } else {
        auto rhs_op = operator_for_rhs();
        auto d_op = operator_for_d();
        printer->fmt_line("vec_rhs[node_id] {} rhs;", rhs_op);
        printer->fmt_line("vec_d[node_id] {} g;", d_op);
    }
}


void CodegenCoreneuronCppVisitor::print_nrn_cur_matrix_shadow_reduction() {
    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    if (info.point_process) {
        printer->add_line("int node_id = node_index[id];");
        printer->fmt_line("vec_rhs[node_id] {} shadow_rhs[id];", rhs_op);
        printer->fmt_line("vec_d[node_id] {} shadow_d[id];", d_op);
    }
}


/**
 * In the current implementation of CPU/CPP backend we need to emit atomic pragma
 * only with PROTECT construct (atomic rduction requirement for other cases on CPU
 * is handled via separate shadow vectors).
 */
void CodegenCoreneuronCppVisitor::print_atomic_reduction_pragma() {
    printer->add_line("#pragma omp atomic update");
}


void CodegenCoreneuronCppVisitor::print_device_method_annotation() {
    // backend specific, nothing for cpu
}


void CodegenCoreneuronCppVisitor::print_global_method_annotation() {
    // backend specific, nothing for cpu
}


void CodegenCoreneuronCppVisitor::print_backend_namespace_start() {
    // no separate namespace for C++ (cpu) backend
}


void CodegenCoreneuronCppVisitor::print_backend_namespace_stop() {
    // no separate namespace for C++ (cpu) backend
}


void CodegenCoreneuronCppVisitor::print_backend_includes() {
    // backend specific, nothing for cpu
}


std::string CodegenCoreneuronCppVisitor::simulator_name() {
    return "CoreNEURON";
}


std::string CodegenCoreneuronCppVisitor::backend_name() const {
    return "C++ (api-compatibility)";
}


bool CodegenCoreneuronCppVisitor::optimize_ion_variable_copies() const {
    return optimize_ionvar_copies;
}


void CodegenCoreneuronCppVisitor::print_memory_allocation_routine() const {
    printer->add_newline(2);
    auto args = "size_t num, size_t size, size_t alignment = 16";
    printer->fmt_push_block("static inline void* mem_alloc({})", args);
    printer->add_line("void* ptr;");
    printer->add_line("posix_memalign(&ptr, alignment, num*size);");
    printer->add_line("memset(ptr, 0, size);");
    printer->add_line("return ptr;");
    printer->pop_block();

    printer->add_newline(2);
    printer->push_block("static inline void mem_free(void* ptr)");
    printer->add_line("free(ptr);");
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_abort_routine() const {
    printer->add_newline(2);
    printer->push_block("static inline void coreneuron_abort()");
    printer->add_line("abort();");
    printer->pop_block();
}


std::string CodegenCoreneuronCppVisitor::compute_method_name(BlockType type) const {
    if (type == BlockType::Initial) {
        return method_name(naming::NRN_INIT_METHOD);
    }
    if (type == BlockType::Constructor) {
        return method_name(naming::NRN_CONSTRUCTOR_METHOD);
    }
    if (type == BlockType::Destructor) {
        return method_name(naming::NRN_DESTRUCTOR_METHOD);
    }
    if (type == BlockType::State) {
        return method_name(naming::NRN_STATE_METHOD);
    }
    if (type == BlockType::Equation) {
        return method_name(naming::NRN_CUR_METHOD);
    }
    if (type == BlockType::Watch) {
        return method_name(naming::NRN_WATCH_CHECK_METHOD);
    }
    throw std::logic_error("compute_method_name not implemented");
}


void CodegenCoreneuronCppVisitor::print_global_var_struct_decl() {
    printer->add_line(global_struct(), ' ', global_struct_instance(), ';');
}

/****************************************************************************************/
/*                         Printing routines for code generation                        */
/****************************************************************************************/


void CodegenCoreneuronCppVisitor::visit_watch_statement(const ast::WatchStatement& /* node */) {
    printer->add_text(fmt::format("nrn_watch_activate(inst, id, pnodecount, {}, v, watch_remove)",
                                  current_watch_statement++));
}


void CodegenCoreneuronCppVisitor::print_function_call(const FunctionCall& node) {
    const auto& name = node.get_node_name();
    auto function_name = name;
    if (defined_method(name)) {
        function_name = method_name(name);
    }

    if (is_net_send(name)) {
        print_net_send_call(node);
        return;
    }

    if (is_net_move(name)) {
        print_net_move_call(node);
        return;
    }

    if (is_net_event(name)) {
        print_net_event_call(node);
        return;
    }

    const auto& arguments = node.get_arguments();
    printer->add_text(function_name, '(');

    if (defined_method(name)) {
        printer->add_text(internal_method_arguments());
        if (!arguments.empty()) {
            printer->add_text(", ");
        }
    }

    print_vector_elements(arguments, ", ");
    printer->add_text(')');
}


void CodegenCoreneuronCppVisitor::print_top_verbatim_blocks() {
    if (info.top_verbatim_blocks.empty()) {
        return;
    }
    print_namespace_stop();

    printer->add_newline(2);
    printer->add_line("using namespace coreneuron;");
    codegen = true;
    printing_top_verbatim_blocks = true;

    for (const auto& block: info.top_blocks) {
        if (block->is_verbatim()) {
            printer->add_newline(2);
            block->accept(*this);
        }
    }

    printing_top_verbatim_blocks = false;
    codegen = false;
    print_namespace_start();
}


void CodegenCoreneuronCppVisitor::print_function_prototypes() {
    if (info.functions.empty() && info.procedures.empty()) {
        return;
    }
    codegen = true;
    printer->add_newline(2);
    for (const auto& node: info.functions) {
        print_function_declaration(*node, node->get_node_name());
        printer->add_text(';');
        printer->add_newline();
    }
    for (const auto& node: info.procedures) {
        print_function_declaration(*node, node->get_node_name());
        printer->add_text(';');
        printer->add_newline();
    }
    codegen = false;
}


static const TableStatement* get_table_statement(const ast::Block& node) {
    // TableStatementVisitor v;

    const auto& table_statements = collect_nodes(node, {AstNodeType::TABLE_STATEMENT});

    if (table_statements.size() != 1) {
        auto message = fmt::format("One table statement expected in {} found {}",
                                   node.get_node_name(),
                                   table_statements.size());
        throw std::runtime_error(message);
    }
    return dynamic_cast<const TableStatement*>(table_statements.front().get());
}


std::tuple<bool, int> CodegenCoreneuronCppVisitor::check_if_var_is_array(const std::string& name) {
    auto symbol = program_symtab->lookup_in_scope(name);
    if (!symbol) {
        throw std::runtime_error(
            fmt::format("CodegenCoreneuronCppVisitor:: {} not found in symbol table!", name));
    }
    if (symbol->is_array()) {
        return {true, symbol->get_length()};
    } else {
        return {false, 0};
    }
}


void CodegenCoreneuronCppVisitor::print_table_check_function(const Block& node) {
    auto statement = get_table_statement(node);
    auto table_variables = statement->get_table_vars();
    auto depend_variables = statement->get_depend_vars();
    const auto& from = statement->get_from();
    const auto& to = statement->get_to();
    auto name = node.get_node_name();
    auto internal_params = internal_method_parameters();
    auto with = statement->get_with()->eval();
    auto use_table_var = get_variable_name(naming::USE_TABLE_VARIABLE);
    auto tmin_name = get_variable_name("tmin_" + name);
    auto mfac_name = get_variable_name("mfac_" + name);
    auto float_type = default_float_data_type();

    printer->add_newline(2);
    print_device_method_annotation();
    printer->fmt_push_block("void check_{}({})",
                            method_name(name),
                            get_parameter_str(internal_params));
    {
        printer->fmt_push_block("if ({} == 0)", use_table_var);
        printer->add_line("return;");
        printer->pop_block();

        printer->add_line("static bool make_table = true;");
        for (const auto& variable: depend_variables) {
            printer->fmt_line("static {} save_{};", float_type, variable->get_node_name());
        }

        for (const auto& variable: depend_variables) {
            const auto& var_name = variable->get_node_name();
            const auto& instance_name = get_variable_name(var_name);
            printer->fmt_push_block("if (save_{} != {})", var_name, instance_name);
            printer->add_line("make_table = true;");
            printer->pop_block();
        }

        printer->push_block("if (make_table)");
        {
            printer->add_line("make_table = false;");

            printer->add_indent();
            printer->add_text(tmin_name, " = ");
            from->accept(*this);
            printer->add_text(';');
            printer->add_newline();

            printer->add_indent();
            printer->add_text("double tmax = ");
            to->accept(*this);
            printer->add_text(';');
            printer->add_newline();


            printer->fmt_line("double dx = (tmax-{}) / {}.;", tmin_name, with);
            printer->fmt_line("{} = 1./dx;", mfac_name);

            printer->fmt_line("double x = {};", tmin_name);
            printer->fmt_push_block("for (std::size_t i = 0; i < {}; x += dx, i++)", with + 1);
            auto function = method_name("f_" + name);
            if (node.is_procedure_block()) {
                printer->fmt_line("{}({}, x);", function, internal_method_arguments());
                for (const auto& variable: table_variables) {
                    auto var_name = variable->get_node_name();
                    auto instance_name = get_variable_name(var_name);
                    auto table_name = get_variable_name("t_" + var_name);
                    auto [is_array, array_length] = check_if_var_is_array(var_name);
                    if (is_array) {
                        for (int j = 0; j < array_length; j++) {
                            printer->fmt_line(
                                "{}[{}][i] = {}[{}];", table_name, j, instance_name, j);
                        }
                    } else {
                        printer->fmt_line("{}[i] = {};", table_name, instance_name);
                    }
                }
            } else {
                auto table_name = get_variable_name("t_" + name);
                printer->fmt_line("{}[i] = {}({}, x);",
                                  table_name,
                                  function,
                                  internal_method_arguments());
            }
            printer->pop_block();

            for (const auto& variable: depend_variables) {
                auto var_name = variable->get_node_name();
                auto instance_name = get_variable_name(var_name);
                printer->fmt_line("save_{} = {};", var_name, instance_name);
            }
        }
        printer->pop_block();
    }
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_table_replacement_function(const ast::Block& node) {
    auto name = node.get_node_name();
    auto statement = get_table_statement(node);
    auto table_variables = statement->get_table_vars();
    auto with = statement->get_with()->eval();
    auto use_table_var = get_variable_name(naming::USE_TABLE_VARIABLE);
    auto tmin_name = get_variable_name("tmin_" + name);
    auto mfac_name = get_variable_name("mfac_" + name);
    auto function_name = method_name("f_" + name);

    printer->add_newline(2);
    print_function_declaration(node, name);
    printer->push_block();
    {
        const auto& params = node.get_parameters();
        printer->fmt_push_block("if ({} == 0)", use_table_var);
        if (node.is_procedure_block()) {
            printer->fmt_line("{}({}, {});",
                              function_name,
                              internal_method_arguments(),
                              params[0].get()->get_node_name());
            printer->add_line("return 0;");
        } else {
            printer->fmt_line("return {}({}, {});",
                              function_name,
                              internal_method_arguments(),
                              params[0].get()->get_node_name());
        }
        printer->pop_block();

        printer->fmt_line("double xi = {} * ({} - {});",
                          mfac_name,
                          params[0].get()->get_node_name(),
                          tmin_name);
        printer->push_block("if (isnan(xi))");
        if (node.is_procedure_block()) {
            for (const auto& var: table_variables) {
                auto var_name = get_variable_name(var->get_node_name());
                auto [is_array, array_length] = check_if_var_is_array(var->get_node_name());
                if (is_array) {
                    for (int j = 0; j < array_length; j++) {
                        printer->fmt_line("{}[{}] = xi;", var_name, j);
                    }
                } else {
                    printer->fmt_line("{} = xi;", var_name);
                }
            }
            printer->add_line("return 0;");
        } else {
            printer->add_line("return xi;");
        }
        printer->pop_block();

        printer->fmt_push_block("if (xi <= 0. || xi >= {}.)", with);
        printer->fmt_line("int index = (xi <= 0.) ? 0 : {};", with);
        if (node.is_procedure_block()) {
            for (const auto& variable: table_variables) {
                auto var_name = variable->get_node_name();
                auto instance_name = get_variable_name(var_name);
                auto table_name = get_variable_name("t_" + var_name);
                auto [is_array, array_length] = check_if_var_is_array(var_name);
                if (is_array) {
                    for (int j = 0; j < array_length; j++) {
                        printer->fmt_line(
                            "{}[{}] = {}[{}][index];", instance_name, j, table_name, j);
                    }
                } else {
                    printer->fmt_line("{} = {}[index];", instance_name, table_name);
                }
            }
            printer->add_line("return 0;");
        } else {
            auto table_name = get_variable_name("t_" + name);
            printer->fmt_line("return {}[index];", table_name);
        }
        printer->pop_block();

        printer->add_line("int i = int(xi);");
        printer->add_line("double theta = xi - double(i);");
        if (node.is_procedure_block()) {
            for (const auto& var: table_variables) {
                auto var_name = var->get_node_name();
                auto instance_name = get_variable_name(var_name);
                auto table_name = get_variable_name("t_" + var_name);
                auto [is_array, array_length] = check_if_var_is_array(var->get_node_name());
                if (is_array) {
                    for (size_t j = 0; j < array_length; j++) {
                        printer->fmt_line(
                            "{0}[{1}] = {2}[{1}][i] + theta*({2}[{1}][i+1]-{2}[{1}][i]);",
                            instance_name,
                            j,
                            table_name);
                    }
                } else {
                    printer->fmt_line("{0} = {1}[i] + theta*({1}[i+1]-{1}[i]);",
                                      instance_name,
                                      table_name);
                }
            }
            printer->add_line("return 0;");
        } else {
            auto table_name = get_variable_name("t_" + name);
            printer->fmt_line("return {0}[i] + theta * ({0}[i+1] - {0}[i]);", table_name);
        }
    }
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_check_table_thread_function() {
    if (info.table_count == 0) {
        return;
    }

    printer->add_newline(2);
    auto name = method_name("check_table_thread");
    auto parameters = external_method_parameters(true);

    printer->fmt_push_block("static void {} ({})", name, parameters);
    printer->add_line("setup_instance(nt, ml);");
    printer->fmt_line("auto* const inst = static_cast<{0}*>(ml->instance);", instance_struct());
    printer->add_line("double v = 0;");

    for (const auto& function: info.functions_with_table) {
        auto method_name_str = method_name("check_" + function->get_node_name());
        auto arguments = internal_method_arguments();
        printer->fmt_line("{}({});", method_name_str, arguments);
    }

    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_function_or_procedure(const ast::Block& node,
                                                              const std::string& name) {
    printer->add_newline(2);
    print_function_declaration(node, name);
    printer->add_text(" ");
    printer->push_block();

    // function requires return variable declaration
    if (node.is_function_block()) {
        auto type = default_float_data_type();
        printer->fmt_line("{} ret_{} = 0.0;", type, name);
    } else {
        printer->fmt_line("int ret_{} = 0;", name);
    }

    print_statement_block(*node.get_statement_block(), false, false);
    printer->fmt_line("return ret_{};", name);
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_function_procedure_helper(const ast::Block& node) {
    codegen = true;
    auto name = node.get_node_name();

    if (info.function_uses_table(name)) {
        auto new_name = "f_" + name;
        print_function_or_procedure(node, new_name);
        print_table_check_function(node);
        print_table_replacement_function(node);
    } else {
        print_function_or_procedure(node, name);
    }

    codegen = false;
}


void CodegenCoreneuronCppVisitor::print_procedure(const ast::ProcedureBlock& node) {
    print_function_procedure_helper(node);
}


void CodegenCoreneuronCppVisitor::print_function(const ast::FunctionBlock& node) {
    auto name = node.get_node_name();

    // name of return variable
    std::string return_var;
    if (info.function_uses_table(name)) {
        return_var = "ret_f_" + name;
    } else {
        return_var = "ret_" + name;
    }

    // first rename return variable name
    auto block = node.get_statement_block().get();
    RenameVisitor v(name, return_var);
    block->accept(v);

    print_function_procedure_helper(node);
}


void CodegenCoreneuronCppVisitor::print_function_tables(const ast::FunctionTableBlock& node) {
    auto name = node.get_node_name();
    const auto& p = node.get_parameters();
    auto params = internal_method_parameters();
    for (const auto& i: p) {
        params.emplace_back("", "double", "", i->get_node_name());
    }
    printer->fmt_line("double {}({})", method_name(name), get_parameter_str(params));
    printer->push_block();
    printer->fmt_line("double _arg[{}];", p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        printer->fmt_line("_arg[{}] = {};", i, p[i]->get_node_name());
    }
    printer->fmt_line("return hoc_func_table({}, {}, _arg);",
                      get_variable_name(std::string("_ptable_" + name), true),
                      p.size());
    printer->pop_block();

    printer->fmt_push_block("double table_{}()", method_name(name));
    printer->fmt_line("hoc_spec_table(&{}, {});",
                      get_variable_name(std::string("_ptable_" + name)),
                      p.size());
    printer->add_line("return 0.;");
    printer->pop_block();
}

/**
 * @brief Checks whether the functor_block generated by sympy solver modifies any variable outside
 * its scope. If it does then return false, so that the operator() of the struct functor of the
 * Eigen Newton solver doesn't have const qualifier.
 *
 * @param variable_block Statement Block of the variables declarations used in the functor struct of
 *                       the solver
 * @param functor_block Actual code being printed in the operator() of the functor struct of the
 *                      solver
 * @return True if operator() is const else False
 */
bool CodegenCoreneuronCppVisitor::is_functor_const(const ast::StatementBlock& variable_block,
                                                   const ast::StatementBlock& functor_block) {
    // Create complete_block with both variable declarations (done in variable_block) and solver
    // part (done in functor_block) to be able to run the SymtabVisitor and DefUseAnalyzeVisitor
    // then and get the proper DUChains for the variables defined in the variable_block
    ast::StatementBlock complete_block(functor_block);
    // Typically variable_block has only one statement, a statement containing the declaration
    // of the local variables
    for (const auto& statement: variable_block.get_statements()) {
        complete_block.insert_statement(complete_block.get_statements().begin(), statement);
    }

    // Create Symbol Table for complete_block
    auto model_symbol_table = std::make_shared<symtab::ModelSymbolTable>();
    SymtabVisitor(model_symbol_table.get()).visit_statement_block(complete_block);
    // Initialize DefUseAnalyzeVisitor to generate the DUChains for the variables defined in the
    // variable_block
    DefUseAnalyzeVisitor v(*complete_block.get_symbol_table());

    // Check the DUChains for all the variables in the variable_block
    // If variable is defined in complete_block don't add const quilifier in operator()
    auto is_functor_const = true;
    const auto& variables = collect_nodes(variable_block, {ast::AstNodeType::LOCAL_VAR});
    for (const auto& variable: variables) {
        const auto& chain = v.analyze(complete_block, variable->get_node_name());
        is_functor_const = !(chain.eval() == DUState::D || chain.eval() == DUState::LD ||
                             chain.eval() == DUState::CD);
        if (!is_functor_const) {
            break;
        }
    }

    return is_functor_const;
}

void CodegenCoreneuronCppVisitor::print_functor_definition(
    const ast::EigenNewtonSolverBlock& node) {
    // functor that evaluates F(X) and J(X) for
    // Newton solver
    auto float_type = default_float_data_type();
    int N = node.get_n_state_vars()->get_value();

    const auto functor_name = info.functor_names[&node];
    printer->fmt_push_block("struct {0}", functor_name);
    printer->add_line("NrnThread* nt;");
    printer->add_line(instance_struct(), "* inst;");
    printer->add_line("int id, pnodecount;");
    printer->add_line("double v;");
    printer->add_line("const Datum* indexes;");
    printer->add_line("double* data;");
    printer->add_line("ThreadDatum* thread;");

    if (ion_variable_struct_required()) {
        print_ion_variable();
    }

    print_statement_block(*node.get_variable_block(), false, false);
    printer->add_newline();

    printer->push_block("void initialize()");
    print_statement_block(*node.get_initialize_block(), false, false);
    printer->pop_block();
    printer->add_newline();

    printer->fmt_line(
        "{0}(NrnThread* nt, {1}* inst, int id, int pnodecount, double v, const Datum* indexes, "
        "double* data, ThreadDatum* thread) : "
        "nt{{nt}}, inst{{inst}}, id{{id}}, pnodecount{{pnodecount}}, v{{v}}, indexes{{indexes}}, "
        "data{{data}}, thread{{thread}} "
        "{{}}",
        functor_name,
        instance_struct());

    printer->add_indent();

    const auto& variable_block = *node.get_variable_block();
    const auto& functor_block = *node.get_functor_block();

    printer->fmt_text(
        "void operator()(const Eigen::Matrix<{0}, {1}, 1>& nmodl_eigen_xm, Eigen::Matrix<{0}, {1}, "
        "1>& nmodl_eigen_fm, "
        "Eigen::Matrix<{0}, {1}, {1}>& nmodl_eigen_jm) {2}",
        float_type,
        N,
        is_functor_const(variable_block, functor_block) ? "const " : "");
    printer->push_block();
    printer->fmt_line("const {}* nmodl_eigen_x = nmodl_eigen_xm.data();", float_type);
    printer->fmt_line("{}* nmodl_eigen_j = nmodl_eigen_jm.data();", float_type);
    printer->fmt_line("{}* nmodl_eigen_f = nmodl_eigen_fm.data();", float_type);
    print_statement_block(functor_block, false, false);
    printer->pop_block();
    printer->add_newline();

    // assign newton solver results in matrix X to state vars
    printer->push_block("void finalize()");
    print_statement_block(*node.get_finalize_block(), false, false);
    printer->pop_block();

    printer->pop_block(";");
}

void CodegenCoreneuronCppVisitor::visit_eigen_newton_solver_block(
    const ast::EigenNewtonSolverBlock& node) {
    // solution vector to store copy of state vars for Newton solver
    printer->add_newline();

    auto float_type = default_float_data_type();
    int N = node.get_n_state_vars()->get_value();
    printer->fmt_line("Eigen::Matrix<{}, {}, 1> nmodl_eigen_xm;", float_type, N);
    printer->fmt_line("{}* nmodl_eigen_x = nmodl_eigen_xm.data();", float_type);

    print_statement_block(*node.get_setup_x_block(), false, false);

    // call newton solver with functor and X matrix that contains state vars
    printer->add_line("// call newton solver");
    printer->fmt_line("{} newton_functor(nt, inst, id, pnodecount, v, indexes, data, thread);",
                      info.functor_names[&node]);
    printer->add_line("newton_functor.initialize();");
    printer->add_line(
        "int newton_iterations = nmodl::newton::newton_solver(nmodl_eigen_xm, newton_functor);");
    printer->add_line(
        "if (newton_iterations < 0) assert(false && \"Newton solver did not converge!\");");

    // assign newton solver results in matrix X to state vars
    print_statement_block(*node.get_update_states_block(), false, false);
    printer->add_line("newton_functor.finalize();");
}

void CodegenCoreneuronCppVisitor::visit_eigen_linear_solver_block(
    const ast::EigenLinearSolverBlock& node) {
    printer->add_newline();

    const std::string float_type = default_float_data_type();
    int N = node.get_n_state_vars()->get_value();
    printer->fmt_line("Eigen::Matrix<{0}, {1}, 1> nmodl_eigen_xm, nmodl_eigen_fm;", float_type, N);
    printer->fmt_line("Eigen::Matrix<{0}, {1}, {1}> nmodl_eigen_jm;", float_type, N);
    if (N <= 4)
        printer->fmt_line("Eigen::Matrix<{0}, {1}, {1}> nmodl_eigen_jm_inv;", float_type, N);
    printer->fmt_line("{}* nmodl_eigen_x = nmodl_eigen_xm.data();", float_type);
    printer->fmt_line("{}* nmodl_eigen_j = nmodl_eigen_jm.data();", float_type);
    printer->fmt_line("{}* nmodl_eigen_f = nmodl_eigen_fm.data();", float_type);
    print_statement_block(*node.get_variable_block(), false, false);
    print_statement_block(*node.get_initialize_block(), false, false);
    print_statement_block(*node.get_setup_x_block(), false, false);

    printer->add_newline();
    print_eigen_linear_solver(float_type, N);
    printer->add_newline();

    print_statement_block(*node.get_update_states_block(), false, false);
    print_statement_block(*node.get_finalize_block(), false, false);
}

void CodegenCoreneuronCppVisitor::print_eigen_linear_solver(const std::string& float_type, int N) {
    if (N <= 4) {
        // Faster compared to LU, given the template specialization in Eigen.
        printer->add_multi_line(R"CODE(
            bool invertible;
            nmodl_eigen_jm.computeInverseWithCheck(nmodl_eigen_jm_inv,invertible);
            nmodl_eigen_xm = nmodl_eigen_jm_inv*nmodl_eigen_fm;
            if (!invertible) assert(false && "Singular or ill-conditioned matrix (Eigen::inverse)!");
        )CODE");
    } else {
        // In Eigen the default storage order is ColMajor.
        // Crout's implementation requires matrices stored in RowMajor order (C++-style arrays).
        // Therefore, the transposeInPlace is critical such that the data() method to give the rows
        // instead of the columns.
        printer->add_line("if (!nmodl_eigen_jm.IsRowMajor) nmodl_eigen_jm.transposeInPlace();");

        // pivot vector
        printer->fmt_line("Eigen::Matrix<int, {}, 1> pivot;", N);
        printer->fmt_line("Eigen::Matrix<{0}, {1}, 1> rowmax;", float_type, N);

        // In-place LU-Decomposition (Crout Algo) : Jm is replaced by its LU-decomposition
        printer->fmt_line(
            "if (nmodl::crout::Crout<{0}>({1}, nmodl_eigen_jm.data(), pivot.data(), rowmax.data()) "
            "< 0) assert(false && \"Singular or ill-conditioned matrix (nmodl::crout)!\");",
            float_type,
            N);

        // Solve the linear system : Forward/Backward substitution part
        printer->fmt_line(
            "nmodl::crout::solveCrout<{0}>({1}, nmodl_eigen_jm.data(), nmodl_eigen_fm.data(), "
            "nmodl_eigen_xm.data(), pivot.data());",
            float_type,
            N);
    }
}

/****************************************************************************************/
/*                           Code-specific helper routines                              */
/****************************************************************************************/


std::string CodegenCoreneuronCppVisitor::internal_method_arguments() {
    if (ion_variable_struct_required()) {
        return "id, pnodecount, inst, ionvar, data, indexes, thread, nt, v";
    }
    return "id, pnodecount, inst, data, indexes, thread, nt, v";
}


/**
 * @todo: figure out how to correctly handle qualifiers
 */
CodegenCoreneuronCppVisitor::ParamVector CodegenCoreneuronCppVisitor::internal_method_parameters() {
    ParamVector params;
    params.emplace_back("", "int", "", "id");
    params.emplace_back("", "int", "", "pnodecount");
    params.emplace_back("", fmt::format("{}*", instance_struct()), "", "inst");
    if (ion_variable_struct_required()) {
        params.emplace_back("", "IonCurVar&", "", "ionvar");
    }
    params.emplace_back("", "double*", "", "data");
    params.emplace_back("const ", "Datum*", "", "indexes");
    params.emplace_back("", "ThreadDatum*", "", "thread");
    params.emplace_back("", "NrnThread*", "", "nt");
    params.emplace_back("", "double", "", "v");
    return params;
}


const char* CodegenCoreneuronCppVisitor::external_method_arguments() noexcept {
    return "id, pnodecount, data, indexes, thread, nt, ml, v";
}


const char* CodegenCoreneuronCppVisitor::external_method_parameters(bool table) noexcept {
    if (table) {
        return "int id, int pnodecount, double* data, Datum* indexes, "
               "ThreadDatum* thread, NrnThread* nt, Memb_list* ml, int tml_id";
    }
    return "int id, int pnodecount, double* data, Datum* indexes, "
           "ThreadDatum* thread, NrnThread* nt, Memb_list* ml, double v";
}


std::string CodegenCoreneuronCppVisitor::nrn_thread_arguments() const {
    if (ion_variable_struct_required()) {
        return "id, pnodecount, ionvar, data, indexes, thread, nt, ml, v";
    }
    return "id, pnodecount, data, indexes, thread, nt, ml, v";
}


/**
 * Function call arguments when function or procedure is defined in the
 * same mod file itself
 */
std::string CodegenCoreneuronCppVisitor::nrn_thread_internal_arguments() {
    if (ion_variable_struct_required()) {
        return "id, pnodecount, inst, ionvar, data, indexes, thread, nt, v";
    }
    return "id, pnodecount, inst, data, indexes, thread, nt, v";
}


/**
 * Replace commonly used variables in the verbatim blocks into their corresponding
 * variable name in the new code generation backend.
 */
std::string CodegenCoreneuronCppVisitor::replace_if_verbatim_variable(std::string name) {
    if (naming::VERBATIM_VARIABLES_MAPPING.find(name) != naming::VERBATIM_VARIABLES_MAPPING.end()) {
        name = naming::VERBATIM_VARIABLES_MAPPING.at(name);
    }

    /**
     * if function is defined the same mod file then the arguments must
     * contain mechanism instance as well.
     */
    if (name == naming::THREAD_ARGS) {
        if (internal_method_call_encountered) {
            name = nrn_thread_internal_arguments();
            internal_method_call_encountered = false;
        } else {
            name = nrn_thread_arguments();
        }
    }
    if (name == naming::THREAD_ARGS_PROTO) {
        name = external_method_parameters();
    }
    return name;
}


/**
 * Processing commonly used constructs in the verbatim blocks.
 * @todo : this is still ad-hoc and requires re-implementation to
 * handle it more elegantly.
 */
std::string CodegenCoreneuronCppVisitor::process_verbatim_text(std::string const& text) {
    parser::CDriver driver;
    driver.scan_string(text);
    auto tokens = driver.all_tokens();
    std::string result;
    for (size_t i = 0; i < tokens.size(); i++) {
        auto token = tokens[i];

        // check if we have function call in the verbatim block where
        // function is defined in the same mod file
        if (program_symtab->is_method_defined(token) && tokens[i + 1] == "(") {
            internal_method_call_encountered = true;
        }
        auto name = process_verbatim_token(token);

        if (token == (std::string("_") + naming::TQITEM_VARIABLE)) {
            name.insert(0, 1, '&');
        }
        if (token == "_STRIDE") {
            name = "pnodecount+id";
        }
        result += name;
    }
    return result;
}


std::string CodegenCoreneuronCppVisitor::register_mechanism_arguments() const {
    auto nrn_cur = nrn_cur_required() ? method_name(naming::NRN_CUR_METHOD) : "nullptr";
    auto nrn_state = nrn_state_required() ? method_name(naming::NRN_STATE_METHOD) : "nullptr";
    auto nrn_alloc = method_name(naming::NRN_ALLOC_METHOD);
    auto nrn_init = method_name(naming::NRN_INIT_METHOD);
    auto const nrn_private_constructor = method_name(naming::NRN_PRIVATE_CONSTRUCTOR_METHOD);
    auto const nrn_private_destructor = method_name(naming::NRN_PRIVATE_DESTRUCTOR_METHOD);
    return fmt::format("mechanism, {}, {}, nullptr, {}, {}, {}, {}, first_pointer_var_index()",
                       nrn_alloc,
                       nrn_cur,
                       nrn_state,
                       nrn_init,
                       nrn_private_constructor,
                       nrn_private_destructor);
}


std::pair<std::string, std::string> CodegenCoreneuronCppVisitor::read_ion_variable_name(
    const std::string& name) {
    return {name, naming::ION_VARNAME_PREFIX + name};
}


std::pair<std::string, std::string> CodegenCoreneuronCppVisitor::write_ion_variable_name(
    const std::string& name) {
    return {naming::ION_VARNAME_PREFIX + name, name};
}


std::string CodegenCoreneuronCppVisitor::conc_write_statement(const std::string& ion_name,
                                                              const std::string& concentration,
                                                              int index) {
    auto conc_var_name = get_variable_name(naming::ION_VARNAME_PREFIX + concentration);
    auto style_var_name = get_variable_name("style_" + ion_name);
    return fmt::format(
        "nrn_wrote_conc({}_type,"
        " &({}),"
        " {},"
        " {},"
        " nrn_ion_global_map,"
        " {},"
        " nt->_ml_list[{}_type]->_nodecount_padded)",
        ion_name,
        conc_var_name,
        index,
        style_var_name,
        get_variable_name(naming::CELSIUS_VARIABLE),
        ion_name);
}


/**
 * If mechanisms dependency level execution is enabled then certain updates
 * like ionic current contributions needs to be atomically updated. In this
 * case we first update current mechanism's shadow vector and then add statement
 * to queue that will be used in reduction queue.
 */
std::string CodegenCoreneuronCppVisitor::process_shadow_update_statement(
    const ShadowUseStatement& statement,
    BlockType /* type */) {
    // when there is no operator or rhs then that statement doesn't need shadow update
    if (statement.op.empty() && statement.rhs.empty()) {
        auto text = statement.lhs + ";";
        return text;
    }

    // return regular statement
    auto lhs = get_variable_name(statement.lhs);
    auto text = fmt::format("{} {} {};", lhs, statement.op, statement.rhs);
    return text;
}


/****************************************************************************************/
/*               Code-specific printing routines for code generation                    */
/****************************************************************************************/


/**
 * NMODL constants from unit database
 *
 */
void CodegenCoreneuronCppVisitor::print_nmodl_constants() {
    if (!info.factor_definitions.empty()) {
        printer->add_newline(2);
        printer->add_line("/** constants used in nmodl from UNITS */");
        for (const auto& it: info.factor_definitions) {
            const std::string format_string = "static const double {} = {};";
            printer->fmt_line(format_string, it->get_node_name(), it->get_value()->get_value());
        }
    }
}


void CodegenCoreneuronCppVisitor::print_first_pointer_var_index_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline int first_pointer_var_index()");
    printer->fmt_line("return {};", info.first_pointer_var_index);
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_num_variable_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline int float_variables_size()");
    printer->fmt_line("return {};", float_variables_size());
    printer->pop_block();

    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline int int_variables_size()");
    printer->fmt_line("return {};", int_variables_size());
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_net_receive_arg_size_getter() {
    if (!net_receive_exist()) {
        return;
    }
    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline int num_net_receive_args()");
    printer->fmt_line("return {};", info.num_net_receive_parameters);
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_mech_type_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline int get_mech_type()");
    // false => get it from the host-only global struct, not the instance structure
    printer->fmt_line("return {};", get_variable_name("mech_type", false));
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_memb_list_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline Memb_list* get_memb_list(NrnThread* nt)");
    printer->push_block("if (!nt->_ml_list)");
    printer->add_line("return nullptr;");
    printer->pop_block();
    printer->add_line("return nt->_ml_list[get_mech_type()];");
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_namespace_start() {
    printer->add_newline(2);
    printer->push_block("namespace coreneuron");
}


void CodegenCoreneuronCppVisitor::print_namespace_stop() {
    printer->pop_block();
}


/**
 * \details There are three types of thread variables currently considered:
 *      - top local thread variables
 *      - thread variables in the mod file
 *      - thread variables for solver
 *
 * These variables are allocated into different thread structures and have
 * corresponding thread ids. Thread id start from 0. In mod2c implementation,
 * thread_data_index is increased at various places and it is used to
 * decide the index of thread.
 */
void CodegenCoreneuronCppVisitor::print_thread_getters() {
    if (info.vectorize && info.derivimplicit_used()) {
        int tid = info.derivimplicit_var_thread_id;
        int list = info.derivimplicit_list_num;

        // clang-format off
        printer->add_newline(2);
        printer->add_line("/** thread specific helper routines for derivimplicit */");

        printer->add_newline(1);
        printer->fmt_push_block("static inline int* deriv{}_advance(ThreadDatum* thread)", list);
        printer->fmt_line("return &(thread[{}].i);", tid);
        printer->pop_block();
        printer->add_newline();

        printer->fmt_push_block("static inline int dith{}()", list);
        printer->fmt_line("return {};", tid+1);
        printer->pop_block();
        printer->add_newline();

        printer->fmt_push_block("static inline void** newtonspace{}(ThreadDatum* thread)", list);
        printer->fmt_line("return &(thread[{}]._pvoid);", tid+2);
        printer->pop_block();
    }

    if (info.vectorize && !info.thread_variables.empty()) {
        printer->add_newline(2);
        printer->add_line("/** tid for thread variables */");
        printer->push_block("static inline int thread_var_tid()");
        printer->fmt_line("return {};", info.thread_var_thread_id);
        printer->pop_block();
    }

    if (info.vectorize && !info.top_local_variables.empty()) {
        printer->add_newline(2);
        printer->add_line("/** tid for top local tread variables */");
        printer->push_block("static inline int top_local_var_tid()");
        printer->fmt_line("return {};", info.top_local_thread_id);
        printer->pop_block();
    }
    // clang-format on
}


/****************************************************************************************/
/*                         Routines for returning variable name                         */
/****************************************************************************************/


std::string CodegenCoreneuronCppVisitor::float_variable_name(const SymbolType& symbol,
                                                             bool use_instance) const {
    auto name = symbol->get_name();
    auto dimension = symbol->get_length();
    auto position = position_of_float_var(name);
    // clang-format off
    if (symbol->is_array()) {
        if (use_instance) {
            return fmt::format("(inst->{}+id*{})", name, dimension);
        }
        return fmt::format("(data + {}*pnodecount + id*{})", position, dimension);
    }
    if (use_instance) {
        return fmt::format("inst->{}[id]", name);
    }
    return fmt::format("data[{}*pnodecount + id]", position);
    // clang-format on
}


std::string CodegenCoreneuronCppVisitor::int_variable_name(const IndexVariableInfo& symbol,
                                                           const std::string& name,
                                                           bool use_instance) const {
    auto position = position_of_int_var(name);
    // clang-format off
    if (symbol.is_index) {
        if (use_instance) {
            return fmt::format("inst->{}[{}]", name, position);
        }
        return fmt::format("indexes[{}]", position);
    }
    if (symbol.is_integer) {
        if (use_instance) {
            return fmt::format("inst->{}[{}*pnodecount+id]", name, position);
        }
        return fmt::format("indexes[{}*pnodecount+id]", position);
    }
    if (use_instance) {
        return fmt::format("inst->{}[indexes[{}*pnodecount + id]]", name, position);
    }
    auto data = symbol.is_vdata ? "_vdata" : "_data";
    return fmt::format("nt->{}[indexes[{}*pnodecount + id]]", data, position);
    // clang-format on
}


std::string CodegenCoreneuronCppVisitor::global_variable_name(const SymbolType& symbol,
                                                              bool use_instance) const {
    if (use_instance) {
        return fmt::format("inst->{}->{}", naming::INST_GLOBAL_MEMBER, symbol->get_name());
    } else {
        return fmt::format("{}.{}", global_struct_instance(), symbol->get_name());
    }
}


std::string CodegenCoreneuronCppVisitor::update_if_ion_variable_name(
    const std::string& name) const {
    std::string result(name);
    if (ion_variable_struct_required()) {
        if (info.is_ion_read_variable(name)) {
            result = naming::ION_VARNAME_PREFIX + name;
        }
        if (info.is_ion_write_variable(name)) {
            result = "ionvar." + name;
        }
        if (info.is_current(name)) {
            result = "ionvar." + name;
        }
    }
    return result;
}


std::string CodegenCoreneuronCppVisitor::get_variable_name(const std::string& name,
                                                           bool use_instance) const {
    const std::string& varname = update_if_ion_variable_name(name);

    // clang-format off
    auto symbol_comparator = [&varname](const SymbolType& sym) {
                            return varname == sym->get_name();
                         };

    auto index_comparator = [&varname](const IndexVariableInfo& var) {
                            return varname == var.symbol->get_name();
                         };
    // clang-format on

    // float variable
    auto f = std::find_if(codegen_float_variables.begin(),
                          codegen_float_variables.end(),
                          symbol_comparator);
    if (f != codegen_float_variables.end()) {
        return float_variable_name(*f, use_instance);
    }

    // integer variable
    auto i =
        std::find_if(codegen_int_variables.begin(), codegen_int_variables.end(), index_comparator);
    if (i != codegen_int_variables.end()) {
        return int_variable_name(*i, varname, use_instance);
    }

    // global variable
    auto g = std::find_if(codegen_global_variables.begin(),
                          codegen_global_variables.end(),
                          symbol_comparator);
    if (g != codegen_global_variables.end()) {
        return global_variable_name(*g, use_instance);
    }

    if (varname == naming::NTHREAD_DT_VARIABLE) {
        return std::string("nt->_") + naming::NTHREAD_DT_VARIABLE;
    }

    // t in net_receive method is an argument to function and hence it should
    // ne used instead of nt->_t which is current time of thread
    if (varname == naming::NTHREAD_T_VARIABLE && !printing_net_receive) {
        return std::string("nt->_") + naming::NTHREAD_T_VARIABLE;
    }

    auto const iter =
        std::find_if(info.neuron_global_variables.begin(),
                     info.neuron_global_variables.end(),
                     [&varname](auto const& entry) { return entry.first->get_name() == varname; });
    if (iter != info.neuron_global_variables.end()) {
        std::string ret;
        if (use_instance) {
            ret = "*(inst->";
        }
        ret.append(varname);
        if (use_instance) {
            ret.append(")");
        }
        return ret;
    }

    // otherwise return original name
    return varname;
}


/****************************************************************************************/
/*                      Main printing routines for code generation                      */
/****************************************************************************************/


void CodegenCoreneuronCppVisitor::print_backend_info() {
    time_t current_time{};
    time(&current_time);
    std::string data_time_str{std::ctime(&current_time)};
    auto version = nmodl::Version::NMODL_VERSION + " [" + nmodl::Version::GIT_REVISION + "]";

    printer->add_line("/*********************************************************");
    printer->add_line("Model Name      : ", info.mod_suffix);
    printer->add_line("Filename        : ", info.mod_file, ".mod");
    printer->add_line("NMODL Version   : ", nmodl_version());
    printer->fmt_line("Vectorized      : {}", info.vectorize);
    printer->fmt_line("Threadsafe      : {}", info.thread_safe);
    printer->add_line("Created         : ", stringutils::trim(data_time_str));
    printer->add_line("Simulator       : ", simulator_name());
    printer->add_line("Backend         : ", backend_name());
    printer->add_line("NMODL Compiler  : ", version);
    printer->add_line("*********************************************************/");
}


void CodegenCoreneuronCppVisitor::print_standard_includes() {
    printer->add_newline();
    printer->add_multi_line(R"CODE(
        #include <math.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
    )CODE");
}


void CodegenCoreneuronCppVisitor::print_coreneuron_includes() {
    printer->add_newline();
    printer->add_multi_line(R"CODE(
        #include <coreneuron/gpu/nrn_acc_manager.hpp>
        #include <coreneuron/mechanism/mech/mod2c_core_thread.hpp>
        #include <coreneuron/mechanism/register_mech.hpp>
        #include <coreneuron/nrnconf.h>
        #include <coreneuron/nrniv/nrniv_decl.h>
        #include <coreneuron/sim/multicore.hpp>
        #include <coreneuron/sim/scopmath/newton_thread.hpp>
        #include <coreneuron/utils/ivocvect.hpp>
        #include <coreneuron/utils/nrnoc_aux.hpp>
        #include <coreneuron/utils/randoms/nrnran123.h>
    )CODE");
    if (info.eigen_newton_solver_exist) {
        printer->add_line("#include <newton/newton.hpp>");
    }
    if (info.eigen_linear_solver_exist) {
        if (std::accumulate(info.state_vars.begin(),
                            info.state_vars.end(),
                            0,
                            [](int l, const SymbolType& variable) {
                                return l += variable->get_length();
                            }) > 4) {
            printer->add_line("#include <crout/crout.hpp>");
        } else {
            printer->add_line("#include <Eigen/Dense>");
            printer->add_line("#include <Eigen/LU>");
        }
    }
}


void CodegenCoreneuronCppVisitor::print_sdlists_init(bool print_initializers) {
    if (info.primes_size == 0) {
        return;
    }
    if (info.primes_size != info.prime_variables_by_order.size()) {
        throw std::runtime_error{
            fmt::format("primes_size = {} differs from prime_variables_by_order.size() = {}, "
                        "this should not happen.",
                        info.primes_size,
                        info.prime_variables_by_order.size())};
    }
    auto const initializer_list = [&](auto const& primes, const char* prefix) -> std::string {
        if (!print_initializers) {
            return {};
        }
        std::string list{"{"};
        for (auto iter = primes.begin(); iter != primes.end(); ++iter) {
            auto const& prime = *iter;
            list.append(std::to_string(position_of_float_var(prefix + prime->get_name())));
            if (std::next(iter) != primes.end()) {
                list.append(", ");
            }
        }
        list.append("}");
        return list;
    };
    printer->fmt_line("int slist1[{}]{};",
                      info.primes_size,
                      initializer_list(info.prime_variables_by_order, ""));
    printer->fmt_line("int dlist1[{}]{};",
                      info.primes_size,
                      initializer_list(info.prime_variables_by_order, "D"));
    codegen_global_variables.push_back(make_symbol("slist1"));
    codegen_global_variables.push_back(make_symbol("dlist1"));
    // additional list for derivimplicit method
    if (info.derivimplicit_used()) {
        auto primes = program_symtab->get_variables_with_properties(NmodlType::prime_name);
        printer->fmt_line("int slist2[{}]{};", info.primes_size, initializer_list(primes, ""));
        codegen_global_variables.push_back(make_symbol("slist2"));
    }
}


/**
 * \details Variables required for type of ion, type of point process etc. are
 * of static int type. For the C++ backend type, it's ok to have
 * these variables as file scoped static variables.
 *
 * Initial values of state variables (h0) are also defined as static
 * variables. Note that the state could be ion variable and it could
 * be also range variable. Hence lookup into symbol table before.
 *
 * When model is not vectorized (shouldn't be the case in coreneuron)
 * the top local variables become static variables.
 *
 * Note that static variables are already initialized to 0. We do the
 * same for some variables to keep same code as neuron.
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CodegenCoreneuronCppVisitor::print_mechanism_global_var_structure(bool print_initializers) {
    const auto value_initialize = print_initializers ? "{}" : "";

    auto float_type = default_float_data_type();
    printer->add_newline(2);
    printer->add_line("/** all global variables */");
    printer->fmt_push_block("struct {}", global_struct());

    for (const auto& ion: info.ions) {
        auto name = fmt::format("{}_type", ion.name);
        printer->fmt_line("int {}{};", name, value_initialize);
        codegen_global_variables.push_back(make_symbol(name));
    }

    if (info.point_process) {
        printer->fmt_line("int point_type{};", value_initialize);
        codegen_global_variables.push_back(make_symbol("point_type"));
    }

    for (const auto& var: info.state_vars) {
        auto name = var->get_name() + "0";
        auto symbol = program_symtab->lookup(name);
        if (symbol == nullptr) {
            printer->fmt_line("{} {}{};", float_type, name, value_initialize);
            codegen_global_variables.push_back(make_symbol(name));
        }
    }

    // Neuron and Coreneuron adds "v" to global variables when vectorize
    // is false. But as v is always local variable and passed as argument,
    // we don't need to use global variable v

    auto& top_locals = info.top_local_variables;
    if (!info.vectorize && !top_locals.empty()) {
        for (const auto& var: top_locals) {
            auto name = var->get_name();
            auto length = var->get_length();
            if (var->is_array()) {
                printer->fmt_line("{} {}[{}] /* TODO init top-local-array */;",
                                  float_type,
                                  name,
                                  length);
            } else {
                printer->fmt_line("{} {} /* TODO init top-local */;", float_type, name);
            }
            codegen_global_variables.push_back(var);
        }
    }

    if (!info.thread_variables.empty()) {
        printer->fmt_line("int thread_data_in_use{};", value_initialize);
        printer->fmt_line("{} thread_data[{}] /* TODO init thread_data */;",
                          float_type,
                          info.thread_var_data_size);
        codegen_global_variables.push_back(make_symbol("thread_data_in_use"));
        auto symbol = make_symbol("thread_data");
        symbol->set_as_array(info.thread_var_data_size);
        codegen_global_variables.push_back(symbol);
    }

    // TODO: remove this entirely?
    printer->fmt_line("int reset{};", value_initialize);
    codegen_global_variables.push_back(make_symbol("reset"));

    printer->fmt_line("int mech_type{};", value_initialize);
    codegen_global_variables.push_back(make_symbol("mech_type"));

    for (const auto& var: info.global_variables) {
        auto name = var->get_name();
        auto length = var->get_length();
        if (var->is_array()) {
            printer->fmt_line("{} {}[{}] /* TODO init const-array */;", float_type, name, length);
        } else {
            double value{};
            if (auto const& value_ptr = var->get_value()) {
                value = *value_ptr;
            }
            printer->fmt_line("{} {}{};",
                              float_type,
                              name,
                              print_initializers ? fmt::format("{{{:g}}}", value) : std::string{});
        }
        codegen_global_variables.push_back(var);
    }

    for (const auto& var: info.constant_variables) {
        auto const name = var->get_name();
        auto* const value_ptr = var->get_value().get();
        double const value{value_ptr ? *value_ptr : 0};
        printer->fmt_line("{} {}{};",
                          float_type,
                          name,
                          print_initializers ? fmt::format("{{{:g}}}", value) : std::string{});
        codegen_global_variables.push_back(var);
    }

    print_sdlists_init(print_initializers);

    if (info.table_count > 0) {
        printer->fmt_line("double usetable{};", print_initializers ? "{1}" : "");
        codegen_global_variables.push_back(make_symbol(naming::USE_TABLE_VARIABLE));

        for (const auto& block: info.functions_with_table) {
            const auto& name = block->get_node_name();
            printer->fmt_line("{} tmin_{}{};", float_type, name, value_initialize);
            printer->fmt_line("{} mfac_{}{};", float_type, name, value_initialize);
            codegen_global_variables.push_back(make_symbol("tmin_" + name));
            codegen_global_variables.push_back(make_symbol("mfac_" + name));
        }

        for (const auto& variable: info.table_statement_variables) {
            auto const name = "t_" + variable->get_name();
            auto const num_values = variable->get_num_values();
            if (variable->is_array()) {
                int array_len = variable->get_length();
                printer->fmt_line(
                    "{} {}[{}][{}]{};", float_type, name, array_len, num_values, value_initialize);
            } else {
                printer->fmt_line("{} {}[{}]{};", float_type, name, num_values, value_initialize);
            }
            codegen_global_variables.push_back(make_symbol(name));
        }
    }

    for (const auto& f: info.function_tables) {
        printer->fmt_line("void* _ptable_{}{{}};", f->get_node_name());
        codegen_global_variables.push_back(make_symbol("_ptable_" + f->get_node_name()));
    }

    if (info.vectorize && info.thread_data_index) {
        printer->fmt_line("ThreadDatum ext_call_thread[{}]{};",
                          info.thread_data_index,
                          value_initialize);
        codegen_global_variables.push_back(make_symbol("ext_call_thread"));
    }

    printer->pop_block(";");

    print_global_var_struct_assertions();
    print_global_var_struct_decl();
}


void CodegenCoreneuronCppVisitor::print_global_var_struct_assertions() const {
    // Assert some things that we assume when copying instances of this struct
    // to the GPU and so on.
    printer->fmt_line("static_assert(std::is_trivially_copy_constructible_v<{}>);",
                      global_struct());
    printer->fmt_line("static_assert(std::is_trivially_move_constructible_v<{}>);",
                      global_struct());
    printer->fmt_line("static_assert(std::is_trivially_copy_assignable_v<{}>);", global_struct());
    printer->fmt_line("static_assert(std::is_trivially_move_assignable_v<{}>);", global_struct());
    printer->fmt_line("static_assert(std::is_trivially_destructible_v<{}>);", global_struct());
}


void CodegenCoreneuronCppVisitor::print_prcellstate_macros() const {
    printer->add_line("#ifndef NRN_PRCELLSTATE");
    printer->add_line("#define NRN_PRCELLSTATE 0");
    printer->add_line("#endif");
}


void CodegenCoreneuronCppVisitor::print_mechanism_info() {
    auto variable_printer = [&](const std::vector<SymbolType>& variables) {
        for (const auto& v: variables) {
            auto name = v->get_name();
            if (!info.point_process) {
                name += "_" + info.mod_suffix;
            }
            if (v->is_array()) {
                name += fmt::format("[{}]", v->get_length());
            }
            printer->add_line(add_escape_quote(name), ",");
        }
    };

    printer->add_newline(2);
    printer->add_line("/** channel information */");
    printer->add_line("static const char *mechanism[] = {");
    printer->increase_indent();
    printer->add_line(add_escape_quote(nmodl_version()), ",");
    printer->add_line(add_escape_quote(info.mod_suffix), ",");
    variable_printer(info.range_parameter_vars);
    printer->add_line("0,");
    variable_printer(info.range_assigned_vars);
    printer->add_line("0,");
    variable_printer(info.range_state_vars);
    printer->add_line("0,");
    variable_printer(info.pointer_variables);
    printer->add_line("0");
    printer->decrease_indent();
    printer->add_line("};");
}


/**
 * Print structs that encapsulate information about scalar and
 * vector elements of type global and thread variables.
 */
void CodegenCoreneuronCppVisitor::print_global_variables_for_hoc() {
    auto variable_printer =
        [&](const std::vector<SymbolType>& variables, bool if_array, bool if_vector) {
            for (const auto& variable: variables) {
                if (variable->is_array() == if_array) {
                    // false => do not use the instance struct, which is not
                    // defined in the global declaration that we are printing
                    auto name = get_variable_name(variable->get_name(), false);
                    auto ename = add_escape_quote(variable->get_name() + "_" + info.mod_suffix);
                    auto length = variable->get_length();
                    if (if_vector) {
                        printer->fmt_line("{{{}, {}, {}}},", ename, name, length);
                    } else {
                        printer->fmt_line("{{{}, &{}}},", ename, name);
                    }
                }
            }
        };

    auto globals = info.global_variables;
    auto thread_vars = info.thread_variables;

    if (info.table_count > 0) {
        globals.push_back(make_symbol(naming::USE_TABLE_VARIABLE));
    }

    printer->add_newline(2);
    printer->add_line("/** connect global (scalar) variables to hoc -- */");
    printer->add_line("static DoubScal hoc_scalar_double[] = {");
    printer->increase_indent();
    variable_printer(globals, false, false);
    variable_printer(thread_vars, false, false);
    printer->add_line("{nullptr, nullptr}");
    printer->decrease_indent();
    printer->add_line("};");

    printer->add_newline(2);
    printer->add_line("/** connect global (array) variables to hoc -- */");
    printer->add_line("static DoubVec hoc_vector_double[] = {");
    printer->increase_indent();
    variable_printer(globals, true, true);
    variable_printer(thread_vars, true, true);
    printer->add_line("{nullptr, nullptr, 0}");
    printer->decrease_indent();
    printer->add_line("};");
}


/**
 * Return registration type for a given BEFORE/AFTER block
 * /param block A BEFORE/AFTER block being registered
 *
 * Depending on a block type i.e. BEFORE or AFTER and also type
 * of it's associated block i.e. BREAKPOINT, INITIAL, SOLVE and
 * STEP, the registration type (as an integer) is calculated.
 * These values are then interpreted by CoreNEURON internally.
 */
static std::string get_register_type_for_ba_block(const ast::Block* block) {
    std::string register_type{};
    BAType ba_type{};
    /// before block have value 10 and after block 20
    if (block->is_before_block()) {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        register_type = "BAType::Before";
        ba_type =
            dynamic_cast<const ast::BeforeBlock*>(block)->get_bablock()->get_type()->get_value();
    } else {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
        register_type = "BAType::After";
        ba_type =
            dynamic_cast<const ast::AfterBlock*>(block)->get_bablock()->get_type()->get_value();
    }

    /// associated blocks have different values (1 to 4) based on type.
    /// These values are based on neuron/coreneuron implementation details.
    if (ba_type == BATYPE_BREAKPOINT) {
        register_type += " + BAType::Breakpoint";
    } else if (ba_type == BATYPE_SOLVE) {
        register_type += " + BAType::Solve";
    } else if (ba_type == BATYPE_INITIAL) {
        register_type += " + BAType::Initial";
    } else if (ba_type == BATYPE_STEP) {
        register_type += " + BAType::Step";
    } else {
        throw std::runtime_error("Unhandled Before/After type encountered during code generation");
    }
    return register_type;
}


/**
 * \details Every mod file has register function to connect with the simulator.
 * Various information about mechanism and callbacks get registered with
 * the simulator using suffix_reg() function.
 *
 * Here are details:
 *  - We should exclude that callback based on the solver, watch statements.
 *  - If nrn_get_mechtype is < -1 means that mechanism is not used in the
 *    context of neuron execution and hence could be ignored in coreneuron
 *    execution.
 *  - Ions are internally defined and their types can be queried similar to
 *    other mechanisms.
 *  - hoc_register_var may not be needed in the context of coreneuron
 *  - We assume net receive buffer is on. This is because generated code is
 *    compatible for cpu as well as gpu target.
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CodegenCoreneuronCppVisitor::print_mechanism_register() {
    printer->add_newline(2);
    printer->add_line("/** register channel with the simulator */");
    printer->fmt_push_block("void _{}_reg()", info.mod_file);

    // type related information
    auto suffix = add_escape_quote(info.mod_suffix);
    printer->add_newline();
    printer->fmt_line("int mech_type = nrn_get_mechtype({});", suffix);
    printer->fmt_line("{} = mech_type;", get_variable_name("mech_type", false));
    printer->push_block("if (mech_type == -1)");
    printer->add_line("return;");
    printer->pop_block();

    printer->add_newline();
    printer->add_line("_nrn_layout_reg(mech_type, 0);");  // 0 for SoA

    // register mechanism
    const auto mech_arguments = register_mechanism_arguments();
    const auto number_of_thread_objects = num_thread_objects();
    if (info.point_process) {
        printer->fmt_line("point_register_mech({}, {}, {}, {});",
                          mech_arguments,
                          info.constructor_node ? method_name(naming::NRN_CONSTRUCTOR_METHOD)
                                                : "nullptr",
                          info.destructor_node ? method_name(naming::NRN_DESTRUCTOR_METHOD)
                                               : "nullptr",
                          number_of_thread_objects);
    } else {
        printer->fmt_line("register_mech({}, {});", mech_arguments, number_of_thread_objects);
        if (info.constructor_node) {
            printer->fmt_line("register_constructor({});",
                              method_name(naming::NRN_CONSTRUCTOR_METHOD));
        }
    }

    // types for ion
    for (const auto& ion: info.ions) {
        printer->fmt_line("{} = nrn_get_mechtype({});",
                          get_variable_name(ion.name + "_type", false),
                          add_escape_quote(ion.name + "_ion"));
    }
    printer->add_newline();

    /*
     *  Register callbacks for thread allocation and cleanup. Note that thread_data_index
     *  represent total number of thread used minus 1 (i.e. index of last thread).
     */
    if (info.vectorize && (info.thread_data_index != 0)) {
        // false to avoid getting the copy from the instance structure
        printer->fmt_line("thread_mem_init({});", get_variable_name("ext_call_thread", false));
    }

    if (!info.thread_variables.empty()) {
        printer->fmt_line("{} = 0;", get_variable_name("thread_data_in_use"));
    }

    if (info.thread_callback_register) {
        printer->add_line("_nrn_thread_reg0(mech_type, thread_mem_cleanup);");
        printer->add_line("_nrn_thread_reg1(mech_type, thread_mem_init);");
    }

    if (info.emit_table_thread()) {
        auto name = method_name("check_table_thread");
        printer->fmt_line("_nrn_thread_table_reg(mech_type, {});", name);
    }

    // register read/write callbacks for pointers
    if (info.bbcore_pointer_used) {
        printer->add_line("hoc_reg_bbcore_read(mech_type, bbcore_read);");
        printer->add_line("hoc_reg_bbcore_write(mech_type, bbcore_write);");
    }

    // register size of double and int elements
    // clang-format off
    printer->add_line("hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());");
    // clang-format on

    // register semantics for index variables
    for (auto& semantic: info.semantics) {
        auto args =
            fmt::format("mech_type, {}, {}", semantic.index, add_escape_quote(semantic.name));
        printer->fmt_line("hoc_register_dparam_semantics({});", args);
    }

    if (info.is_watch_used()) {
        auto watch_fun = compute_method_name(BlockType::Watch);
        printer->fmt_line("hoc_register_watch_check({}, mech_type);", watch_fun);
    }

    if (info.write_concentration) {
        printer->add_line("nrn_writes_conc(mech_type, 0);");
    }

    // register various information for point process type
    if (info.net_event_used) {
        printer->add_line("add_nrn_has_net_event(mech_type);");
    }
    if (info.artificial_cell) {
        printer->fmt_line("add_nrn_artcell(mech_type, {});", info.tqitem_index);
    }
    if (net_receive_buffering_required()) {
        printer->fmt_line("hoc_register_net_receive_buffering({}, mech_type);",
                          method_name("net_buf_receive"));
    }
    if (info.num_net_receive_parameters != 0) {
        auto net_recv_init_arg = "nullptr";
        if (info.net_receive_initial_node != nullptr) {
            net_recv_init_arg = "net_init";
        }
        printer->fmt_line("set_pnt_receive(mech_type, {}, {}, num_net_receive_args());",
                          method_name("net_receive"),
                          net_recv_init_arg);
    }
    if (info.for_netcon_used) {
        // index where information about FOR_NETCON is stored in the integer array
        const auto index =
            std::find_if(info.semantics.begin(), info.semantics.end(), [](const IndexSemantics& a) {
                return a.name == naming::FOR_NETCON_SEMANTIC;
            })->index;
        printer->fmt_line("add_nrn_fornetcons(mech_type, {});", index);
    }

    if (info.net_event_used || info.net_send_used) {
        printer->add_line("hoc_register_net_send_buffering(mech_type);");
    }

    /// register all before/after blocks
    for (size_t i = 0; i < info.before_after_blocks.size(); i++) {
        // register type and associated function name for the block
        const auto& block = info.before_after_blocks[i];
        std::string register_type = get_register_type_for_ba_block(block);
        std::string function_name = method_name(fmt::format("nrn_before_after_{}", i));
        printer->fmt_line("hoc_reg_ba(mech_type, {}, {});", function_name, register_type);
    }

    // register variables for hoc
    printer->add_line("hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);");
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_thread_memory_callbacks() {
    if (!info.thread_callback_register) {
        return;
    }

    // thread_mem_init callback
    printer->add_newline(2);
    printer->add_line("/** thread memory allocation callback */");
    printer->push_block("static void thread_mem_init(ThreadDatum* thread) ");

    if (info.vectorize && info.derivimplicit_used()) {
        printer->fmt_line("thread[dith{}()].pval = nullptr;", info.derivimplicit_list_num);
    }
    if (info.vectorize && (info.top_local_thread_size != 0)) {
        auto length = info.top_local_thread_size;
        auto allocation = fmt::format("(double*)mem_alloc({}, sizeof(double))", length);
        printer->fmt_line("thread[top_local_var_tid()].pval = {};", allocation);
    }
    if (info.thread_var_data_size != 0) {
        auto length = info.thread_var_data_size;
        auto thread_data = get_variable_name("thread_data");
        auto thread_data_in_use = get_variable_name("thread_data_in_use");
        auto allocation = fmt::format("(double*)mem_alloc({}, sizeof(double))", length);
        printer->fmt_push_block("if ({})", thread_data_in_use);
        printer->fmt_line("thread[thread_var_tid()].pval = {};", allocation);
        printer->chain_block("else");
        printer->fmt_line("thread[thread_var_tid()].pval = {};", thread_data);
        printer->fmt_line("{} = 1;", thread_data_in_use);
        printer->pop_block();
    }
    printer->pop_block();
    printer->add_newline(2);


    // thread_mem_cleanup callback
    printer->add_line("/** thread memory cleanup callback */");
    printer->push_block("static void thread_mem_cleanup(ThreadDatum* thread) ");

    // clang-format off
    if (info.vectorize && info.derivimplicit_used()) {
        int n = info.derivimplicit_list_num;
        printer->fmt_line("free(thread[dith{}()].pval);", n);
        printer->fmt_line("nrn_destroy_newtonspace(static_cast<NewtonSpace*>(*newtonspace{}(thread)));", n);
    }
    // clang-format on

    if (info.top_local_thread_size != 0) {
        auto line = "free(thread[top_local_var_tid()].pval);";
        printer->add_line(line);
    }
    if (info.thread_var_data_size != 0) {
        auto thread_data = get_variable_name("thread_data");
        auto thread_data_in_use = get_variable_name("thread_data_in_use");
        printer->fmt_push_block("if (thread[thread_var_tid()].pval == {})", thread_data);
        printer->fmt_line("{} = 0;", thread_data_in_use);
        printer->chain_block("else");
        printer->add_line("free(thread[thread_var_tid()].pval);");
        printer->pop_block();
    }
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_mechanism_range_var_structure(bool print_initializers) {
    auto const value_initialize = print_initializers ? "{}" : "";
    auto int_type = default_int_data_type();
    printer->add_newline(2);
    printer->add_line("/** all mechanism instance variables and global variables */");
    printer->fmt_push_block("struct {} ", instance_struct());

    for (auto const& [var, type]: info.neuron_global_variables) {
        auto const name = var->get_name();
        printer->fmt_line("{}* {}{};",
                          type,
                          name,
                          print_initializers ? fmt::format("{{&coreneuron::{}}}", name)
                                             : std::string{});
    }
    for (auto& var: codegen_float_variables) {
        const auto& name = var->get_name();
        auto type = get_range_var_float_type(var);
        auto qualifier = is_constant_variable(name) ? "const " : "";
        printer->fmt_line("{}{}* {}{};", qualifier, type, name, value_initialize);
    }
    for (auto& var: codegen_int_variables) {
        const auto& name = var.symbol->get_name();
        if (var.is_index || var.is_integer) {
            auto qualifier = var.is_constant ? "const " : "";
            printer->fmt_line("{}{}* {}{};", qualifier, int_type, name, value_initialize);
        } else {
            auto qualifier = var.is_constant ? "const " : "";
            auto type = var.is_vdata ? "void*" : default_float_data_type();
            printer->fmt_line("{}{}* {}{};", qualifier, type, name, value_initialize);
        }
    }

    printer->fmt_line("{}* {}{};",
                      global_struct(),
                      naming::INST_GLOBAL_MEMBER,
                      print_initializers ? fmt::format("{{&{}}}", global_struct_instance())
                                         : std::string{});
    printer->pop_block(";");
}


void CodegenCoreneuronCppVisitor::print_ion_var_structure() {
    if (!ion_variable_struct_required()) {
        return;
    }
    printer->add_newline(2);
    printer->add_line("/** ion write variables */");
    printer->push_block("struct IonCurVar");

    std::string float_type = default_float_data_type();
    std::vector<std::string> members;

    for (auto& ion: info.ions) {
        for (auto& var: ion.writes) {
            printer->fmt_line("{} {};", float_type, var);
            members.push_back(var);
        }
    }
    for (auto& var: info.currents) {
        if (!info.is_ion_variable(var)) {
            printer->fmt_line("{} {};", float_type, var);
            members.push_back(var);
        }
    }

    print_ion_var_constructor(members);

    printer->pop_block(";");
}


void CodegenCoreneuronCppVisitor::print_ion_var_constructor(
    const std::vector<std::string>& members) {
    // constructor
    printer->add_newline();
    printer->add_indent();
    printer->add_text("IonCurVar() : ");
    for (int i = 0; i < members.size(); i++) {
        printer->fmt_text("{}(0)", members[i]);
        if (i + 1 < members.size()) {
            printer->add_text(", ");
        }
    }
    printer->add_text(" {}");
    printer->add_newline();
}


void CodegenCoreneuronCppVisitor::print_ion_variable() {
    printer->add_line("IonCurVar ionvar;");
}


void CodegenCoreneuronCppVisitor::print_global_variable_device_update_annotation() {
    // nothing for cpu
}


void CodegenCoreneuronCppVisitor::print_setup_range_variable() {
    auto type = float_data_type();
    printer->add_newline(2);
    printer->add_line("/** allocate and setup array for range variable */");
    printer->fmt_push_block("static inline {}* setup_range_variable(double* variable, int n)",
                            type);
    printer->fmt_line("{0}* data = ({0}*) mem_alloc(n, sizeof({0}));", type);
    printer->push_block("for(size_t i = 0; i < n; i++)");
    printer->add_line("data[i] = variable[i];");
    printer->pop_block();
    printer->add_line("return data;");
    printer->pop_block();
}


/**
 * \details If floating point type like "float" is specified on command line then
 * we can't turn all variables to new type. This is because certain variables
 * are pointers to internal variables (e.g. ions). Hence, we check if given
 * variable can be safely converted to new type. If so, return new type.
 */
std::string CodegenCoreneuronCppVisitor::get_range_var_float_type(const SymbolType& symbol) {
    // clang-format off
    auto with   =   NmodlType::read_ion_var
                    | NmodlType::write_ion_var
                    | NmodlType::pointer_var
                    | NmodlType::bbcore_pointer_var
                    | NmodlType::extern_neuron_variable;
    // clang-format on
    bool need_default_type = symbol->has_any_property(with);
    if (need_default_type) {
        return default_float_data_type();
    }
    return float_data_type();
}


void CodegenCoreneuronCppVisitor::print_instance_variable_setup() {
    if (range_variable_setup_required()) {
        print_setup_range_variable();
    }

    printer->add_newline();
    printer->add_line("// Allocate instance structure");
    printer->fmt_push_block("static void {}(NrnThread* nt, Memb_list* ml, int type)",
                            method_name(naming::NRN_PRIVATE_CONSTRUCTOR_METHOD));
    printer->add_line("assert(!ml->instance);");
    printer->add_line("assert(!ml->global_variables);");
    printer->add_line("assert(ml->global_variables_size == 0);");
    printer->fmt_line("auto* const inst = new {}{{}};", instance_struct());
    printer->fmt_line("assert(inst->{} == &{});",
                      naming::INST_GLOBAL_MEMBER,
                      global_struct_instance());
    printer->add_line("ml->instance = inst;");
    printer->fmt_line("ml->global_variables = inst->{};", naming::INST_GLOBAL_MEMBER);
    printer->fmt_line("ml->global_variables_size = sizeof({});", global_struct());
    printer->pop_block();
    printer->add_newline();

    auto const cast_inst_and_assert_validity = [&]() {
        printer->fmt_line("auto* const inst = static_cast<{}*>(ml->instance);", instance_struct());
        printer->add_line("assert(inst);");
        printer->fmt_line("assert(inst->{});", naming::INST_GLOBAL_MEMBER);
        printer->fmt_line("assert(inst->{} == &{});",
                          naming::INST_GLOBAL_MEMBER,
                          global_struct_instance());
        printer->fmt_line("assert(inst->{} == ml->global_variables);", naming::INST_GLOBAL_MEMBER);
        printer->fmt_line("assert(ml->global_variables_size == sizeof({}));", global_struct());
    };

    // Must come before print_instance_struct_copy_to_device and
    // print_instance_struct_delete_from_device
    print_instance_struct_transfer_routine_declarations();

    printer->add_line("// Deallocate the instance structure");
    printer->fmt_push_block("static void {}(NrnThread* nt, Memb_list* ml, int type)",
                            method_name(naming::NRN_PRIVATE_DESTRUCTOR_METHOD));
    cast_inst_and_assert_validity();
    print_instance_struct_delete_from_device();
    printer->add_multi_line(R"CODE(
        delete inst;
        ml->instance = nullptr;
        ml->global_variables = nullptr;
        ml->global_variables_size = 0;
    )CODE");
    printer->pop_block();
    printer->add_newline();


    printer->add_line("/** initialize mechanism instance variables */");
    printer->push_block("static inline void setup_instance(NrnThread* nt, Memb_list* ml)");
    cast_inst_and_assert_validity();

    std::string stride;
    printer->add_line("int pnodecount = ml->_nodecount_padded;");
    stride = "*pnodecount";

    printer->add_line("Datum* indexes = ml->pdata;");

    auto const float_type = default_float_data_type();

    int id = 0;
    std::vector<std::string> ptr_members{naming::INST_GLOBAL_MEMBER};
    for (auto const& [var, type]: info.neuron_global_variables) {
        ptr_members.push_back(var->get_name());
    }
    ptr_members.reserve(ptr_members.size() + codegen_float_variables.size() +
                        codegen_int_variables.size());
    for (auto& var: codegen_float_variables) {
        auto name = var->get_name();
        auto range_var_type = get_range_var_float_type(var);
        if (float_type == range_var_type) {
            auto const variable = fmt::format("ml->data+{}{}", id, stride);
            printer->fmt_line("inst->{} = {};", name, variable);
        } else {
            // TODO what MOD file exercises this?
            printer->fmt_line("inst->{} = setup_range_variable(ml->data+{}{}, pnodecount);",
                              name,
                              id,
                              stride);
        }
        ptr_members.push_back(std::move(name));
        id += var->get_length();
    }

    for (auto& var: codegen_int_variables) {
        auto name = var.symbol->get_name();
        auto const variable = [&var]() {
            if (var.is_index || var.is_integer) {
                return "ml->pdata";
            } else if (var.is_vdata) {
                return "nt->_vdata";
            } else {
                return "nt->_data";
            }
        }();
        printer->fmt_line("inst->{} = {};", name, variable);
        ptr_members.push_back(std::move(name));
    }
    print_instance_struct_copy_to_device();
    printer->pop_block();  // setup_instance
    printer->add_newline();

    print_instance_struct_transfer_routines(ptr_members);
}


void CodegenCoreneuronCppVisitor::print_initial_block(const InitialBlock* node) {
    if (info.artificial_cell) {
        printer->add_line("double v = 0.0;");
    } else {
        printer->add_line("int node_id = node_index[id];");
        printer->add_line("double v = voltage[node_id];");
        print_v_unused();
    }

    if (ion_variable_struct_required()) {
        printer->add_line("IonCurVar ionvar;");
    }

    // read ion statements
    auto read_statements = ion_read_statements(BlockType::Initial);
    for (auto& statement: read_statements) {
        printer->add_line(statement);
    }

    // initialize state variables (excluding ion state)
    for (auto& var: info.state_vars) {
        auto name = var->get_name();
        if (!info.is_ionic_conc(name)) {
            auto lhs = get_variable_name(name);
            auto rhs = get_variable_name(name + "0");
            if (var->is_array()) {
                printer->fmt_push_block(
                    "for (std::size_t arr_index = 0; arr_index < {}; ++arr_index)",
                    var->get_length());
                printer->add_line(lhs, "[arr_index] = ", rhs, ";");
                printer->pop_block();
            } else {
                printer->fmt_line("{} = {};", lhs, rhs);
            }
        }
    }

    // initial block
    if (node != nullptr) {
        const auto& block = node->get_statement_block();
        print_statement_block(*block, false, false);
    }

    // write ion statements
    auto write_statements = ion_write_statements(BlockType::Initial);
    for (auto& statement: write_statements) {
        auto text = process_shadow_update_statement(statement, BlockType::Initial);
        printer->add_line(text);
    }
}


void CodegenCoreneuronCppVisitor::print_global_function_common_code(
    BlockType type,
    const std::string& function_name) {
    std::string method;
    if (function_name.empty()) {
        method = compute_method_name(type);
    } else {
        method = function_name;
    }
    auto args = "NrnThread* nt, Memb_list* ml, int type";

    // watch statement function doesn't have type argument
    if (type == BlockType::Watch) {
        args = "NrnThread* nt, Memb_list* ml";
    }

    print_global_method_annotation();
    printer->fmt_push_block("void {}({})", method, args);
    if (type != BlockType::Destructor && type != BlockType::Constructor) {
        // We do not (currently) support DESTRUCTOR and CONSTRUCTOR blocks
        // running anything on the GPU.
        print_kernel_data_present_annotation_block_begin();
    } else {
        /// TODO: Remove this when the code generation is propery done
        /// Related to https://github.com/BlueBrain/nmodl/issues/692
        printer->add_line("#ifndef CORENEURON_BUILD");
    }
    printer->add_multi_line(R"CODE(
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* node_index = ml->nodeindices;
        double* data = ml->data;
        const double* voltage = nt->_actual_v;
    )CODE");

    if (type == BlockType::Equation) {
        printer->add_line("double* vec_rhs = nt->_actual_rhs;");
        printer->add_line("double* vec_d = nt->_actual_d;");
        print_rhs_d_shadow_variables();
    }
    printer->add_line("Datum* indexes = ml->pdata;");
    printer->add_line("ThreadDatum* thread = ml->_thread;");

    if (type == BlockType::Initial) {
        printer->add_newline();
        printer->add_line("setup_instance(nt, ml);");
    }
    printer->fmt_line("auto* const inst = static_cast<{}*>(ml->instance);", instance_struct());
    printer->add_newline(1);
}

void CodegenCoreneuronCppVisitor::print_nrn_init(bool skip_init_check) {
    codegen = true;
    printer->add_newline(2);
    printer->add_line("/** initialize channel */");

    print_global_function_common_code(BlockType::Initial);
    if (info.derivimplicit_used()) {
        printer->add_newline();
        int nequation = info.num_equations;
        int list_num = info.derivimplicit_list_num;
        // clang-format off
        printer->fmt_line("int& deriv_advance_flag = *deriv{}_advance(thread);", list_num);
        printer->add_line("deriv_advance_flag = 0;");
        print_deriv_advance_flag_transfer_to_device();
        printer->fmt_line("auto ns = newtonspace{}(thread);", list_num);
        printer->fmt_line("auto& th = thread[dith{}()];", list_num);
        printer->push_block("if (*ns == nullptr)");
        printer->fmt_line("int vec_size = 2*{}*pnodecount*sizeof(double);", nequation);
        printer->fmt_line("double* vec = makevector(vec_size);", nequation);
        printer->fmt_line("th.pval = vec;", list_num);
        printer->fmt_line("*ns = nrn_cons_newtonspace({}, pnodecount);", nequation);
        print_newtonspace_transfer_to_device();
        printer->pop_block();
        // clang-format on
    }

    // update global variable as those might be updated via python/hoc API
    // NOTE: CoreNEURON has enough information to do this on its own, which
    // would be neater.
    print_global_variable_device_update_annotation();

    if (skip_init_check) {
        printer->push_block("if (_nrn_skip_initmodel == 0)");
    }

    if (!info.changed_dt.empty()) {
        printer->fmt_line("double _save_prev_dt = {};",
                          get_variable_name(naming::NTHREAD_DT_VARIABLE));
        printer->fmt_line("{} = {};",
                          get_variable_name(naming::NTHREAD_DT_VARIABLE),
                          info.changed_dt);
        print_dt_update_to_device();
    }

    print_channel_iteration_block_parallel_hint(BlockType::Initial, info.initial_node);
    printer->push_block("for (int id = 0; id < nodecount; id++)");

    if (info.net_receive_node != nullptr) {
        printer->fmt_line("{} = -1e20;", get_variable_name("tsave"));
    }

    print_initial_block(info.initial_node);
    printer->pop_block();

    if (!info.changed_dt.empty()) {
        printer->fmt_line("{} = _save_prev_dt;", get_variable_name(naming::NTHREAD_DT_VARIABLE));
        print_dt_update_to_device();
    }

    printer->pop_block();

    if (info.derivimplicit_used()) {
        printer->add_line("deriv_advance_flag = 1;");
        print_deriv_advance_flag_transfer_to_device();
    }

    if (info.net_send_used && !info.artificial_cell) {
        print_send_event_move();
    }

    print_kernel_data_present_annotation_block_end();
    if (skip_init_check) {
        printer->pop_block();
    }
    codegen = false;
}

void CodegenCoreneuronCppVisitor::print_before_after_block(const ast::Block* node,
                                                           size_t block_id) {
    codegen = true;

    std::string ba_type;
    std::shared_ptr<ast::BABlock> ba_block;

    if (node->is_before_block()) {
        ba_block = dynamic_cast<const ast::BeforeBlock*>(node)->get_bablock();
        ba_type = "BEFORE";
    } else {
        ba_block = dynamic_cast<const ast::AfterBlock*>(node)->get_bablock();
        ba_type = "AFTER";
    }

    std::string ba_block_type = ba_block->get_type()->eval();

    /// name of the before/after function
    std::string function_name = method_name(fmt::format("nrn_before_after_{}", block_id));

    /// print common function code like init/state/current
    printer->add_newline(2);
    printer->fmt_line("/** {} of block type {} # {} */", ba_type, ba_block_type, block_id);
    print_global_function_common_code(BlockType::BeforeAfter, function_name);

    print_channel_iteration_block_parallel_hint(BlockType::BeforeAfter, node);
    printer->push_block("for (int id = 0; id < nodecount; id++)");

    printer->add_line("int node_id = node_index[id];");
    printer->add_line("double v = voltage[node_id];");
    print_v_unused();

    // read ion statements
    const auto& read_statements = ion_read_statements(BlockType::Equation);
    for (auto& statement: read_statements) {
        printer->add_line(statement);
    }

    /// print main body
    printer->add_indent();
    print_statement_block(*ba_block->get_statement_block());
    printer->add_newline();

    // write ion statements
    const auto& write_statements = ion_write_statements(BlockType::Equation);
    for (auto& statement: write_statements) {
        auto text = process_shadow_update_statement(statement, BlockType::Equation);
        printer->add_line(text);
    }

    /// loop end including data annotation block
    printer->pop_block();
    printer->pop_block();
    print_kernel_data_present_annotation_block_end();

    codegen = false;
}

void CodegenCoreneuronCppVisitor::print_nrn_constructor() {
    printer->add_newline(2);
    print_global_function_common_code(BlockType::Constructor);
    if (info.constructor_node != nullptr) {
        const auto& block = info.constructor_node->get_statement_block();
        print_statement_block(*block, false, false);
    }
    printer->add_line("#endif");
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_nrn_destructor() {
    printer->add_newline(2);
    print_global_function_common_code(BlockType::Destructor);
    if (info.destructor_node != nullptr) {
        const auto& block = info.destructor_node->get_statement_block();
        print_statement_block(*block, false, false);
    }
    printer->add_line("#endif");
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_functors_definitions() {
    codegen = true;
    for (const auto& functor_name: info.functor_names) {
        printer->add_newline(2);
        print_functor_definition(*functor_name.first);
    }
    codegen = false;
}


void CodegenCoreneuronCppVisitor::print_nrn_alloc() {
    printer->add_newline(2);
    auto method = method_name(naming::NRN_ALLOC_METHOD);
    printer->fmt_push_block("static void {}(double* data, Datum* indexes, int type)", method);
    printer->add_line("// do nothing");
    printer->pop_block();
}

/**
 * \todo Number of watch could be more than number of statements
 * according to grammar. Check if this is correctly handled in neuron
 * and coreneuron.
 */
void CodegenCoreneuronCppVisitor::print_watch_activate() {
    if (info.watch_statements.empty()) {
        return;
    }
    codegen = true;
    printer->add_newline(2);
    auto inst = fmt::format("{}* inst", instance_struct());

    printer->fmt_push_block(
        "static void nrn_watch_activate({}, int id, int pnodecount, int watch_id, "
        "double v, bool &watch_remove)",
        inst);

    // initialize all variables only during first watch statement
    printer->push_block("if (watch_remove == false)");
    for (int i = 0; i < info.watch_count; i++) {
        auto name = get_variable_name(fmt::format("watch{}", i + 1));
        printer->fmt_line("{} = 0;", name);
    }
    printer->add_line("watch_remove = true;");
    printer->pop_block();

    /**
     * \todo Similar to neuron/coreneuron we are using
     * first watch and ignoring rest.
     */
    for (int i = 0; i < info.watch_statements.size(); i++) {
        auto statement = info.watch_statements[i];
        printer->fmt_push_block("if (watch_id == {})", i);

        auto varname = get_variable_name(fmt::format("watch{}", i + 1));
        printer->add_indent();
        printer->fmt_text("{} = 2 + (", varname);
        auto watch = statement->get_statements().front();
        watch->get_expression()->visit_children(*this);
        printer->add_text(");");
        printer->add_newline();

        printer->pop_block();
    }
    printer->pop_block();
    codegen = false;
}


/**
 * \todo Similar to print_watch_activate, we are using only
 * first watch. need to verify with neuron/coreneuron about rest.
 */
void CodegenCoreneuronCppVisitor::print_watch_check() {
    if (info.watch_statements.empty()) {
        return;
    }
    codegen = true;
    printer->add_newline(2);
    printer->add_line("/** routine to check watch activation */");
    print_global_function_common_code(BlockType::Watch);

    // WATCH statements appears in NET_RECEIVE block and while printing
    // net_receive function we already check if it contains any MUTEX/PROTECT
    // constructs. As WATCH is not a top level block but list of statements,
    // we don't need to have ivdep pragma related check
    print_channel_iteration_block_parallel_hint(BlockType::Watch, nullptr);

    printer->push_block("for (int id = 0; id < nodecount; id++)");

    if (info.is_voltage_used_by_watch_statements()) {
        printer->add_line("int node_id = node_index[id];");
        printer->add_line("double v = voltage[node_id];");
        print_v_unused();
    }

    // flat to make sure only one WATCH statement can be triggered at a time
    printer->add_line("bool watch_untriggered = true;");

    for (int i = 0; i < info.watch_statements.size(); i++) {
        auto statement = info.watch_statements[i];
        const auto& watch = statement->get_statements().front();
        const auto& varname = get_variable_name(fmt::format("watch{}", i + 1));

        // start block 1
        printer->fmt_push_block("if ({}&2 && watch_untriggered)", varname);

        // start block 2
        printer->add_indent();
        printer->add_text("if (");
        watch->get_expression()->accept(*this);
        printer->add_text(") {");
        printer->add_newline();
        printer->increase_indent();

        // start block 3
        printer->fmt_push_block("if (({}&1) == 0)", varname);

        printer->add_line("watch_untriggered = false;");

        const auto& tqitem = get_variable_name("tqitem");
        const auto& point_process = get_variable_name("point_process");
        printer->add_indent();
        printer->add_text("net_send_buffering(");
        const auto& t = get_variable_name("t");
        printer->fmt_text("nt, ml->_net_send_buffer, 0, {}, -1, {}, {}+0.0, ",
                          tqitem,
                          point_process,
                          t);
        watch->get_value()->accept(*this);
        printer->add_text(");");
        printer->add_newline();
        printer->pop_block();

        printer->add_line(varname, " = 3;");
        // end block 3

        // start block 3
        printer->decrease_indent();
        printer->push_block("} else");
        printer->add_line(varname, " = 2;");
        printer->pop_block();
        // end block 3

        printer->pop_block();
        // end block 1
    }

    printer->pop_block();
    print_send_event_move();
    print_kernel_data_present_annotation_block_end();
    printer->pop_block();
    codegen = false;
}


void CodegenCoreneuronCppVisitor::print_net_receive_common_code(const Block& node,
                                                                bool need_mech_inst) {
    printer->add_multi_line(R"CODE(
        int tid = pnt->_tid;
        int id = pnt->_i_instance;
        double v = 0;
    )CODE");

    if (info.artificial_cell || node.is_initial_block()) {
        printer->add_line("NrnThread* nt = nrn_threads + tid;");
        printer->add_line("Memb_list* ml = nt->_ml_list[pnt->_type];");
    }
    if (node.is_initial_block()) {
        print_kernel_data_present_annotation_block_begin();
    }

    printer->add_multi_line(R"CODE(
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        double* data = ml->data;
        double* weights = nt->weights;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
    )CODE");
    if (need_mech_inst) {
        printer->fmt_line("auto* const inst = static_cast<{0}*>(ml->instance);", instance_struct());
    }

    if (node.is_initial_block()) {
        print_net_init_acc_serial_annotation_block_begin();
    }

    // rename variables but need to see if they are actually used
    auto parameters = info.net_receive_node->get_parameters();
    if (!parameters.empty()) {
        int i = 0;
        printer->add_newline();
        for (auto& parameter: parameters) {
            auto name = parameter->get_node_name();
            bool var_used = VarUsageVisitor().variable_used(node, "(*" + name + ")");
            if (var_used) {
                printer->fmt_line("double* {} = weights + weight_index + {};", name, i);
                RenameVisitor vr(name, "*" + name);
                node.visit_children(vr);
            }
            i++;
        }
    }
}


void CodegenCoreneuronCppVisitor::print_net_send_call(const FunctionCall& node) {
    auto const& arguments = node.get_arguments();
    const auto& tqitem = get_variable_name("tqitem");
    std::string weight_index = "weight_index";
    std::string pnt = "pnt";

    // for functions not generated from NET_RECEIVE blocks (i.e. top level INITIAL block)
    // the weight_index argument is 0.
    if (!printing_net_receive && !printing_net_init) {
        weight_index = "0";
        auto var = get_variable_name("point_process");
        if (info.artificial_cell) {
            pnt = "(Point_process*)" + var;
        }
    }

    // artificial cells don't use spike buffering
    // clang-format off
    if (info.artificial_cell) {
        printer->fmt_text("artcell_net_send(&{}, {}, {}, nt->_t+", tqitem, weight_index, pnt);
    } else {
        const auto& point_process = get_variable_name("point_process");
        const auto& t = get_variable_name("t");
        printer->add_text("net_send_buffering(");
        printer->fmt_text("nt, ml->_net_send_buffer, 0, {}, {}, {}, {}+", tqitem, weight_index, point_process, t);
    }
    // clang-format off
    print_vector_elements(arguments, ", ");
    printer->add_text(')');
}


void CodegenCoreneuronCppVisitor::print_net_move_call(const FunctionCall& node) {
    if (!printing_net_receive && !printing_net_init) {
        throw std::runtime_error("Error : net_move only allowed in NET_RECEIVE block");
    }

    auto const& arguments = node.get_arguments();
    const auto& tqitem = get_variable_name("tqitem");
    std::string weight_index = "-1";
    std::string pnt = "pnt";

    // artificial cells don't use spike buffering
    // clang-format off
    if (info.artificial_cell) {
        printer->fmt_text("artcell_net_move(&{}, {}, ", tqitem, pnt);
        print_vector_elements(arguments, ", ");
        printer->add_text(")");
    } else {
        const auto& point_process = get_variable_name("point_process");
        printer->add_text("net_send_buffering(");
        printer->fmt_text("nt, ml->_net_send_buffer, 2, {}, {}, {}, ", tqitem, weight_index, point_process);
        print_vector_elements(arguments, ", ");
        printer->add_text(", 0.0");
        printer->add_text(")");
    }
}


void CodegenCoreneuronCppVisitor::print_net_event_call(const FunctionCall& node) {
    const auto& arguments = node.get_arguments();
    if (info.artificial_cell) {
        printer->add_text("net_event(pnt, ");
        print_vector_elements(arguments, ", ");
    } else {
        const auto& point_process = get_variable_name("point_process");
        printer->add_text("net_send_buffering(");
        printer->fmt_text("nt, ml->_net_send_buffer, 1, -1, -1, {}, ", point_process);
        print_vector_elements(arguments, ", ");
        printer->add_text(", 0.0");
    }
    printer->add_text(")");
}

/**
 * Rename arguments to NET_RECEIVE block with corresponding pointer variable
 *
 * Arguments to NET_RECEIVE block are packed and passed via weight vector. These
 * variables need to be replaced with corresponding pointer variable. For example,
 * if mod file is like
 *
 * \code{.mod}
 *      NET_RECEIVE (weight, R){
 *          INITIAL {
 *              R=1
 *          }
 *      }
 * \endcode
 *
 * then generated code for initial block should be:
 *
 * \code{.cpp}
 *      double* R = weights + weight_index + 0;
 *      (*R) = 1.0;
 * \endcode
 *
 * So, the `R` in AST needs to be renamed with `(*R)`.
 */
static void rename_net_receive_arguments(const ast::NetReceiveBlock& net_receive_node, const ast::Node& node) {
    const auto& parameters = net_receive_node.get_parameters();
    for (auto& parameter: parameters) {
        const auto& name = parameter->get_node_name();
        auto var_used = VarUsageVisitor().variable_used(node, name);
        if (var_used) {
            RenameVisitor vr(name, "(*" + name + ")");
            node.get_statement_block()->visit_children(vr);
        }
    }
}


void CodegenCoreneuronCppVisitor::print_net_init() {
    const auto node = info.net_receive_initial_node;
    if (node == nullptr) {
        return;
    }

    // rename net_receive arguments used in the initial block of net_receive
    rename_net_receive_arguments(*info.net_receive_node, *node);

    codegen = true;
    printing_net_init = true;
    auto args = "Point_process* pnt, int weight_index, double flag";
    printer->add_newline(2);
    printer->add_line("/** initialize block for net receive */");
    printer->fmt_push_block("static void net_init({})", args);
    auto block = node->get_statement_block().get();
    if (block->get_statements().empty()) {
        printer->add_line("// do nothing");
    } else {
        print_net_receive_common_code(*node);
        print_statement_block(*block, false, false);
        if (node->is_initial_block()) {
            print_net_init_acc_serial_annotation_block_end();
            print_kernel_data_present_annotation_block_end();
            printer->add_line("auto& nsb = ml->_net_send_buffer;");
            print_net_send_buf_update_to_host();
        }
    }
    printer->pop_block();
    codegen = false;
    printing_net_init = false;
}


void CodegenCoreneuronCppVisitor::print_send_event_move() {
    printer->add_newline();
    printer->add_line("NetSendBuffer_t* nsb = ml->_net_send_buffer;");
    print_net_send_buf_update_to_host();
    printer->push_block("for (int i=0; i < nsb->_cnt; i++)");
    printer->add_multi_line(R"CODE(
        int type = nsb->_sendtype[i];
        int tid = nt->id;
        double t = nsb->_nsb_t[i];
        double flag = nsb->_nsb_flag[i];
        int vdata_index = nsb->_vdata_index[i];
        int weight_index = nsb->_weight_index[i];
        int point_index = nsb->_pnt_index[i];
        net_sem_from_gpu(type, vdata_index, weight_index, tid, point_index, t, flag);
    )CODE");
    printer->pop_block();
    printer->add_line("nsb->_cnt = 0;");
    print_net_send_buf_count_update_to_device();
}


std::string CodegenCoreneuronCppVisitor::net_receive_buffering_declaration() {
    return fmt::format("void {}(NrnThread* nt)", method_name("net_buf_receive"));
}


void CodegenCoreneuronCppVisitor::print_get_memb_list() {
    printer->add_line("Memb_list* ml = get_memb_list(nt);");
    printer->push_block("if (!ml)");
    printer->add_line("return;");
    printer->pop_block();
    printer->add_newline();
}


void CodegenCoreneuronCppVisitor::print_net_receive_loop_begin() {
    printer->add_line("int count = nrb->_displ_cnt;");
    print_channel_iteration_block_parallel_hint(BlockType::NetReceive, info.net_receive_node);
    printer->push_block("for (int i = 0; i < count; i++)");
}


void CodegenCoreneuronCppVisitor::print_net_receive_loop_end() {
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_net_receive_buffering(bool need_mech_inst) {
    if (!net_receive_required() || info.artificial_cell) {
        return;
    }
    printer->add_newline(2);
    printer->push_block(net_receive_buffering_declaration());

    print_get_memb_list();

    const auto& net_receive = method_name("net_receive_kernel");

    print_kernel_data_present_annotation_block_begin();

    printer->add_line("NetReceiveBuffer_t* nrb = ml->_net_receive_buffer;");
    if (need_mech_inst) {
        printer->fmt_line("auto* const inst = static_cast<{0}*>(ml->instance);", instance_struct());
    }
    print_net_receive_loop_begin();
    printer->add_line("int start = nrb->_displ[i];");
    printer->add_line("int end = nrb->_displ[i+1];");
    printer->push_block("for (int j = start; j < end; j++)");
    printer->add_multi_line(R"CODE(
        int index = nrb->_nrb_index[j];
        int offset = nrb->_pnt_index[index];
        double t = nrb->_nrb_t[index];
        int weight_index = nrb->_weight_index[index];
        double flag = nrb->_nrb_flag[index];
        Point_process* point_process = nt->pntprocs + offset;
    )CODE");
    printer->add_line(net_receive, "(t, point_process, inst, nt, ml, weight_index, flag);");
    printer->pop_block();
    print_net_receive_loop_end();

    print_device_stream_wait();
    printer->add_line("nrb->_displ_cnt = 0;");
    printer->add_line("nrb->_cnt = 0;");

    if (info.net_send_used || info.net_event_used) {
        print_send_event_move();
    }

    print_kernel_data_present_annotation_block_end();
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_net_send_buffering_cnt_update() const {
    printer->add_line("i = nsb->_cnt++;");
}


void CodegenCoreneuronCppVisitor::print_net_send_buffering_grow() {
    printer->push_block("if (i >= nsb->_size)");
    printer->add_line("nsb->grow();");
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_net_send_buffering() {
    if (!net_send_buffer_required()) {
        return;
    }

    printer->add_newline(2);
    print_device_method_annotation();
    auto args =
        "const NrnThread* nt, NetSendBuffer_t* nsb, int type, int vdata_index, "
        "int weight_index, int point_index, double t, double flag";
    printer->fmt_push_block("static inline void net_send_buffering({})", args);
    printer->add_line("int i = 0;");
    print_net_send_buffering_cnt_update();
    print_net_send_buffering_grow();
    printer->push_block("if (i < nsb->_size)");
    printer->add_multi_line(R"CODE(
         nsb->_sendtype[i] = type;
         nsb->_vdata_index[i] = vdata_index;
         nsb->_weight_index[i] = weight_index;
         nsb->_pnt_index[i] = point_index;
         nsb->_nsb_t[i] = t;
         nsb->_nsb_flag[i] = flag;
    )CODE");
    printer->pop_block();
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_net_receive_kernel() {
    if (!net_receive_required()) {
        return;
    }
    codegen = true;
    printing_net_receive = true;
    const auto node = info.net_receive_node;

    // rename net_receive arguments used in the block itself
    rename_net_receive_arguments(*info.net_receive_node, *node);

    std::string name;
    ParamVector params;
    if (!info.artificial_cell) {
        name = method_name("net_receive_kernel");
        params.emplace_back("", "double", "", "t");
        params.emplace_back("", "Point_process*", "", "pnt");
        params.emplace_back("", fmt::format("{}*", instance_struct()),
                            "", "inst");
        params.emplace_back("", "NrnThread*", "", "nt");
        params.emplace_back("", "Memb_list*", "", "ml");
        params.emplace_back("", "int", "", "weight_index");
        params.emplace_back("", "double", "", "flag");
    } else {
        name = method_name("net_receive");
        params.emplace_back("", "Point_process*", "", "pnt");
        params.emplace_back("", "int", "", "weight_index");
        params.emplace_back("", "double", "", "flag");
    }

    printer->add_newline(2);
    printer->fmt_push_block("static inline void {}({})", name, get_parameter_str(params));
    print_net_receive_common_code(*node, info.artificial_cell);
    if (info.artificial_cell) {
        printer->add_line("double t = nt->_t;");
    }

    // set voltage variable if it is used in the block (e.g. for WATCH statement)
    auto v_used = VarUsageVisitor().variable_used(*node->get_statement_block(), "v");
    if (v_used) {
        printer->add_line("int node_id = ml->nodeindices[id];");
        printer->add_line("v = nt->_actual_v[node_id];");
    }

    printer->fmt_line("{} = t;", get_variable_name("tsave"));

    if (info.is_watch_used()) {
        printer->add_line("bool watch_remove = false;");
    }

    printer->add_indent();
    node->get_statement_block()->accept(*this);
    printer->add_newline();
    printer->pop_block();

    printing_net_receive = false;
    codegen = false;
}


void CodegenCoreneuronCppVisitor::print_net_receive() {
    if (!net_receive_required()) {
        return;
    }
    codegen = true;
    printing_net_receive = true;
    if (!info.artificial_cell) {
        const auto& name = method_name("net_receive");
        ParamVector params;
        params.emplace_back("", "Point_process*", "", "pnt");
        params.emplace_back("", "int", "", "weight_index");
        params.emplace_back("", "double", "", "flag");
        printer->add_newline(2);
        printer->fmt_push_block("static void {}({})", name, get_parameter_str(params));
        printer->add_line("NrnThread* nt = nrn_threads + pnt->_tid;");
        printer->add_line("Memb_list* ml = get_memb_list(nt);");
        printer->add_line("NetReceiveBuffer_t* nrb = ml->_net_receive_buffer;");
        printer->push_block("if (nrb->_cnt >= nrb->_size)");
        printer->add_line("realloc_net_receive_buffer(nt, ml);");
        printer->pop_block();
        printer->add_multi_line(R"CODE(
            int id = nrb->_cnt;
            nrb->_pnt_index[id] = pnt-nt->pntprocs;
            nrb->_weight_index[id] = weight_index;
            nrb->_nrb_t[id] = nt->_t;
            nrb->_nrb_flag[id] = flag;
            nrb->_cnt++;
        )CODE");
        printer->pop_block();
    }
    printing_net_receive = false;
    codegen = false;
}


/**
 * \todo Data is not derived. Need to add instance into instance struct?
 * data used here is wrong in AoS because as in original implementation,
 * data is not incremented every iteration for AoS. May be better to derive
 * actual variable names? [resolved now?]
 * slist needs to added as local variable
 */
void CodegenCoreneuronCppVisitor::print_derivimplicit_kernel(const Block& block) {
    auto ext_args = external_method_arguments();
    auto ext_params = external_method_parameters();
    auto suffix = info.mod_suffix;
    auto list_num = info.derivimplicit_list_num;
    auto block_name = block.get_node_name();
    auto primes_size = info.primes_size;
    auto stride = "*pnodecount+id";

    printer->add_newline(2);

    printer->push_block("namespace");
            printer->fmt_push_block("struct _newton_{}_{}", block_name, info.mod_suffix);
            printer->fmt_push_block("int operator()({}) const", external_method_parameters());
    auto const instance = fmt::format("auto* const inst = static_cast<{0}*>(ml->instance);",
                                      instance_struct());
    auto const slist1 = fmt::format("auto const& slist{} = {};",
                                    list_num,
                                    get_variable_name(fmt::format("slist{}", list_num)));
    auto const slist2 = fmt::format("auto& slist{} = {};",
                                    list_num + 1,
                                    get_variable_name(fmt::format("slist{}", list_num + 1)));
    auto const dlist1 = fmt::format("auto const& dlist{} = {};",
                                    list_num,
                                    get_variable_name(fmt::format("dlist{}", list_num)));
    auto const dlist2 = fmt::format(
        "double* dlist{} = static_cast<double*>(thread[dith{}()].pval) + ({}*pnodecount);",
        list_num + 1,
        list_num,
        info.primes_size);
    printer->add_line(instance);
    if (ion_variable_struct_required()) {
        print_ion_variable();
    }
    printer->fmt_line("double* savstate{} = static_cast<double*>(thread[dith{}()].pval);",
                      list_num,
                      list_num);
    printer->add_line(slist1);
    printer->add_line(dlist1);
    printer->add_line(dlist2);
    codegen = true;
    print_statement_block(*block.get_statement_block(), false, false);
    codegen = false;
    printer->add_line("int counter = -1;");
            printer->fmt_push_block("for (int i=0; i<{}; i++)", info.num_primes);
            printer->fmt_push_block("if (*deriv{}_advance(thread))", list_num);
    printer->fmt_line(
        "dlist{0}[(++counter){1}] = "
        "data[dlist{2}[i]{1}]-(data[slist{2}[i]{1}]-savstate{2}[i{1}])/nt->_dt;",
        list_num + 1,
        stride,
        list_num);
            printer->chain_block("else");
    printer->fmt_line("dlist{0}[(++counter){1}] = data[slist{2}[i]{1}]-savstate{2}[i{1}];",
                      list_num + 1,
                      stride,
                      list_num);
    printer->pop_block();
    printer->pop_block();
    printer->add_line("return 0;");
    printer->pop_block();  // operator()
    printer->pop_block(";");   // struct
    printer->pop_block();  // namespace
    printer->add_newline();
    printer->fmt_push_block("int {}_{}({})", block_name, suffix, ext_params);
    printer->add_line(instance);
    printer->fmt_line("double* savstate{} = (double*) thread[dith{}()].pval;", list_num, list_num);
    printer->add_line(slist1);
    printer->add_line(slist2);
    printer->add_line(dlist2);
    printer->fmt_push_block("for (int i=0; i<{}; i++)", info.num_primes);
    printer->fmt_line("savstate{}[i{}] = data[slist{}[i]{}];", list_num, stride, list_num, stride);
    printer->pop_block();
    printer->fmt_line(
        "int reset = nrn_newton_thread(static_cast<NewtonSpace*>(*newtonspace{}(thread)), {}, "
        "slist{}, _newton_{}_{}{{}}, dlist{}, {});",
        list_num,
        primes_size,
        list_num + 1,
        block_name,
        suffix,
        list_num + 1,
        ext_args);
    printer->add_line("return reset;");
    printer->pop_block();
    printer->add_newline(2);
}


void CodegenCoreneuronCppVisitor::print_newtonspace_transfer_to_device() const {
    // nothing to do on cpu
}


/****************************************************************************************/
/*                                Print nrn_state routine                                */
/****************************************************************************************/


void CodegenCoreneuronCppVisitor::print_nrn_state() {
    if (!nrn_state_required()) {
        return;
    }
    codegen = true;

    printer->add_newline(2);
    printer->add_line("/** update state */");
    print_global_function_common_code(BlockType::State);
    print_channel_iteration_block_parallel_hint(BlockType::State, info.nrn_state_block);
    printer->push_block("for (int id = 0; id < nodecount; id++)");

    printer->add_line("int node_id = node_index[id];");
    printer->add_line("double v = voltage[node_id];");
    print_v_unused();

    /**
     * \todo Eigen solver node also emits IonCurVar variable in the functor
     * but that shouldn't update ions in derivative block
     */
    if (ion_variable_struct_required()) {
        print_ion_variable();
    }

    auto read_statements = ion_read_statements(BlockType::State);
    for (auto& statement: read_statements) {
        printer->add_line(statement);
    }

    if (info.nrn_state_block) {
        info.nrn_state_block->visit_children(*this);
    }

    if (info.currents.empty() && info.breakpoint_node != nullptr) {
        auto block = info.breakpoint_node->get_statement_block();
        print_statement_block(*block, false, false);
    }

    const auto& write_statements = ion_write_statements(BlockType::State);
    for (auto& statement: write_statements) {
        const auto& text = process_shadow_update_statement(statement, BlockType::State);
        printer->add_line(text);
    }
    printer->pop_block();

    print_kernel_data_present_annotation_block_end();

    printer->pop_block();
    codegen = false;
}


/****************************************************************************************/
/*                            Print nrn_cur related routines                            */
/****************************************************************************************/


void CodegenCoreneuronCppVisitor::print_nrn_current(const BreakpointBlock& node) {
    const auto& args = internal_method_parameters();
    const auto& block = node.get_statement_block();
    printer->add_newline(2);
    print_device_method_annotation();
            printer->fmt_push_block("inline double nrn_current_{}({})",
                                    info.mod_suffix,
                                    get_parameter_str(args));
    printer->add_line("double current = 0.0;");
    print_statement_block(*block, false, false);
    for (auto& current: info.currents) {
        const auto& name = get_variable_name(current);
        printer->fmt_line("current += {};", name);
    }
    printer->add_line("return current;");
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_nrn_cur_conductance_kernel(const BreakpointBlock& node) {
    const auto& block = node.get_statement_block();
    print_statement_block(*block, false, false);
    if (!info.currents.empty()) {
        std::string sum;
        for (const auto& current: info.currents) {
            auto var = breakpoint_current(current);
            sum += get_variable_name(var);
            if (&current != &info.currents.back()) {
                sum += "+";
            }
        }
        printer->fmt_line("double rhs = {};", sum);
    }

    std::string sum;
    for (const auto& conductance: info.conductances) {
        auto var = breakpoint_current(conductance.variable);
        sum += get_variable_name(var);
        if (&conductance != &info.conductances.back()) {
            sum += "+";
        }
    }
    printer->fmt_line("double g = {};", sum);

    for (const auto& conductance: info.conductances) {
        if (!conductance.ion.empty()) {
            const auto& lhs = std::string(naming::ION_VARNAME_PREFIX) + "di" + conductance.ion + "dv";
            const auto& rhs = get_variable_name(conductance.variable);
            const ShadowUseStatement statement{lhs, "+=", rhs};
            const auto& text = process_shadow_update_statement(statement, BlockType::Equation);
            printer->add_line(text);
        }
    }
}


void CodegenCoreneuronCppVisitor::print_nrn_cur_non_conductance_kernel() {
    printer->fmt_line("double g = nrn_current_{}({}+0.001);",
                                  info.mod_suffix,
                                  internal_method_arguments());
    for (auto& ion: info.ions) {
        for (auto& var: ion.writes) {
            if (ion.is_ionic_current(var)) {
                const auto& name = get_variable_name(var);
                printer->fmt_line("double di{} = {};", ion.name, name);
            }
        }
    }
    printer->fmt_line("double rhs = nrn_current_{}({});",
                                  info.mod_suffix,
                                  internal_method_arguments());
    printer->add_line("g = (g-rhs)/0.001;");
    for (auto& ion: info.ions) {
        for (auto& var: ion.writes) {
            if (ion.is_ionic_current(var)) {
                const auto& lhs = std::string(naming::ION_VARNAME_PREFIX) + "di" + ion.name + "dv";
                auto rhs = fmt::format("(di{}-{})/0.001", ion.name, get_variable_name(var));
                if (info.point_process) {
                    auto area = get_variable_name(naming::NODE_AREA_VARIABLE);
                    rhs += fmt::format("*1.e2/{}", area);
                }
                const ShadowUseStatement statement{lhs, "+=", rhs};
                const auto& text = process_shadow_update_statement(statement, BlockType::Equation);
                printer->add_line(text);
            }
        }
    }
}


void CodegenCoreneuronCppVisitor::print_nrn_cur_kernel(const BreakpointBlock& node) {
    printer->add_line("int node_id = node_index[id];");
    printer->add_line("double v = voltage[node_id];");
    print_v_unused();
    if (ion_variable_struct_required()) {
        print_ion_variable();
    }

    const auto& read_statements = ion_read_statements(BlockType::Equation);
    for (auto& statement: read_statements) {
        printer->add_line(statement);
    }

    if (info.conductances.empty()) {
        print_nrn_cur_non_conductance_kernel();
    } else {
        print_nrn_cur_conductance_kernel(node);
    }

    const auto& write_statements = ion_write_statements(BlockType::Equation);
    for (auto& statement: write_statements) {
        auto text = process_shadow_update_statement(statement, BlockType::Equation);
        printer->add_line(text);
    }

    if (info.point_process) {
        const auto& area = get_variable_name(naming::NODE_AREA_VARIABLE);
        printer->fmt_line("double mfactor = 1.e2/{};", area);
        printer->add_line("g = g*mfactor;");
        printer->add_line("rhs = rhs*mfactor;");
    }

    print_g_unused();
}


void CodegenCoreneuronCppVisitor::print_fast_imem_calculation() {
    if (!info.electrode_current) {
        return;
    }
    std::string rhs, d;
    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    if (info.point_process) {
        rhs = "shadow_rhs[id]";
        d = "shadow_d[id]";
    } else {
        rhs = "rhs";
        d = "g";
    }

    printer->push_block("if (nt->nrn_fast_imem)");
    if (nrn_cur_reduction_loop_required()) {
        printer->push_block("for (int id = 0; id < nodecount; id++)");
        printer->add_line("int node_id = node_index[id];");
    }
    printer->fmt_line("nt->nrn_fast_imem->nrn_sav_rhs[node_id] {} {};", rhs_op, rhs);
    printer->fmt_line("nt->nrn_fast_imem->nrn_sav_d[node_id] {} {};", d_op, d);
    if (nrn_cur_reduction_loop_required()) {
        printer->pop_block();
    }
    printer->pop_block();
}


void CodegenCoreneuronCppVisitor::print_nrn_cur() {
    if (!nrn_cur_required()) {
        return;
    }

    codegen = true;
    if (info.conductances.empty()) {
        print_nrn_current(*info.breakpoint_node);
    }

    printer->add_newline(2);
    printer->add_line("/** update current */");
    print_global_function_common_code(BlockType::Equation);
    print_channel_iteration_block_parallel_hint(BlockType::Equation, info.breakpoint_node);
    printer->push_block("for (int id = 0; id < nodecount; id++)");
    print_nrn_cur_kernel(*info.breakpoint_node);
    print_nrn_cur_matrix_shadow_update();
    if (!nrn_cur_reduction_loop_required()) {
        print_fast_imem_calculation();
    }
    printer->pop_block();

    if (nrn_cur_reduction_loop_required()) {
        printer->push_block("for (int id = 0; id < nodecount; id++)");
        print_nrn_cur_matrix_shadow_reduction();
        printer->pop_block();
        print_fast_imem_calculation();
    }

    print_kernel_data_present_annotation_block_end();
    printer->pop_block();
    codegen = false;
}


/****************************************************************************************/
/*                            Main code printing entry points                            */
/****************************************************************************************/

void CodegenCoreneuronCppVisitor::print_headers_include() {
    print_standard_includes();
    print_backend_includes();
    print_coreneuron_includes();
}


void CodegenCoreneuronCppVisitor::print_namespace_begin() {
    print_namespace_start();
    print_backend_namespace_start();
}


void CodegenCoreneuronCppVisitor::print_namespace_end() {
    print_backend_namespace_stop();
    print_namespace_stop();
}


void CodegenCoreneuronCppVisitor::print_common_getters() {
    print_first_pointer_var_index_getter();
    print_net_receive_arg_size_getter();
    print_thread_getters();
    print_num_variable_getter();
    print_mech_type_getter();
    print_memb_list_getter();
}


void CodegenCoreneuronCppVisitor::print_data_structures(bool print_initializers) {
    print_mechanism_global_var_structure(print_initializers);
    print_mechanism_range_var_structure(print_initializers);
    print_ion_var_structure();
}


void CodegenCoreneuronCppVisitor::print_v_unused() const {
    if (!info.vectorize) {
        return;
    }
    printer->add_multi_line(R"CODE(
        #if NRN_PRCELLSTATE
        inst->v_unused[id] = v;
        #endif
    )CODE");
}


void CodegenCoreneuronCppVisitor::print_g_unused() const {
    printer->add_multi_line(R"CODE(
        #if NRN_PRCELLSTATE
        inst->g_unused[id] = g;
        #endif
    )CODE");
}


void CodegenCoreneuronCppVisitor::print_compute_functions() {
    print_top_verbatim_blocks();
    for (const auto& procedure: info.procedures) {
        print_procedure(*procedure);
    }
    for (const auto& function: info.functions) {
        print_function(*function);
    }
    for (const auto& function: info.function_tables) {
        print_function_tables(*function);
    }
    for (size_t i = 0; i < info.before_after_blocks.size(); i++) {
        print_before_after_block(info.before_after_blocks[i], i);
    }
    for (const auto& callback: info.derivimplicit_callbacks) {
        const auto& block = *callback->get_node_to_solve();
        print_derivimplicit_kernel(block);
    }
    print_net_send_buffering();
    print_net_init();
    print_watch_activate();
    print_watch_check();
    print_net_receive_kernel();
    print_net_receive();
    print_net_receive_buffering();
    print_nrn_init();
    print_nrn_cur();
    print_nrn_state();
}


void CodegenCoreneuronCppVisitor::print_codegen_routines() {
    codegen = true;
    print_backend_info();
    print_headers_include();
    print_namespace_begin();
    print_nmodl_constants();
    print_prcellstate_macros();
    print_mechanism_info();
    print_data_structures(true);
    print_global_variables_for_hoc();
    print_common_getters();
    print_memory_allocation_routine();
    print_abort_routine();
    print_thread_memory_callbacks();
    print_instance_variable_setup();
    print_nrn_alloc();
    print_nrn_constructor();
    print_nrn_destructor();
    print_function_prototypes();
    print_functors_definitions();
    print_compute_functions();
    print_check_table_thread_function();
    print_mechanism_register();
    print_namespace_end();
    codegen = false;
}


/****************************************************************************************/
/*                            Overloaded visitor routines                               */
/****************************************************************************************/


void CodegenCoreneuronCppVisitor::visit_derivimplicit_callback(const ast::DerivimplicitCallback& node) {
    if (!codegen) {
        return;
    }
    printer->fmt_line("{}_{}({});",
                      node.get_node_to_solve()->get_node_name(),
                      info.mod_suffix,
                      external_method_arguments());
}


void CodegenCoreneuronCppVisitor::visit_solution_expression(const SolutionExpression& node) {
    auto block = node.get_node_to_solve().get();
    if (block->is_statement_block()) {
        auto statement_block = dynamic_cast<ast::StatementBlock*>(block);
        print_statement_block(*statement_block, false, false);
    } else {
        block->accept(*this);
    }
}


void CodegenCoreneuronCppVisitor::visit_for_netcon(const ast::ForNetcon& node) {
    // For_netcon should take the same arguments as net_receive and apply the operations
    // in the block to the weights of the netcons. Since all the weights are on the same vector,
    // weights, we have a mask of operations that we apply iteratively, advancing the offset
    // to the next netcon.
    const auto& args = node.get_parameters();
    RenameVisitor v;
    const auto& statement_block = node.get_statement_block();
    for (size_t i_arg = 0; i_arg < args.size(); ++i_arg) {
        // sanitize node_name since we want to substitute names like (*w) as they are
        auto old_name =
            std::regex_replace(args[i_arg]->get_node_name(), regex_special_chars, R"(\$&)");
        const auto& new_name = fmt::format("weights[{} + nt->_fornetcon_weight_perm[i]]", i_arg);
        v.set(old_name, new_name);
        statement_block->accept(v);
    }

    const auto index =
        std::find_if(info.semantics.begin(), info.semantics.end(), [](const IndexSemantics& a) {
            return a.name == naming::FOR_NETCON_SEMANTIC;
        })->index;

    printer->fmt_text("const size_t offset = {}*pnodecount + id;", index);
    printer->add_newline();
    printer->add_line(
        "const size_t for_netcon_start = nt->_fornetcon_perm_indices[indexes[offset]];");
    printer->add_line(
        "const size_t for_netcon_end = nt->_fornetcon_perm_indices[indexes[offset] + 1];");

    printer->add_line("for (auto i = for_netcon_start; i < for_netcon_end; ++i) {");
    printer->increase_indent();
    print_statement_block(*statement_block, false, false);
    printer->decrease_indent();

    printer->add_line("}");
}

}  // namespace codegen
}  // namespace nmodl
