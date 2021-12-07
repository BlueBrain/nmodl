/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/codegen_c_visitor.hpp"

#include <algorithm>
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

using namespace fmt::literals;

namespace nmodl {
namespace codegen {

using namespace ast;

using visitor::DefUseAnalyzeVisitor;
using visitor::DUChain;
using visitor::DUState;
using visitor::RenameVisitor;
using visitor::SymtabVisitor;
using visitor::VarUsageVisitor;

using symtab::syminfo::NmodlType;
using SymbolType = std::shared_ptr<symtab::Symbol>;

using nmodl::utils::UseNumbersInString;
namespace codegen_utils = nmodl::codegen::utils;

/****************************************************************************************/
/*                            Overloaded visitor routines                               */
/****************************************************************************************/

static const std::regex regex_special_chars{R"([-[\]{}()*+?.,\^$|#\s])"};

void CodegenCVisitor::visit_string(const String& node) {
    if (!codegen) {
        return;
    }
    std::string name = node.eval();
    if (enable_variable_name_lookup) {
        name = get_variable_name(name);
    }
    printer->add_text(name);
}


void CodegenCVisitor::visit_integer(const Integer& node) {
    if (!codegen) {
        return;
    }
    const auto& value = node.get_value();
    printer->add_text(std::to_string(value));
}


void CodegenCVisitor::visit_float(const Float& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(format_float_string(node.get_value()));
}


void CodegenCVisitor::visit_double(const Double& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(format_double_string(node.get_value()));
}


void CodegenCVisitor::visit_boolean(const Boolean& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(std::to_string(static_cast<int>(node.eval())));
}


void CodegenCVisitor::visit_name(const Name& node) {
    if (!codegen) {
        return;
    }
    node.visit_children(*this);
}


void CodegenCVisitor::visit_unit(const ast::Unit& node) {
    // do not print units
}


void CodegenCVisitor::visit_prime_name(const PrimeName& node) {
    throw std::runtime_error("PRIME encountered during code generation, ODEs not solved?");
}


/**
 * \todo : Validate how @ is being handled in neuron implementation
 */
void CodegenCVisitor::visit_var_name(const VarName& node) {
    if (!codegen) {
        return;
    }
    const auto& name = node.get_name();
    const auto& at_index = node.get_at();
    const auto& index = node.get_index();
    name->accept(*this);
    if (at_index) {
        printer->add_text("@");
        at_index->accept(*this);
    }
    if (index) {
        printer->add_text("[");
        printer->add_text("static_cast<int>(");
        index->accept(*this);
        printer->add_text(")");
        printer->add_text("]");
    }
}


void CodegenCVisitor::visit_indexed_name(const IndexedName& node) {
    if (!codegen) {
        return;
    }
    node.get_name()->accept(*this);
    printer->add_text("[");
    printer->add_text("static_cast<int>(");
    node.get_length()->accept(*this);
    printer->add_text(")");
    printer->add_text("]");
}


void CodegenCVisitor::visit_local_list_statement(const LocalListStatement& node) {
    if (!codegen) {
        return;
    }
    auto type = local_var_type() + " ";
    printer->add_text(type);
    print_vector_elements(node.get_variables(), ", ");
}


void CodegenCVisitor::visit_if_statement(const IfStatement& node) {
    if (!codegen) {
        return;
    }
    printer->add_text("if (");
    node.get_condition()->accept(*this);
    printer->add_text(") ");
    node.get_statement_block()->accept(*this);
    print_vector_elements(node.get_elseifs(), "");
    const auto& elses = node.get_elses();
    if (elses) {
        elses->accept(*this);
    }
}


void CodegenCVisitor::visit_else_if_statement(const ElseIfStatement& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(" else if (");
    node.get_condition()->accept(*this);
    printer->add_text(") ");
    node.get_statement_block()->accept(*this);
}


void CodegenCVisitor::visit_else_statement(const ElseStatement& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(" else ");
    node.visit_children(*this);
}


void CodegenCVisitor::visit_while_statement(const WhileStatement& node) {
    printer->add_text("while (");
    node.get_condition()->accept(*this);
    printer->add_text(") ");
    node.get_statement_block()->accept(*this);
}


void CodegenCVisitor::visit_from_statement(const ast::FromStatement& node) {
    if (!codegen) {
        return;
    }
    auto name = node.get_node_name();
    const auto& from = node.get_from();
    const auto& to = node.get_to();
    const auto& inc = node.get_increment();
    const auto& block = node.get_statement_block();
    printer->add_text("for(int {}="_format(name));
    from->accept(*this);
    printer->add_text("; {}<="_format(name));
    to->accept(*this);
    if (inc) {
        printer->add_text("; {}+="_format(name));
        inc->accept(*this);
    } else {
        printer->add_text("; {}++"_format(name));
    }
    printer->add_text(")");
    block->accept(*this);
}


void CodegenCVisitor::visit_paren_expression(const ParenExpression& node) {
    if (!codegen) {
        return;
    }
    printer->add_text("(");
    node.get_expression()->accept(*this);
    printer->add_text(")");
}


void CodegenCVisitor::visit_binary_expression(const BinaryExpression& node) {
    if (!codegen) {
        return;
    }
    auto op = node.get_op().eval();
    const auto& lhs = node.get_lhs();
    const auto& rhs = node.get_rhs();
    if (op == "^") {
        printer->add_text("pow(");
        lhs->accept(*this);
        printer->add_text(", ");
        rhs->accept(*this);
        printer->add_text(")");
    } else {
        lhs->accept(*this);
        printer->add_text(" " + op + " ");
        rhs->accept(*this);
    }
}


void CodegenCVisitor::visit_binary_operator(const BinaryOperator& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(node.eval());
}


void CodegenCVisitor::visit_unary_operator(const UnaryOperator& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(" " + node.eval());
}


/**
 * \details Statement block is top level construct (for every nmodl block).
 * Sometime we want to analyse ast nodes even if code generation is
 * false. Hence we visit children even if code generation is false.
 */
void CodegenCVisitor::visit_statement_block(const StatementBlock& node) {
    if (!codegen) {
        node.visit_children(*this);
        return;
    }
    print_statement_block(node);
}


void CodegenCVisitor::visit_function_call(const FunctionCall& node) {
    if (!codegen) {
        return;
    }
    print_function_call(node);
}


void CodegenCVisitor::visit_verbatim(const Verbatim& node) {
    if (!codegen) {
        return;
    }
    auto text = node.get_statement()->eval();
    auto result = process_verbatim_text(text);

    auto statements = stringutils::split_string(result, '\n');
    for (auto& statement: statements) {
        stringutils::trim_newline(statement);
        if (statement.find_first_not_of(' ') != std::string::npos) {
            printer->add_line(statement);
        }
    }
}

void CodegenCVisitor::visit_update_dt(const ast::UpdateDt& node) {
    // dt change statement should be pulled outside already
}

/****************************************************************************************/
/*                               Common helper routines                                 */
/****************************************************************************************/


/**
 * \details Certain statements like unit, comment, solve can/need to be skipped
 * during code generation. Note that solve block is wrapped in expression
 * statement and hence we have to check inner expression. It's also true
 * for the initial block defined inside net receive block.
 */
bool CodegenCVisitor::statement_to_skip(const Statement& node) const {
    // clang-format off
    if (node.is_unit_state()
        || node.is_line_comment()
        || node.is_block_comment()
        || node.is_solve_block()
        || node.is_conductance_hint()
        || node.is_table_statement()) {
        return true;
    }
    // clang-format on
    if (node.is_expression_statement()) {
        auto expression = dynamic_cast<const ExpressionStatement*>(&node)->get_expression();
        if (expression->is_solve_block()) {
            return true;
        }
        if (expression->is_initial_block()) {
            return true;
        }
    }
    return false;
}


bool CodegenCVisitor::net_send_buffer_required() const noexcept {
    if (net_receive_required() && !info.artificial_cell) {
        if (info.net_event_used || info.net_send_used || info.is_watch_used()) {
            return true;
        }
    }
    return false;
}


bool CodegenCVisitor::net_receive_buffering_required() const noexcept {
    return info.point_process && !info.artificial_cell && info.net_receive_node != nullptr;
}


bool CodegenCVisitor::nrn_state_required() const noexcept {
    if (info.artificial_cell) {
        return false;
    }
    return info.nrn_state_block != nullptr || info.currents.empty();
}


bool CodegenCVisitor::nrn_cur_required() const noexcept {
    return info.breakpoint_node != nullptr && !info.currents.empty();
}


bool CodegenCVisitor::net_receive_exist() const noexcept {
    return info.net_receive_node != nullptr;
}


bool CodegenCVisitor::breakpoint_exist() const noexcept {
    return info.breakpoint_node != nullptr;
}


bool CodegenCVisitor::net_receive_required() const noexcept {
    return net_receive_exist();
}


/**
 * \details When floating point data type is not default (i.e. double) then we
 * have to copy old array to new type (for range variables).
 */
bool CodegenCVisitor::range_variable_setup_required() const noexcept {
    return codegen::naming::DEFAULT_FLOAT_TYPE != float_data_type();
}


bool CodegenCVisitor::state_variable(const std::string& name) const {
    // clang-format off
    auto result = std::find_if(info.state_vars.begin(),
                               info.state_vars.end(),
                               [&name](const SymbolType& sym) {
                                   return name == sym->get_name();
                               }
    );
    // clang-format on
    return result != info.state_vars.end();
}


int CodegenCVisitor::position_of_float_var(const std::string& name) const {
    int index = 0;
    for (const auto& var: codegen_float_variables) {
        if (var->get_name() == name) {
            return index;
        }
        index += var->get_length();
    }
    throw std::logic_error(name + " variable not found");
}


int CodegenCVisitor::position_of_int_var(const std::string& name) const {
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
 * \details We can directly print value but if user specify value as integer then
 * then it gets printed as an integer. To avoid this, we use below wrapper.
 * If user has provided integer then it gets printed as 1.0 (similar to mod2c
 * and neuron where ".0" is appended). Otherwise we print double variables as
 * they are represented in the mod file by user. If the value is in scientific
 * representation (1e+20, 1E-15) then keep it as it is.
 */
std::string CodegenCVisitor::format_double_string(const std::string& s_value) {
    return codegen_utils::format_double_string<CodegenCVisitor>(s_value);
}


std::string CodegenCVisitor::format_float_string(const std::string& s_value) {
    return codegen_utils::format_float_string<CodegenCVisitor>(s_value);
}


/**
 * \details Statements like if, else etc. don't need semicolon at the end.
 * (Note that it's valid to have "extraneous" semicolon). Also, statement
 * block can appear as statement using expression statement which need to
 * be inspected.
 */
bool CodegenCVisitor::need_semicolon(Statement* node) const {
    // clang-format off
    if (node->is_if_statement()
        || node->is_else_if_statement()
        || node->is_else_statement()
        || node->is_from_statement()
        || node->is_verbatim()
        || node->is_for_all_statement()
        || node->is_from_statement()
        || node->is_conductance_hint()
        || node->is_while_statement()) {
        return false;
    }
    if (node->is_expression_statement()) {
        auto expression = dynamic_cast<ExpressionStatement*>(node)->get_expression();
        if (expression->is_statement_block()
            || expression->is_eigen_newton_solver_block()
            || expression->is_eigen_linear_solver_block()
            || expression->is_solution_expression()
            || expression->is_for_netcon()) {
            return false;
        }
    }
    // clang-format on
    return true;
}


// check if there is a function or procedure defined with given name
bool CodegenCVisitor::defined_method(const std::string& name) const {
    const auto& function = program_symtab->lookup(name);
    auto properties = NmodlType::function_block | NmodlType::procedure_block;
    return function && function->has_any_property(properties);
}


/**
 * \details Current variable used in breakpoint block could be local variable.
 * In this case, neuron has already renamed the variable name by prepending
 * "_l". In our implementation, the variable could have been renamed by
 * one of the pass. And hence, we search all local variables and check if
 * the variable is renamed. Note that we have to look into the symbol table
 * of statement block and not breakpoint.
 */
std::string CodegenCVisitor::breakpoint_current(std::string current) const {
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


int CodegenCVisitor::float_variables_size() const {
    auto count_length = [](int l, const SymbolType& variable) {
        return l += variable->get_length();
    };

    int float_size = std::accumulate(info.range_parameter_vars.begin(),
                                     info.range_parameter_vars.end(),
                                     0,
                                     count_length);
    float_size += std::accumulate(info.range_assigned_vars.begin(),
                                  info.range_assigned_vars.end(),
                                  0,
                                  count_length);
    float_size += std::accumulate(info.range_state_vars.begin(),
                                  info.range_state_vars.end(),
                                  0,
                                  count_length);
    float_size +=
        std::accumulate(info.assigned_vars.begin(), info.assigned_vars.end(), 0, count_length);

    /// all state variables for which we add Dstate variables
    float_size += std::accumulate(info.state_vars.begin(), info.state_vars.end(), 0, count_length);

    /// for v_unused variable
    if (info.vectorize) {
        float_size++;
    }
    /// for g_unused variable
    if (breakpoint_exist()) {
        float_size++;
    }
    /// for tsave variable
    if (net_receive_exist()) {
        float_size++;
    }
    return float_size;
}


int CodegenCVisitor::int_variables_size() const {
    int num_variables = 0;
    for (const auto& semantic: info.semantics) {
        num_variables += semantic.size;
    }
    return num_variables;
}


/**
 * \details Depending upon the block type, we have to print read/write ion variables
 * during code generation. Depending on block/procedure being printed, this
 * method return statements as vector. As different code backends could have
 * different variable names, we rely on backend-specific read_ion_variable_name
 * and write_ion_variable_name method which will be overloaded.
 *
 * \todo After looking into mod2c and neuron implementation, it seems like
 * Ode block type is not used (?). Need to look into implementation details.
 */
std::vector<std::string> CodegenCVisitor::ion_read_statements(BlockType type) {
    if (optimize_ion_variable_copies()) {
        return ion_read_statements_optimized(type);
    }
    std::vector<std::string> statements;
    for (const auto& ion: info.ions) {
        auto name = ion.name;
        for (const auto& var: ion.reads) {
            if (type == BlockType::Ode && ion.is_ionic_conc(var) && state_variable(var)) {
                continue;
            }
            auto variable_names = read_ion_variable_name(var);
            auto first = get_variable_name(variable_names.first);
            auto second = get_variable_name(variable_names.second);
            statements.push_back("{} = {};"_format(first, second));
        }
        for (const auto& var: ion.writes) {
            if (type == BlockType::Ode && ion.is_ionic_conc(var) && state_variable(var)) {
                continue;
            }
            if (ion.is_ionic_conc(var)) {
                auto variables = read_ion_variable_name(var);
                auto first = get_variable_name(variables.first);
                auto second = get_variable_name(variables.second);
                statements.push_back("{} = {};"_format(first, second));
            }
        }
    }
    return statements;
}


std::vector<std::string> CodegenCVisitor::ion_read_statements_optimized(BlockType type) {
    std::vector<std::string> statements;
    for (const auto& ion: info.ions) {
        for (const auto& var: ion.writes) {
            if (type == BlockType::Ode && ion.is_ionic_conc(var) && state_variable(var)) {
                continue;
            }
            if (ion.is_ionic_conc(var)) {
                auto variables = read_ion_variable_name(var);
                auto first = "ionvar." + variables.first;
                auto second = get_variable_name(variables.second);
                statements.push_back("{} = {};"_format(first, second));
            }
        }
    }
    return statements;
}


std::vector<ShadowUseStatement> CodegenCVisitor::ion_write_statements(BlockType type) {
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
                        rhs += "*(1.e2/{})"_format(area);
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
                throw std::logic_error("codegen error for {} ion"_format(ion.name));
            }
            auto ion_type_name = "{}_type"_format(ion.name);
            auto lhs = "int {}"_format(ion_type_name);
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
std::string CodegenCVisitor::process_verbatim_token(const std::string& token) {
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


bool CodegenCVisitor::ion_variable_struct_required() const {
    return optimize_ion_variable_copies() && info.ion_has_write_variable();
}


/**
 * \details This can be override in the backend. For example, parameters can be constant
 * except in INITIAL block where they are set to 0. As initial block is/can be
 * executed on c/cpu backend, gpu/cuda backend can mark the parameter as constant.
 */
bool CodegenCVisitor::is_constant_variable(const std::string& name) const {
    auto symbol = program_symtab->lookup_in_scope(name);
    bool is_constant = false;
    if (symbol != nullptr) {
        // per mechanism ion variables needs to be updated from neuron/coreneuron values
        if (info.is_ion_variable(name)) {
            is_constant = false;
        } else if (symbol->has_any_property(NmodlType::param_assign) &&
                   symbol->get_write_count() == 0) {
            is_constant = true;
        }
    }
    return is_constant;
}


/**
 * \details Once variables are populated, update index semantics to register with coreneuron
 */
void CodegenCVisitor::update_index_semantics() {
    int index = 0;
    info.semantics.clear();

    if (info.point_process) {
        info.semantics.emplace_back(index++, naming::AREA_SEMANTIC, 1);
        info.semantics.emplace_back(index++, naming::POINT_PROCESS_SEMANTIC, 1);
    }
    for (const auto& ion: info.ions) {
        for (auto i = 0; i < ion.reads.size(); ++i) {
            info.semantics.emplace_back(index++, ion.name + "_ion", 1);
        }
        for (const auto& var: ion.writes) {
            /// add if variable is not present in the read list
            if (std::find(ion.reads.begin(), ion.reads.end(), var) == ion.reads.end()) {
                info.semantics.emplace_back(index++, ion.name + "_ion", 1);
            }
            if (ion.is_ionic_current(var)) {
                info.semantics.emplace_back(index++, ion.name + "_ion", 1);
            }
        }
        if (ion.need_style) {
            info.semantics.emplace_back(index++, "#{}_ion"_format(ion.name), 1);
        }
    }
    for (auto& var: info.pointer_variables) {
        if (info.first_pointer_var_index == -1) {
            info.first_pointer_var_index = index;
        }
        int size = var->get_length();
        if (var->has_any_property(NmodlType::pointer_var)) {
            info.semantics.emplace_back(index, naming::POINTER_SEMANTIC, size);
        } else {
            info.semantics.emplace_back(index, naming::CORE_POINTER_SEMANTIC, size);
        }
        index += size;
    }

    if (info.diam_used) {
        info.semantics.emplace_back(index++, naming::DIAM_VARIABLE, 1);
    }

    if (info.area_used) {
        info.semantics.emplace_back(index++, naming::AREA_VARIABLE, 1);
    }

    if (info.net_send_used) {
        info.semantics.emplace_back(index++, naming::NET_SEND_SEMANTIC, 1);
    }

    /*
     * Number of semantics for watch is one greater than number of
     * actual watch statements in the mod file
     */
    if (!info.watch_statements.empty()) {
        for (int i = 0; i < info.watch_statements.size() + 1; i++) {
            info.semantics.emplace_back(index++, naming::WATCH_SEMANTIC, 1);
        }
    }

    if (info.for_netcon_used) {
        info.semantics.emplace_back(index++, naming::FOR_NETCON_SEMANTIC, 1);
    }
}


std::vector<SymbolType> CodegenCVisitor::get_float_variables() {
    // sort with definition order
    auto comparator = [](const SymbolType& first, const SymbolType& second) -> bool {
        return first->get_definition_order() < second->get_definition_order();
    };

    auto assigned = info.assigned_vars;
    auto states = info.state_vars;

    // each state variable has corresponding Dstate variable
    for (auto& state: states) {
        auto name = "D" + state->get_name();
        auto symbol = make_symbol(name);
        if (state->is_array()) {
            symbol->set_as_array(state->get_length());
        }
        symbol->set_definition_order(state->get_definition_order());
        assigned.push_back(symbol);
    }
    std::sort(assigned.begin(), assigned.end(), comparator);

    auto variables = info.range_parameter_vars;
    variables.insert(variables.end(),
                     info.range_assigned_vars.begin(),
                     info.range_assigned_vars.end());
    variables.insert(variables.end(), info.range_state_vars.begin(), info.range_state_vars.end());
    variables.insert(variables.end(), assigned.begin(), assigned.end());

    if (info.vectorize) {
        variables.push_back(make_symbol(naming::VOLTAGE_UNUSED_VARIABLE));
    }
    if (breakpoint_exist()) {
        std::string name = info.vectorize ? naming::CONDUCTANCE_UNUSED_VARIABLE
                                          : naming::CONDUCTANCE_VARIABLE;
        variables.push_back(make_symbol(name));
    }
    if (net_receive_exist()) {
        variables.push_back(make_symbol(naming::T_SAVE_VARIABLE));
    }
    return variables;
}


/**
 * IndexVariableInfo has following constructor arguments:
 *      - symbol
 *      - is_vdata   (false)
 *      - is_index   (false
 *      - is_integer (false)
 *
 * Which variables are constant qualified?
 *
 *  - node area is read only
 *  - read ion variables are read only
 *  - style_ionname is index / offset
 */
std::vector<IndexVariableInfo> CodegenCVisitor::get_int_variables() {
    std::vector<IndexVariableInfo> variables;
    if (info.point_process) {
        variables.emplace_back(make_symbol(naming::NODE_AREA_VARIABLE));
        variables.back().is_constant = true;
        /// note that this variable is not printed in neuron implementation
        if (info.artificial_cell) {
            variables.emplace_back(make_symbol(naming::POINT_PROCESS_VARIABLE), true);
        } else {
            variables.emplace_back(make_symbol(naming::POINT_PROCESS_VARIABLE), false, false, true);
            variables.back().is_constant = true;
        }
    }

    for (const auto& ion: info.ions) {
        bool need_style = false;
        std::unordered_map<std::string, int> ion_vars;  // used to keep track of the variables to
                                                        // not have doubles between read/write. Same
                                                        // name variables are allowed
        for (const auto& var: ion.reads) {
            const std::string name = naming::ION_VARNAME_PREFIX + var;
            variables.emplace_back(make_symbol(name));
            variables.back().is_constant = true;
            ion_vars[name] = variables.size() - 1;
        }

        /// symbol for di_ion_dv var
        std::shared_ptr<symtab::Symbol> ion_di_dv_var = nullptr;

        for (const auto& var: ion.writes) {
            const std::string name = naming::ION_VARNAME_PREFIX + var;

            const auto ion_vars_it = ion_vars.find(name);
            if (ion_vars_it != ion_vars.end()) {
                variables[ion_vars_it->second].is_constant = false;
            } else {
                variables.emplace_back(make_symbol(naming::ION_VARNAME_PREFIX + var));
            }
            if (ion.is_ionic_current(var)) {
                ion_di_dv_var = make_symbol(std::string(naming::ION_VARNAME_PREFIX) + "di" +
                                            ion.name + "dv");
            }
            if (ion.is_intra_cell_conc(var) || ion.is_extra_cell_conc(var)) {
                need_style = true;
            }
        }

        /// insert after read/write variables but before style ion variable
        if (ion_di_dv_var != nullptr) {
            variables.emplace_back(ion_di_dv_var);
        }

        if (need_style) {
            variables.emplace_back(make_symbol("style_" + ion.name), false, true);
            variables.back().is_constant = true;
        }
    }

    for (const auto& var: info.pointer_variables) {
        auto name = var->get_name();
        if (var->has_any_property(NmodlType::pointer_var)) {
            variables.emplace_back(make_symbol(name));
        } else {
            variables.emplace_back(make_symbol(name), true);
        }
    }

    if (info.diam_used) {
        variables.emplace_back(make_symbol(naming::DIAM_VARIABLE));
    }

    if (info.area_used) {
        variables.emplace_back(make_symbol(naming::AREA_VARIABLE));
    }

    // for non-artificial cell, when net_receive buffering is enabled
    // then tqitem is an offset
    if (info.net_send_used) {
        if (info.artificial_cell) {
            variables.emplace_back(make_symbol(naming::TQITEM_VARIABLE), true);
        } else {
            variables.emplace_back(make_symbol(naming::TQITEM_VARIABLE), false, false, true);
            variables.back().is_constant = true;
        }
        info.tqitem_index = variables.size() - 1;
    }

    /**
     * \note Variables for watch statements : there is one extra variable
     * used in coreneuron compared to actual watch statements for compatibility
     * with neuron (which uses one extra Datum variable)
     */
    if (!info.watch_statements.empty()) {
        for (int i = 0; i < info.watch_statements.size() + 1; i++) {
            variables.emplace_back(make_symbol("watch{}"_format(i)), false, false, true);
        }
    }
    return variables;
}


/**
 * \details When we enable fine level parallelism at channel level, we have do updates
 * to ion variables in atomic way. As cpus don't have atomic instructions in
 * simd loop, we have to use shadow vectors for every ion variables. Here
 * we return list of all such variables.
 *
 * \todo If conductances are specified, we don't need all below variables
 */
std::vector<SymbolType> CodegenCVisitor::get_shadow_variables() {
    std::vector<SymbolType> variables;
    for (const auto& ion: info.ions) {
        for (const auto& var: ion.writes) {
            variables.push_back({make_symbol(shadow_varname(naming::ION_VARNAME_PREFIX + var))});
            if (ion.is_ionic_current(var)) {
                variables.push_back({make_symbol(shadow_varname(
                    std::string(naming::ION_VARNAME_PREFIX) + "di" + ion.name + "dv"))});
            }
        }
    }
    variables.push_back({make_symbol("ml_rhs")});
    variables.push_back({make_symbol("ml_d")});
    return variables;
}


/****************************************************************************************/
/*                      Routines must be overloaded in backend                          */
/****************************************************************************************/

std::string CodegenCVisitor::get_parameter_str(const ParamVector& params) {
    std::string param{};
    for (auto iter = params.begin(); iter != params.end(); iter++) {
        param += "{}{} {}{}"_format(std::get<0>(*iter),
                                    std::get<1>(*iter),
                                    std::get<2>(*iter),
                                    std::get<3>(*iter));
        if (!nmodl::utils::is_last(iter, params)) {
            param += ", ";
        }
    }
    return param;
}


void CodegenCVisitor::print_channel_iteration_task_begin(BlockType type) {
    // backend specific, do nothing
}


void CodegenCVisitor::print_channel_iteration_task_end() {
    // backend specific, do nothing
}


void CodegenCVisitor::print_channel_iteration_tiling_block_begin(BlockType type) {
    // no tiling for cpu backend, just get loop bounds
    printer->add_line("int start = 0;");
    printer->add_line("int end = nodecount;");
}


void CodegenCVisitor::print_channel_iteration_tiling_block_end() {
    // backend specific, do nothing
}

void CodegenCVisitor::print_instance_variable_transfer_to_device() const {
    // backend specific, do nothing
}

void CodegenCVisitor::print_deriv_advance_flag_transfer_to_device() const {
    // backend specific, do nothing
}

void CodegenCVisitor::print_device_atomic_capture_annotation() const {
    // backend specific, do nothing
}

void CodegenCVisitor::print_net_send_buf_count_update_to_host() const {
    // backend specific, do nothing
}

void CodegenCVisitor::print_net_send_buf_update_to_host() const {
    // backend specific, do nothing
}

void CodegenCVisitor::print_net_send_buf_count_update_to_device() const {
    // backend specific, do nothing
}

void CodegenCVisitor::print_dt_update_to_device() const {
    // backend specific, do nothing
}

void CodegenCVisitor::print_device_stream_wait() const {
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
void CodegenCVisitor::print_kernel_data_present_annotation_block_begin() {
    // backend specific, do nothing
}


void CodegenCVisitor::print_kernel_data_present_annotation_block_end() {
    // backend specific, do nothing
}

void CodegenCVisitor::print_net_init_acc_serial_annotation_block_begin() {
    // backend specific, do nothing
}

void CodegenCVisitor::print_net_init_acc_serial_annotation_block_end() {
    // backend specific, do nothing
}

/**
 * \details Depending programming model and compiler, we print compiler hint
 * for parallelization. For example:
 *
 * \code
 *      #pragma ivdep
 *      for(int id = 0; id < nodecount; id++) {
 *
 *      #pragma acc parallel loop
 *      for(int id = 0; id < nodecount; id++) {
 * \endcode
 */
void CodegenCVisitor::print_channel_iteration_block_parallel_hint(BlockType type) {
    printer->add_line("#pragma ivdep");
}


bool CodegenCVisitor::nrn_cur_reduction_loop_required() {
    return channel_task_dependency_enabled() || info.point_process;
}


bool CodegenCVisitor::shadow_vector_setup_required() {
    return (channel_task_dependency_enabled() && !codegen_shadow_variables.empty());
}


/**
 * \details For CPU backend we iterate over all node counts. For cuda we use thread
 * index to check if block needs to be executed or not.
 */
void CodegenCVisitor::print_channel_iteration_block_begin(BlockType type) {
    print_channel_iteration_block_parallel_hint(type);
    printer->start_block("for (int id = start; id < end; id++)");
}


void CodegenCVisitor::print_channel_iteration_block_end() {
    printer->end_block(1);
}


void CodegenCVisitor::print_rhs_d_shadow_variables() {
    if (info.point_process) {
        printer->add_line("double* shadow_rhs = nt->{};"_format(naming::NTHREAD_RHS_SHADOW));
        printer->add_line("double* shadow_d = nt->{};"_format(naming::NTHREAD_D_SHADOW));
    }
}


void CodegenCVisitor::print_nrn_cur_matrix_shadow_update() {
    if (channel_task_dependency_enabled()) {
        auto rhs = get_variable_name("ml_rhs");
        auto d = get_variable_name("ml_d");
        printer->add_line("{} = rhs;"_format(rhs));
        printer->add_line("{} = g;"_format(d));
    } else {
        if (info.point_process) {
            printer->add_line("shadow_rhs[id] = rhs;");
            printer->add_line("shadow_d[id] = g;");
        } else {
            auto rhs_op = operator_for_rhs();
            auto d_op = operator_for_d();
            print_atomic_reduction_pragma();
            printer->add_line("vec_rhs[node_id] {} rhs;"_format(rhs_op));
            print_atomic_reduction_pragma();
            printer->add_line("vec_d[node_id] {} g;"_format(d_op));
        }
    }
}


void CodegenCVisitor::print_nrn_cur_matrix_shadow_reduction() {
    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    if (channel_task_dependency_enabled()) {
        auto rhs = get_variable_name("ml_rhs");
        auto d = get_variable_name("ml_d");
        printer->add_line("int node_id = node_index[id];");
        print_atomic_reduction_pragma();
        printer->add_line("vec_rhs[node_id] {} {};"_format(rhs_op, rhs));
        print_atomic_reduction_pragma();
        printer->add_line("vec_d[node_id] {} {};"_format(d_op, d));
    } else {
        if (info.point_process) {
            printer->add_line("int node_id = node_index[id];");
            print_atomic_reduction_pragma();
            printer->add_line("vec_rhs[node_id] {} shadow_rhs[id];"_format(rhs_op));
            print_atomic_reduction_pragma();
            printer->add_line("vec_d[node_id] {} shadow_d[id];"_format(d_op));
        }
    }
}


void CodegenCVisitor::print_atomic_reduction_pragma() {
    // backend specific, do nothing
}


void CodegenCVisitor::print_shadow_reduction_block_begin() {
    printer->start_block("for (int id = start; id < end; id++)");
}


void CodegenCVisitor::print_shadow_reduction_statements() {
    for (const auto& statement: shadow_statements) {
        print_atomic_reduction_pragma();
        auto lhs = get_variable_name(statement.lhs);
        auto rhs = get_variable_name(shadow_varname(statement.lhs));
        auto text = "{} {} {};"_format(lhs, statement.op, rhs);
        printer->add_line(text);
    }
    shadow_statements.clear();
}


void CodegenCVisitor::print_shadow_reduction_block_end() {
    printer->end_block(1);
}


void CodegenCVisitor::print_device_method_annotation() {
    // backend specific, nothing for cpu
}


void CodegenCVisitor::print_global_method_annotation() {
    // backend specific, nothing for cpu
}


void CodegenCVisitor::print_backend_namespace_start() {
    // no separate namespace for C (cpu) backend
}


void CodegenCVisitor::print_backend_namespace_stop() {
    // no separate namespace for C (cpu) backend
}


void CodegenCVisitor::print_backend_includes() {
    // backend specific, nothing for cpu
}


std::string CodegenCVisitor::backend_name() const {
    return "C (api-compatibility)";
}


bool CodegenCVisitor::block_require_shadow_update(BlockType type) {
    return false;
}


bool CodegenCVisitor::channel_task_dependency_enabled() {
    return false;
}


bool CodegenCVisitor::optimize_ion_variable_copies() const {
    return optimize_ionvar_copies;
}


void CodegenCVisitor::print_memory_allocation_routine() const {
    printer->add_newline(2);
    auto args = "size_t num, size_t size, size_t alignment = 16";
    printer->add_line("static inline void* mem_alloc({}) {}"_format(args, "{"));
    printer->add_line("    void* ptr;");
    printer->add_line("    posix_memalign(&ptr, alignment, num*size);");
    printer->add_line("    memset(ptr, 0, size);");
    printer->add_line("    return ptr;");
    printer->add_line("}");

    printer->add_newline(2);
    printer->add_line("static inline void mem_free(void* ptr) {");
    printer->add_line("    free(ptr);");
    printer->add_line("}");
}


void CodegenCVisitor::print_abort_routine() const {
    printer->add_newline(2);
    printer->add_line("static inline void coreneuron_abort() {");
    printer->add_line("    abort();");
    printer->add_line("}");
}


std::string CodegenCVisitor::compute_method_name(BlockType type) const {
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


// note extra empty space for pretty-printing if we skip the symbol
std::string CodegenCVisitor::ptr_type_qualifier() {
    return "__restrict__ ";
}

/// Useful in ispc so that variables in the global struct get "uniform "
std::string CodegenCVisitor::global_var_struct_type_qualifier() {
    return "";
}

void CodegenCVisitor::print_global_var_struct_decl() {
    printer->add_line("{} {}_global;"_format(global_struct(), info.mod_suffix));
}

std::string CodegenCVisitor::k_const() {
    return "const ";
}


/****************************************************************************************/
/*              printing routines for code generation                                   */
/****************************************************************************************/


void CodegenCVisitor::visit_watch_statement(const ast::WatchStatement& node) {
    printer->add_text("nrn_watch_activate(inst, id, pnodecount, {}, v, watch_remove)"_format(
        current_watch_statement++));
}


void CodegenCVisitor::print_statement_block(const ast::StatementBlock& node,
                                            bool open_brace,
                                            bool close_brace) {
    if (open_brace) {
        printer->start_block();
    }

    auto statements = node.get_statements();
    for (const auto& statement: statements) {
        if (statement_to_skip(*statement)) {
            continue;
        }
        /// not necessary to add indent for verbatim block (pretty-printing)
        if (!statement->is_verbatim()) {
            printer->add_indent();
        }
        statement->accept(*this);
        if (need_semicolon(statement.get())) {
            printer->add_text(";");
        }
        printer->add_newline();
    }

    if (close_brace) {
        printer->end_block();
    }
}


void CodegenCVisitor::print_function_call(const FunctionCall& node) {
    auto name = node.get_node_name();
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

    auto arguments = node.get_arguments();
    printer->add_text("{}("_format(function_name));

    if (defined_method(name)) {
        printer->add_text(internal_method_arguments());
        if (!arguments.empty()) {
            printer->add_text(", ");
        }
    } else if (nmodl::details::needs_neuron_thread_first_arg(function_name) &&
               arguments.front()->get_node_name() != "nt") {
        arguments.insert(arguments.begin(), std::make_shared<ast::String>("nt"));
    }

    print_vector_elements(arguments, ", ");
    printer->add_text(")");
}


void CodegenCVisitor::print_top_verbatim_blocks() {
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


/**
 * \todo Issue with verbatim renaming. e.g. pattern.mod has info struct with
 * index variable. If we use "index" instead of "indexes" as default argument
 * then during verbatim replacement we don't know the index is which one. This
 * is because verbatim renaming pass has already stripped out prefixes from
 * the text.
 */
void CodegenCVisitor::rename_function_arguments() {
    auto default_arguments = stringutils::split_string(nrn_thread_arguments(), ',');
    for (auto& arg: default_arguments) {
        stringutils::trim(arg);
        RenameVisitor v(arg, "arg_" + arg);
        for (const auto& function: info.functions) {
            if (has_parameter_of_name(function, arg)) {
                function->accept(v);
            }
        }
        for (const auto& function: info.procedures) {
            if (has_parameter_of_name(function, arg)) {
                function->accept(v);
            }
        }
    }
}


void CodegenCVisitor::print_function_prototypes() {
    if (info.functions.empty() && info.procedures.empty()) {
        return;
    }
    codegen = true;
    printer->add_newline(2);
    for (const auto& node: info.functions) {
        print_function_declaration(*node, node->get_node_name());
        printer->add_text(";");
        printer->add_newline();
    }
    for (const auto& node: info.procedures) {
        print_function_declaration(*node, node->get_node_name());
        printer->add_text(";");
        printer->add_newline();
    }
    codegen = false;
}


static const TableStatement* get_table_statement(const ast::Block& node) {
    // TableStatementVisitor v;

    const auto& table_statements = collect_nodes(node, {AstNodeType::TABLE_STATEMENT});

    if (table_statements.size() != 1) {
        auto message =
            "One table statement expected in {} found {}"_format(node.get_node_name(),
                                                                 table_statements.size());
        throw std::runtime_error(message);
    }
    return dynamic_cast<const TableStatement*>(table_statements.front().get());
}


void CodegenCVisitor::print_table_check_function(const Block& node) {
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
    printer->start_block(
        "void check_{}({})"_format(method_name(name), get_parameter_str(internal_params)));
    {
        printer->add_line("if ( {} == 0) {}"_format(use_table_var, "{"));
        printer->add_line("    return;");
        printer->add_line("}");

        printer->add_line("static bool make_table = true;");
        for (const auto& variable: depend_variables) {
            printer->add_line("static {} save_{};"_format(float_type, variable->get_node_name()));
        }

        for (const auto& variable: depend_variables) {
            auto name = variable->get_node_name();
            auto instance_name = get_variable_name(name);
            printer->add_line("if (save_{} != {}) {}"_format(name, instance_name, "{"));
            printer->add_line("    make_table = true;");
            printer->add_line("}");
        }

        printer->start_block("if (make_table)");
        {
            printer->add_line("make_table = false;");

            printer->add_indent();
            printer->add_text("{} = "_format(tmin_name));
            from->accept(*this);
            printer->add_text(";");
            printer->add_newline();

            printer->add_indent();
            printer->add_text("double tmax = ");
            to->accept(*this);
            printer->add_text(";");
            printer->add_newline();


            printer->add_line("double dx = (tmax-{})/{}.0;"_format(tmin_name, with));
            printer->add_line("{} = 1.0/dx;"_format(mfac_name));

            printer->add_line("int i = 0;");
            printer->add_line("double x = 0;");
            printer->add_line(
                "for(i = 0, x = {}; i < {}; x += dx, i++) {}"_format(tmin_name, with + 1, "{"));
            auto function = method_name("f_" + name);
            printer->add_line("    {}({}, x);"_format(function, internal_method_arguments()));
            for (const auto& variable: table_variables) {
                auto name = variable->get_node_name();
                auto instance_name = get_variable_name(name);
                auto table_name = get_variable_name("t_" + name);
                printer->add_line("    {}[i] = {};"_format(table_name, instance_name));
            }
            printer->add_line("}");

            for (const auto& variable: depend_variables) {
                auto name = variable->get_node_name();
                auto instance_name = get_variable_name(name);
                printer->add_line("save_{} = {};"_format(name, instance_name));
            }
        }
        printer->end_block(1);
    }
    printer->end_block(1);
}


void CodegenCVisitor::print_table_replacement_function(const ast::Block& node) {
    auto name = node.get_node_name();
    auto statement = get_table_statement(node);
    auto table_variables = statement->get_table_vars();
    auto with = statement->get_with()->eval();
    auto use_table_var = get_variable_name(naming::USE_TABLE_VARIABLE);
    auto float_type = default_float_data_type();
    auto tmin_name = get_variable_name("tmin_" + name);
    auto mfac_name = get_variable_name("mfac_" + name);
    auto function_name = method_name("f_" + name);

    printer->add_newline(2);
    print_function_declaration(node, name);
    printer->start_block();
    {
        const auto& params = node.get_parameters();
        printer->add_line("if ( {} == 0) {{"_format(use_table_var));
        printer->add_line("    {}({}, {});"_format(function_name,
                                                   internal_method_arguments(),
                                                   params[0].get()->get_node_name()));
        printer->add_line("     return 0;");
        printer->add_line("}");

        printer->add_line("double xi = {} * ({} - {});"_format(mfac_name,
                                                               params[0].get()->get_node_name(),
                                                               tmin_name));
        printer->add_line("if (isnan(xi)) {");
        for (const auto& var: table_variables) {
            auto name = get_variable_name(var->get_node_name());
            printer->add_line("    {} = xi;"_format(name));
        }
        printer->add_line("    return 0;");
        printer->add_line("}");

        printer->add_line("if (xi <= 0.0 || xi >= {}) {}"_format(with, "{"));
        printer->add_line("    int index = (xi <= 0.0) ? 0 : {};"_format(with));
        for (const auto& variable: table_variables) {
            auto name = variable->get_node_name();
            auto instance_name = get_variable_name(name);
            auto table_name = get_variable_name("t_" + name);
            printer->add_line("    {} = {}[index];"_format(instance_name, table_name));
        }
        printer->add_line("    return 0;");
        printer->add_line("}");

        printer->add_line("int i = int(xi);");
        printer->add_line("double theta = xi - double(i);");
        for (const auto& var: table_variables) {
            auto instance_name = get_variable_name(var->get_node_name());
            auto table_name = get_variable_name("t_" + var->get_node_name());
            printer->add_line(
                "{0} = {1}[i] + theta*({1}[i+1]-{1}[i]);"_format(instance_name, table_name));
        }

        printer->add_line("return 0;");
    }
    printer->end_block(1);
}


void CodegenCVisitor::print_check_table_thread_function() {
    if (info.table_count == 0) {
        return;
    }

    printer->add_newline(2);
    auto name = method_name("check_table_thread");
    auto parameters = external_method_parameters(true);

    printer->add_line("static void {} ({}) {}"_format(name, parameters, "{"));
    printer->add_line("    Memb_list* ml = nt->_ml_list[tml_id];");
    printer->add_line("    setup_instance(nt, ml);");
    printer->add_line("    {0}* inst = ({0}*) ml->instance;"_format(instance_struct()));
    printer->add_line("    double v = 0;");

    for (const auto& function: info.functions_with_table) {
        auto name = method_name("check_" + function->get_node_name());
        auto arguments = internal_method_arguments();
        printer->add_line("    {}({});"_format(name, arguments));
    }

    /**
     * \todo `check_table_thread` is called multiple times from coreneuron including
     * after `finitialize`. If we cleaup the instance then it will result in segfault
     * but if we don't then there is memory leak
     */
    printer->add_line("    // cleanup_instance(ml);");
    printer->add_line("}");
}


void CodegenCVisitor::print_function_or_procedure(const ast::Block& node, const std::string& name) {
    printer->add_newline(2);
    print_function_declaration(node, name);
    printer->add_text(" ");
    printer->start_block();

    // function requires return variable declaration
    if (node.is_function_block()) {
        auto type = default_float_data_type();
        printer->add_line("{} ret_{} = 0.0;"_format(type, name));
    } else {
        printer->add_line("int ret_{} = 0;"_format(name));
    }

    print_statement_block(*node.get_statement_block(), false, false);
    printer->add_line("return ret_{};"_format(name));
    printer->end_block(1);
}


void CodegenCVisitor::print_function_procedure_helper(const ast::Block& node) {
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


void CodegenCVisitor::print_procedure(const ast::ProcedureBlock& node) {
    print_function_procedure_helper(node);
}


void CodegenCVisitor::print_function(const ast::FunctionBlock& node) {
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


std::string CodegenCVisitor::find_var_unique_name(const std::string& original_name) const {
    auto& singleton_random_string_class = nmodl::utils::SingletonRandomString<4>::instance();
    std::string unique_name = original_name;
    if (singleton_random_string_class.random_string_exists(original_name)) {
        unique_name = original_name;
        unique_name += "_" + singleton_random_string_class.get_random_string(original_name);
    };
    return unique_name;
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
bool is_functor_const(const ast::StatementBlock& variable_block,
                      const ast::StatementBlock& functor_block) {
    // Save DUChain for every variable in variable_block
    std::unordered_map<std::string, DUChain> chains;

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
        if (!is_functor_const)
            break;
    }

    return is_functor_const;
}

void CodegenCVisitor::visit_eigen_newton_solver_block(const ast::EigenNewtonSolverBlock& node) {
    // solution vector to store copy of state vars for Newton solver
    printer->add_newline();

    // Check if there is a variable defined in the mod file as X, J, Jm or F and if yes
    // try to use a different string for the matrices created by sympy in the form
    // X_<random_number>, J_<random_number>, Jm_<random_number> and F_<random_number>
    std::string X = find_var_unique_name("X");
    std::string Xm = find_var_unique_name("Xm");
    std::string J = find_var_unique_name("J");
    std::string Jm = find_var_unique_name("Jm");
    std::string F = find_var_unique_name("F");
    std::string Fm = find_var_unique_name("Fm");

    auto float_type = default_float_data_type();
    int N = node.get_n_state_vars()->get_value();
    printer->add_line("Eigen::Matrix<{}, {}, 1> {};"_format(float_type, N, Xm));
    printer->add_line("{}* {} = {}.data();"_format(float_type, X, Xm));

    print_statement_block(*node.get_setup_x_block(), false, false);

    // functor that evaluates F(X) and J(X) for
    // Newton solver
    printer->start_block("struct functor");
    printer->add_line("NrnThread* nt;");
    printer->add_line("{0}* inst;"_format(instance_struct()));
    printer->add_line("int id, pnodecount;");
    printer->add_line("double v;");
    printer->add_line("Datum* indexes;");
    if (ion_variable_struct_required()) {
        print_ion_variable();
    }

    print_statement_block(*node.get_variable_block(), false, false);
    printer->add_newline();

    printer->start_block("void initialize()");
    print_statement_block(*node.get_initialize_block(), false, false);
    printer->end_block(2);

    printer->add_line(
        "functor(NrnThread* nt, {}* inst, int id, int pnodecount, double v, Datum* indexes) : nt(nt), inst(inst), id(id), pnodecount(pnodecount), v(v), indexes(indexes) {}"_format(
            instance_struct(), "{}"));

    printer->add_indent();

    const auto& variable_block = *node.get_variable_block();
    const auto& functor_block = *node.get_functor_block();

    printer->add_text(
        "void operator()(const Eigen::Matrix<{0}, {1}, 1>& {2}, Eigen::Matrix<{0}, {1}, "
        "1>& {3}, "
        "Eigen::Matrix<{0}, {1}, {1}>& {4}) {5}"_format(
            float_type,
            N,
            Xm,
            Fm,
            Jm,
            is_functor_const(variable_block, functor_block) ? "const " : ""));
    printer->start_block();
    printer->add_line("const {}* {} = {}.data();"_format(float_type, X, Xm));
    printer->add_line("{}* {} = {}.data();"_format(float_type, J, Jm));
    printer->add_line("{}* {} = {}.data();"_format(float_type, F, Fm));
    print_statement_block(functor_block, false, false);
    printer->end_block(2);

    // assign newton solver results in matrix X to state vars
    printer->start_block("void finalize()");
    print_statement_block(*node.get_finalize_block(), false, false);
    printer->end_block(1);

    printer->end_block(0);
    printer->add_text(";");
    printer->add_newline();

    // call newton solver with functor and X matrix that contains state vars
    printer->add_line("// call newton solver");
    printer->add_line("functor newton_functor(nt, inst, id, pnodecount, v, indexes);");
    printer->add_line("newton_functor.initialize();");
    printer->add_line(
        "int newton_iterations = nmodl::newton::newton_solver({}, newton_functor);"_format(Xm));

    // assign newton solver results in matrix X to state vars
    print_statement_block(*node.get_update_states_block(), false, false);
    printer->add_line("newton_functor.finalize();");
}

void CodegenCVisitor::visit_eigen_linear_solver_block(const ast::EigenLinearSolverBlock& node) {
    printer->add_newline();

    // Check if there is a variable defined in the mod file as X, J, Jm or F and if yes
    // try to use a different string for the matrices created by sympy in the form
    // X_<random_number>, J_<random_number>, Jm_<random_number> and F_<random_number>
    std::string X = find_var_unique_name("X");
    std::string Xm = find_var_unique_name("Xm");
    std::string J = find_var_unique_name("J");
    std::string Jm = find_var_unique_name("Jm");
    std::string F = find_var_unique_name("F");
    std::string Fm = find_var_unique_name("Fm");

    const std::string float_type = default_float_data_type();
    int N = node.get_n_state_vars()->get_value();
    printer->add_line("Eigen::Matrix<{0}, {1}, 1> {2}, {3};"_format(float_type, N, Xm, Fm));
    printer->add_line("Eigen::Matrix<{0}, {1}, {1}> {2};"_format(float_type, N, Jm));
    printer->add_line("{}* {} = {}.data();"_format(float_type, X, Xm));
    printer->add_line("{}* {} = {}.data();"_format(float_type, J, Jm));
    printer->add_line("{}* {} = {}.data();"_format(float_type, F, Fm));
    print_statement_block(*node.get_variable_block(), false, false);
    print_statement_block(*node.get_initialize_block(), false, false);
    print_statement_block(*node.get_setup_x_block(), false, false);

    printer->add_newline();
    print_eigen_linear_solver(float_type, N, Xm, Jm, Fm);
    printer->add_newline();

    print_statement_block(*node.get_update_states_block(), false, false);
    print_statement_block(*node.get_finalize_block(), false, false);
}

void CodegenCVisitor::print_eigen_linear_solver(const std::string& float_type,
                                                int N,
                                                const std::string& Xm,
                                                const std::string& Jm,
                                                const std::string& Fm) {
    if (N <= 4) {
        // Faster compared to LU, given the template specialization in Eigen.
        printer->add_line("{0} = {1}.inverse()*{2};"_format(Xm, Jm, Fm));
    } else {
        printer->add_line(
            "{0} = Eigen::PartialPivLU<Eigen::Ref<Eigen::Matrix<{1}, {2}, {2}>>>({3}).solve({4});"_format(
                Xm, float_type, N, Jm, Fm));
    }
}

/****************************************************************************************/
/*                           Code-specific helper routines                              */
/****************************************************************************************/


std::string CodegenCVisitor::internal_method_arguments() {
    if (ion_variable_struct_required()) {
        return "id, pnodecount, inst, ionvar, data, indexes, thread, nt, v";
    }
    return "id, pnodecount, inst, data, indexes, thread, nt, v";
}


std::string CodegenCVisitor::param_type_qualifier() {
    return "";
}


std::string CodegenCVisitor::param_ptr_qualifier() {
    return "";
}


/**
 * @todo: figure out how to correctly handle qualifiers
 */
CodegenCVisitor::ParamVector CodegenCVisitor::internal_method_parameters() {
    auto params = ParamVector();
    params.emplace_back("", "int", "", "id");
    params.emplace_back(param_type_qualifier(), "int", "", "pnodecount");
    params.emplace_back(param_type_qualifier(),
                        "{}*"_format(instance_struct()),
                        param_ptr_qualifier(),
                        "inst");
    if (ion_variable_struct_required()) {
        params.emplace_back("", "IonCurVar&", "", "ionvar");
    }
    params.emplace_back("", "double*", "", "data");
    params.emplace_back(k_const(), "Datum*", "", "indexes");
    params.emplace_back(param_type_qualifier(), "ThreadDatum*", "", "thread");
    params.emplace_back(param_type_qualifier(), "NrnThread*", param_ptr_qualifier(), "nt");
    params.emplace_back("", "double", "", "v");
    return params;
}


std::string CodegenCVisitor::external_method_arguments() const {
    return "id, pnodecount, data, indexes, thread, nt, v";
}


std::string CodegenCVisitor::external_method_parameters(bool table) const {
    if (table) {
        return "int id, int pnodecount, double* data, Datum* indexes, "
               "ThreadDatum* thread, NrnThread* nt, int tml_id";
    }
    return "int id, int pnodecount, double* data, Datum* indexes, "
           "ThreadDatum* thread, NrnThread* nt, double v";
}


std::string CodegenCVisitor::nrn_thread_arguments() {
    if (ion_variable_struct_required()) {
        return "id, pnodecount, ionvar, data, indexes, thread, nt, v";
    }
    return "id, pnodecount, data, indexes, thread, nt, v";
}


/**
 * Function call arguments when function or procedure is defined in the
 * same mod file itself
 */
std::string CodegenCVisitor::nrn_thread_internal_arguments() {
    if (ion_variable_struct_required()) {
        return "id, pnodecount, inst, ionvar, data, indexes, thread, nt, v";
    }
    return "id, pnodecount, inst, data, indexes, thread, nt, v";
}


/**
 * Replace commonly used variables in the verbatim blocks into their corresponding
 * variable name in the new code generation backend.
 */
std::string CodegenCVisitor::replace_if_verbatim_variable(std::string name) {
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
std::string CodegenCVisitor::process_verbatim_text(std::string text) {
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
            name = "&" + name;
        }
        if (token == "_STRIDE") {
            name = "pnodecount+id";
        }
        result += name;
    }
    return result;
}


std::string CodegenCVisitor::register_mechanism_arguments() const {
    auto nrn_cur = nrn_cur_required() ? method_name(naming::NRN_CUR_METHOD) : "NULL";
    auto nrn_state = nrn_state_required() ? method_name(naming::NRN_STATE_METHOD) : "NULL";
    auto nrn_alloc = method_name(naming::NRN_ALLOC_METHOD);
    auto nrn_init = method_name(naming::NRN_INIT_METHOD);
    return "mechanism, {}, {}, NULL, {}, {}, first_pointer_var_index()"
           ""_format(nrn_alloc, nrn_cur, nrn_state, nrn_init);
}


std::pair<std::string, std::string> CodegenCVisitor::read_ion_variable_name(
    const std::string& name) const {
    return {name, naming::ION_VARNAME_PREFIX + name};
}


std::pair<std::string, std::string> CodegenCVisitor::write_ion_variable_name(
    const std::string& name) const {
    return {naming::ION_VARNAME_PREFIX + name, name};
}


std::string CodegenCVisitor::conc_write_statement(const std::string& ion_name,
                                                  const std::string& concentration,
                                                  int index) {
    auto conc_var_name = get_variable_name(naming::ION_VARNAME_PREFIX + concentration);
    auto style_var_name = get_variable_name("style_" + ion_name);
    return "nrn_wrote_conc({}_type,"
           " &({}),"
           " {},"
           " {},"
           " nrn_ion_global_map,"
           " celsius,"
           " nt->_ml_list[{}_type]->_nodecount_padded)"
           ""_format(ion_name, conc_var_name, index, style_var_name, ion_name);
}


/**
 * If mechanisms dependency level execution is enabled then certain updates
 * like ionic current contributions needs to be atomically updated. In this
 * case we first update current mechanism's shadow vector and then add statement
 * to queue that will be used in reduction queue.
 */
std::string CodegenCVisitor::process_shadow_update_statement(ShadowUseStatement& statement,
                                                             BlockType type) {
    // when there is no operator or rhs then that statement doesn't need shadow update
    if (statement.op.empty() && statement.rhs.empty()) {
        auto text = statement.lhs + ";";
        return text;
    }

    // blocks like initial doesn't use shadow update (e.g. due to wrote_conc call)
    if (block_require_shadow_update(type)) {
        shadow_statements.push_back(statement);
        auto lhs = get_variable_name(shadow_varname(statement.lhs));
        auto rhs = statement.rhs;
        auto text = "{} = {};"_format(lhs, rhs);
        return text;
    }

    // return regular statement
    auto lhs = get_variable_name(statement.lhs);
    auto text = "{} {} {};"_format(lhs, statement.op, statement.rhs);
    return text;
}


/****************************************************************************************/
/*               Code-specific printing routines for code generation                    */
/****************************************************************************************/


/**
 * NMODL constants from unit database
 *
 */
void CodegenCVisitor::print_nmodl_constants() {
    if (!info.factor_definitions.empty()) {
        printer->add_newline(2);
        printer->add_line("/** constants used in nmodl from UNITS */");
        for (const auto& it: info.factor_definitions) {
#ifdef USE_LEGACY_UNITS
            const std::string format_string = "static const double {} = {:g};";
#else
            const std::string format_string = "static const double {} = {:.18g};";
#endif
            printer->add_line(fmt::format(format_string,
                                          it->get_node_name(),
                                          stod(it->get_value()->get_value())));
        }
    }
}


void CodegenCVisitor::print_first_pointer_var_index_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->add_line("static inline int first_pointer_var_index() {");
    printer->add_line("    return {};"_format(info.first_pointer_var_index));
    printer->add_line("}");
}


void CodegenCVisitor::print_num_variable_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->add_line("static inline int float_variables_size() {");
    printer->add_line("    return {};"_format(float_variables_size()));
    printer->add_line("}");

    printer->add_newline(2);
    print_device_method_annotation();
    printer->add_line("static inline int int_variables_size() {");
    printer->add_line("    return {};"_format(int_variables_size()));
    printer->add_line("}");
}


void CodegenCVisitor::print_net_receive_arg_size_getter() {
    if (!net_receive_exist()) {
        return;
    }
    printer->add_newline(2);
    print_device_method_annotation();
    printer->add_line("static inline int num_net_receive_args() {");
    printer->add_line("    return {};"_format(info.num_net_receive_parameters));
    printer->add_line("}");
}


void CodegenCVisitor::print_mech_type_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->add_line("static inline int get_mech_type() {");
    printer->add_line("    return {};"_format(get_variable_name("mech_type")));
    printer->add_line("}");
}


void CodegenCVisitor::print_memb_list_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->add_line("static inline Memb_list* get_memb_list(NrnThread* nt) {");
    printer->add_line("    if (nt->_ml_list == NULL) {");
    printer->add_line("        return NULL;");
    printer->add_line("    }");
    printer->add_line("    return nt->_ml_list[get_mech_type()];");
    printer->add_line("}");
}


void CodegenCVisitor::print_namespace_start() {
    printer->add_newline(2);
    printer->start_block("namespace coreneuron");
}


void CodegenCVisitor::print_namespace_stop() {
    printer->end_block(1);
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

void CodegenCVisitor::print_thread_getters() {
    if (info.vectorize && info.derivimplicit_used()) {
        int tid = info.derivimplicit_var_thread_id;
        int list = info.derivimplicit_list_num;

        // clang-format off
        printer->add_newline(2);
        printer->add_line("/** thread specific helper routines for derivimplicit */");

        printer->add_newline(1);
        printer->add_line("static inline int* deriv{}_advance(ThreadDatum* thread) {}"_format(list, "{"));
        printer->add_line("    return &(thread[{}].i);"_format(tid));
        printer->add_line("}");

        printer->add_newline(1);
        printer->add_line("static inline int dith{}() {}"_format(list, "{"));
        printer->add_line("    return {};"_format(tid+1));
        printer->add_line("}");

        printer->add_newline(1);
        printer->add_line("static inline void** newtonspace{}(ThreadDatum* thread) {}"_format(list, "{"));
        printer->add_line("    return &(thread[{}]._pvoid);"_format(tid+2));
        printer->add_line("}");
    }

    if (info.vectorize && !info.thread_variables.empty()) {
        printer->add_newline(2);
        printer->add_line("/** tid for thread variables */");
        printer->add_line("static inline int thread_var_tid() {");
        printer->add_line("    return {};"_format(info.thread_var_thread_id));
        printer->add_line("}");
    }

    if (info.vectorize && !info.top_local_variables.empty()) {
        printer->add_newline(2);
        printer->add_line("/** tid for top local tread variables */");
        printer->add_line("static inline int top_local_var_tid() {");
        printer->add_line("    return {};"_format(info.top_local_thread_id));
        printer->add_line("}");
    }
    // clang-format on
}


/****************************************************************************************/
/*                         Routines for returning variable name                         */
/****************************************************************************************/


std::string CodegenCVisitor::float_variable_name(const SymbolType& symbol,
                                                 bool use_instance) const {
    auto name = symbol->get_name();
    auto dimension = symbol->get_length();
    auto position = position_of_float_var(name);
    // clang-format off
    if (symbol->is_array()) {
        if (use_instance) {
            return "(inst->{}+id*{})"_format(name, dimension);
        }
        return "(data + {}*pnodecount + id*{})"_format(position, dimension);
    }
    if (use_instance) {
        return "inst->{}[id]"_format(name);
    }
    return "data[{}*pnodecount + id]"_format(position);
    // clang-format on
}


std::string CodegenCVisitor::int_variable_name(const IndexVariableInfo& symbol,
                                               const std::string& name,
                                               bool use_instance) const {
    auto position = position_of_int_var(name);
    // clang-format off
    if (symbol.is_index) {
        if (use_instance) {
            return "inst->{}[{}]"_format(name, position);
        }
        return "indexes[{}]"_format(position);
    }
    if (symbol.is_integer) {
        if (use_instance) {
            return "inst->{}[{}*pnodecount+id]"_format(name, position);
        }
        return "indexes[{}*pnodecount+id]"_format(position);
    }
    if (use_instance) {
        return "inst->{}[indexes[{}*pnodecount + id]]"_format(name, position);
    }
    auto data = symbol.is_vdata ? "_vdata" : "_data";
    return "nt->{}[indexes[{}*pnodecount + id]]"_format(data, position);
    // clang-format on
}


std::string CodegenCVisitor::global_variable_name(const SymbolType& symbol) const {
    return "{}_global.{}"_format(info.mod_suffix, symbol->get_name());
}


std::string CodegenCVisitor::ion_shadow_variable_name(const SymbolType& symbol) const {
    return "inst->{}[id]"_format(symbol->get_name());
}


std::string CodegenCVisitor::update_if_ion_variable_name(const std::string& name) const {
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


std::string CodegenCVisitor::get_variable_name(const std::string& name, bool use_instance) const {
    std::string varname = update_if_ion_variable_name(name);

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
        return global_variable_name(*g);
    }

    // shadow variable
    auto s = std::find_if(codegen_shadow_variables.begin(),
                          codegen_shadow_variables.end(),
                          symbol_comparator);
    if (s != codegen_shadow_variables.end()) {
        return ion_shadow_variable_name(*s);
    }

    if (varname == naming::NTHREAD_DT_VARIABLE) {
        return std::string("nt->_") + naming::NTHREAD_DT_VARIABLE;
    }

    // t in net_receive method is an argument to function and hence it should
    // ne used instead of nt->_t which is current time of thread
    if (varname == naming::NTHREAD_T_VARIABLE && !printing_net_receive) {
        return std::string("nt->_") + naming::NTHREAD_T_VARIABLE;
    }

    // otherwise return original name
    return varname;
}


/****************************************************************************************/
/*                      Main printing routines for code generation                      */
/****************************************************************************************/


void CodegenCVisitor::print_backend_info() {
    time_t tr;
    time(&tr);
    auto date = std::string(asctime(localtime(&tr)));
    auto version = nmodl::Version::NMODL_VERSION + " [" + nmodl::Version::GIT_REVISION + "]";

    printer->add_line("/*********************************************************");
    printer->add_line("Model Name      : {}"_format(info.mod_suffix));
    printer->add_line("Filename        : {}"_format(info.mod_file + ".mod"));
    printer->add_line("NMODL Version   : {}"_format(nmodl_version()));
    printer->add_line("Vectorized      : {}"_format(info.vectorize));
    printer->add_line("Threadsafe      : {}"_format(info.thread_safe));
    printer->add_line("Created         : {}"_format(stringutils::trim(date)));
    printer->add_line("Backend         : {}"_format(backend_name()));
    printer->add_line("NMODL Compiler  : {}"_format(version));
    printer->add_line("*********************************************************/");
}


void CodegenCVisitor::print_standard_includes() {
    printer->add_newline();
    printer->add_line("#include <math.h>");
    printer->add_line("#include \"nmodl/fast_math.hpp\" // extend math with some useful functions");
    printer->add_line("#include <stdio.h>");
    printer->add_line("#include <stdlib.h>");
    printer->add_line("#include <string.h>");
}


void CodegenCVisitor::print_coreneuron_includes() {
    printer->add_newline();
    printer->add_line("#include <coreneuron/mechanism/mech/cfile/scoplib.h>");
    printer->add_line("#include <coreneuron/nrnconf.h>");
    printer->add_line("#include <coreneuron/sim/multicore.hpp>");
    printer->add_line("#include <coreneuron/mechanism/register_mech.hpp>");
    printer->add_line("#include <coreneuron/gpu/nrn_acc_manager.hpp>");
    printer->add_line("#include <coreneuron/utils/randoms/nrnran123.h>");
    printer->add_line("#include <coreneuron/nrniv/nrniv_decl.h>");
    printer->add_line("#include <coreneuron/utils/ivocvect.hpp>");
    printer->add_line("#include <coreneuron/utils/nrnoc_aux.hpp>");
    printer->add_line("#include <coreneuron/mechanism/mech/mod2c_core_thread.hpp>");
    printer->add_line("#include <coreneuron/sim/scopmath/newton_struct.h>");
    printer->add_line("#include \"_kinderiv.h\"");
    if (info.eigen_newton_solver_exist) {
        printer->add_line("#include <newton/newton.hpp>");
    }
    if (info.eigen_linear_solver_exist) {
        printer->add_line("#include <Eigen/LU>");
    }
}


/**
 * \details Variables required for type of ion, type of point process etc. are
 * of static int type. For any backend type (C,C++), it's ok to have
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
void CodegenCVisitor::print_mechanism_global_var_structure() {
    const auto qualifier = global_var_struct_type_qualifier();

    auto float_type = default_float_data_type();
    printer->add_newline(2);
    printer->add_line("/** all global variables */");
    printer->add_line("struct {} {}"_format(global_struct(), "{"));
    printer->increase_indent();

    if (!info.ions.empty()) {
        for (const auto& ion: info.ions) {
            auto name = "{}_type"_format(ion.name);
            printer->add_line("{}int {};"_format(qualifier, name));
            codegen_global_variables.push_back(make_symbol(name));
        }
    }

    if (info.point_process) {
        printer->add_line("{}int point_type;"_format(qualifier));
        codegen_global_variables.push_back(make_symbol("point_type"));
    }

    if (!info.state_vars.empty()) {
        for (const auto& var: info.state_vars) {
            auto name = var->get_name() + "0";
            auto symbol = program_symtab->lookup(name);
            if (symbol == nullptr) {
                printer->add_line("{}{} {};"_format(qualifier, float_type, name));
                codegen_global_variables.push_back(make_symbol(name));
            }
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
                printer->add_line("{}{} {}[{}];"_format(qualifier, float_type, name, length));
            } else {
                printer->add_line("{}{} {};"_format(qualifier, float_type, name));
            }
            codegen_global_variables.push_back(var);
        }
    }

    if (!info.thread_variables.empty()) {
        printer->add_line("{}int thread_data_in_use;"_format(qualifier));
        printer->add_line(
            "{}{} thread_data[{}];"_format(qualifier, float_type, info.thread_var_data_size));
        codegen_global_variables.push_back(make_symbol("thread_data_in_use"));
        auto symbol = make_symbol("thread_data");
        symbol->set_as_array(info.thread_var_data_size);
        codegen_global_variables.push_back(symbol);
    }

    printer->add_line("{}int reset;"_format(qualifier));
    codegen_global_variables.push_back(make_symbol("reset"));

    printer->add_line("{}int mech_type;"_format(qualifier));
    codegen_global_variables.push_back(make_symbol("mech_type"));

    auto& globals = info.global_variables;
    auto& constants = info.constant_variables;

    if (!globals.empty()) {
        for (const auto& var: globals) {
            auto name = var->get_name();
            auto length = var->get_length();
            if (var->is_array()) {
                printer->add_line("{}{} {}[{}];"_format(qualifier, float_type, name, length));
            } else {
                printer->add_line("{}{} {};"_format(qualifier, float_type, name));
            }
            codegen_global_variables.push_back(var);
        }
    }

    if (!constants.empty()) {
        for (const auto& var: constants) {
            auto name = var->get_name();
            auto value_ptr = var->get_value();
            printer->add_line("{}{} {};"_format(qualifier, float_type, name));
            codegen_global_variables.push_back(var);
        }
    }

    if (info.primes_size != 0) {
        printer->add_line("int* {}slist1;"_format(qualifier));
        printer->add_line("int* {}dlist1;"_format(qualifier));
        codegen_global_variables.push_back(make_symbol("slist1"));
        codegen_global_variables.push_back(make_symbol("dlist1"));
        if (info.derivimplicit_used()) {
            printer->add_line("int* {}slist2;"_format(qualifier));
            codegen_global_variables.push_back(make_symbol("slist2"));
        }
    }

    if (info.table_count > 0) {
        printer->add_line("{}double usetable;"_format(qualifier));
        codegen_global_variables.push_back(make_symbol(naming::USE_TABLE_VARIABLE));

        for (const auto& block: info.functions_with_table) {
            auto name = block->get_node_name();
            printer->add_line("{}{} tmin_{};"_format(qualifier, float_type, name));
            printer->add_line("{}{} mfac_{};"_format(qualifier, float_type, name));
            codegen_global_variables.push_back(make_symbol("tmin_" + name));
            codegen_global_variables.push_back(make_symbol("mfac_" + name));
        }

        for (const auto& variable: info.table_statement_variables) {
            auto name = "t_" + variable->get_name();
            printer->add_line("{}* {}{};"_format(float_type, qualifier, name));
            codegen_global_variables.push_back(make_symbol(name));
        }
    }

    if (info.vectorize) {
        printer->add_line("ThreadDatum* {}ext_call_thread;"_format(qualifier));
        codegen_global_variables.push_back(make_symbol("ext_call_thread"));
    }

    printer->decrease_indent();
    printer->add_line("};");

    printer->add_newline(1);
    printer->add_line("/** holds object of global variable */");
    print_global_variable_device_create_annotation_pre();
    print_global_var_struct_decl();

    // create copy on the device
    print_global_variable_device_create_annotation_post();
}


void CodegenCVisitor::print_prcellstate_macros() const {
    printer->add_line("#ifndef NRN_PRCELLSTATE");
    printer->add_line("#define NRN_PRCELLSTATE 0");
    printer->add_line("#endif");
}


void CodegenCVisitor::print_mechanism_info() {
    auto variable_printer = [&](std::vector<SymbolType>& variables) {
        for (const auto& v: variables) {
            auto name = v->get_name();
            if (!info.point_process) {
                name += "_" + info.mod_suffix;
            }
            if (v->is_array()) {
                name += "[{}]"_format(v->get_length());
            }
            printer->add_line(add_escape_quote(name) + ",");
        }
    };

    printer->add_newline(2);
    printer->add_line("/** channel information */");
    printer->add_line("static const char *mechanism[] = {");
    printer->increase_indent();
    printer->add_line(add_escape_quote(nmodl_version()) + ",");
    printer->add_line(add_escape_quote(info.mod_suffix) + ",");
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
void CodegenCVisitor::print_global_variables_for_hoc() {
    auto variable_printer =
        [&](const std::vector<SymbolType>& variables, bool if_array, bool if_vector) {
            for (const auto& variable: variables) {
                if (variable->is_array() == if_array) {
                    auto name = get_variable_name(variable->get_name());
                    auto ename = add_escape_quote(variable->get_name() + "_" + info.mod_suffix);
                    auto length = variable->get_length();
                    if (if_vector) {
                        printer->add_line("{}, {}, {},"_format(ename, name, length));
                    } else {
                        printer->add_line("{}, &{},"_format(ename, name));
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
    printer->add_line("0, 0");
    printer->decrease_indent();
    printer->add_line("};");

    printer->add_newline(2);
    printer->add_line("/** connect global (array) variables to hoc -- */");
    printer->add_line("static DoubVec hoc_vector_double[] = {");
    printer->increase_indent();
    variable_printer(globals, true, true);
    variable_printer(thread_vars, true, true);
    printer->add_line("0, 0, 0");
    printer->decrease_indent();
    printer->add_line("};");
}


/**
 * \details Every mod file has register function to connect with the simulator.
 * Various information about mechanism and callbacks get registered with
 * the simulator using suffix_reg() function.
 *
 * Here are details:
 *  - setup_global_variables function used to create vectors necessary for specific
 *    solvers like euler and derivimplicit. All global variables are initialized as well.
 *    We should exclude that callback based on the solver, watch statements.
 *  - If nrn_get_mechtype is < -1 means that mechanism is not used in the
 *    context of neuron execution and hence could be ignored in coreneuron
 *    execution.
 *  - Ions are internally defined and their types can be queried similar to
 *    other mechanisms.
 *  - hoc_register_var may not be needed in the context of coreneuron
 *  - We assume net receive buffer is on. This is because generated code is
 *    compatible for cpu as well as gpu target.
 */
void CodegenCVisitor::print_mechanism_register() {
    printer->add_newline(2);
    printer->add_line("/** register channel with the simulator */");
    printer->start_block("void _{}_reg() "_format(info.mod_file));

    // type related information
    auto mech_type = get_variable_name("mech_type");
    auto suffix = add_escape_quote(info.mod_suffix);
    printer->add_newline();
    printer->add_line("int mech_type = nrn_get_mechtype({});"_format(suffix));
    printer->add_line("{} = mech_type;"_format(mech_type));
    printer->add_line("if (mech_type == -1) {");
    printer->add_line("    return;");
    printer->add_line("}");

    printer->add_newline();
    printer->add_line("_nrn_layout_reg(mech_type, 0);");  // 0 for SoA

    // register mechanism
    auto args = register_mechanism_arguments();
    auto nobjects = num_thread_objects();
    if (info.point_process) {
        printer->add_line("point_register_mech({}, {}, {}, {});"_format(
            args,
            info.constructor_node ? method_name(naming::NRN_CONSTRUCTOR_METHOD) : "NULL",
            info.destructor_node ? method_name(naming::NRN_DESTRUCTOR_METHOD) : "NULL",
            nobjects));
    } else {
        printer->add_line("register_mech({}, {});"_format(args, nobjects));
        if (info.constructor_node) {
            printer->add_line(
                "register_constructor({});"_format(method_name(naming::NRN_CONSTRUCTOR_METHOD)));
        }
    }

    // types for ion
    for (const auto& ion: info.ions) {
        const auto& type = get_variable_name(ion.name + "_type");
        const auto& name = add_escape_quote(ion.name + "_ion");
        printer->add_line(type + " = nrn_get_mechtype(" + name + ");");
    }
    printer->add_newline();

    // allocate global variables
    printer->add_line("setup_global_variables();");

    /*
     *  If threads are used then memory is allocated in setup_global_variables.
     *  Register callbacks for thread allocation and cleanup. Note that thread_data_index
     *  represent total number of thread used minus 1 (i.e. index of last thread).
     */
    if (info.vectorize && (info.thread_data_index != 0)) {
        auto name = get_variable_name("ext_call_thread");
        printer->add_line("thread_mem_init({});"_format(name));
    }

    if (!info.thread_variables.empty()) {
        printer->add_line("{} = 0;"_format(get_variable_name("thread_data_in_use")));
    }

    if (info.thread_callback_register) {
        printer->add_line("_nrn_thread_reg0(mech_type, thread_mem_cleanup);");
        printer->add_line("_nrn_thread_reg1(mech_type, thread_mem_init);");
    }

    if (info.emit_table_thread()) {
        auto name = method_name("check_table_thread");
        printer->add_line("_nrn_thread_table_reg(mech_type, {});"_format(name));
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
        auto args = "mech_type, {}, {}"_format(semantic.index, add_escape_quote(semantic.name));
        printer->add_line("hoc_register_dparam_semantics({});"_format(args));
    }

    if (info.is_watch_used()) {
        auto watch_fun = compute_method_name(BlockType::Watch);
        printer->add_line("hoc_register_watch_check({}, mech_type);"_format(watch_fun));
    }

    if (info.write_concentration) {
        printer->add_line("nrn_writes_conc(mech_type, 0);");
    }

    // register various information for point process type
    if (info.net_event_used) {
        printer->add_line("add_nrn_has_net_event(mech_type);");
    }
    if (info.artificial_cell) {
        printer->add_line("add_nrn_artcell(mech_type, {});"_format(info.tqitem_index));
    }
    if (net_receive_buffering_required()) {
        printer->add_line("hoc_register_net_receive_buffering({}, mech_type);"_format(
            method_name("net_buf_receive")));
    }
    if (info.num_net_receive_parameters != 0) {
        auto net_recv_init_arg = "nullptr";
        if (info.net_receive_initial_node != nullptr) {
            net_recv_init_arg = "net_init";
        }
        auto pnt_recline = "set_pnt_receive(mech_type, {}, {}, num_net_receive_args());"_format(
            method_name("net_receive"), net_recv_init_arg);
        printer->add_line(pnt_recline);
    }
    if (info.for_netcon_used) {
        // index where information about FOR_NETCON is stored in the integer array
        const auto index =
            std::find_if(info.semantics.begin(), info.semantics.end(), [](const IndexSemantics& a) {
                return a.name == naming::FOR_NETCON_SEMANTIC;
            })->index;
        printer->add_line("add_nrn_fornetcons(mech_type, {});"_format(index));
    }

    if (info.net_event_used || info.net_send_used) {
        printer->add_line("hoc_register_net_send_buffering(mech_type);");
    }

    // register variables for hoc
    printer->add_line("hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);");
    printer->end_block(1);
}


void CodegenCVisitor::print_thread_memory_callbacks() {
    if (!info.thread_callback_register) {
        return;
    }

    // thread_mem_init callback
    printer->add_newline(2);
    printer->add_line("/** thread memory allocation callback */");
    printer->start_block("static void thread_mem_init(ThreadDatum* thread) ");

    if (info.vectorize && info.derivimplicit_used()) {
        printer->add_line("thread[dith{}()].pval = NULL;"_format(info.derivimplicit_list_num));
    }
    if (info.vectorize && (info.top_local_thread_size != 0)) {
        auto length = info.top_local_thread_size;
        auto allocation = "(double*)mem_alloc({}, sizeof(double))"_format(length);
        auto line = "thread[top_local_var_tid()].pval = {};"_format(allocation);
        printer->add_line(line);
    }
    if (info.thread_var_data_size != 0) {
        auto length = info.thread_var_data_size;
        auto thread_data = get_variable_name("thread_data");
        auto thread_data_in_use = get_variable_name("thread_data_in_use");
        auto allocation = "(double*)mem_alloc({}, sizeof(double))"_format(length);
        printer->add_line("if ({}) {}"_format(thread_data_in_use, "{"));
        printer->add_line("    thread[thread_var_tid()].pval = {};"_format(allocation));
        printer->add_line("} else {");
        printer->add_line("    thread[thread_var_tid()].pval = {};"_format(thread_data));
        printer->add_line("    {} = 1;"_format(thread_data_in_use));
        printer->add_line("}");
    }
    printer->end_block(3);


    // thread_mem_cleanup callback
    printer->add_line("/** thread memory cleanup callback */");
    printer->start_block("static void thread_mem_cleanup(ThreadDatum* thread) ");

    // clang-format off
    if (info.vectorize && info.derivimplicit_used()) {
        int n = info.derivimplicit_list_num;
        printer->add_line("free(thread[dith{}()].pval);"_format(n));
        printer->add_line("nrn_destroy_newtonspace(static_cast<NewtonSpace*>(*newtonspace{}(thread)));"_format(n));
    }
    // clang-format on

    if (info.top_local_thread_size != 0) {
        auto line = "free(thread[top_local_var_tid()].pval);";
        printer->add_line(line);
    }
    if (info.thread_var_data_size != 0) {
        auto thread_data = get_variable_name("thread_data");
        auto thread_data_in_use = get_variable_name("thread_data_in_use");
        printer->add_line("if (thread[thread_var_tid()].pval == {}) {}"_format(thread_data, "{"));
        printer->add_line("    {} = 0;"_format(thread_data_in_use));
        printer->add_line("} else {");
        printer->add_line("    free(thread[thread_var_tid()].pval);");
        printer->add_line("}");
    }
    printer->end_block(1);
}


void CodegenCVisitor::print_mechanism_range_var_structure() {
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


void CodegenCVisitor::print_ion_var_structure() {
    if (!ion_variable_struct_required()) {
        return;
    }
    printer->add_newline(2);
    printer->add_line("/** ion write variables */");
    printer->start_block("struct IonCurVar");

    std::string float_type = default_float_data_type();
    std::vector<std::string> members;

    for (auto& ion: info.ions) {
        for (auto& var: ion.writes) {
            printer->add_line("{} {};"_format(float_type, var));
            members.push_back(var);
        }
    }
    for (auto& var: info.currents) {
        if (!info.is_ion_variable(var)) {
            printer->add_line("{} {};"_format(float_type, var));
            members.push_back(var);
        }
    }

    print_ion_var_constructor(members);

    printer->end_block();
    printer->add_text(";");
    printer->add_newline();
}


void CodegenCVisitor::print_ion_var_constructor(const std::vector<std::string>& members) {
    // constructor
    printer->add_newline();
    printer->add_line("IonCurVar() : ", 0);
    for (int i = 0; i < members.size(); i++) {
        printer->add_text("{}(0)"_format(members[i]));
        if (i + 1 < members.size()) {
            printer->add_text(", ");
        }
    }
    printer->add_text(" {}");
    printer->add_newline();
}


void CodegenCVisitor::print_ion_variable() {
    printer->add_line("IonCurVar ionvar;");
}


void CodegenCVisitor::print_global_variable_device_create_annotation_pre() {
    // nothing for cpu
}

void CodegenCVisitor::print_global_variable_device_create_annotation_post() {
    // nothing for cpu
}


void CodegenCVisitor::print_global_variable_device_update_annotation() {
    // nothing for cpu
}


void CodegenCVisitor::print_global_variable_setup() {
    std::vector<std::string> allocated_variables;

    printer->add_newline(2);
    printer->add_line("/** initialize global variables */");
    printer->start_block("static inline void setup_global_variables() ");

    printer->add_line("static int setup_done = 0;");
    printer->add_line("if (setup_done) {");
    printer->add_line("    return;");
    printer->add_line("}");

    // offsets for state variables
    if (info.primes_size != 0) {
        if (info.primes_size != info.prime_variables_by_order.size()) {
            throw std::runtime_error{
                "primes_size = {} differs from prime_variables_by_order.size() = {}, this should not happen."_format(
                    info.primes_size, info.prime_variables_by_order.size())};
        }
        auto slist1 = get_variable_name("slist1");
        auto dlist1 = get_variable_name("dlist1");
        auto n = info.primes_size;
        printer->add_line("{} = (int*) mem_alloc({}, sizeof(int));"_format(slist1, n));
        printer->add_line("{} = (int*) mem_alloc({}, sizeof(int));"_format(dlist1, n));
        allocated_variables.push_back(slist1);
        allocated_variables.push_back(dlist1);

        int id = 0;
        for (auto& prime: info.prime_variables_by_order) {
            auto name = prime->get_name();
            printer->add_line("{}[{}] = {};"_format(slist1, id, position_of_float_var(name)));
            printer->add_line("{}[{}] = {};"_format(dlist1, id, position_of_float_var("D" + name)));
            id++;
        }
    }

    // additional list for derivimplicit method
    if (info.derivimplicit_used()) {
        auto primes = program_symtab->get_variables_with_properties(NmodlType::prime_name);
        auto slist2 = get_variable_name("slist2");
        auto nprimes = info.primes_size;
        printer->add_line("{} = (int*) mem_alloc({}, sizeof(int));"_format(slist2, nprimes));
        int id = 0;
        for (auto& variable: primes) {
            auto name = variable->get_name();
            printer->add_line("{}[{}] = {};"_format(slist2, id, position_of_float_var(name)));
            id++;
        }
        allocated_variables.push_back(slist2);
    }

    // memory for thread member
    if (info.vectorize && (info.thread_data_index != 0)) {
        auto n = info.thread_data_index;
        auto alloc = "(ThreadDatum*) mem_alloc({}, sizeof(ThreadDatum))"_format(n);
        auto name = get_variable_name("ext_call_thread");
        printer->add_line("{} = {};"_format(name, alloc));
        allocated_variables.push_back(name);
    }

    // initialize global variables
    for (auto& var: info.state_vars) {
        auto name = var->get_name() + "0";
        auto symbol = program_symtab->lookup(name);
        if (symbol == nullptr) {
            auto global_name = get_variable_name(name);
            printer->add_line("{} = 0.0;"_format(global_name));
        }
    }

    // note : v is not needed in global structure for nmodl even if vectorize is false

    if (!info.thread_variables.empty()) {
        printer->add_line("{} = 0;"_format(get_variable_name("thread_data_in_use")));
    }

    // initialize global variables
    for (auto& var: info.global_variables) {
        if (!var->is_array()) {
            auto name = get_variable_name(var->get_name());
            double value = 0;
            auto value_ptr = var->get_value();
            if (value_ptr != nullptr) {
                value = *value_ptr;
            }
            /// use %g to be same as nocmodl in neuron
            printer->add_line("{} = {:g};"_format(name, value));
        }
    }

    // initialize constant variables
    for (auto& var: info.constant_variables) {
        auto name = get_variable_name(var->get_name());
        auto value_ptr = var->get_value();
        double value = 0;
        if (value_ptr != nullptr) {
            value = *value_ptr;
        }
        /// use %g to be same as nocmodl in neuron
        printer->add_line("{} = {:g};"_format(name, value));
    }

    if (info.table_count > 0) {
        auto name = get_variable_name(naming::USE_TABLE_VARIABLE);
        printer->add_line("{} = 1;"_format(name));

        for (auto& variable: info.table_statement_variables) {
            auto name = get_variable_name("t_" + variable->get_name());
            int num_values = variable->get_num_values();
            printer->add_line(
                "{} = (double*) mem_alloc({}, sizeof(double));"_format(name, num_values));
        }
    }

    // update device copy
    print_global_variable_device_update_annotation();

    printer->add_newline();
    printer->add_line("setup_done = 1;");
    printer->end_block(3);

    printer->add_line("/** free global variables */");
    printer->start_block("static inline void free_global_variables() ");
    if (allocated_variables.empty()) {
        printer->add_line("// do nothing");
    } else {
        for (auto& var: allocated_variables) {
            printer->add_line("mem_free({});"_format(var));
        }
    }
    printer->end_block(1);
}


void CodegenCVisitor::print_shadow_vector_setup() {
    printer->add_newline(2);
    printer->add_line("/** allocate and initialize shadow vector */");
    auto args = "{}* inst, Memb_list* ml"_format(instance_struct());
    printer->start_block("static inline void setup_shadow_vectors({}) "_format(args));
    if (channel_task_dependency_enabled()) {
        printer->add_line("int nodecount = ml->nodecount;");
        for (auto& var: codegen_shadow_variables) {
            auto name = var->get_name();
            auto type = default_float_data_type();
            auto allocation = "({0}*) mem_alloc(nodecount, sizeof({0}))"_format(type);
            printer->add_line("inst->{0} = {1};"_format(name, allocation));
        }
    }
    printer->end_block(3);

    printer->add_line("/** free shadow vector */");
    args = "{}* inst"_format(instance_struct());
    printer->start_block("static inline void free_shadow_vectors({}) "_format(args));
    if (channel_task_dependency_enabled()) {
        for (auto& var: codegen_shadow_variables) {
            auto name = var->get_name();
            printer->add_line("mem_free(inst->{});"_format(name));
        }
    }
    printer->end_block(1);
}


void CodegenCVisitor::print_setup_range_variable() {
    auto type = float_data_type();
    printer->add_newline(2);
    printer->add_line("/** allocate and setup array for range variable */");
    printer->start_block(
        "static inline {}* setup_range_variable(double* variable, int n) "_format(type));
    printer->add_line("{0}* data = ({0}*) mem_alloc(n, sizeof({0}));"_format(type));
    printer->add_line("for(size_t i = 0; i < n; i++) {");
    printer->add_line("    data[i] = variable[i];");
    printer->add_line("}");
    printer->add_line("return data;");
    printer->end_block(1);
}


/**
 * \details If floating point type like "float" is specified on command line then
 * we can't turn all variables to new type. This is because certain variables
 * are pointers to internal variables (e.g. ions). Hence, we check if given
 * variable can be safely converted to new type. If so, return new type.
 */
std::string CodegenCVisitor::get_range_var_float_type(const SymbolType& symbol) {
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

/**
 * \details For CPU/Host target there is no device pointer. In this case
 * just use the host variable name directly.
 */
std::string CodegenCVisitor::get_variable_device_pointer(const std::string& variable,
                                                         const std::string& /*type*/) const {
    return variable;
}


void CodegenCVisitor::print_instance_variable_setup() {
    if (range_variable_setup_required()) {
        print_setup_range_variable();
    }

    if (shadow_vector_setup_required()) {
        print_shadow_vector_setup();
    }
    printer->add_newline(2);
    printer->add_line("/** initialize mechanism instance variables */");
    printer->start_block("static inline void setup_instance(NrnThread* nt, Memb_list* ml) ");
    printer->add_line("{0}* inst = ({0}*) mem_alloc(1, sizeof({0}));"_format(instance_struct()));
    if (channel_task_dependency_enabled() && !codegen_shadow_variables.empty()) {
        printer->add_line("setup_shadow_vectors(inst, ml);");
    }

    std::string stride;
    printer->add_line("int pnodecount = ml->_nodecount_padded;");
    stride = "*pnodecount";

    printer->add_line("Datum* indexes = ml->pdata;");

    std::string float_type = default_float_data_type();
    std::string int_type = default_int_data_type();
    std::string float_type_pointer = float_type + "*";
    std::string int_type_pointer = int_type + "*";

    int id = 0;
    std::vector<std::string> variables_to_free;

    for (auto& var: codegen_float_variables) {
        auto name = var->get_name();
        auto range_var_type = get_range_var_float_type(var);
        if (float_type == range_var_type) {
            auto variable = "ml->data+{}{}"_format(id, stride);
            auto device_variable = get_variable_device_pointer(variable, float_type_pointer);
            printer->add_line("inst->{} = {};"_format(name, device_variable));
        } else {
            printer->add_line("inst->{} = setup_range_variable(ml->data+{}{}, pnodecount);"_format(
                name, id, stride));
            variables_to_free.push_back(name);
        }
        id += var->get_length();
    }

    for (auto& var: codegen_int_variables) {
        auto name = var.symbol->get_name();
        std::string variable = name;
        std::string type = "";
        if (var.is_index || var.is_integer) {
            variable = "ml->pdata";
            type = int_type_pointer;
        } else if (var.is_vdata) {
            variable = "nt->_vdata";
            type = "void**";
        } else {
            variable = "nt->_data";
            type = info.artificial_cell ? "void*" : float_type_pointer;
        }
        auto device_variable = get_variable_device_pointer(variable, type);
        printer->add_line("inst->{} = {};"_format(name, device_variable));
    }

    printer->add_line("ml->instance = (void*) inst;");
    print_instance_variable_transfer_to_device();
    printer->end_block(3);

    printer->add_line("/** cleanup mechanism instance variables */");
    printer->start_block("static inline void cleanup_instance(Memb_list* ml) ");
    printer->add_line("{0}* inst = ({0}*) ml->instance;"_format(instance_struct()));
    if (range_variable_setup_required()) {
        for (auto& var: variables_to_free) {
            printer->add_line("mem_free((void*)inst->{});"_format(var));
        }
    }
    printer->add_line("mem_free((void*)inst);");
    printer->end_block(1);
}


void CodegenCVisitor::print_initial_block(const InitialBlock* node) {
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
            printer->add_line("{} = {};"_format(lhs, rhs));
        }
    }

    // initial block
    if (node != nullptr) {
        const auto& block = node->get_statement_block();
        print_statement_block(*block.get(), false, false);
    }

    // write ion statements
    auto write_statements = ion_write_statements(BlockType::Initial);
    for (auto& statement: write_statements) {
        auto text = process_shadow_update_statement(statement, BlockType::Initial);
        printer->add_line(text);
    }
}


void CodegenCVisitor::print_global_function_common_code(BlockType type) {
    std::string method = compute_method_name(type);
    auto args = "NrnThread* nt, Memb_list* ml, int type";

    // watch statement function doesn't have type argument
    if (type == BlockType::Watch) {
        args = "NrnThread* nt, Memb_list* ml";
    }

    print_global_method_annotation();
    printer->start_block("void {}({})"_format(method, args));
    if (type != BlockType::Destructor && type != BlockType::Constructor) {
        // We do not (currently) support DESTRUCTOR and CONSTRUCTOR blocks
        // running anything on the GPU.
        print_kernel_data_present_annotation_block_begin();
    } else {
        /// TODO: Remove this when the code generation is propery done
        /// Related to https://github.com/BlueBrain/nmodl/issues/692
        printer->add_line("#ifndef CORENEURON_BUILD");
    }
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


void CodegenCVisitor::print_nrn_init(bool skip_init_check) {
    codegen = true;
    printer->add_newline(2);
    printer->add_line("/** initialize channel */");

    print_global_function_common_code(BlockType::Initial);
    if (info.derivimplicit_used()) {
        printer->add_newline();
        int nequation = info.num_equations;
        int list_num = info.derivimplicit_list_num;
        // clang-format off
        printer->add_line("int& deriv_advance_flag = *deriv{}_advance(thread);"_format(list_num));
        printer->add_line("deriv_advance_flag = 0;");
        print_deriv_advance_flag_transfer_to_device();
        printer->add_line("auto ns = newtonspace{}(thread);"_format(list_num));
        printer->add_line("auto& th = thread[dith{}()];"_format(list_num));

        printer->add_line("if (*ns == nullptr) {");
        printer->add_line("    int vec_size = 2*{}*pnodecount*sizeof(double);"_format(nequation));
        printer->add_line("    double* vec = makevector(vec_size);"_format(nequation));
        printer->add_line("    th.pval = vec;"_format(list_num));
        printer->add_line("    *ns = nrn_cons_newtonspace({}, pnodecount);"_format(nequation));
        print_newtonspace_transfer_to_device();
        printer->add_line("}");
        // clang-format on
    }

    // update global variable as those might be updated via python/hoc API
    print_global_variable_device_update_annotation();

    if (skip_init_check) {
        printer->start_block("if (_nrn_skip_initmodel == 0)");
    }

    if (!info.changed_dt.empty()) {
        printer->add_line(
            "double _save_prev_dt = {};"_format(get_variable_name(naming::NTHREAD_DT_VARIABLE)));
        printer->add_line(
            "{} = {};"_format(get_variable_name(naming::NTHREAD_DT_VARIABLE), info.changed_dt));
        print_dt_update_to_device();
    }

    print_channel_iteration_tiling_block_begin(BlockType::Initial);
    print_channel_iteration_block_begin(BlockType::Initial);

    if (info.net_receive_node != nullptr) {
        printer->add_line("{} = -1e20;"_format(get_variable_name("tsave")));
    }

    print_initial_block(info.initial_node);
    print_channel_iteration_block_end();
    print_shadow_reduction_statements();
    print_channel_iteration_tiling_block_end();

    if (!info.changed_dt.empty()) {
        printer->add_line(
            "{} = _save_prev_dt;"_format(get_variable_name(naming::NTHREAD_DT_VARIABLE)));
        print_dt_update_to_device();
    }

    printer->end_block(1);

    if (info.derivimplicit_used()) {
        printer->add_line("deriv_advance_flag = 1;");
        print_deriv_advance_flag_transfer_to_device();
    }

    if (info.net_send_used && !info.artificial_cell) {
        print_send_event_move();
    }

    print_kernel_data_present_annotation_block_end();
    if (skip_init_check) {
        printer->end_block(1);
    }
    codegen = false;
}


void CodegenCVisitor::print_nrn_constructor() {
    printer->add_newline(2);
    print_global_function_common_code(BlockType::Constructor);
    if (info.constructor_node != nullptr) {
        const auto& block = info.constructor_node->get_statement_block();
        print_statement_block(*block.get(), false, false);
    }
    printer->add_line("#endif");
    printer->end_block(1);
}


void CodegenCVisitor::print_nrn_destructor() {
    printer->add_newline(2);
    print_global_function_common_code(BlockType::Destructor);
    if (info.destructor_node != nullptr) {
        const auto& block = info.destructor_node->get_statement_block();
        print_statement_block(*block.get(), false, false);
    }
    printer->add_line("#endif");
    printer->end_block(1);
}


void CodegenCVisitor::print_nrn_alloc() {
    printer->add_newline(2);
    auto method = method_name(naming::NRN_ALLOC_METHOD);
    printer->start_block("static void {}(double* data, Datum* indexes, int type) "_format(method));
    printer->add_line("// do nothing");
    printer->end_block(1);
}

/**
 * \todo Number of watch could be more than number of statements
 * according to grammar. Check if this is correctly handled in neuron
 * and coreneuron.
 */
void CodegenCVisitor::print_watch_activate() {
    if (info.watch_statements.empty()) {
        return;
    }
    codegen = true;
    printer->add_newline(2);
    auto inst = "{}* inst"_format(instance_struct());

    printer->start_block(
        "static void nrn_watch_activate({}, int id, int pnodecount, int watch_id, double v, bool &watch_remove) "_format(
            inst));

    // initialize all variables only during first watch statement
    printer->add_line("if (watch_remove == false) {");
    for (int i = 0; i < info.watch_count; i++) {
        auto name = get_variable_name("watch{}"_format(i + 1));
        printer->add_line("    {} = 0;"_format(name));
    }
    printer->add_line("    watch_remove = true;");
    printer->add_line("}");

    /**
     * \todo Similar to neuron/coreneuron we are using
     * first watch and ignoring rest.
     */
    for (int i = 0; i < info.watch_statements.size(); i++) {
        auto statement = info.watch_statements[i];
        printer->start_block("if (watch_id == {})"_format(i));

        auto varname = get_variable_name("watch{}"_format(i + 1));
        printer->add_indent();
        printer->add_text("{} = 2 + ("_format(varname));
        auto watch = statement->get_statements().front();
        watch->get_expression()->visit_children(*this);
        printer->add_text(");");
        printer->add_newline();

        printer->end_block(1);
    }
    printer->end_block(1);
    codegen = false;
}


/**
 * \todo Similar to print_watch_activate, we are using only
 * first watch. need to verify with neuron/coreneuron about rest.
 */
void CodegenCVisitor::print_watch_check() {
    if (info.watch_statements.empty()) {
        return;
    }
    codegen = true;
    printer->add_newline(2);
    printer->add_line("/** routine to check watch activation */");
    print_global_function_common_code(BlockType::Watch);
    print_channel_iteration_tiling_block_begin(BlockType::Watch);
    print_channel_iteration_block_begin(BlockType::Watch);

    if (info.is_voltage_used_by_watch_statements()) {
        printer->add_line("int node_id = node_index[id];");
        printer->add_line("double v = voltage[node_id];");
        print_v_unused();
    }

    // flat to make sure only one WATCH statement can be triggered at a time
    printer->add_line("bool watch_untriggered = true;");

    for (int i = 0; i < info.watch_statements.size(); i++) {
        auto statement = info.watch_statements[i];
        auto watch = statement->get_statements().front();
        auto varname = get_variable_name("watch{}"_format(i + 1));

        // start block 1
        printer->start_block("if ({}&2 && watch_untriggered)"_format(varname));

        // start block 2
        printer->add_indent();
        printer->add_text("if (");
        watch->get_expression()->accept(*this);
        printer->add_text(") {");
        printer->add_newline();
        printer->increase_indent();

        // start block 3
        printer->start_block("if (({}&1) == 0)"_format(varname));

        printer->add_line("watch_untriggered = false;");

        auto tqitem = get_variable_name("tqitem");
        auto point_process = get_variable_name("point_process");
        printer->add_indent();
        printer->add_text("net_send_buffering(");
        auto t = get_variable_name("t");
        printer->add_text(
            "ml->_net_send_buffer, 0, {}, 0, {}, {}+0.0, "_format(tqitem, point_process, t));
        watch->get_value()->accept(*this);
        printer->add_text(");");
        printer->add_newline();
        printer->end_block(1);

        printer->add_line("{} = 3;"_format(varname));
        // end block 3

        // start block 3
        printer->decrease_indent();
        printer->start_block("} else");
        printer->add_line("{} = 2;"_format(varname));
        printer->end_block(1);
        // end block 3

        printer->end_block(1);
        // end block 1
    }

    print_channel_iteration_block_end();
    print_send_event_move();
    print_channel_iteration_tiling_block_end();
    print_kernel_data_present_annotation_block_end();
    printer->end_block(1);
    codegen = false;
}


void CodegenCVisitor::print_net_receive_common_code(const Block& node, bool need_mech_inst) {
    printer->add_line("int tid = pnt->_tid;");
    printer->add_line("int id = pnt->_i_instance;");
    printer->add_line("double v = 0;");
    if (info.artificial_cell || node.is_initial_block()) {
        printer->add_line("NrnThread* nt = nrn_threads + tid;");
        printer->add_line("Memb_list* ml = nt->_ml_list[pnt->_type];");
    }
    if (node.is_initial_block()) {
        print_kernel_data_present_annotation_block_begin();
    }

    printer->add_line("{}int nodecount = ml->nodecount;"_format(param_type_qualifier()));
    printer->add_line("{}int pnodecount = ml->_nodecount_padded;"_format(param_type_qualifier()));
    printer->add_line("double* data = ml->data;");
    printer->add_line("double* weights = nt->weights;");
    printer->add_line("Datum* indexes = ml->pdata;");
    printer->add_line("ThreadDatum* thread = ml->_thread;");
    if (need_mech_inst) {
        printer->add_line("{0}* inst = ({0}*) ml->instance;"_format(instance_struct()));
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
                auto statement = "double* {} = weights + weight_index + {};"_format(name, i);
                printer->add_line(statement);
                RenameVisitor vr(name, "*" + name);
                node.visit_children(vr);
            }
            i++;
        }
    }
}


void CodegenCVisitor::print_net_send_call(const FunctionCall& node) {
    auto arguments = node.get_arguments();
    auto tqitem = get_variable_name("tqitem");
    std::string weight_index = "weight_index";
    std::string pnt = "pnt";

    // for non-net_receieve functions i.e. initial block, the weight_index argument is 0.
    if (!printing_net_receive) {
        weight_index = "0";
        auto var = get_variable_name("point_process");
        if (info.artificial_cell) {
            pnt = "(Point_process*)" + var;
        }
    }

    // artificial cells don't use spike buffering
    // clang-format off
    if (info.artificial_cell) {
        printer->add_text("artcell_net_send(&{}, {}, {}, nt->_t+"_format(tqitem, weight_index, pnt));
    } else {
        auto point_process = get_variable_name("point_process");
        std::string t = get_variable_name("t");
        printer->add_text("net_send_buffering(");
        printer->add_text("ml->_net_send_buffer, 0, {}, {}, {}, {}+"_format(tqitem, weight_index, point_process, t));
    }
    // clang-format off
    print_vector_elements(arguments, ", ");
    printer->add_text(")");
}


void CodegenCVisitor::print_net_move_call(const FunctionCall& node) {
    if (!printing_net_receive) {
        std::cout << "Error : net_move only allowed in NET_RECEIVE block" << std::endl;
        abort();
    }

    auto arguments = node.get_arguments();
    auto tqitem = get_variable_name("tqitem");
    std::string weight_index = "-1";
    std::string pnt = "pnt";

    // artificial cells don't use spike buffering
    // clang-format off
    if (info.artificial_cell) {
        printer->add_text("artcell_net_move(&{}, {}, nt->_t+"_format(tqitem, pnt));
        print_vector_elements(arguments, ", ");
        printer->add_text(")");
    } else {
        auto point_process = get_variable_name("point_process");
        std::string t = get_variable_name("t");
        printer->add_text("net_send_buffering(");
        printer->add_text("ml->_net_send_buffer, 2, {}, {}, {}, "_format(tqitem, weight_index, point_process));
        print_vector_elements(arguments, ", ");
        printer->add_text(", 0.0");
        printer->add_text(")");
    }
}


void CodegenCVisitor::print_net_event_call(const FunctionCall& node) {
    const auto& arguments = node.get_arguments();
    if (info.artificial_cell) {
        printer->add_text("net_event(pnt, ");
        print_vector_elements(arguments, ", ");
    } else {
        auto point_process = get_variable_name("point_process");
        printer->add_text("net_send_buffering(");
        printer->add_text("ml->_net_send_buffer, 1, -1, -1, {}, "_format(point_process));
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
    auto parameters = net_receive_node.get_parameters();
    for (auto& parameter: parameters) {
        auto name = parameter->get_node_name();
        auto var_used = VarUsageVisitor().variable_used(node, name);
        if (var_used) {
            RenameVisitor vr(name, "(*" + name + ")");
            node.get_statement_block()->visit_children(vr);
        }
    }
}


void CodegenCVisitor::print_net_init() {
    const auto node = info.net_receive_initial_node;
    if (node == nullptr) {
        return;
    }

    // rename net_receive arguments used in the initial block of net_receive
    rename_net_receive_arguments(*info.net_receive_node, *node);

    codegen = true;
    auto args = "Point_process* pnt, int weight_index, double flag";
    printer->add_newline(2);
    printer->add_line("/** initialize block for net receive */");
    printer->start_block("static void net_init({}) "_format(args));
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
    printer->end_block(1);
    codegen = false;
}


void CodegenCVisitor::print_send_event_move() {
    printer->add_newline();
    printer->add_line("NetSendBuffer_t* nsb = ml->_net_send_buffer;");
    print_net_send_buf_update_to_host();
    printer->add_line("for (int i=0; i < nsb->_cnt; i++) {");
    printer->add_line("    int type = nsb->_sendtype[i];");
    printer->add_line("    int tid = nt->id;");
    printer->add_line("    double t = nsb->_nsb_t[i];");
    printer->add_line("    double flag = nsb->_nsb_flag[i];");
    printer->add_line("    int vdata_index = nsb->_vdata_index[i];");
    printer->add_line("    int weight_index = nsb->_weight_index[i];");
    printer->add_line("    int point_index = nsb->_pnt_index[i];");
    // clang-format off
    printer->add_line("    net_sem_from_gpu(type, vdata_index, weight_index, tid, point_index, t, flag);");
    // clang-format on
    printer->add_line("}");
    printer->add_line("nsb->_cnt = 0;");
    print_net_send_buf_count_update_to_device();
}


std::string CodegenCVisitor::net_receive_buffering_declaration() {
    return "void {}(NrnThread* nt)"_format(method_name("net_buf_receive"));
}


void CodegenCVisitor::print_get_memb_list() {
    printer->add_line("Memb_list* ml = get_memb_list(nt);");
    printer->add_line("if (ml == NULL) {");
    printer->add_line("    return;");
    printer->add_line("}");
    printer->add_newline();
}


void CodegenCVisitor::print_net_receive_loop_begin() {
    printer->add_line("int count = nrb->_displ_cnt;");
    print_channel_iteration_block_parallel_hint(BlockType::NetReceive);
    printer->start_block("for (int i = 0; i < count; i++)");
}


void CodegenCVisitor::print_net_receive_loop_end() {
    printer->end_block(1);
}


void CodegenCVisitor::print_net_receive_buffering(bool need_mech_inst) {
    if (!net_receive_required() || info.artificial_cell) {
        return;
    }
    printer->add_newline(2);
    printer->start_block(net_receive_buffering_declaration());

    print_get_memb_list();

    auto net_receive = method_name("net_receive_kernel");

    print_kernel_data_present_annotation_block_begin();

    printer->add_line(
        "NetReceiveBuffer_t* {}nrb = ml->_net_receive_buffer;"_format(ptr_type_qualifier()));
    if (need_mech_inst) {
        printer->add_line("{0}* inst = ({0}*) ml->instance;"_format(instance_struct()));
    }
    print_net_receive_loop_begin();
    printer->add_line("int start = nrb->_displ[i];");
    printer->add_line("int end = nrb->_displ[i+1];");
    printer->start_block("for (int j = start; j < end; j++)");
    printer->add_line("int index = nrb->_nrb_index[j];");
    printer->add_line("int offset = nrb->_pnt_index[index];");
    printer->add_line("double t = nrb->_nrb_t[index];");
    printer->add_line("int weight_index = nrb->_weight_index[index];");
    printer->add_line("double flag = nrb->_nrb_flag[index];");
    printer->add_line("Point_process* point_process = nt->pntprocs + offset;");
    printer->add_line(
        "{}(t, point_process, inst, nt, ml, weight_index, flag);"_format(net_receive));
    printer->end_block(1);
    print_net_receive_loop_end();

    print_device_stream_wait();
    printer->add_line("nrb->_displ_cnt = 0;");
    printer->add_line("nrb->_cnt = 0;");

    if (info.net_send_used || info.net_event_used) {
        print_send_event_move();
    }

    printer->add_newline();
    print_kernel_data_present_annotation_block_end();
    printer->end_block(1);
}

void CodegenCVisitor::print_net_send_buffering_grow() {
    printer->add_line("if(i >= nsb->_size) {");
    printer->add_line("    nsb->grow();");
    printer->add_line("}");
}

void CodegenCVisitor::print_net_send_buffering() {
    if (!net_send_buffer_required()) {
        return;
    }

    printer->add_newline(2);
    print_device_method_annotation();
    auto args =
        "NetSendBuffer_t* nsb, int type, int vdata_index, "
        "int weight_index, int point_index, double t, double flag";
    printer->start_block("static inline void net_send_buffering({}) "_format(args));
    printer->add_line("int i = 0;");
    print_device_atomic_capture_annotation();
    printer->add_line("i = nsb->_cnt++;");
    print_net_send_buffering_grow();
    printer->add_line("if(i < nsb->_size) {");
    printer->add_line("    nsb->_sendtype[i] = type;");
    printer->add_line("    nsb->_vdata_index[i] = vdata_index;");
    printer->add_line("    nsb->_weight_index[i] = weight_index;");
    printer->add_line("    nsb->_pnt_index[i] = point_index;");
    printer->add_line("    nsb->_nsb_t[i] = t;");
    printer->add_line("    nsb->_nsb_flag[i] = flag;");
    printer->add_line("}");
    printer->end_block(1);
}


void CodegenCVisitor::visit_for_netcon(const ast::ForNetcon& node) {
    // For_netcon should take the same arguments as net_receive and apply the operations
    // in the block to the weights of the netcons. Since all the weights are on the same vector,
    // weights, we have a mask of operations that we apply iteratively, advancing the offset
    // to the next netcon.
    const auto& args = node.get_parameters();
    RenameVisitor v;
    auto& statement_block = node.get_statement_block();
    for (size_t i_arg = 0; i_arg < args.size(); ++i_arg) {
        // sanitize node_name since we want to substitute names like (*w) as they are
        auto old_name =
            std::regex_replace(args[i_arg]->get_node_name(), regex_special_chars, R"(\$&)");
        auto new_name = "weights[{} + nt->_fornetcon_weight_perm[i]]"_format(i_arg);
        v.set(old_name, new_name);
        statement_block->accept(v);
    }

    const auto index =
        std::find_if(info.semantics.begin(), info.semantics.end(), [](const IndexSemantics& a) {
            return a.name == naming::FOR_NETCON_SEMANTIC;
        })->index;

    printer->add_text("const size_t offset = {}*pnodecount + id;"_format(index));
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

void CodegenCVisitor::print_net_receive_kernel() {
    if (!net_receive_required()) {
        return;
    }
    codegen = true;
    printing_net_receive = true;
    const auto node = info.net_receive_node;

    // rename net_receive arguments used in the block itself
    rename_net_receive_arguments(*info.net_receive_node, *node);

    std::string name;
    auto params = ParamVector();
    if (!info.artificial_cell) {
        name = method_name("net_receive_kernel");
        params.emplace_back("", "double", "", "t");
        params.emplace_back("", "Point_process*", "", "pnt");
        params.emplace_back(param_type_qualifier(),
                            "{}*"_format(instance_struct()),
                            param_ptr_qualifier(),
                            "inst");
        params.emplace_back(param_type_qualifier(), "NrnThread*", param_ptr_qualifier(), "nt");
        params.emplace_back(param_type_qualifier(), "Memb_list*", param_ptr_qualifier(), "ml");
        params.emplace_back("", "int", "", "weight_index");
        params.emplace_back("", "double", "", "flag");
    } else {
        name = method_name("net_receive");
        params.emplace_back("", "Point_process*", "", "pnt");
        params.emplace_back("", "int", "", "weight_index");
        params.emplace_back("", "double", "", "flag");
    }

    printer->add_newline(2);
    printer->start_block("static inline void {}({}) "_format(name, get_parameter_str(params)));
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

    printer->add_line("{} = t;"_format(get_variable_name("tsave")));

    if (info.is_watch_used()) {
        printer->add_line("bool watch_remove = false;");
    }

    printer->add_indent();
    node->get_statement_block()->accept(*this);
    printer->add_newline();
    printer->end_block();
    printer->add_newline();

    printing_net_receive = false;
    codegen = false;
}


void CodegenCVisitor::print_net_receive() {
    if (!net_receive_required()) {
        return;
    }
    codegen = true;
    printing_net_receive = true;
    if (!info.artificial_cell) {
        std::string name = method_name("net_receive");
        auto params = ParamVector();
        params.emplace_back("", "Point_process*", "", "pnt");
        params.emplace_back("", "int", "", "weight_index");
        params.emplace_back("", "double", "", "flag");
        printer->add_newline(2);
        printer->start_block("static void {}({}) "_format(name, get_parameter_str(params)));
        printer->add_line("NrnThread* nt = nrn_threads + pnt->_tid;");
        printer->add_line("Memb_list* ml = get_memb_list(nt);");
        printer->add_line("NetReceiveBuffer_t* nrb = ml->_net_receive_buffer;");
        printer->add_line("if (nrb->_cnt >= nrb->_size) {");
        printer->add_line("    realloc_net_receive_buffer(nt, ml);");
        printer->add_line("}");
        printer->add_line("int id = nrb->_cnt;");
        printer->add_line("nrb->_pnt_index[id] = pnt-nt->pntprocs;");
        printer->add_line("nrb->_weight_index[id] = weight_index;");
        printer->add_line("nrb->_nrb_t[id] = nt->_t;");
        printer->add_line("nrb->_nrb_flag[id] = flag;");
        printer->add_line("nrb->_cnt++;");
        printer->end_block(1);
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
void CodegenCVisitor::print_derivimplicit_kernel(Block* block) {
    auto ext_args = external_method_arguments();
    auto ext_params = external_method_parameters();
    auto suffix = info.mod_suffix;
    auto list_num = info.derivimplicit_list_num;
    auto block_name = block->get_node_name();
    auto primes_size = info.primes_size;
    auto stride = "*pnodecount+id";

    printer->add_newline(2);

    // clang-format off
    printer->start_block("int {}_{}({})"_format(block_name, suffix, ext_params));
    auto instance = "{0}* inst = ({0}*)get_memb_list(nt)->instance;"_format(instance_struct());
    auto slist1 = "int* slist{} = {};"_format(list_num, get_variable_name("slist{}"_format(list_num)));
    auto slist2 = "int* slist{} = {};"_format(list_num+1, get_variable_name("slist{}"_format(list_num+1)));
    auto dlist1 = "int* dlist{} = {};"_format(list_num, get_variable_name("dlist{}"_format(list_num)));
    auto dlist2 = "double* dlist{} = (double*) thread[dith{}()].pval + ({}*pnodecount);"_format(list_num + 1, list_num, info.primes_size);

    printer->add_line(instance);
    printer->add_line("double* savstate{} = (double*) thread[dith{}()].pval;"_format(list_num, list_num));
    printer->add_line(slist1);
    printer->add_line(slist2);
    printer->add_line(dlist2);
    printer->add_line("for (int i=0; i<{}; i++) {}"_format(info.num_primes, "{"));
    printer->add_line("    savstate{}[i{}] = data[slist{}[i]{}];"_format(list_num, stride, list_num, stride));
    printer->add_line("}");

    auto argument = "{}, slist{}, _derivimplicit_{}_{}, dlist{}, {}"_format(primes_size, list_num+1, block_name, suffix, list_num + 1, ext_args);
    printer->add_line("int reset = nrn_newton_thread(static_cast<NewtonSpace*>(*newtonspace{}(thread)), {});"_format(list_num, argument));
    printer->add_line("return reset;");
    printer->end_block(3);

    /**
     * \todo To be backward compatible with mod2c we have to generate below
     * comment marker in the generated cpp file for kinderiv.py to
     * process it and generate correct _kinderiv.h
     */
    printer->add_line("/* _derivimplicit_ {} _{} */"_format(block_name, info.mod_suffix));
    printer->add_newline(1);

    printer->start_block("int _newton_{}_{}({}) "_format(block_name, info.mod_suffix, external_method_parameters()));
    printer->add_line(instance);
    if (ion_variable_struct_required()) {
        print_ion_variable();
    }
    printer->add_line("double* savstate{} = (double*) thread[dith{}()].pval;"_format(list_num, list_num));
    printer->add_line(slist1);
    printer->add_line(dlist1);
    printer->add_line(dlist2);
    codegen = true;
    print_statement_block(*block->get_statement_block(), false, false);
    codegen = false;
    printer->add_line("int counter = -1;");
    printer->add_line("for (int i=0; i<{}; i++) {}"_format(info.num_primes, "{"));
    printer->add_line("    if (*deriv{}_advance(thread)) {}"_format(list_num, "{"));
    printer->add_line("        dlist{0}[(++counter){1}] = data[dlist{2}[i]{1}]-(data[slist{2}[i]{1}]-savstate{2}[i{1}])/nt->_dt;"_format(list_num + 1, stride, list_num));
    printer->add_line("    }");
    printer->add_line("    else {");
    printer->add_line("        dlist{0}[(++counter){1}] = data[slist{2}[i]{1}]-savstate{2}[i{1}];"_format(list_num + 1, stride, list_num));
    printer->add_line("    }");
    printer->add_line("}");
    printer->add_line("return 0;");
    printer->end_block();
    // clang-format on
}


void CodegenCVisitor::print_newtonspace_transfer_to_device() const {
    // nothing to do on cpu
}


void CodegenCVisitor::visit_derivimplicit_callback(const ast::DerivimplicitCallback& node) {
    if (!codegen) {
        return;
    }
    auto thread_args = external_method_arguments();
    auto num_primes = info.num_primes;
    auto suffix = info.mod_suffix;
    int num = info.derivimplicit_list_num;
    auto slist = get_variable_name("slist{}"_format(num));
    auto dlist = get_variable_name("dlist{}"_format(num));
    auto block_name = node.get_node_to_solve()->get_node_name();

    auto args =
        "{}, {}, {}, _derivimplicit_{}_{}, {}"
        ""_format(num_primes, slist, dlist, block_name, suffix, thread_args);
    auto statement = "derivimplicit_thread({});"_format(args);
    printer->add_line(statement);
}

void CodegenCVisitor::visit_solution_expression(const SolutionExpression& node) {
    auto block = node.get_node_to_solve().get();
    if (block->is_statement_block()) {
        auto statement_block = dynamic_cast<ast::StatementBlock*>(block);
        print_statement_block(*statement_block, false, false);
    } else {
        block->accept(*this);
    }
}


/****************************************************************************************/
/*                                Print nrn_state routine                                */
/****************************************************************************************/


void CodegenCVisitor::print_nrn_state() {
    if (!nrn_state_required()) {
        return;
    }
    codegen = true;

    printer->add_newline(2);
    printer->add_line("/** update state */");
    print_global_function_common_code(BlockType::State);
    print_channel_iteration_tiling_block_begin(BlockType::State);
    print_channel_iteration_block_begin(BlockType::State);

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

    auto write_statements = ion_write_statements(BlockType::State);
    for (auto& statement: write_statements) {
        auto text = process_shadow_update_statement(statement, BlockType::State);
        printer->add_line(text);
    }
    print_channel_iteration_block_end();
    if (!shadow_statements.empty()) {
        print_shadow_reduction_block_begin();
        print_shadow_reduction_statements();
        print_shadow_reduction_block_end();
    }
    print_channel_iteration_tiling_block_end();

    print_kernel_data_present_annotation_block_end();
    printer->end_block(1);
    codegen = false;
}


/****************************************************************************************/
/*                            Print nrn_cur related routines                            */
/****************************************************************************************/


void CodegenCVisitor::print_nrn_current(const BreakpointBlock& node) {
    auto args = internal_method_parameters();
    const auto& block = node.get_statement_block();
    printer->add_newline(2);
    print_device_method_annotation();
    printer->start_block("static inline double nrn_current({})"_format(get_parameter_str(args)));
    printer->add_line("double current = 0.0;");
    print_statement_block(*block, false, false);
    for (auto& current: info.currents) {
        auto name = get_variable_name(current);
        printer->add_line("current += {};"_format(name));
    }
    printer->add_line("return current;");
    printer->end_block(1);
}


void CodegenCVisitor::print_nrn_cur_conductance_kernel(const BreakpointBlock& node) {
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
        printer->add_line("double rhs = {};"_format(sum));
    }

    std::string sum;
    for (const auto& conductance: info.conductances) {
        auto var = breakpoint_current(conductance.variable);
        sum += get_variable_name(var);
        if (&conductance != &info.conductances.back()) {
            sum += "+";
        }
    }
    printer->add_line("double g = {};"_format(sum));

    for (const auto& conductance: info.conductances) {
        if (!conductance.ion.empty()) {
            auto lhs = std::string(naming::ION_VARNAME_PREFIX) + "di" + conductance.ion + "dv";
            auto rhs = get_variable_name(conductance.variable);
            ShadowUseStatement statement{lhs, "+=", rhs};
            auto text = process_shadow_update_statement(statement, BlockType::Equation);
            printer->add_line(text);
        }
    }
}


void CodegenCVisitor::print_nrn_cur_non_conductance_kernel() {
    printer->add_line("double g = nrn_current({}+0.001);"_format(internal_method_arguments()));
    for (auto& ion: info.ions) {
        for (auto& var: ion.writes) {
            if (ion.is_ionic_current(var)) {
                auto name = get_variable_name(var);
                printer->add_line("double di{} = {};"_format(ion.name, name));
            }
        }
    }
    printer->add_line("double rhs = nrn_current({});"_format(internal_method_arguments()));
    printer->add_line("g = (g-rhs)/0.001;");
    for (auto& ion: info.ions) {
        for (auto& var: ion.writes) {
            if (ion.is_ionic_current(var)) {
                auto lhs = std::string(naming::ION_VARNAME_PREFIX) + "di" + ion.name + "dv";
                auto rhs = "(di{}-{})/0.001"_format(ion.name, get_variable_name(var));
                if (info.point_process) {
                    auto area = get_variable_name(naming::NODE_AREA_VARIABLE);
                    rhs += "*1.e2/{}"_format(area);
                }
                ShadowUseStatement statement{lhs, "+=", rhs};
                auto text = process_shadow_update_statement(statement, BlockType::Equation);
                printer->add_line(text);
            }
        }
    }
}


void CodegenCVisitor::print_nrn_cur_kernel(const BreakpointBlock& node) {
    printer->add_line("int node_id = node_index[id];");
    printer->add_line("double v = voltage[node_id];");
    print_v_unused();
    if (ion_variable_struct_required()) {
        print_ion_variable();
    }

    auto read_statements = ion_read_statements(BlockType::Equation);
    for (auto& statement: read_statements) {
        printer->add_line(statement);
    }

    if (info.conductances.empty()) {
        print_nrn_cur_non_conductance_kernel();
    } else {
        print_nrn_cur_conductance_kernel(node);
    }

    auto write_statements = ion_write_statements(BlockType::Equation);
    for (auto& statement: write_statements) {
        auto text = process_shadow_update_statement(statement, BlockType::Equation);
        printer->add_line(text);
    }

    if (info.point_process) {
        auto area = get_variable_name(naming::NODE_AREA_VARIABLE);
        printer->add_line("double mfactor = 1.e2/{};"_format(area));
        printer->add_line("g = g*mfactor;");
        printer->add_line("rhs = rhs*mfactor;");
    }

    print_g_unused();
}

void CodegenCVisitor::print_fast_imem_calculation() {
    if (!info.electrode_current) {
        return;
    }
    std::string rhs, d;
    auto rhs_op = operator_for_rhs();
    auto d_op = operator_for_d();
    if (channel_task_dependency_enabled()) {
        rhs = get_variable_name("ml_rhs");
        d = get_variable_name("ml_d");
    } else if (info.point_process) {
        rhs = "shadow_rhs[id]";
        d = "shadow_d[id]";
    } else {
        rhs = "rhs";
        d = "g";
    }

    printer->start_block("if (nt->nrn_fast_imem)");
    if (nrn_cur_reduction_loop_required()) {
        print_shadow_reduction_block_begin();
        printer->add_line("int node_id = node_index[id];");
    }
    print_atomic_reduction_pragma();
    printer->add_line("nt->nrn_fast_imem->nrn_sav_rhs[node_id] {} {};"_format(rhs_op, rhs));
    print_atomic_reduction_pragma();
    printer->add_line("nt->nrn_fast_imem->nrn_sav_d[node_id] {} {};"_format(d_op, d));
    if (nrn_cur_reduction_loop_required()) {
        print_shadow_reduction_block_end();
    }
    printer->end_block(1);
}

void CodegenCVisitor::print_nrn_cur() {
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
    print_channel_iteration_tiling_block_begin(BlockType::Equation);
    print_channel_iteration_block_begin(BlockType::Equation);
    print_nrn_cur_kernel(*info.breakpoint_node);
    print_nrn_cur_matrix_shadow_update();
    if (!nrn_cur_reduction_loop_required()) {
        print_fast_imem_calculation();
    }
    print_channel_iteration_block_end();

    if (nrn_cur_reduction_loop_required()) {
        print_shadow_reduction_block_begin();
        print_nrn_cur_matrix_shadow_reduction();
        print_shadow_reduction_statements();
        print_shadow_reduction_block_end();
        print_fast_imem_calculation();
    }

    print_channel_iteration_tiling_block_end();
    print_kernel_data_present_annotation_block_end();
    printer->end_block(1);
    codegen = false;
}


/****************************************************************************************/
/*                            Main code printing entry points                            */
/****************************************************************************************/

void CodegenCVisitor::print_headers_include() {
    print_standard_includes();
    print_backend_includes();
    print_coreneuron_includes();
}


void CodegenCVisitor::print_namespace_begin() {
    print_namespace_start();
    print_backend_namespace_start();
}


void CodegenCVisitor::print_namespace_end() {
    print_backend_namespace_stop();
    print_namespace_stop();
}


void CodegenCVisitor::print_common_getters() {
    print_first_pointer_var_index_getter();
    print_net_receive_arg_size_getter();
    print_thread_getters();
    print_num_variable_getter();
    print_mech_type_getter();
    print_memb_list_getter();
}


void CodegenCVisitor::print_data_structures() {
    print_mechanism_global_var_structure();
    print_mechanism_range_var_structure();
    print_ion_var_structure();
}

void CodegenCVisitor::print_v_unused() const {
    printer->add_line("#if NRN_PRCELLSTATE");
    printer->add_line("inst->v_unused[id] = v;");
    printer->add_line("#endif");
}

void CodegenCVisitor::print_g_unused() const {
    printer->add_line("#if NRN_PRCELLSTATE");
    printer->add_line("inst->g_unused[id] = g;");
    printer->add_line("#endif");
}

void CodegenCVisitor::print_compute_functions() {
    print_top_verbatim_blocks();
    print_function_prototypes();
    for (const auto& procedure: info.procedures) {
        print_procedure(*procedure);
    }
    for (const auto& function: info.functions) {
        print_function(*function);
    }
    for (const auto& callback: info.derivimplicit_callbacks) {
        auto block = callback->get_node_to_solve().get();
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


void CodegenCVisitor::print_codegen_routines() {
    codegen = true;
    print_backend_info();
    print_headers_include();
    print_namespace_begin();
    print_nmodl_constants();
    print_prcellstate_macros();
    print_mechanism_info();
    print_data_structures();
    print_global_variables_for_hoc();
    print_common_getters();
    print_memory_allocation_routine();
    print_abort_routine();
    print_thread_memory_callbacks();
    print_global_variable_setup();
    print_instance_variable_setup();
    print_nrn_alloc();
    print_nrn_constructor();
    print_nrn_destructor();
    print_compute_functions();
    print_check_table_thread_function();
    print_mechanism_register();
    print_namespace_end();
    codegen = false;
}


void CodegenCVisitor::print_wrapper_routines() {
    // nothing to do
}


void CodegenCVisitor::set_codegen_global_variables(std::vector<SymbolType>& global_vars) {
    codegen_global_variables = global_vars;
}


void CodegenCVisitor::setup(const Program& node) {
    program_symtab = node.get_symbol_table();

    CodegenHelperVisitor v;
    info = v.analyze(node);
    info.mod_file = mod_filename;

    if (!info.vectorize) {
        logger->warn("CodegenCVisitor : MOD file uses non-thread safe constructs of NMODL");
    }

    codegen_float_variables = get_float_variables();
    codegen_int_variables = get_int_variables();
    codegen_shadow_variables = get_shadow_variables();

    update_index_semantics();
    rename_function_arguments();
}


void CodegenCVisitor::visit_program(const Program& node) {
    setup(node);
    print_codegen_routines();
    print_wrapper_routines();
}

}  // namespace codegen
}  // namespace nmodl
