/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "codegen/codegen_cpp_visitor.hpp"

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

CodegenCppVisitor::CodegenCppVisitor(std::string mod_filename,
                      const std::string& output_dir,
                      std::string float_type,
                      const bool optimize_ionvar_copies,
                      const std::string& extension)
        : printer(std::make_unique<CodePrinter>(output_dir + "/" + mod_filename + extension))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}

/// This constructor is private, see the public section below to find how to create an instance
/// of this class.
CodegenCppVisitor::CodegenCppVisitor(std::string mod_filename,
                std::ostream& stream,
                std::string float_type,
                const bool optimize_ionvar_copies)
    : printer(std::make_unique<CodePrinter>(stream))
    , mod_filename(std::move(mod_filename))
    , float_type(std::move(float_type))
    , optimize_ionvar_copies(optimize_ionvar_copies) {}

CodegenCppVisitor::~CodegenCppVisitor() {};

/****************************************************************************************/
/*                            Overloaded visitor routines                               */
/****************************************************************************************/

static const std::regex regex_special_chars{R"([-[\]{}()*+?.,\^$|#\s])"};

void CodegenCppVisitor::visit_string(const String& node) {
    if (!codegen) {
        return;
    }
    std::string name = node.eval();
    if (enable_variable_name_lookup) {
        name = get_variable_name(name);
    }
    printer->add_text(name);
}


void CodegenCppVisitor::visit_integer(const Integer& node) {
    if (!codegen) {
        return;
    }
    const auto& value = node.get_value();
    printer->add_text(std::to_string(value));
}


void CodegenCppVisitor::visit_float(const Float& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(format_float_string(node.get_value()));
}


void CodegenCppVisitor::visit_double(const Double& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(format_double_string(node.get_value()));
}


void CodegenCppVisitor::visit_boolean(const Boolean& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(std::to_string(static_cast<int>(node.eval())));
}


void CodegenCppVisitor::visit_name(const Name& node) {
    if (!codegen) {
        return;
    }
    node.visit_children(*this);
}


void CodegenCppVisitor::visit_unit(const ast::Unit& node) {
    // do not print units
}


void CodegenCppVisitor::visit_prime_name(const PrimeName& /* node */) {
    throw std::runtime_error("PRIME encountered during code generation, ODEs not solved?");
}


/**
 * \todo : Validate how @ is being handled in neuron implementation
 */
void CodegenCppVisitor::visit_var_name(const VarName& node) {
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


void CodegenCppVisitor::visit_indexed_name(const IndexedName& node) {
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


void CodegenCppVisitor::visit_local_list_statement(const LocalListStatement& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(local_var_type(), ' ');
    print_vector_elements(node.get_variables(), ", ");
}


void CodegenCppVisitor::visit_if_statement(const IfStatement& node) {
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


void CodegenCppVisitor::visit_else_if_statement(const ElseIfStatement& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(" else if (");
    node.get_condition()->accept(*this);
    printer->add_text(") ");
    node.get_statement_block()->accept(*this);
}


void CodegenCppVisitor::visit_else_statement(const ElseStatement& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(" else ");
    node.visit_children(*this);
}


void CodegenCppVisitor::visit_while_statement(const WhileStatement& node) {
    printer->add_text("while (");
    node.get_condition()->accept(*this);
    printer->add_text(") ");
    node.get_statement_block()->accept(*this);
}


void CodegenCppVisitor::visit_from_statement(const ast::FromStatement& node) {
    if (!codegen) {
        return;
    }
    auto name = node.get_node_name();
    const auto& from = node.get_from();
    const auto& to = node.get_to();
    const auto& inc = node.get_increment();
    const auto& block = node.get_statement_block();
    printer->fmt_text("for (int {} = ", name);
    from->accept(*this);
    printer->fmt_text("; {} <= ", name);
    to->accept(*this);
    if (inc) {
        printer->fmt_text("; {} += ", name);
        inc->accept(*this);
    } else {
        printer->fmt_text("; {}++", name);
    }
    printer->add_text(") ");
    block->accept(*this);
}


void CodegenCppVisitor::visit_paren_expression(const ParenExpression& node) {
    if (!codegen) {
        return;
    }
    printer->add_text("(");
    node.get_expression()->accept(*this);
    printer->add_text(")");
}


void CodegenCppVisitor::visit_binary_expression(const BinaryExpression& node) {
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


void CodegenCppVisitor::visit_binary_operator(const BinaryOperator& node) {
    if (!codegen) {
        return;
    }
    printer->add_text(node.eval());
}


void CodegenCppVisitor::visit_unary_operator(const UnaryOperator& node) {
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
void CodegenCppVisitor::visit_statement_block(const StatementBlock& node) {
    if (!codegen) {
        node.visit_children(*this);
        return;
    }
    print_statement_block(node);
}


void CodegenCppVisitor::visit_function_call(const FunctionCall& node) {
    if (!codegen) {
        return;
    }
    print_function_call(node);
}


void CodegenCppVisitor::visit_verbatim(const Verbatim& node) {
    if (!codegen) {
        return;
    }
    const auto& text = node.get_statement()->eval();
    const auto& result = process_verbatim_text(text);

    const auto& statements = stringutils::split_string(result, '\n');
    for (const auto& statement: statements) {
        const auto& trimed_stmt = stringutils::trim_newline(statement);
        if (trimed_stmt.find_first_not_of(' ') != std::string::npos) {
            printer->add_line(trimed_stmt);
        }
    }
}

void CodegenCppVisitor::visit_update_dt(const ast::UpdateDt& node) {
    // dt change statement should be pulled outside already
}

void CodegenCppVisitor::visit_protect_statement(const ast::ProtectStatement& node) {
    print_atomic_reduction_pragma();
    printer->add_indent();
    node.get_expression()->accept(*this);
    printer->add_text(";");
}

void CodegenCppVisitor::visit_mutex_lock(const ast::MutexLock& node) {
    printer->fmt_line("#pragma omp critical ({})", info.mod_suffix);
    printer->add_indent();
    printer->push_block();
}

void CodegenCppVisitor::visit_mutex_unlock(const ast::MutexUnlock& node) {
    printer->pop_block();
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
bool CodegenCppVisitor::statement_to_skip(const Statement& node) {
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


bool CodegenCppVisitor::net_send_buffer_required() const noexcept {
    if (net_receive_required() && !info.artificial_cell) {
        if (info.net_event_used || info.net_send_used || info.is_watch_used()) {
            return true;
        }
    }
    return false;
}


bool CodegenCppVisitor::net_receive_buffering_required() const noexcept {
    return info.point_process && !info.artificial_cell && info.net_receive_node != nullptr;
}


bool CodegenCppVisitor::nrn_state_required() const noexcept {
    if (info.artificial_cell) {
        return false;
    }
    return info.nrn_state_block != nullptr || breakpoint_exist();
}


bool CodegenCppVisitor::nrn_cur_required() const noexcept {
    return info.breakpoint_node != nullptr && !info.currents.empty();
}


bool CodegenCppVisitor::net_receive_exist() const noexcept {
    return info.net_receive_node != nullptr;
}


bool CodegenCppVisitor::breakpoint_exist() const noexcept {
    return info.breakpoint_node != nullptr;
}


bool CodegenCppVisitor::net_receive_required() const noexcept {
    return net_receive_exist();
}


/**
 * \details When floating point data type is not default (i.e. double) then we
 * have to copy old array to new type (for range variables).
 */
bool CodegenCppVisitor::range_variable_setup_required() const noexcept {
    return codegen::naming::DEFAULT_FLOAT_TYPE != float_data_type();
}


int CodegenCppVisitor::position_of_float_var(const std::string& name) const {
    int index = 0;
    for (const auto& var: codegen_float_variables) {
        if (var->get_name() == name) {
            return index;
        }
        index += var->get_length();
    }
    throw std::logic_error(name + " variable not found");
}


int CodegenCppVisitor::position_of_int_var(const std::string& name) const {
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
std::string CodegenCppVisitor::format_double_string(const std::string& s_value) {
    return utils::format_double_string<CodegenCppVisitor>(s_value);
}


std::string CodegenCppVisitor::format_float_string(const std::string& s_value) {
    return utils::format_float_string<CodegenCppVisitor>(s_value);
}


/**
 * \details Statements like if, else etc. don't need semicolon at the end.
 * (Note that it's valid to have "extraneous" semicolon). Also, statement
 * block can appear as statement using expression statement which need to
 * be inspected.
 */
bool CodegenCppVisitor::need_semicolon(const Statement& node) {
    // clang-format off
    if (node.is_if_statement()
        || node.is_else_if_statement()
        || node.is_else_statement()
        || node.is_from_statement()
        || node.is_verbatim()
        || node.is_from_statement()
        || node.is_conductance_hint()
        || node.is_while_statement()
        || node.is_protect_statement()
        || node.is_mutex_lock()
        || node.is_mutex_unlock()) {
        return false;
    }
    if (node.is_expression_statement()) {
        auto expression = dynamic_cast<const ExpressionStatement&>(node).get_expression();
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
bool CodegenCppVisitor::defined_method(const std::string& name) const {
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
std::string CodegenCppVisitor::breakpoint_current(std::string current) const {
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


int CodegenCppVisitor::float_variables_size() const {
    return codegen_float_variables.size();
}


int CodegenCppVisitor::int_variables_size() const {
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
std::vector<std::string> CodegenCppVisitor::ion_read_statements(BlockType type) const {
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


std::vector<std::string> CodegenCppVisitor::ion_read_statements_optimized(BlockType type) const {
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
std::vector<ShadowUseStatement> CodegenCppVisitor::ion_write_statements(BlockType type) {
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
std::string CodegenCppVisitor::process_verbatim_token(const std::string& token) {
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


bool CodegenCppVisitor::ion_variable_struct_required() const {
    return optimize_ion_variable_copies() && info.ion_has_write_variable();
}


/**
 * \details This can be override in the backend. For example, parameters can be constant
 * except in INITIAL block where they are set to 0. As initial block is/can be
 * executed on c++/cpu backend, gpu backend can mark the parameter as constant.
 */
bool CodegenCppVisitor::is_constant_variable(const std::string& name) const {
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


/**
 * \details Once variables are populated, update index semantics to register with coreneuron
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CodegenCppVisitor::update_index_semantics() {
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
            info.semantics.emplace_back(index++, fmt::format("{}_ion", ion.name), 1);
            info.semantics.emplace_back(index++, fmt::format("#{}_ion", ion.name), 1);
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


std::vector<CodegenCppVisitor::SymbolType> CodegenCppVisitor::get_float_variables() const {
    // sort with definition order
    auto comparator = [](const SymbolType& first, const SymbolType& second) -> bool {
        return first->get_definition_order() < second->get_definition_order();
    };

    auto assigned = info.assigned_vars;
    auto states = info.state_vars;

    // each state variable has corresponding Dstate variable
    for (const auto& state: states) {
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

        // make sure conductance variable like `g` is not already defined
        if (auto r = std::find_if(variables.cbegin(),
                                  variables.cend(),
                                  [&](const auto& s) { return name == s->get_name(); });
            r == variables.cend()) {
            variables.push_back(make_symbol(name));
        }
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
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::vector<IndexVariableInfo> CodegenCppVisitor::get_int_variables() {
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

    for (auto& ion: info.ions) {
        bool need_style = false;
        std::unordered_map<std::string, int> ion_vars;  // used to keep track of the variables to
                                                        // not have doubles between read/write. Same
                                                        // name variables are allowed
        // See if we need to add extra readion statements to match NEURON with SoA data
        auto const has_var = [&ion](const char* suffix) -> bool {
            auto const pred = [name = ion.name + suffix](auto const& x) { return x == name; };
            return std::any_of(ion.reads.begin(), ion.reads.end(), pred) ||
                   std::any_of(ion.writes.begin(), ion.writes.end(), pred);
        };
        auto const add_implicit_read = [&ion](const char* suffix) {
            auto name = ion.name + suffix;
            ion.reads.push_back(name);
            ion.implicit_reads.push_back(std::move(name));
        };
        bool const have_ionin{has_var("i")}, have_ionout{has_var("o")};
        if (have_ionin && !have_ionout) {
            add_implicit_read("o");
        } else if (have_ionout && !have_ionin) {
            add_implicit_read("i");
        }
        for (const auto& var: ion.reads) {
            const std::string name = naming::ION_VARNAME_PREFIX + var;
            variables.emplace_back(make_symbol(name));
            variables.back().is_constant = true;
            ion_vars[name] = static_cast<int>(variables.size() - 1);
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
            variables.emplace_back(make_symbol(naming::ION_VARNAME_PREFIX + ion.name + "_erev"));
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
        info.tqitem_index = static_cast<int>(variables.size() - 1);
    }

    /**
     * \note Variables for watch statements : there is one extra variable
     * used in coreneuron compared to actual watch statements for compatibility
     * with neuron (which uses one extra Datum variable)
     */
    if (!info.watch_statements.empty()) {
        for (int i = 0; i < info.watch_statements.size() + 1; i++) {
            variables.emplace_back(make_symbol(fmt::format("watch{}", i)), false, false, true);
        }
    }
    return variables;
}


/****************************************************************************************/
/*                      Routines must be overloaded in backend                          */
/****************************************************************************************/

std::string CodegenCppVisitor::get_parameter_str(const ParamVector& params) {
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
void CodegenCppVisitor::print_channel_iteration_block_parallel_hint(BlockType /* type */,
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





/**
 * In the current implementation of CPU/CPP backend we need to emit atomic pragma
 * only with PROTECT construct (atomic rduction requirement for other cases on CPU
 * is handled via separate shadow vectors).
 */
void CodegenCppVisitor::print_atomic_reduction_pragma() {
    printer->add_line("#pragma omp atomic update");
}


bool CodegenCppVisitor::optimize_ion_variable_copies() const {
    return optimize_ionvar_copies;
}


/****************************************************************************************/
/*              printing routines for code generation                                   */
/****************************************************************************************/


void CodegenCppVisitor::visit_watch_statement(const ast::WatchStatement& /* node */) {
    printer->add_text(fmt::format("nrn_watch_activate(inst, id, pnodecount, {}, v, watch_remove)",
                                  current_watch_statement++));
}


void CodegenCppVisitor::print_statement_block(const ast::StatementBlock& node,
                                              bool open_brace,
                                              bool close_brace) {
    if (open_brace) {
        printer->push_block();
    }

    const auto& statements = node.get_statements();
    for (const auto& statement: statements) {
        if (statement_to_skip(*statement)) {
            continue;
        }
        /// not necessary to add indent for verbatim block (pretty-printing)
        if (!statement->is_verbatim() && !statement->is_mutex_lock() &&
            !statement->is_mutex_unlock() && !statement->is_protect_statement()) {
            printer->add_indent();
        }
        statement->accept(*this);
        if (need_semicolon(*statement)) {
            printer->add_text(';');
        }
        if (!statement->is_mutex_lock() && !statement->is_mutex_unlock()) {
            printer->add_newline();
        }
    }

    if (close_brace) {
        printer->pop_block_nl(0);
    }
}


void CodegenCppVisitor::print_function_call(const FunctionCall& node) {
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


void CodegenCppVisitor::print_top_verbatim_blocks() {
    if (info.top_verbatim_blocks.empty()) {
        return;
    }
    print_namespace_stop();

    printer->add_newline(2);
    printer->add_line("using namespace ", namespace_name, ";");
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
void CodegenCppVisitor::rename_function_arguments() {
    const auto& default_arguments = stringutils::split_string(nrn_thread_arguments(), ',');
    for (const auto& dirty_arg: default_arguments) {
        const auto& arg = stringutils::trim(dirty_arg);
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


void CodegenCppVisitor::print_function_prototypes() {
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


std::tuple<bool, int> CodegenCppVisitor::check_if_var_is_array(const std::string& name) {
    auto symbol = program_symtab->lookup_in_scope(name);
    if (!symbol) {
        throw std::runtime_error(
            fmt::format("CodegenCppVisitor:: {} not found in symbol table!", name));
    }
    if (symbol->is_array()) {
        return {true, symbol->get_length()};
    } else {
        return {false, 0};
    }
}








void CodegenCppVisitor::print_function_or_procedure(const ast::Block& node,
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


void CodegenCppVisitor::print_function_procedure_helper(const ast::Block& node) {
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





void CodegenCppVisitor::print_function(const ast::FunctionBlock& node) {
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

void CodegenCppVisitor::print_functor_definition(const ast::EigenNewtonSolverBlock& node) {
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

void CodegenCppVisitor::visit_eigen_newton_solver_block(const ast::EigenNewtonSolverBlock& node) {
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

void CodegenCppVisitor::visit_eigen_linear_solver_block(const ast::EigenLinearSolverBlock& node) {
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

void CodegenCppVisitor::print_eigen_linear_solver(const std::string& float_type, int N) {
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


/**
 * Replace commonly used variables in the verbatim blocks into their corresponding
 * variable name in the new code generation backend.
 */
std::string CodegenCppVisitor::replace_if_verbatim_variable(std::string name) {
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
// This will need changes for NEURON but should be more or less the same
std::string CodegenCppVisitor::process_verbatim_text(std::string const& text) {
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


std::pair<std::string, std::string> CodegenCppVisitor::read_ion_variable_name(
    const std::string& name) {
    return {name, naming::ION_VARNAME_PREFIX + name};
}


std::pair<std::string, std::string> CodegenCppVisitor::write_ion_variable_name(
    const std::string& name) {
    return {naming::ION_VARNAME_PREFIX + name, name};
}


std::string CodegenCppVisitor::conc_write_statement(const std::string& ion_name,
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
std::string CodegenCppVisitor::process_shadow_update_statement(const ShadowUseStatement& statement,
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
void CodegenCppVisitor::print_nmodl_constants() {
    if (!info.factor_definitions.empty()) {
        printer->add_newline(2);
        printer->add_line("/** constants used in nmodl from UNITS */");
        for (const auto& it: info.factor_definitions) {
            const std::string format_string = "static const double {} = {};";
            printer->fmt_line(format_string, it->get_node_name(), it->get_value()->get_value());
        }
    }
}





void CodegenCppVisitor::print_num_variable_getter() {
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


void CodegenCppVisitor::print_net_receive_arg_size_getter() {
    if (!net_receive_exist()) {
        return;
    }
    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline int num_net_receive_args()");
    printer->fmt_line("return {};", info.num_net_receive_parameters);
    printer->pop_block();
}


void CodegenCppVisitor::print_mech_type_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline int get_mech_type()");
    // false => get it from the host-only global struct, not the instance structure
    printer->fmt_line("return {};", get_variable_name("mech_type", false));
    printer->pop_block();
}


void CodegenCppVisitor::print_memb_list_getter() {
    printer->add_newline(2);
    print_device_method_annotation();
    printer->push_block("static inline Memb_list* get_memb_list(NrnThread* nt)");
    printer->push_block("if (!nt->_ml_list)");
    printer->add_line("return nullptr;");
    printer->pop_block();
    printer->add_line("return nt->_ml_list[get_mech_type()];");
    printer->pop_block();
}


void CodegenCppVisitor::print_namespace_start() {
    printer->add_newline(2);
    printer->fmt_push_block("namespace {}", namespace_name);
}


void CodegenCppVisitor::print_namespace_stop() {
    printer->pop_block();
}


/****************************************************************************************/
/*                         Routines for returning variable name                         */
/****************************************************************************************/


std::string CodegenCppVisitor::float_variable_name(const SymbolType& symbol,
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


std::string CodegenCppVisitor::int_variable_name(const IndexVariableInfo& symbol,
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


std::string CodegenCppVisitor::global_variable_name(const SymbolType& symbol,
                                                    bool use_instance) const {
    if (use_instance) {
        return fmt::format("inst->{}->{}", naming::INST_GLOBAL_MEMBER, symbol->get_name());
    } else {
        return fmt::format("{}.{}", global_struct_instance(), symbol->get_name());
    }
}


std::string CodegenCppVisitor::update_if_ion_variable_name(const std::string& name) const {
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

// This will need changes for NEURON but should be more or less the same
std::string CodegenCppVisitor::get_variable_name(const std::string& name, bool use_instance) const {
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


void CodegenCppVisitor::print_backend_info() {
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
    printer->add_line("Simulator       : {}", simulator_name());
    printer->add_line("Backend         : ", backend_name());
    printer->add_line("NMODL Compiler  : ", version);
    printer->add_line("*********************************************************/");
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
void CodegenCppVisitor::print_mechanism_global_var_structure(bool print_initializers) {
    const auto value_initialize = print_initializers ? "{}" : "";
    const auto qualifier = global_var_struct_type_qualifier();

    auto float_type = default_float_data_type();
    printer->add_newline(2);
    printer->add_line("/** all global variables */");
    printer->fmt_push_block("struct {}", global_struct());

    for (const auto& ion: info.ions) {
        auto name = fmt::format("{}_type", ion.name);
        printer->fmt_line("{}int {}{};", qualifier, name, value_initialize);
        codegen_global_variables.push_back(make_symbol(name));
    }

    if (info.point_process) {
        printer->fmt_line("{}int point_type{};", qualifier, value_initialize);
        codegen_global_variables.push_back(make_symbol("point_type"));
    }

    for (const auto& var: info.state_vars) {
        auto name = var->get_name() + "0";
        auto symbol = program_symtab->lookup(name);
        if (symbol == nullptr) {
            printer->fmt_line("{}{} {}{};", qualifier, float_type, name, value_initialize);
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
                printer->fmt_line("{}{} {}[{}] /* TODO init top-local-array */;",
                                  qualifier,
                                  float_type,
                                  name,
                                  length);
            } else {
                printer->fmt_line("{}{} {} /* TODO init top-local */;",
                                  qualifier,
                                  float_type,
                                  name);
            }
            codegen_global_variables.push_back(var);
        }
    }

    if (!info.thread_variables.empty()) {
        printer->fmt_line("{}int thread_data_in_use{};", qualifier, value_initialize);
        printer->fmt_line("{}{} thread_data[{}] /* TODO init thread_data */;",
                          qualifier,
                          float_type,
                          info.thread_var_data_size);
        codegen_global_variables.push_back(make_symbol("thread_data_in_use"));
        auto symbol = make_symbol("thread_data");
        symbol->set_as_array(info.thread_var_data_size);
        codegen_global_variables.push_back(symbol);
    }

    // TODO: remove this entirely?
    printer->fmt_line("{}int reset{};", qualifier, value_initialize);
    codegen_global_variables.push_back(make_symbol("reset"));

    printer->fmt_line("{}int mech_type{};", qualifier, value_initialize);
    codegen_global_variables.push_back(make_symbol("mech_type"));

    for (const auto& var: info.global_variables) {
        auto name = var->get_name();
        auto length = var->get_length();
        if (var->is_array()) {
            printer->fmt_line(
                "{}{} {}[{}] /* TODO init const-array */;", qualifier, float_type, name, length);
        } else {
            double value{};
            if (auto const& value_ptr = var->get_value()) {
                value = *value_ptr;
            }
            printer->fmt_line("{}{} {}{};",
                              qualifier,
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
        printer->fmt_line("{}{} {}{};",
                          qualifier,
                          float_type,
                          name,
                          print_initializers ? fmt::format("{{{:g}}}", value) : std::string{});
        codegen_global_variables.push_back(var);
    }

    if (info.primes_size != 0) {
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
        printer->fmt_line("{}int slist1[{}]{};",
                          qualifier,
                          info.primes_size,
                          initializer_list(info.prime_variables_by_order, ""));
        printer->fmt_line("{}int dlist1[{}]{};",
                          qualifier,
                          info.primes_size,
                          initializer_list(info.prime_variables_by_order, "D"));
        codegen_global_variables.push_back(make_symbol("slist1"));
        codegen_global_variables.push_back(make_symbol("dlist1"));
        // additional list for derivimplicit method
        if (info.derivimplicit_used()) {
            auto primes = program_symtab->get_variables_with_properties(NmodlType::prime_name);
            printer->fmt_line("{}int slist2[{}]{};",
                              qualifier,
                              info.primes_size,
                              initializer_list(primes, ""));
            codegen_global_variables.push_back(make_symbol("slist2"));
        }
    }

    if (info.table_count > 0) {
        printer->fmt_line("{}double usetable{};", qualifier, print_initializers ? "{1}" : "");
        codegen_global_variables.push_back(make_symbol(naming::USE_TABLE_VARIABLE));

        for (const auto& block: info.functions_with_table) {
            const auto& name = block->get_node_name();
            printer->fmt_line("{}{} tmin_{}{};", qualifier, float_type, name, value_initialize);
            printer->fmt_line("{}{} mfac_{}{};", qualifier, float_type, name, value_initialize);
            codegen_global_variables.push_back(make_symbol("tmin_" + name));
            codegen_global_variables.push_back(make_symbol("mfac_" + name));
        }

        for (const auto& variable: info.table_statement_variables) {
            auto const name = "t_" + variable->get_name();
            auto const num_values = variable->get_num_values();
            if (variable->is_array()) {
                int array_len = variable->get_length();
                printer->fmt_line("{}{} {}[{}][{}]{};",
                                  qualifier,
                                  float_type,
                                  name,
                                  array_len,
                                  num_values,
                                  value_initialize);
            } else {
                printer->fmt_line(
                    "{}{} {}[{}]{};", qualifier, float_type, name, num_values, value_initialize);
            }
            codegen_global_variables.push_back(make_symbol(name));
        }
    }

    for (const auto& f: info.function_tables) {
        printer->fmt_line("void* _ptable_{}{{}};", f->get_node_name());
        codegen_global_variables.push_back(make_symbol("_ptable_" + f->get_node_name()));
    }

    if (info.vectorize && info.thread_data_index) {
        printer->fmt_line("{}ThreadDatum ext_call_thread[{}]{};",
                          qualifier,
                          info.thread_data_index,
                          value_initialize);
        codegen_global_variables.push_back(make_symbol("ext_call_thread"));
    }

    printer->pop_block(";");

    print_global_var_struct_assertions();
    print_global_var_struct_decl();
}




void CodegenCppVisitor::print_prcellstate_macros() const {
    printer->add_line("#ifndef NRN_PRCELLSTATE");
    printer->add_line("#define NRN_PRCELLSTATE 0");
    printer->add_line("#endif");
}


void CodegenCppVisitor::print_mechanism_info() {
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
void CodegenCppVisitor::print_global_variables_for_hoc() {
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


void CodegenCppVisitor::print_thread_memory_callbacks() {
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


/**
 * \details If floating point type like "float" is specified on command line then
 * we can't turn all variables to new type. This is because certain variables
 * are pointers to internal variables (e.g. ions). Hence, we check if given
 * variable can be safely converted to new type. If so, return new type.
 */
std::string CodegenCppVisitor::get_range_var_float_type(const SymbolType& symbol) {
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





void CodegenCppVisitor::print_functors_definitions() {
    codegen = true;
    for (const auto& functor_name: info.functor_names) {
        printer->add_newline(2);
        print_functor_definition(*functor_name.first);
    }
    codegen = false;
}








void CodegenCppVisitor::print_net_send_call(const FunctionCall& node) {
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


void CodegenCppVisitor::print_net_move_call(const FunctionCall& node) {
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


void CodegenCppVisitor::print_net_event_call(const FunctionCall& node) {
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


void CodegenCppVisitor::visit_for_netcon(const ast::ForNetcon& node) {
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




void CodegenCppVisitor::set_codegen_global_variables(const std::vector<SymbolType>& global_vars) {
    codegen_global_variables = global_vars;
}


void CodegenCppVisitor::setup(const Program& node) {
    program_symtab = node.get_symbol_table();

    CodegenHelperVisitor v;
    info = v.analyze(node);
    info.mod_file = mod_filename;

    if (!info.vectorize) {
        logger->warn("CodegenCppVisitor : MOD file uses non-thread safe constructs of NMODL");
    }

    codegen_float_variables = get_float_variables();
    codegen_int_variables = get_int_variables();

    update_index_semantics();
    rename_function_arguments();
}


void CodegenCppVisitor::visit_program(const Program& node) {
    setup(node);
    print_codegen_routines();
}

}  // namespace codegen
}  // namespace nmodl
