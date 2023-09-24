/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "codegen/codegen_cpp_visitor.hpp"

#include "ast/all.hpp"
#include "codegen/codegen_utils.hpp"
#include "visitors/rename_visitor.hpp"

namespace nmodl {
namespace codegen {

using namespace ast;

using visitor::RenameVisitor;


/****************************************************************************************/
/*                            Overloaded visitor routines                               */
/****************************************************************************************/

extern const std::regex regex_special_chars{R"([-[\]{}()*+?.,\^$|#\s])"};

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


template <typename T>
bool CodegenCppVisitor::has_parameter_of_name(const T& node, const std::string& name) {
    auto parameters = node->get_parameters();
    return std::any_of(parameters.begin(),
                       parameters.end(),
                       [&name](const decltype(*parameters.begin()) arg) {
                           return arg->get_node_name() == name;
                       });
}


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

}  // namespace codegen
}  // namespace nmodl