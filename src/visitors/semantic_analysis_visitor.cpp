/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/semantic_analysis_visitor.hpp"
#include "ast/breakpoint_block.hpp"
#include "ast/function_block.hpp"
#include "ast/function_call.hpp"
#include "ast/function_table_block.hpp"
#include "ast/independent_block.hpp"
#include "ast/procedure_block.hpp"
#include "ast/program.hpp"
#include "ast/string.hpp"
#include "ast/suffix.hpp"
#include "ast/table_statement.hpp"
#include "symtab/symbol_properties.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

using symtab::syminfo::NmodlType;

bool SemanticAnalysisVisitor::check(const ast::Program& node) {
    check_fail = false;

    /// <-- This code is for check 2
    const auto& suffix_node = collect_nodes(node, {ast::AstNodeType::SUFFIX});
    if (!suffix_node.empty()) {
        const auto& suffix = std::dynamic_pointer_cast<const ast::Suffix>(suffix_node[0]);
        const auto& type = suffix->get_type()->get_node_name();
        is_point_process = (type == "POINT_PROCESS" || type == "ARTIFICIAL_CELL");
    }
    /// -->

    /// <-- This code is for check 4
    using namespace symtab::syminfo;
    const auto& with_prop = NmodlType::read_ion_var | NmodlType::write_ion_var;

    psymtab = node.get_symbol_table();
    assert(psymtab != nullptr);

    // get all ion variables
    const auto& ion_variables = psymtab->get_variables_with_properties(with_prop, false);

    /// make sure ion variables aren't redefined in a `CONSTANT` block.
    for (const auto& var: ion_variables) {
        if (var->has_any_property(NmodlType::constant_var)) {
            logger->critical(
                fmt::format("SemanticAnalysisVisitor :: ion variable {} from the USEION statement "
                            "can not be re-declared in a CONSTANT block",
                            var->get_name()));
            check_fail = true;
        }
    }
    /// -->

    visit_program(node);
    return check_fail;
}

void SemanticAnalysisVisitor::visit_program(const ast::Program& node) {
    psymtab = node.get_symbol_table();
    /// <-- This code is for check 8
    const auto& derivative_block_nodes = collect_nodes(node, {ast::AstNodeType::DERIVATIVE_BLOCK});
    if (derivative_block_nodes.size() > 1) {
        logger->critical("It is not supported to have several DERIVATIVE blocks");
        check_fail = true;
    }
    /// -->
    node.visit_children(*this);
}

void SemanticAnalysisVisitor::visit_function_call(const ast::FunctionCall& node) {
    /// <-- This code is for check 9
    if (visited_function_or_procedure_blocks.empty()) {
        return;
    }
    const auto func_name = node.get_node_name();
    const auto func_already_visited = std::find_if(visited_function_or_procedure_blocks.begin(),
                                                   visited_function_or_procedure_blocks.end(),
                                                   [&func_name](const ast::Block* block) {
                                                       return func_name == block->get_node_name();
                                                   });
    if (func_already_visited != visited_function_or_procedure_blocks.end()) {
        logger->critical(
            fmt::format("SemanticAnalysisVisitor :: Recursive function call of \"{}\" in {}",
                        func_name,
                        node.get_token()->position()));
        check_fail = true;
        return;
    }
    auto func_symbol = psymtab->lookup(func_name);
    // If symbol is not found or there are no AST nodes for it or it's not a function or procedure
    // return
    if (!func_symbol || func_symbol->get_nodes().empty() ||
        !func_symbol->has_any_property(NmodlType::function_block | NmodlType::procedure_block)) {
        return;
    }
    const auto func_block = func_symbol->get_nodes()[0];
    func_block->accept(*this);
    /// -->
}

void SemanticAnalysisVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    /// <-- This code is for check 1
    visited_function_or_procedure_blocks.push_back(&node);
    one_arg_in_procedure_function = node.get_parameters().size() == 1;
    node.visit_children(*this);
    visited_function_or_procedure_blocks.pop_back();
    /// -->
}

void SemanticAnalysisVisitor::visit_function_block(const ast::FunctionBlock& node) {
    /// <-- This code is for check 1
    visited_function_or_procedure_blocks.push_back(&node);
    one_arg_in_procedure_function = node.get_parameters().size() == 1;
    node.visit_children(*this);
    visited_function_or_procedure_blocks.pop_back();
    /// -->
}

void SemanticAnalysisVisitor::visit_table_statement(const ast::TableStatement& tableStmt) {
    /// <-- This code is for check 1
    if (!visited_function_or_procedure_blocks.empty() && !one_arg_in_procedure_function) {
        logger->critical(
            "SemanticAnalysisVisitor :: The procedure or function containing the TABLE statement "
            "should contains exactly one argument.");
        check_fail = true;
    }
    /// -->
    /// <-- This code is for check 3
    const auto& table_vars = tableStmt.get_table_vars();
    const auto first_element = (*visited_function_or_procedure_blocks.cbegin());
    const auto in_function = !visited_function_or_procedure_blocks.empty() &&
                             first_element->is_function_block();
    if (in_function && !table_vars.empty()) {
        logger->critical(fmt::format(
            "SemanticAnalysisVisitor :: TABLE statement in FUNCTION {} cannot have a table name "
            "list.",
            first_element->get_node_name()));
    }
    const auto in_procedure = !visited_function_or_procedure_blocks.empty() &&
                              first_element->is_procedure_block();
    if (in_procedure && table_vars.empty()) {
        logger->critical(
            fmt::format("SemanticAnalysisVisitor :: TABLE statement in PROCEDURE {} must have a "
                        "table name list.",
                        first_element->get_node_name()));
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_destructor_block(const ast::DestructorBlock& /* node */) {
    /// <-- This code is for check 2
    if (!is_point_process) {
        logger->warn(
            "SemanticAnalysisVisitor :: This mod file is not point process but contains a "
            "destructor.");
        check_fail = true;
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_independent_block(const ast::IndependentBlock& node) {
    /// <-- This code is for check 5
    for (const auto& n: node.get_variables()) {
        if (n->get_value()->get_value() != "t") {
            logger->warn(
                "SemanticAnalysisVisitor :: '{}' cannot be used as an independent variable, only "
                "'t' is allowed.",
                n->get_value()->get_value());
        }
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_function_table_block(const ast::FunctionTableBlock& node) {
    /// <-- This code is for check 7
    if (node.get_parameters().size() < 1) {
        logger->critical(
            "SemanticAnalysisVisitor :: Function table '{}' must have one or more arguments.",
            node.get_node_name());
        check_fail = true;
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_protect_statement(const ast::ProtectStatement& /* node */) {
    /// <-- This code is for check 6
    if (accel_backend) {
        logger->error("PROTECT statement is not supported with GPU execution");
    }
    if (in_mutex) {
        logger->warn("SemanticAnalysisVisitor :: Find a PROTECT inside a already locked part.");
    }
    /// -->
}

void SemanticAnalysisVisitor::visit_mutex_lock(const ast::MutexLock& /* node */) {
    /// <-- This code is for check 6
    if (accel_backend) {
        logger->error("MUTEXLOCK statement is not supported with GPU execution");
    }
    if (in_mutex) {
        logger->warn("SemanticAnalysisVisitor :: Found a MUTEXLOCK inside an already locked part.");
    }
    in_mutex = true;
    /// -->
}

void SemanticAnalysisVisitor::visit_mutex_unlock(const ast::MutexUnlock& /* node */) {
    /// <-- This code is for check 6
    if (accel_backend) {
        logger->error("MUTEXUNLOCK statement is not supported with GPU execution");
    }
    if (!in_mutex) {
        logger->warn("SemanticAnalysisVisitor :: Found a MUTEXUNLOCK outside a locked part.");
    }
    in_mutex = false;
    /// -->
}

}  // namespace visitor
}  // namespace nmodl
