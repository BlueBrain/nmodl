/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>
#include <memory>

#include "ast/ast.hpp"
#include "global_var_visitor.hpp"
/**
 * \file
 * \brief AST Visitor to change nmodl::symtab::syminfo::NmodlType::global_var
 * variables in nmodl::symtab to nmodl::symtab::syminfo::NmodlType::range_var
 * so that they can be handled by Codegen as RANGE variables
 */

namespace nmodl {
namespace visitor {

using symtab::syminfo::NmodlType;

void GlobalToRangeVisitor::visit_statement_block(ast::StatementBlock& node) {
    auto& statements = node.get_statements();
    ast::RangeVarVector range_variables;
    std::vector<std::string> global_vars_to_remove;
    for(auto& statement : statements) {
        std::cout << statement->get_node_type_name() << std::endl;
        if(statement->is_global()) {
            auto& global_variables = std::static_pointer_cast<ast::Global>(statement)->get_variables();
            global_vars_to_remove.clear();
            for (auto& global_variable : global_variables) {
                auto variable_name = global_variable->get_node_name();
                if (ast->get_symbol_table()->lookup(variable_name)->get_write_count() > 0) {
                    range_variables.emplace_back(new ast::RangeVar(global_variable->get_name()));
                    global_vars_to_remove.emplace_back(variable_name);
                }
            }
            for(auto& global_var_to_remove : global_vars_to_remove) {
                std::static_pointer_cast<ast::Global>(statement)->erase_global_var(find_if(global_variables.begin(), global_variables.end(), [global_var_to_remove] (std::shared_ptr<nmodl::ast::GlobalVar> var) { return var->get_node_name() == global_var_to_remove; } ));
            }
        }
    }
    if (!range_variables.empty()) {
        auto range_statement = new ast::Range(range_variables);
        node.emplace_back_statement(range_statement);
    }
}

}  // namespace visitor
}  // namespace nmodl
