/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>
#include <memory>

#include "ast/global.hpp"
#include "ast/program.hpp"
#include "ast/range.hpp"
#include "ast/statement_block.hpp"
#include "visitors/global_var_visitor.hpp"
/**
 * \file
 * \brief AST Visitor to change ast::GlobalVar to ast::Range
 * variables in nmodl::symtab to nmodl::symtab::syminfo::NmodlType::range_var
 * so that they can be handled by Codegen as RANGE variables
 */

namespace nmodl {
namespace visitor {

void GlobalToRangeVisitor::visit_statement_block(ast::StatementBlock& node) {
    const auto& statements = node.get_statements();
    ast::RangeVarVector range_variables;
    std::vector<std::string> global_vars_to_remove;
    for (const auto& statement: statements) {
        if (statement->is_global()) {
            const auto& global_variables =
                std::static_pointer_cast<ast::Global>(statement)->get_variables();
            global_vars_to_remove.clear();
            for (const auto& global_variable: global_variables) {
                auto variable_name = global_variable->get_node_name();
                if (ast->get_symbol_table()->lookup(variable_name)->get_write_count() > 0) {
                    range_variables.emplace_back(new ast::RangeVar(global_variable->get_name()));
                    global_vars_to_remove.emplace_back(variable_name);
                }
            }
            for (const auto& global_var_to_remove: global_vars_to_remove) {
                const auto& global_var_iter =
                    find_if(global_variables.begin(),
                            global_variables.end(),
                            [&](const std::shared_ptr<nmodl::ast::GlobalVar>& var) {
                                return var->get_node_name() == global_var_to_remove;
                            });
                std::static_pointer_cast<ast::Global>(statement)->erase_global_var(global_var_iter);
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
