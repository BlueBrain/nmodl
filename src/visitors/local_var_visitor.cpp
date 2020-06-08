/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>
#include <memory>
#include <unordered_set>

#include "ast/assigned_block.hpp"
#include "ast/assigned_definition.hpp"
#include "ast/local_list_statement.hpp"
#include "ast/local_var.hpp"
#include "ast/node.hpp"
#include "ast/program.hpp"
#include "visitors/local_var_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

void LocalToAssignedVisitor::visit_program(ast::Program& node) {
    ast::AssignedDefinitionVector assigned_variables;
    std::unordered_set<ast::LocalVar*> local_variables_to_remove;
    std::unordered_set<ast::Node*> local_nodes_to_remove;
    std::shared_ptr<ast::AssignedBlock> assigned_block;

    const auto& top_level_nodes = node.get_blocks();
    const auto& symbol_table = node.get_symbol_table();

    for (auto& top_level_node: top_level_nodes) {
        /// only process local_list statements
        if (top_level_node->is_local_list_statement()) {
            const auto& local_variables =
                std::static_pointer_cast<ast::LocalListStatement>(top_level_node)->get_variables();
            for (auto& local_variable: local_variables) {
                auto variable_name = local_variable->get_node_name();
                /// check if local variable is being updated in the mod file
                if (symbol_table->lookup(variable_name)->get_write_count() > 0) {
                    assigned_variables.emplace_back(
                        new ast::AssignedDefinition(local_variable->get_name(),
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr,
                                                    nullptr));
                    local_variables_to_remove.emplace(local_variable.get());
                }
            }

            /// remove offending local variables
            std::static_pointer_cast<ast::LocalListStatement>(top_level_node)
                ->erase_local_var(local_variables_to_remove);

            /// add empty local statements to local_statements_to_remove
            if (local_variables.empty()) {
                local_nodes_to_remove.emplace(top_level_node.get());
            }
        }
        /// save pointer to assigned block to add the assigned variables
        if (top_level_node->is_assigned_block()) {
            assigned_block = std::static_pointer_cast<ast::AssignedBlock>(top_level_node);
        }
    }

    /// remove offending local statements if empty
    node.erase_node(local_nodes_to_remove);

    /// if no assigned block found add one to the node otherwise emplace back new assigned variables
    if (assigned_block == nullptr) {
        assigned_block = std::make_shared<ast::AssignedBlock>(assigned_variables);
        node.emplace_back_node(std::static_pointer_cast<ast::Node>(assigned_block));
    } else {
        for (auto& assigned_variable: assigned_variables) {
            assigned_block->emplace_back_assigned_definition(assigned_variable);
        }
    }
}

}  // namespace visitor
}  // namespace nmodl
