/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>
#include <memory>
#include <unordered_set>

#include "ast/identifier.hpp"
#include "ast/local_list_statement.hpp"
#include "ast/local_var.hpp"
#include "ast/name.hpp"
#include "ast/neuron_block.hpp"
#include "ast/node.hpp"
#include "ast/program.hpp"
#include "ast/range.hpp"
#include "ast/range_var.hpp"
#include "ast/statement_block.hpp"
#include "ast/string.hpp"
#include "visitors/local_var_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

void LocalToRangeVisitor::visit_program(ast::Program& node) {
    ast::RangeVarVector range_variables;
    std::unordered_set<ast::LocalVar*> local_variables_to_remove;
    std::unordered_set<ast::Node*> local_nodes_to_remove;
    std::shared_ptr<ast::NeuronBlock> neuron_block;

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
                    range_variables.emplace_back(
                        new ast::RangeVar(get_name_from_string(local_variable->get_node_name())));
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
        /// save pointer to neuron block to add the range variables
        if (top_level_node->is_neuron_block()) {
            neuron_block = std::static_pointer_cast<ast::NeuronBlock>(top_level_node);
        }
    }

    /// remove offending local statements if empty
    node.erase_node(local_nodes_to_remove);

    /// insert new range variables to statement block of the neuron block
    if (!range_variables.empty()) {
        auto range_statement = new ast::Range(range_variables);
        neuron_block->get_statement_block()->emplace_back_statement(range_statement);
    }
}

}  // namespace visitor
}  // namespace nmodl
