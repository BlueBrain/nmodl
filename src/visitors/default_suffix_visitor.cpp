/*
 * Copyright 2024 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "default_suffix_visitor.hpp"

#include "visitor_utils.hpp"

namespace nmodl {
namespace visitor {

void DefaultSuffixVisitor::visit_neuron_block(ast::NeuronBlock& node) {
    const auto& suffix_nodes = collect_nodes(node, {ast::AstNodeType::SUFFIX});
    if (suffix_nodes.empty()) {
        auto default_suffix = mod_path.stem().string();

        auto type = std::make_shared<ast::Name>(std::make_shared<ast::String>("SUFFIX"));
        auto suffix_name = std::make_shared<ast::Name>(
            std::make_shared<ast::String>(default_suffix));
        auto suffix = std::make_shared<ast::Suffix>(type, suffix_name);

        auto stmt_block = node.get_statement_block();
        auto& stmts = stmt_block->get_statements();
        stmt_block->insert_statement(stmts.begin(), suffix);
    }
}

}  // namespace visitor
}  // namespace nmodl
