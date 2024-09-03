/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/derivative_original_visitor.hpp"

#include "ast/all.hpp"

namespace nmodl {
namespace visitor {


void DerivativeOriginalVisitor::visit_derivative_block(ast::DerivativeBlock& node) {
    derivative_original = node.clone();
    node.visit_children(*this);
}

void DerivativeOriginalVisitor::visit_program(ast::Program& node) {
    node.visit_children(*this);
    if (derivative_original) {
        auto der_node =
            new ast::DerivativeOriginalBlock(derivative_original->get_name(),
                                             derivative_original->get_statement_block());
        node.emplace_back_node(der_node);
    }
}

}  // namespace visitor
}  // namespace nmodl
