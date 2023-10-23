/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/sympy_rename_visitor.hpp"

#include "ast/all.hpp"
#include "utils/logger.hpp"


namespace nmodl {
namespace visitor {

void SympyRenameVisitor::visit_indexed_name(ast::IndexedName& node) {
    /// TODO: Do this for indexed names as well
}

void SympyRenameVisitor::visit_prime_name(ast::PrimeName& node) {
    if (!under_sympy) {
        return;
    }
    if (rename_to_sympy) {
        logger->debug("SympyRenameVisitor :: Renaming variable {} to {}", node.get_node_name(), node.get_node_name() + sympy_suffix);
        node.set_value(std::make_shared<ast::String>(node.get_node_name() + sympy_suffix));
        return;
    }
    if (rename_from_sympy) {
        const auto var_name = node.get_node_name();
        if (var_name.length() > sympy_suffix.length() && var_name.substr(var_name.length() - sympy_suffix.length()) == sympy_suffix) {
            logger->debug("SympyRenameVisitor :: Renaming variable {} to {}", node.get_node_name(), var_name.substr(0, var_name.length() - sympy_suffix.length()));
            node.set_value(std::make_shared<ast::String>(var_name.substr(0, var_name.length() - sympy_suffix.length())));
        }
        return;
    }
}

void SympyRenameVisitor::visit_name(ast::Name& node) {
    if (!under_sympy) {
        return;
    }
    if (rename_to_sympy) {
        logger->debug("SympyRenameVisitor :: Renaming variable {} to {}", node.get_node_name(), node.get_node_name() + sympy_suffix);
        node.set_name(node.get_node_name() + sympy_suffix);
        return;
    }
    if (rename_from_sympy) {
        const auto var_name = node.get_node_name();
        if (var_name.length() > sympy_suffix.length() && var_name.substr(var_name.length() - sympy_suffix.length()) == sympy_suffix) {
            logger->debug("SympyRenameVisitor :: Renaming variable {} to {}", node.get_node_name(), var_name.substr(0, var_name.length() - sympy_suffix.length()));
            node.set_name(var_name.substr(0, var_name.length() - sympy_suffix.length()));
        }
        return;
    }
}

void SympyRenameVisitor::visit_solve_block(ast::SolveBlock& node) {
    under_sympy = false;
    node.visit_children(*this);
    under_sympy = true;
}

void SympyRenameVisitor::visit_derivative_block(ast::DerivativeBlock& node) {
    under_sympy = true;
    node.visit_children(*this);
    under_sympy = false;
}

void SympyRenameVisitor::visit_linear_block(ast::LinearBlock& node) {
    under_sympy = true;
    node.visit_children(*this);
    under_sympy = false;
}

void SympyRenameVisitor::visit_non_linear_block(ast::NonLinearBlock& node) {
    under_sympy = true;
    node.visit_children(*this);
    under_sympy = false;
}

}  // namespace visitor
}  // namespace nmodl
