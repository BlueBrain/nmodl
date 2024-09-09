/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/derivative_original_visitor.hpp"

#include "ast/all.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {


void DerivativeOriginalVisitor::visit_derivative_block(ast::DerivativeBlock& node) {
    // TODO figure out what happens when we have both a KINETIC and a DERIVATIVE block in the same
    // mod file
    node.visit_children(*this);
    derivative_original = node.clone();
}


void DerivativeOriginalVisitor::visit_derivative_original_block(
    ast::DerivativeOriginalBlock& node) {
    derivative_block = true;
    node.visit_children(*this);
    derivative_block = false;
}


void DerivativeOriginalVisitor::visit_diff_eq_expression(ast::DiffEqExpression& node) {
    differential_equation = true;
    node.visit_children(*this);
    differential_equation = false;
}


void DerivativeOriginalVisitor::visit_binary_expression(ast::BinaryExpression& node) {
    const auto& lhs = node.get_lhs();

    /// we have to only solve ODEs under original derivative block where lhs is variable
    if (!derivative_block || !differential_equation || !lhs->is_var_name()) {
        return;
    }

    auto name = std::dynamic_pointer_cast<ast::VarName>(lhs)->get_name();

    if (name->is_prime_name()) {
        auto varname = "D" + name->get_node_name();
        node.set_lhs(std::make_shared<ast::Name>(new ast::String(varname)));
        if (program_symtab->lookup(varname) == nullptr) {
            auto symbol = std::make_shared<symtab::Symbol>(varname, ModToken());
            symbol->set_original_name(name->get_node_name());
            // TODO check if we need below line
            // symbol->created_from_state();
            program_symtab->insert(symbol);
        }
    }
}

void DerivativeOriginalVisitor::visit_program(ast::Program& node) {
    program_symtab = node.get_symbol_table();
    node.visit_children(*this);
    if (derivative_original) {
        auto der_node =
            new ast::DerivativeOriginalBlock(derivative_original->get_name(),
                                             derivative_original->get_statement_block());
        node.emplace_back_node(der_node);
    }
    // re-visit the AST since we now inserted the DERIVATIVE_ORIGINAL block
    node.visit_children(*this);
}

}  // namespace visitor
}  // namespace nmodl
