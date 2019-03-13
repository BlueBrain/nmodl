/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/constant_folder_visitor.hpp"
#include "visitors/visitor_utils.hpp"


namespace nmodl {

static bool is_number(ast::Expression* node) {
    return node->is_integer() || node->is_double();
}

static double get_value(ast::Expression* node) {
    if (node->is_integer()) {
        return dynamic_cast<ast::Integer*>(node)->eval();
    }
    if (node->is_double()) {
        return dynamic_cast<ast::Double*>(node)->eval();
    }
    throw std::runtime_error("Invalid type passed to is_number()");
}

static double compute(double lhs, ast::BinaryOp op, double rhs) {
    switch (op) {
    case ast::BOP_ADDITION:
        return lhs + rhs;

    case ast::BOP_SUBTRACTION:
        return lhs - rhs;

    case ast::BOP_MULTIPLICATION:
        return lhs * rhs;

    case ast::BOP_DIVISION:
        return lhs / rhs;

    default:
        throw std::logic_error("Invalid binary operator in constant folding");
    }
}


void ConstantFolderVisitor::visit_wrapped_expression(ast::WrappedExpression* node) {
    std::cout << "Before -> " << to_nmodl(node) << "\n";

    visited_wrapped_expressions.push(node);
    node->visit_children(this);

    auto expr = node->get_expression();
    if (!expr->is_binary_expression()) {
        return;
    }

    auto binary_expr = std::dynamic_pointer_cast<ast::BinaryExpression>(expr);
    auto lhs = binary_expr->get_lhs();
    auto rhs = binary_expr->get_rhs();
    auto op = binary_expr->get_op().get_value();

    if (lhs->is_paren_expression()) {
        auto e = std::dynamic_pointer_cast<ast::ParenExpression>(lhs);
        binary_expr->set_lhs(e->get_expression());
        lhs = binary_expr->get_lhs();
    }

    if (rhs->is_paren_expression()) {
        auto e = std::dynamic_pointer_cast<ast::ParenExpression>(rhs);
        binary_expr->set_rhs(e->get_expression());
        rhs = binary_expr->get_rhs();
    }

    if (lhs->is_wrapped_expression()) {
        auto e = std::dynamic_pointer_cast<ast::WrappedExpression>(lhs);
        binary_expr->set_lhs(e->get_expression());
        lhs = binary_expr->get_lhs();
    }

    if (rhs->is_wrapped_expression()) {
        auto e = std::dynamic_pointer_cast<ast::WrappedExpression>(rhs);
        binary_expr->set_rhs(e->get_expression());
        rhs = binary_expr->get_rhs();
    }

    if (!is_number(lhs.get()) || !is_number(rhs.get())) {
        return;
    }

    auto value = compute(get_value(lhs.get()), op, get_value(rhs.get()));

    if (lhs->is_integer() && rhs->is_integer()) {
        node->set_expression(std::make_shared<ast::Integer>(int(value), nullptr));
    } else {
        node->set_expression(std::make_shared<ast::Double>(value));
    }
    std::cout << "After -> " << to_nmodl(node) << "\n";
}


}  // namespace nmodl