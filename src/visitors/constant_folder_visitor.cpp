/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/constant_folder_visitor.hpp"


namespace nmodl {

/// check if given expression is a number
/// note that the DEFINE node is already expanded to integer
static bool is_number(const std::shared_ptr<ast::Expression>& node) {
    return node->is_integer() || node->is_double() || node->is_float();
}

/// get value of a number node
/// TODO : eval method can be added to virtual base class
static double get_value(const std::shared_ptr<ast::Expression>& node) {
    if (node->is_integer()) {
        return std::dynamic_pointer_cast<ast::Integer>(node)->eval();
    }
    if (node->is_float()) {
        return std::dynamic_pointer_cast<ast::Float>(node)->eval();
    }
    if (node->is_double()) {
        return std::dynamic_pointer_cast<ast::Double>(node)->eval();
    }
    throw std::runtime_error("Invalid type passed to is_number()");
}

/// operators that currently implemented
static bool supported_operator(ast::BinaryOp op) {
    return op == ast::BOP_ADDITION || op == ast::BOP_SUBTRACTION || op == ast::BOP_MULTIPLICATION ||
           op == ast::BOP_DIVISION;
}

/// Evaluate binary operation
/// TODO : add support for other binary operators like ^ (pow)
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

/**
 * Visit parenthesis expression and simplify it
 * @param node AST node representing an expression with parenthesis
 *
 * AST could has expression like (1+2). In this case, it has following
 * for in the AST :
 *
 *  parenthesis_exp => wrapped_expr => binary_expression => ...
 *
 * To make constant folding simple, we can remove intermediate wrapped_expr
 * and directly replace binary_expression inside parenthesis_exp :
 *
 *  parenthesis_exp => binary_expression => ...
 */
void ConstantFolderVisitor::visit_paren_expression(ast::ParenExpression* node) {
    node->visit_children(this);
    auto expr = node->get_expression();
    if (expr->is_wrapped_expression()) {
        auto e = std::dynamic_pointer_cast<ast::WrappedExpression>(expr);
        node->set_expression(e->get_expression());
    }
}

/**
 * Visit wrapped node type and perform constant folding
 * @param node AST node that wrap other node types
 *
 * MOD file has expressions like
 *
 * a = 1 + 2
 * DEFINE NN 10
 * FROM i=0 TO NANN-2 {
 *
 * }
 *
 * which need to be turned into
 *
 * a = 1 + 2
 * DEFINE NN 10
 * FROM i=0 TO 8 {
 *
 * }
 */
void ConstantFolderVisitor::visit_wrapped_expression(ast::WrappedExpression* node) {
    node->visit_children(this);

    /// first expression which is wrapped
    auto expr = node->get_expression();

    /// opposite to visit_paren_expression, we might have
    /// a = (2+1)
    /// in this case we can eliminate paren expression and eliminate
    if (expr->is_paren_expression()) {
        auto e = std::dynamic_pointer_cast<ast::ParenExpression>(expr);
        node->set_expression(e->get_expression());
        expr = node->get_expression();
    }

    /// we want to simplify binary expressions only
    if (!expr->is_binary_expression()) {
        return;
    }

    auto binary_expr = std::dynamic_pointer_cast<ast::BinaryExpression>(expr);
    auto lhs = binary_expr->get_lhs();
    auto rhs = binary_expr->get_rhs();
    auto op = binary_expr->get_op().get_value();

    /// in case of expression like
    /// a = 2 + ((1) + (3))
    /// we are in the innermost expression i.e. ((1) + (3))
    /// where (1) and (2) are wrapped expression themself. we can
    /// remove these extra wrapped expressions

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

    /// once we simplify, lhs and rhs must be numbers for constant folding
    if (!is_number(lhs) || !is_number(rhs) || !supported_operator(op)) {
        return;
    }

    /// compute the value of expression
    auto value = compute(get_value(lhs), op, get_value(rhs));

    /// if both operands are not integers or floats, result is double
    if (lhs->is_integer() && rhs->is_integer()) {
        node->set_expression(std::make_shared<ast::Integer>(int(value), nullptr));
    } else if (lhs->is_float() && rhs->is_float()) {
        node->set_expression(std::make_shared<ast::Float>(float(value)));
    } else {
        node->set_expression(std::make_shared<ast::Double>(value));
    }
}


}  // namespace nmodl