/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "ast/ast.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {

/**
 * \class SteadystateVisitor
 * \brief Visitor for STEADYSTATE solve statements
 *
 * For each STEADYSTATE solve statement, creates a new
 * LINEAR or NONLINEAR block whose solution
 * gives the steady state solution of the
 * corresponding derivative block.
 *
 * If the derivative block was called "X", the
 * new (NON)LINEAR block will be called "X_steadystate"
 *
 * Updates the solve statement to point to the new
 * (NON)LINEAR block "X_steadystate"
 *
 */

class SteadystateVisitor: public AstVisitor {
  private:
    /// set of steadystate sparse (i.e. linear) derivative block nodes
    std::set<ast::DerivativeBlock*> ss_linear_blocks;

    /// set of steadystate derivimplicit(i.e. nonlinear) derivative block nodes
    std::set<ast::DerivativeBlock*> ss_nonlinear_blocks;

    /// true if we are visiting an ss deriv block
    bool is_ss_deriv_block = false;

    /// statements to remove from block
    std::set<ast::Node*> statements_to_remove;

    /// current statement block being visited
    ast::StatementBlock* current_statement_block = nullptr;

    /// current expression statement being visited
    ast::ExpressionStatement* current_expression_statement = nullptr;

    /// new equations to add to block
    std::vector<std::string> new_eqs;

  public:
    SteadystateVisitor() = default;

    void visit_diff_eq_expression(ast::DiffEqExpression* node) override;
    void visit_expression_statement(ast::ExpressionStatement* node) override;
    void visit_statement_block(ast::StatementBlock* node) override;
    void visit_derivative_block(ast::DerivativeBlock* node) override;
    void visit_program(ast::Program* node) override;
};

}  // namespace nmodl