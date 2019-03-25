/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <set>
#include <vector>

#include "ast/ast.hpp"
#include "symtab/symbol.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/lookup_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {

/**
 * \class SympySolverVisitor
 * \brief Visitor for systems of algebraic and differential equations
 *
 * For DERIVATIVE block, solver method "cnexp":
 *  - replace each ODE with its analytic solution
 *  - optionally using the (1,1) order Pade approximant in dt
 *
 * For DERIVATIVE block, solver method "euler":
 *  - replace each ODE with forwards Euler timestep
 *
 * For DERIVATIVE block, solver method "sparse":
 *  - construct backwards Euler timestep linear system
 *  - for small systems: solves resulting linear algebraic equation by
 *    Gaussian elimination, replaces differential equations
 *    with explicit solution of backwards Euler equations
 *  - for large systems, returns matrix and vector of linear system
 *    to be solved by e.g. LU factorization
 *
 * For DERIVATIVE block, solver method "derivimplicit":
 *  - construct backwards Euler timestep non-linear system
 *  - return function F and its Jacobian J to be solved by newton solver
 *
 * For LINEAR blocks:
 *  - for small systems: solve linear system of algebraic equations by
 *    Gaussian elimination, replace equations with solutions
 *  - for large systems: return matrix and vector of linear system
 *    to be solved by e.g. LU factorization
 *
 * For NON_LINEAR blocks:
 *  - return function F and its Jacobian J to be solved by newton solver
 *
 */

class SympySolverVisitor: public AstVisitor {
  private:
    /// clear any data from previous block
    void clear_previous_block_data();

    /// replace binary expression with new expression provided as string
    static void replace_diffeq_expression(ast::DiffEqExpression* expr, const std::string& new_expr);

    // raise error if kinetic/ode/(non)linear statements are spread over multiple blocks
    void check_expr_statements_in_same_block();

    // return iterator pointing to where solution should be inserted in statement block
    ast::StatementVector::iterator get_solution_location_iterator(ast::StatementVector& statements);

    std::shared_ptr<ast::EigenNewtonSolverBlock> construct_eigen_newton_solver_block(
        const std::vector<std::string>& setup_x,
        const std::vector<std::string>& functor,
        const std::vector<std::string>& update_state);

    /// solve linear system (for "sparse" and "LINEAR")
    void solve_linear_system(const std::vector<std::string>& pre_solve_statements = {});

    /// solve non-linear system (for "derivimplicit" and "NONLINEAR")
    void solve_non_linear_system(const std::vector<std::string>& pre_solve_statements = {});

    /// return NMODL string version of node, excluding any units
    static std::string to_nmodl_for_sympy(ast::AST* node) {
        return nmodl::to_nmodl(node, {ast::AstNodeType::UNIT, ast::AstNodeType::UNIT_DEF});
    }

    /// global variables
    std::set<std::string> global_vars;

    /// local variables in current block + globals
    std::set<std::string> vars;

    /// map between derivative block names and associated solver method
    std::map<std::string, std::string> derivative_block_solve_method{};

    /// expression statements appearing in the block
    /// (these can be of type DiffEqExpression, LinEquation or NonLinEquation)
    std::set<ast::Node*> expression_statements;

    /// current expression statement being visited (to track ODEs / (non)lineqs)
    ast::ExpressionStatement* current_expression_statement;

    /// last expression statement visited (to know where to insert solutions in statement block)
    ast::ExpressionStatement* last_expression_statement = nullptr;

    /// current statement block being visited
    ast::StatementBlock* current_statement_block = nullptr;

    /// block where expression statements appear (to check there is only one)
    ast::StatementBlock* block_with_expression_statements = nullptr;

    /// method specified in solve block
    std::string solve_method;

    /// vector of {ODE, linear eq, non-linear eq} system to solve
    std::vector<std::string> eq_system;

    /// state variables vector
    std::vector<std::string> state_vars;

    /// optionally replace cnexp solution with (1,1) pade approx
    bool use_pade_approx;

    // optionally do CSE (common subexpression elimination) for sparse solver
    bool elimination;

    /// max number of state vars allowed for small system linear solver
    int SMALL_LINEAR_SYSTEM_MAX_STATES;

  public:
    SympySolverVisitor(bool use_pade_approx = false,
                       bool elimination = true,
                       int SMALL_LINEAR_SYSTEM_MAX_STATES = 3)
        : use_pade_approx(use_pade_approx)
        , elimination(elimination)
        , SMALL_LINEAR_SYSTEM_MAX_STATES(SMALL_LINEAR_SYSTEM_MAX_STATES){};

    void visit_diff_eq_expression(ast::DiffEqExpression* node) override;
    void visit_derivative_block(ast::DerivativeBlock* node) override;
    void visit_lin_equation(ast::LinEquation* node) override;
    void visit_linear_block(ast::LinearBlock* node) override;
    void visit_non_lin_equation(ast::NonLinEquation* node) override;
    void visit_non_linear_block(ast::NonLinearBlock* node) override;
    void visit_expression_statement(ast::ExpressionStatement* node) override;
    void visit_statement_block(ast::StatementBlock* node) override;
    void visit_program(ast::Program* node) override;
};

}  // namespace nmodl