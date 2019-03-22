/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "ast/ast.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {

/**
 * \class CnexpSolveVisitor
 * \brief Visitor that solves and replaces ODEs using cnexp method
 *
 * This pass solves ODEs in derivative block if cnexp method is used.
 * The original ODEs get replaced with the solution. This transformation
 * is performed at ast level. This is useful for performance modeling
 * purpose where we want to measure performance metrics using perfvisitor
 * pass.
 */

class CnexpSolveVisitor: public AstVisitor {
  private:
    /// true while visiting differential equation
    bool differential_equation = false;

    std::map<std::string, bool> keep_derivative_block;

    /// name of the cnexp method
    const std::string cnexp_method = "cnexp";

    /// name of the derivimplicit method
    const std::string derivimplicit_method = "derivimplicit";

    /// name of the euler method
    const std::string euler_method = "euler";

    symtab::SymbolTable* program_symtab = nullptr;

    /// a map holding solve block names and methods
    std::map<std::string, std::string> solve_blocks;

    /// the derivate name currently being considered
    std::string derivative_block_name;

  public:
    CnexpSolveVisitor() = default;

    void visit_solve_block(ast::SolveBlock* node) override;
    void visit_diff_eq_expression(ast::DiffEqExpression* node) override;
    void visit_binary_expression(ast::BinaryExpression* node) override;
    void visit_program(ast::Program* node) override;
    void visit_derivative_block(ast::DerivativeBlock* node) override;
};

}  // namespace nmodl