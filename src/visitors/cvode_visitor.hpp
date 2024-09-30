/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::CvodeVisitor
 */

#include "symtab/decl.hpp"
#include "visitors/ast_visitor.hpp"
#include <string>

namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class CvodeVisitor
 * \brief Visitor used for generating the necessary AST nodes for CVODE
 */
class CvodeVisitor: public AstVisitor {
  private:
    /// The copy of the derivative block of a given mod file
    std::shared_ptr<ast::DerivativeBlock> der_block = nullptr;

    /// true while visiting differential equation
    bool differential_equation = false;

    /// global symbol table
    symtab::SymbolTable* program_symtab = nullptr;

    /// visiting derivative block
    bool derivative_block = false;

  public:
    void visit_derivative_block(ast::DerivativeBlock& node) override;
    void visit_program(ast::Program& node) override;
    void visit_cvode_block(ast::CvodeBlock& node) override;
    void visit_diff_eq_expression(ast::DiffEqExpression& node) override;
    void visit_binary_expression(ast::BinaryExpression& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
