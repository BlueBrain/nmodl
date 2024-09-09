/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::DerivativeOriginalVisitor
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
 * \class DerivativeOriginalVisitor
 * \brief Make a copy of the `DERIVATIVE` block (if it exists), and insert back as
 * `DERIVATIVE_ORIGINAL` block.
 *
 * If \ref SympySolverVisitor runs successfully, it replaces the original
 * solution. This block is inserted before that to prevent losing access to
 * information about the block.
 */
class DerivativeOriginalVisitor: public AstVisitor {
  private:
    /// The copy of the derivative block we are solving
    ast::DerivativeBlock* derivative_original = nullptr;

    /// true while visiting differential equation
    bool differential_equation = false;

    /// global symbol table
    symtab::SymbolTable* program_symtab = nullptr;

    /// visiting derivative block
    bool derivative_block = false;

  public:
    void visit_derivative_block(ast::DerivativeBlock& node) override;
    void visit_program(ast::Program& node) override;
    void visit_derivative_original_block(ast::DerivativeOriginalBlock& node) override;
    void visit_diff_eq_expression(ast::DiffEqExpression& node) override;
    void visit_binary_expression(ast::BinaryExpression& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
