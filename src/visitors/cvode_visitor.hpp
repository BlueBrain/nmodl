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
#include <unordered_set>

namespace nmodl {
namespace visitor {

enum class BlockIndex { FUNCTION = 0, JACOBIAN = 1 };

inline BlockIndex& operator++(BlockIndex& index) {
    if (index == BlockIndex::FUNCTION) {
        index = BlockIndex::JACOBIAN;
    } else {
        index = BlockIndex::FUNCTION;
    }
    return index;
}
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
    std::shared_ptr<ast::DerivativeBlock> derivative_block = nullptr;

    /// true while visiting differential equation
    bool in_differential_equation = false;

    /// global symbol table
    symtab::SymbolTable* program_symtab = nullptr;

    /// true while we are visiting a CVODE block
    bool in_cvode_block = false;

    /// index of the block to modify
    BlockIndex block_index = BlockIndex::FUNCTION;

    /// list of conserve equations encountered
    std::unordered_set<ast::Statement*> conserve_equations;

  public:
    void visit_derivative_block(ast::DerivativeBlock& node) override;
    void visit_program(ast::Program& node) override;
    void visit_cvode_block(ast::CvodeBlock& node) override;
    void visit_diff_eq_expression(ast::DiffEqExpression& node) override;
    void visit_binary_expression(ast::BinaryExpression& node) override;
    void visit_statement_block(ast::StatementBlock& node) override;
    void visit_conserve(ast::Conserve& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
