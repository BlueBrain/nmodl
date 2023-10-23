/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::SympyRenameVisitor
 */

#include "ast/ast.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

enum SympyRenameType {
  to_sympy,
  from_sympy
};

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class SympyRenameVisitor
 * \brief %Visitor for adding a suffix to all \c Name s related to Sympy
 *
 */
class SympyRenameVisitor: public AstVisitor {
  private: 
    /// string suffix to add to all the variables before passing to sympy
    const std::string sympy_suffix = "_sympyvar";

    /// rename all VarName nodes before passing them to sympy with "_sympyvar"
    bool rename_to_sympy{false};
  
    /// remove the "_sympyvar" suffix from all the VarName nodes after sympy solver
    bool rename_from_sympy{false};

    /// make sure that we are inside AST node related to Sympy to avoid chaning all \c Name s
    bool under_sympy{false};

  public:

    SympyRenameVisitor() = default;

    explicit SympyRenameVisitor(SympyRenameType to_from_sympy) {
      if (to_from_sympy == SympyRenameType::to_sympy) {
        rename_to_sympy = true;
      } else {
        rename_from_sympy = true;
      }
    };

    ~SympyRenameVisitor() = default;

    std::string get_suffix() const {
      return sympy_suffix;
    }

    void visit_prime_name(ast::PrimeName& node) override;
    void visit_name(ast::Name& node) override;
    void visit_solve_block(ast::SolveBlock& node) override;
    void visit_derivative_block(ast::DerivativeBlock& node) override;
    void visit_linear_block(ast::LinearBlock& node) override;
    void visit_non_linear_block(ast::NonLinearBlock& node) override;
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
