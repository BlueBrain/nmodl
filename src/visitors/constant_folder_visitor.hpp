/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <stack>
#include <string>

#include "ast/ast.hpp"
#include "symtab/symbol_table.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {

/**
 * \class ConstantFolderVisitor
 * \brief Perform constant folding of inter/float expressions
 *
 * Add....
 *
 */

class ConstantFolderVisitor: public AstVisitor {
  private:
    std::stack<ast::WrappedExpression*> visited_wrapped_expressions;

  public:
    ConstantFolderVisitor() = default;
    void visit_wrapped_expression(ast::WrappedExpression* node) override;
    void visit_paren_expression(ast::ParenExpression* node) override;
};

}  // namespace nmodl