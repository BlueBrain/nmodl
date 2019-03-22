/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "ast/ast.hpp"
#include "symtab/symbol_table.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {

class SolveBlockVisitor: public AstVisitor {
  private:
    std::shared_ptr<ast::Program> program;

    bool in_breakpoint_block = false;
    ast::StatementVector solve_expresisons;

  public:
    explicit SolveBlockVisitor(std::shared_ptr<ast::Program> program)
        : program(std::move(program))
        , solve_expresisons() {}


    void visit_breakpoint_block(ast::BreakpointBlock* node) override;

    void visit_expression_statement(ast::ExpressionStatement* node) override;
};

}  // namespace nmodl
