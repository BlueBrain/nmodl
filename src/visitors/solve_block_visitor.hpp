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


/**
 * \class SolveBlockVisitor
 * \brief Replace solve block statements with actual solution node in the AST
 *
 * Once sympy or cnexp passes are run, solve blocks can be replaced with solver
 * expression node that represent solution that is going to be solved. All solve
 * statements appearing in breakpoint block are transferred to NrnState block.
 */

class SolveBlockVisitor: public AstVisitor {
  private:
    symtab::SymbolTable* symtab = nullptr;

    bool in_breakpoint_block = false;

    /// solve expression statements for NrnState block
    ast::StatementVector nrn_state_solve_statements;

    ast::SolveExpression* create_solve_expression(ast::SolveBlock* solve_block);

  public:
    SolveBlockVisitor() = default;
    void visit_breakpoint_block(ast::BreakpointBlock* node) override;
    void visit_expression_statement(ast::ExpressionStatement* node) override;
    void visit_program(ast::Program* node) override;
};

}  // namespace nmodl
