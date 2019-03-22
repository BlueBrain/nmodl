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
    symtab::SymbolTable* symtab = nullptr;
    ast::StatementVector solve_expresisons;
    bool in_breakpoint_block = false;

  public:
    SolveBlockVisitor() = default;

    void visit_breakpoint_block(ast::BreakpointBlock* node) override;
    void visit_expression_statement(ast::ExpressionStatement* node) override;

    virtual void visit_program(ast::Program* node) override {
        symtab = node->get_symbol_table();
        node->visit_children(this);
        auto nrn_state = new ast::NrnStateBlock(solve_expresisons);
        node->addNode(nrn_state);
    }
};

}  // namespace nmodl
