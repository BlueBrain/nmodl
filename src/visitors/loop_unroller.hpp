/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "ast/ast.hpp"
#include "symtab/symbol_table.hpp"
#include "visitors/ast_visitor.hpp"


namespace nmodl {

/**
 * \class LoopUnrollVisitor
 * \brief "Blindly" rename given variable to new name
 *
 * During inlining related passes we have to rename variables
 * to avoid name conflicts. This pass "blindly" rename any given
 * variable to new name. The error handling / legality checks are
 * supposed to be done by other higher level passes. For example,
 * local renaming pass should be done from inner-most block to top
 * level block;
 *
 * \todo : Add log/warning messages.
 */

class LoopUnrollVisitor: public AstVisitor {
  private:
    // unroll even if verbatim block exist
    bool unroll_verbatim = true;

  public:
    LoopUnrollVisitor() = default;

    LoopUnrollVisitor(bool unroll_verbatim)
        : unroll_verbatim(unroll_verbatim) {}

    virtual void visit_statement_block(ast::StatementBlock* node) override;
};

}  // namespace nmodl