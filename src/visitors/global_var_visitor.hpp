/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::GlobalToRangeVisitor
 */

#include <sstream>
#include <string>
#include <vector>

#include "ast/ast.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class GlobalToRangeVisitor
 * \brief Visitor for Units blocks of AST
 *
 * We override AstVisitor::visit_global_var to visit the
 * ast::GlobalVar nodes of the ast
 */

class GlobalToRangeVisitor: public AstVisitor {
  private:
    /// ast::Ast* node
    std::shared_ptr<ast::Program> ast;

  public:
    /// \name Ctor & dtor
    /// \{

    /// Default UnitsVisitor constructor
    GlobalToRangeVisitor() = default;

    /// Constructor that takes as parameter the AST
    explicit GlobalToRangeVisitor(std::shared_ptr<ast::Program> node)
        : ast(node) {}

    /// \}

    /// Visit ast::StatementBlock nodes to check if there is any GLOBAL
    /// variables defined in them that are written in any part of the code.
    /// This is checked by reading the write_count member of the variable in
    /// the symtab::SymbolTable. If it's written it removes the variable from
    /// the ast::Global node and adds it to the ast::Range node of the
    /// ast::StatementBlock
    void visit_statement_block(ast::StatementBlock& node) override;
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
