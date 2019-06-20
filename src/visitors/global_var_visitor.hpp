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
    ast::Ast* ast;

  public:
    /// \name Ctor & dtor
    /// \{

    /// Default UnitsVisitor constructor
    GlobalToRangeVisitor() = default;

    /// Constructor that takes as parameter the AST
    explicit GlobalToRangeVisitor(ast::Ast* node)
        : ast(node) {}

    /// \}

    /// Function to visit all the ast::GlobalVar nodes in order to check if
    /// any of the GLOBAL vars are written in any part of the code. This is
    /// checked by reading the write_count member of the variable in the
    /// symtab::SymbolTable. If the variable is writted, then its property
    /// nmodl::symtab::syminfo::NmodlType::global_var is changed to
    /// nmodl::symtab::syminfo::NmodlType::range_var in order to become
    /// part of the Instance of the mechanism in the \c .cpp generated file
    void visit_global_var(ast::GlobalVar* node) override;
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
