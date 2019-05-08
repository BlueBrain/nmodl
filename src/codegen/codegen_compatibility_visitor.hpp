/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::codegen::CodegenCompatibilityVisitor
 */

#include "ast/ast.hpp"
#include "symtab/symbol_table.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace codegen {

using namespace ast;

/**
 * @addtogroup codegen_backends
 * @{
 */

/**
 * \class CodegenCompatibilityVisitor
 * \brief %Visitor for printing compatibility issues of the mod file
 */
class CodegenCompatibilityVisitor: public visitor::AstVisitor {
    /// Vector that stores all the ast::Nodes that are incompatible
    /// with NMODL \c C++ code generator
    std::vector<std::shared_ptr<ast::Ast>> incompatible_ast_nodes;

  public:
    /// \name Ctor & dtor
    /// \{

    /// Default CodegenCompatibilityVisitor constructor
    CodegenCompatibilityVisitor() = default;

    /// \}

    /// Function that searches the ast::Ast for nodes that
    /// are incompatible with NMODL \c C++ code generator
    bool find_incompatible_ast_nodes(Ast* node);
};

/** @} */  // end of codegen_backends

}  // namespace codegen
}  // namespace nmodl
