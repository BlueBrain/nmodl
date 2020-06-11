/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::IspcRenameVisitor
 */

#include <regex>
#include <string>

#include "visitors/ast_visitor.hpp"


namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class IspcRenameVisitor
 * \brief Rename variable names to match ISPC standards
 *
 * Here are some examples of double constants that
 * ISPC compiler parses: 3.14d, 31.4d-1, 1.d, 1.0d,
 * 1d-2. Rename variables based on regex that matches
 * those presentations.
 *
 */
class IspcRenameVisitor: public AstVisitor {
  private:
    /// ast::Ast* node
    std::shared_ptr<ast::Program> ast;

    /// regex that matches double constant expressions
    const std::regex double_regex = std::move(
        std::regex("([0-9\\.]*d[\\-0-9]+)|([0-9\\.]+d[\\-0-9]*)"));

    /// new name
    const std::string new_var_name_prefix = "var_";

    // rename verbatim blocks as well
    bool rename_verbatim = true;

  public:
    /// Default constructor
    IspcRenameVisitor() = delete;

    /// Constructor that takes as parameter the AST
    explicit IspcRenameVisitor(std::shared_ptr<ast::Program> node)
        : ast(std::move(node)) {}

    void enable_verbatim(bool state) {
        rename_verbatim = state;
    }

    void visit_name(ast::Name& node) override;
    void visit_prime_name(ast::PrimeName& node) override;
    void visit_verbatim(ast::Verbatim& node) override;
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
