/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::TestVisitor
 */

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

class TestVisitor: public AstVisitor {
  private:
    std::string indexed_name;
    std::pair<std::string, std::unordered_set<std::string>> dependencies;

  public:
    /// \name Ctor & dtor
    /// \{

    /// Default constructor
    TestVisitor() = default;

    /// \}

    void visit_indexed_name(ast::IndexedName& node) override;
    void visit_diff_eq_expression(ast::DiffEqExpression& node) override;
    void visit_program(ast::Program& node) override;

    std::string get_indexed_name();
    std::pair<std::string, std::unordered_set<std::string>> get_dependencies();
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
