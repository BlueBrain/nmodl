/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::IndexedNameVisitor
 */

#include <string>
#include <unordered_set>

#include "ast/diff_eq_expression.hpp"
#include "ast/indexed_name.hpp"
#include "ast/program.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class IndexedNameVisitor
 * \brief Get node name with indexed for the IndexedName node and
 * the dependencies of DiffEqExpression node
 */
class IndexedNameVisitor: public AstVisitor {
  private:
    std::string indexed_name;
    std::pair<std::string, std::unordered_set<std::string>> dependencies;

  public:
    /// \name Ctor & dtor
    /// \{

    /// Default constructor
    IndexedNameVisitor() = default;

    /// \}

    /// Get node name with index for the IndexedName node
    void visit_indexed_name(ast::IndexedName& node) override;

    /// Get dependencies for the DiffEqExpression node
    void visit_diff_eq_expression(ast::DiffEqExpression& node) override;

    void visit_program(ast::Program& node) override;

    /// get the attribute indexed_name
    std::string get_indexed_name();

    /// get the attribute dependencies
    std::pair<std::string, std::unordered_set<std::string>> get_dependencies();
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
