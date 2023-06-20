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

///
/// THIS FILE IS GENERATED AT BUILD TIME AND SHALL NOT BE EDITED.
///

#pragma once

#include "ast/ast_decl.hpp"

namespace nmodl {
/// Implementation of different AST visitors
namespace visitor {

/**
 * \defgroup visitor Visitor Implementation
 * \brief All visitors related implementation details
 *
 * \defgroup visitor_classes Visitors
 * \ingroup visitor
 * \brief Different visitors implemented in NMODL
 * \{
 */

/**
 * \brief Abstract base class for all visitors implementation
 *
 * This class defines interface for all concrete visitors implementation.
 * Note that this class only provides interface that could be implemented
 * by concrete visitors like ast::AstVisitor.
 *
 * \sa ast::AstVisitor
 */
class Visitor {
  public:
    virtual ~Visitor() = default;

    {% for node in nodes %}
      /// visit node of type ast::{{ node.class_name }}
      virtual void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}& node) = 0;
    {% endfor %}
};

/**
 * \brief Abstract base class for all constant visitors implementation
 *
 * This class defines interface for all concrete constant visitors implementation.
 * Note that this class only provides interface that could be implemented
 * by concrete visitors like ast::ConstAstVisitor.
 *
 * \sa ast::ConstAstVisitor
 */
class ConstVisitor {
  public:
    virtual ~ConstVisitor() = default;

    {% for node in nodes %}
      /// visit node of type ast::{{ node.class_name }}
      virtual void visit_{{ node.class_name|snake_case }}(const ast::{{ node.class_name }}& node) = 0;
    {% endfor %}
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
