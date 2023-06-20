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

/**
 *
 * \dir
 * \brief Auto generated visitors
 *
 * \file
 * \brief \copybrief nmodl::visitor::AstVisitor
 */

#include "visitors/visitor.hpp"


namespace nmodl {
namespace visitor {

/**
 * \ingroup visitor_classes
 * \{
 */

/**
 * \brief Concrete visitor for all AST classes
 */
class AstVisitor: public Visitor {
  public:
    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}& node) override;
    {% endfor %}
};

/**
 * \brief Concrete constant visitor for all AST classes
 */
class ConstAstVisitor: public ConstVisitor {
  public:
    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(const ast::{{ node.class_name }}& node) override;
    {% endfor %}
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl

