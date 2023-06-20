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

#include "visitors/ast_visitor.hpp"

#include "ast/all.hpp"


namespace nmodl {
namespace visitor {

using namespace ast;

{% for node in nodes %}
void AstVisitor::visit_{{ node.class_name|snake_case }}({{ node.class_name }}& node) {
    node.visit_children(*this);
}
{% endfor %}

{% for node in nodes %}
void ConstAstVisitor::visit_{{ node.class_name|snake_case }}(const {{ node.class_name }}& node) {
    node.visit_children(*this);
}
{% endfor %}

}  // namespace visitor
}  // namespace nmodl

