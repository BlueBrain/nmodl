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

#include "visitors/checkparent_visitor.hpp"

#include <string>

#include "ast/all.hpp"
#include "utils/logger.hpp"

namespace nmodl {
namespace visitor {
namespace test {

using namespace ast;

int CheckParentVisitor::check_ast(const Ast& node) {

    parent = nullptr;

    node.accept(*this);

    return 0;
}

void CheckParentVisitor::check_parent(const ast::Ast& node) const {
    if (!parent) {
        if (is_root_with_null_parent && node.get_parent()) {
            const auto& parent_type = parent->get_node_type_name();
            throw std::runtime_error(
                fmt::format("root->parent: {} is set when it should be nullptr", parent_type));
        }
    } else {
        if (parent != node.get_parent()) {
            const std::string parent_type = (parent  == nullptr) ? "nullptr" : parent->get_node_type_name();
            const std::string node_parent_type = (node.get_parent() == nullptr) ? "nullptr" : node.get_parent()->get_node_type_name();
            throw std::runtime_error(fmt::format("parent: {} and child->parent: {} missmatch",
                                                 parent_type,
                                                 node_parent_type));
        }
    }
}


{% for node in nodes %}
void CheckParentVisitor::visit_{{ node.class_name|snake_case }}(const {{ node.class_name }}& node) {
    // check the node
    check_parent(node);

    // Set this node as parent. and go down the tree
    parent = &node;

    // visit its children
    node.visit_children(*this);

    // I am done with these children, I go up the tree. The parent of this node is the parent
    parent = node.get_parent();
}

{% endfor %}

}  // namespace test
}  // namespace visitor
}  // namespace nmodl
