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

#include "visitors/var_usage_visitor.hpp"

#include <utility>

#include "ast/name.hpp"


namespace nmodl {
namespace visitor {

/// rename matching variable
void VarUsageVisitor::visit_name(const ast::Name& node) {
    const auto& name = node.get_node_name();
    if (name == var_name) {
        used = true;
    }
}

bool VarUsageVisitor::variable_used(const ast::Node& node, std::string name) {
    used = false;
    var_name = std::move(name);
    node.visit_children(*this);
    return used;
}

}  // namespace visitor
}  // namespace nmodl
