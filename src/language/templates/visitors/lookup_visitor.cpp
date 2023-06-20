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

#include "visitors/lookup_visitor.hpp"

#include <algorithm>

#include "ast/all.hpp"


namespace nmodl {
namespace visitor {

using namespace ast;

{% for node in nodes %}
template <typename DefaultVisitor>
void MetaAstLookupVisitor<DefaultVisitor>::visit_{{ node.class_name|snake_case }}(typename visit_arg_trait<{{ node.class_name }}>::type& node) {
    const auto type = node.get_node_type();
    if (std::find(types.begin(), types.end(), type) != types.end()) {
        nodes.push_back(node.get_shared_ptr());
    }
    node.visit_children(*this);
}
{% endfor %}

template <typename DefaultVisitor>
const typename MetaAstLookupVisitor<DefaultVisitor>::nodes_t&
MetaAstLookupVisitor<DefaultVisitor>::lookup(typename MetaAstLookupVisitor<DefaultVisitor>::ast_t& node,
                                             const std::vector<AstNodeType>& t_types) {
    clear();
    this->types = t_types;
    node.accept(*this);
    return nodes;
}

template <typename DefaultVisitor>
const typename MetaAstLookupVisitor<DefaultVisitor>::nodes_t&
MetaAstLookupVisitor<DefaultVisitor>::lookup(typename MetaAstLookupVisitor<DefaultVisitor>::ast_t& node,
                                             AstNodeType type) {
    clear();
    this->types.push_back(type);
    node.accept(*this);
    return nodes;
}

template <typename DefaultVisitor>
const typename MetaAstLookupVisitor<DefaultVisitor>::nodes_t&
MetaAstLookupVisitor<DefaultVisitor>::lookup(typename MetaAstLookupVisitor<DefaultVisitor>::ast_t& node) {
    nodes.clear();
    node.accept(*this);
    return nodes;
}

// explicit template instantiation definitions
template class MetaAstLookupVisitor<Visitor>;
template class MetaAstLookupVisitor<ConstVisitor>;

}  // namespace visitor
}  // namespace nmodl
