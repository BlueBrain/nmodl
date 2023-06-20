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

#include "visitors/json_visitor.hpp"

#include "ast/all.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

using namespace ast;

{% for node in nodes %}
void JSONVisitor::visit_{{ node.class_name|snake_case }}(const {{ node.class_name }}& node) {
    {% if node.has_children() %}
    printer->push_block(node.get_node_type_name());
    if (embed_nmodl) {
        printer->add_block_property("nmodl", to_nmodl(node));
    }
    node.visit_children(*this);
    {% if node.is_data_type_node %}
            {% if node.is_integer_node %}
    if(!node.get_macro()) {
        std::stringstream ss;
        ss << node.eval();
        printer->add_node(ss.str());
    }
            {% else %}
    std::stringstream ss;
    ss << node.eval();
    printer->add_node(ss.str());
            {% endif %}
        {% endif %}
    printer->pop_block();
        {% if node.is_program_node %}
    if (node.get_parent() == nullptr) {
        flush();
    }
        {% endif %}
    {% else %}
    (void)node;
    printer->add_node("{{ node.class_name }}");
    {% endif %}
}

{% endfor %}

}  // namespace visitor
}  // namespace nmodl

