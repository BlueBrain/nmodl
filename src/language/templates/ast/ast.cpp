/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "ast/ast.hpp"
#include "symtab/symbol_table.hpp"

/**
 * \file
 * \brief Auto generated AST classes implementations
 */

namespace nmodl {
namespace ast {

    {% for node in nodes %}

    ///
    /// {{node.class_name}} member functions definition
    ///

    void {{ node.class_name }}::visit_children(visitor::Visitor& v) {
    {% for child in node.non_base_members %}
        {% if child.is_vector %}
        /// visit each element of vector
        for (auto& item : this->{{ child.varname }}) {
            item->accept(v);
        }
        {% elif child.optional %}
        /// optional member could be nullptr
        if (this->{{ child.varname }}) {
            this->{{ child.varname }}->accept(v);
        }
        {% elif child.is_pointer_node %}
        /// use -> for pointer member
        {{ child.varname }}->accept(v);
        {% else %}
        /// use . for object member
        {{ child.varname }}.accept(v);
        {% endif %}
        (void)v;
    {% endfor %}
    }

    {% if node.children %}

    {{ node.ctor_definition() }}

    {% if node.has_ptr_children() %}
        {{ node.ctor_shrptr_definition() }}
    {% endif %}

    /// copy constructor implementation
    {{ node.class_name }}::{{ node.class_name }}(const {{ node.class_name }}& obj) {
        {% for child in node.children %}
            {% if child.is_vector %}
            /// copy each element of vector
            for (auto& item : obj.{{ child.varname }}) {
                this->{{ child.varname }}.emplace_back(item->clone());
            }
            {% elif child.is_pointer_node or child.optional %}
            /// pointer member must be reseted with the new copy
            if (obj.{{ child.varname }}) {
                this->{{ child.varname }}.reset(obj.{{ child.varname }}->clone());
            }
            {% else %}
            /// object member can be just copied by value
            this->{{ child.varname }} = obj.{{ child.varname }};
            {% endif %}
        {% endfor %}

        {% if node.has_token %}
        /// if there is a token, make copy
        if (obj.token) {
            this-> token = std::shared_ptr<ModToken>(obj.token->clone());
        }
        {% endif %}

        /// set parents
        set_parent_in_children();
    }

    /// set this parent in the children
    void {{ node.class_name }}::set_parent_in_children() {

        const auto parentShrPtr = get_shared_ptr();

        {% for child in node.children %}
            {% if child.is_vector %}
            /// set parent for each element of the vector
            for (auto& item : {{ child.varname }}) {
                // this check could be superfluous, may we add nullptr as children?
                if (item) {
                    item->set_parent(parentShrPtr);
                }
            }
            {% elif child.is_pointer_node or child.optional %}
            // this check could be superfluous, may we add nullptr as children?
            if ({{ child.varname }}) {
                {{ child.varname }}->set_parent(parentShrPtr);
            }
            {% endif %}
        {% endfor %}

    }




    {% endif %}

    {% for child in node.children %}
    {{ child.get_setter_method_definition(node.class_name) }}
    {% endfor %}

    {% endfor %}

}  // namespace ast
}  // namespace nmodl
