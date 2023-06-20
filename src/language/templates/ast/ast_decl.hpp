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

#include <memory>
#include <vector>

/// \file
/// \brief Auto generated  AST node types and aliases declaration

namespace nmodl {
namespace ast {

/// forward declaration of ast nodes

struct Ast;
{% for node in nodes %}
class {{ node.class_name }};
{% endfor %}

/**
 * @defgroup ast_type AST Node Types
 * @ingroup ast
 * @brief Enum node types for all ast nodes
 * @{
 */

/**
 * \brief Enum type for every AST node type
 *
 * Every node in the ast has associated type represented by this
 * enum class.
 *
 * \sa ast::Ast::get_node_type ast::Ast::get_node_type_name
 */
enum class AstNodeType {
    {% for node in nodes %}
    {{ node.class_name|snake_case|upper }}, ///< type of ast::{{ node.class_name }}
    {% endfor %}
};

/** @} */  // end of ast_type

/**
 * @defgroup ast_vec_type AST Vector Type Aliases
 * @ingroup ast
 * @brief Vector type alias for AST node
 * @{
 */
{% for node in nodes %}
using {{ node.class_name }}Vector = std::vector<std::shared_ptr<{{ node.class_name }}>>;
{% endfor %}

/** @} */  // end of ast_vec_type

}  // namespace ast
}  // namespace nmodl

