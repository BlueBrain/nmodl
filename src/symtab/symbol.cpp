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

#include "symtab/symbol.hpp"
#include "utils/logger.hpp"
#include <ast/ast.hpp>

namespace nmodl {
namespace symtab {

using syminfo::NmodlType;
using syminfo::Status;


bool Symbol::is_variable() const noexcept {
    // if symbol has one of the following property then it
    // is considered as variable in the NMODL
    // clang-format off
        NmodlType var_properties = NmodlType::local_var
                                    | NmodlType::global_var
                                    | NmodlType::range_var
                                    | NmodlType::param_assign
                                    | NmodlType::pointer_var
                                    | NmodlType::bbcore_pointer_var
                                    | NmodlType::extern_var
                                    | NmodlType::assigned_definition
                                    | NmodlType::read_ion_var
                                    | NmodlType::write_ion_var
                                    | NmodlType::nonspecific_cur_var
                                    | NmodlType::electrode_cur_var
                                    | NmodlType::argument
                                    | NmodlType::extern_neuron_variable;
    // clang-format on
    return has_any_property(var_properties);
}

std::string Symbol::to_string() const {
    std::string s(name);
    if (properties != NmodlType::empty) {
        s += fmt::format(" [Properties : {}]", syminfo::to_string(properties));
    }
    if (status != Status::empty) {
        s += fmt::format(" [Status : {}]", syminfo::to_string(status));
    }
    return s;
}

std::vector<ast::Ast*> Symbol::get_nodes_by_type(
    std::initializer_list<ast::AstNodeType> l) const noexcept {
    std::vector<ast::Ast*> _nodes;
    for (const auto& n: nodes) {
        for (const auto& m: l) {
            if (n->get_node_type() == m) {
                _nodes.push_back(n);
                break;
            }
        }
    }
    return _nodes;
}

}  // namespace symtab
}  // namespace nmodl
