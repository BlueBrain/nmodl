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

#include "visitors/indexedname_visitor.hpp"
#include "ast/binary_expression.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

void IndexedNameVisitor::visit_indexed_name(ast::IndexedName& node) {
    indexed_name = nmodl::get_indexed_name(node);
}

void IndexedNameVisitor::visit_diff_eq_expression(ast::DiffEqExpression& node) {
    node.visit_children(*this);
    const auto& bin_exp = std::static_pointer_cast<ast::BinaryExpression>(node.get_expression());
    auto lhs = bin_exp->get_lhs();
    auto rhs = bin_exp->get_rhs();
    dependencies = nmodl::statement_dependencies(lhs, rhs);
}

void IndexedNameVisitor::visit_program(ast::Program& node) {
    node.visit_children(*this);
}
std::pair<std::string, std::unordered_set<std::string>> IndexedNameVisitor::get_dependencies() {
    return dependencies;
}
std::string IndexedNameVisitor::get_indexed_name() {
    return indexed_name;
}

}  // namespace visitor
}  // namespace nmodl
