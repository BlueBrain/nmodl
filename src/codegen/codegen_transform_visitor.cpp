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

#include <memory>

#include "ast/ast_decl.hpp"
#include "ast/function_block.hpp"
#include "ast/string.hpp"
#include "ast/table_statement.hpp"
#include "codegen/codegen_transform_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
using namespace ast;

void CodegenTransformVisitor::visit_function_block(FunctionBlock& node) {
    auto table_statements = collect_nodes(node, {AstNodeType::TABLE_STATEMENT});
    for (auto t: table_statements) {
        auto t_ = std::dynamic_pointer_cast<TableStatement>(t);
        t_->set_table_vars({std::make_shared<Name>(new String(node.get_node_name()))});
    }
}
}  // namespace nmodl
