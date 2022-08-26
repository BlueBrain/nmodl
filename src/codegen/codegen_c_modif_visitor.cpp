/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>
#include <memory>
#include <unordered_set>

#include "ast/ast_decl.hpp"
#include "ast/function_block.hpp"
#include "ast/string.hpp"
#include "ast/table_statement.hpp"
#include "codegen/codegen_c_modif_visitor.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
using namespace ast;

void CodegenCModifVisitor::visit_function_block(FunctionBlock& node) {
    auto table_statements = collect_nodes(node, {AstNodeType::TABLE_STATEMENT});
    for (auto t: table_statements) {
        auto t_ = std::dynamic_pointer_cast<TableStatement>(t);
        t_->set_table_vars({std::make_shared<Name>(new String(node.get_node_name()))});
    }
}
}  // namespace nmodl
