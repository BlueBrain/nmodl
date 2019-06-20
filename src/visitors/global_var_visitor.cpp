/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <iostream>
#include <memory>

#include "ast/ast.hpp"
#include "global_var_visitor.hpp"
/**
 * \file
 * \brief AST Visitor to change nmodl::symtab::syminfo::NmodlType::global_var
 * variables in nmodl::symtab to nmodl::symtab::syminfo::NmodlType::range_var
 * so that they can be handled by Codegen as RANGE variables
 */

namespace nmodl {
namespace visitor {

using symtab::syminfo::NmodlType;

void GlobalToRangeVisitor::visit_global_var(ast::GlobalVar* node) {
    if (ast->get_symbol_table()->lookup(node->get_node_name())->get_write_count() > 0) {
        ast->get_symbol_table()
            ->lookup(node->get_node_name())
            ->remove_property(NmodlType::global_var);
        ast->get_symbol_table()->lookup(node->get_node_name())->add_property(NmodlType::range_var);
    }
}

}  // namespace visitor
}  // namespace nmodl
