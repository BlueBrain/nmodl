/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/solve_block_visitor.hpp"
#include "utils/logger.hpp"
#include "visitors/lookup_visitor.hpp"

namespace nmodl {


void SolveBlockVisitor::visit_breakpoint_block(ast::BreakpointBlock* node) {
    in_breakpoint_block = true;
    node->visit_children(this);
    in_breakpoint_block = false;
}


void SolveBlockVisitor::visit_expression_statement(ast::ExpressionStatement* node) {
    node->visit_children(this);
    if (node->get_expression()->is_solve_block()) {
        auto solve_block = std::dynamic_pointer_cast<ast::SolveBlock>(node->get_expression());
        auto sb_name = solve_block->get_block_name();
        auto solve_func = symtab->lookup(sb_name->get_node_name());
        auto block_to_solve = solve_func->get_node()->get_statement_block();
        auto sexp = new ast::SolveExpression(solve_block, block_to_solve);
        if (in_breakpoint_block) {
            logger->debug("found in breakpoint solve block and func {}", solve_func->get_name());
            solve_expresisons.emplace_back(new ast::ExpressionStatement(sexp));
        } else {
            node->set_expression(std::shared_ptr<ast::SolveExpression>(sexp));
        }
    }
}

}  // namespace nmodl
