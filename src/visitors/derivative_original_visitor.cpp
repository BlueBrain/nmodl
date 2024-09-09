/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visitors/derivative_original_visitor.hpp"

#include "ast/all.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

using symtab::syminfo::NmodlType;

void DerivativeOriginalVisitor::visit_derivative_block(ast::DerivativeBlock& node) {
    // TODO figure out what happens when we have both a KINETIC and a DERIVATIVE block in the same mod file
    // TODO replace x' with Dx
    node.visit_children(*this);
    derivative_original = node.clone();
}

void DerivativeOriginalVisitor::visit_program(ast::Program& node) {
    node.visit_children(*this);
    if (derivative_original != nullptr) {
        auto der_node =
            new ast::DerivativeOriginalBlock(derivative_original->get_name(),
                                             derivative_original->get_statement_block());
        node.emplace_back_node(der_node);
        // get the statement block
        auto statement_block = std::dynamic_pointer_cast<ast::StatementBlock>(collect_nodes(*der_node, {ast::AstNodeType::STATEMENT_BLOCK})[0]);
        // get all statement expressions
        auto all_statement_exprs = collect_nodes(*der_node, {ast::AstNodeType::EXPRESSION_STATEMENT});

        for (auto& statement: all_statement_exprs){
            auto result = nmodl::to_nmodl(*statement);
            std::cout << result << std::endl;
            std::cout << "Is diffeq: " << statement->is_diff_eq_expression() << std::endl;
        }
        // now that the original DERIVATIVE block is copied, we visit it again and replace x' etc.
        // with Dx etc.
        // TODO add error checking to make sure block is not malformed
        // get the differential equations in the block
        //auto diff_eqs = collect_nodes(*der_node, {ast::AstNodeType::DIFF_EQ_EXPRESSION});
        //for (auto& diff_eq : diff_eqs){
        //    // get all of the statements as strings
        //    auto eq_str = to_nmodl(diff_eq);
        //    auto split_eq = stringutils::split_string(eq_str, '=');
        //    auto x_prime_split = stringutils::split_string(split_eq[0], '\'');
        //    // the actual name of the variable we will replace with Dvariable
        //    auto x = stringutils::trim(x_prime_split[0]);
        //    std::cout << x << std::endl;
        //}
    }
}

}  // namespace visitor
}  // namespace nmodl
