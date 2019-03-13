/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/loop_unroller.hpp"
#include "parser/c11_driver.hpp"
#include "visitors/visitor_utils.hpp"


namespace nmodl {

using namespace ast;

static ast::ExpressionStatement* unroll_for_loop(ast::FromStatement* node) {
    const auto& from = node->get_from();
    const auto& to = node->get_to();
    const auto& increment = node->get_increment();


    if (increment != nullptr && !increment->is_integer()) {
        return nullptr;
    }

    if (!from->is_integer() || !to->is_integer()) {
        return nullptr;
    }

    int n_from = std::dynamic_pointer_cast<ast::Integer>(from)->eval();
    int n_to = std::dynamic_pointer_cast<ast::Integer>(to)->eval();
    int n_increment = 1;

    if (increment != nullptr) {
        n_increment = std::dynamic_pointer_cast<ast::Integer>(increment)->eval();
    }

    int num_unroll = (n_to - n_from) / n_increment;

    std::cout << "Got for unroll : " << num_unroll << " times \n";
    auto block = node->get_statement_block();

    ast::StatementVector statements;
    for (int i = 0; i < num_unroll; i++) {
        auto s = std::make_shared<ExpressionStatement>(block->clone());
        statements.push_back(s);
    }

    auto b = new StatementBlock(std::move(statements));
    auto e = new ExpressionStatement(b);

    std::cout << to_nmodl(e);
}

/**
 * Parse verbatim blocks and rename variable if it is used.
 */
void LoopUnrollVisitor::visit_statement_block(ast::StatementBlock* node) {
    for (auto& statement: node->statements) {
        std::cout << "..";
        if (statement->is_from_statement()) {
            auto s = dynamic_cast<ast::FromStatement*>(statement.get());
            unroll_for_loop(s);
        }
    }
}

}  // namespace nmodl