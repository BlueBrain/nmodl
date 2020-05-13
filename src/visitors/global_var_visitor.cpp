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

void GlobalToRangeVisitor::visit_global_var(ast::GlobalVar& node) {
    if (ast->get_symbol_table()->lookup(node.get_node_name())->get_write_count() > 0) {
        ast->get_symbol_table()
            ->lookup(node.get_node_name())
            ->remove_property(NmodlType::global_var);
        ast->get_symbol_table()->lookup(node.get_node_name())->add_property(NmodlType::range_var);
    }
}

void GlobalToRangeVisitor::visit_statement_block(ast::StatementBlock& node) {
    auto& statements = node.get_statements();
    ast::RangeVarVector range_variables;
    for(auto& statement : statements) {
        std::cout << statement->get_node_type_name() << std::endl;
        if(statement->is_global()) {
            auto global_variables = std::static_pointer_cast<ast::Global>(statement)->get_variables();
            for (auto global_variable = global_variables.begin();
                 global_variable != global_variables.end(); global_variable++) {
                auto variable_name = (*global_variable)->get_node_name();
                if (ast->get_symbol_table()->lookup(variable_name)->get_write_count() > 0) {
                    /*ast->get_symbol_table()
                            ->lookup(variable_name)
                            ->remove_property(NmodlType::global_var);
                    ast->get_symbol_table()->lookup(variable_name)->add_property(NmodlType::range_var);*/
                    global_variables.erase(global_variable);
                    /*auto range_declaration =
                    R"(NEURON {
                        RANGE )" + variable_name + R"(
                    })";
                    std::cout << range_declaration << std::endl;
                    auto range_statement = create_statement(range_declaration);
                    node.insert_statement(statements.begin(), range_statement);*/
                    //auto range_statement = std::make_shared<ast::Statement>(ast::RangeVarVector());
                    //node.emplace_back_statement(range_statement);
                    range_variables.emplace_back(new ast::RangeVar((*global_variable)->get_name()));
                }
            }
        }
    }
    if (!range_variables.empty()) {
        auto range_statement = new ast::Range(range_variables);
        node.emplace_back_statement(range_statement);
    }
    for(auto& statement : statements) {
        std::cout << statement->get_node_type_name() << std::endl;
        if(statement->is_global()) {
            auto global_variables = std::dynamic_pointer_cast<ast::Global>(statement)->get_variables();
            for (auto & global_variable : global_variables) {
                auto variable_name = global_variable->get_node_name();
                std::cout << variable_name << std::endl;
            }
        }
        if(statement->is_range()) {
            auto range_variables = std::dynamic_pointer_cast<ast::Range>(statement)->get_variables();
            for (auto & range_variable : range_variables) {
                auto variable_name = range_variable->get_node_name();
                std::cout << variable_name << std::endl;
            }
        }
    }
}

}  // namespace visitor
}  // namespace nmodl
