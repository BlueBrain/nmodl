/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/sympy_replace_solutions_visitor.hpp"
#include "visitors/lookup_visitor.hpp"


#include "ast/all.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {

SympyReplaceSolutionsVisitor::SympyReplaceSolutionsVisitor(
    const std::vector<std::string>& pre_solve_statements,
    const std::vector<std::string>& solutions,
    const std::unordered_set<ast::Statement*>& to_be_removed)
    : pre_solve_statements_(pre_solve_statements.begin(), pre_solve_statements.end())
    , to_be_removed_(&to_be_removed) {
    const auto ss_tmp_delimeter =
        std::find_if(solutions.begin(), solutions.end(), [](const std::string& statement) {
            return statement.substr(0, 3) != "tmp";
        });
    tmp_statements_ = SolutionSorter(solutions.begin(), ss_tmp_delimeter);
    solutions_ = SolutionSorter(ss_tmp_delimeter, solutions.end());

    replacements_.clear();
}

void SympyReplaceSolutionsVisitor::visit_statement_block(ast::StatementBlock& node) {
    const bool dump_at_the_end = is_statement_block_root_;
    is_statement_block_root_ = false;

    policy_ = ReplacePolicy::VALUE;
    node.visit_children(*this);

    if (!solutions_.is_all_untagged()) {
        policy_ = ReplacePolicy::GREEDY;
        node.visit_children(*this);
    }

    const auto& old_statements = node.get_statements();

    ast::StatementVector new_statements;
    new_statements.reserve(2 * old_statements.size());
    for (const auto& old_statement: old_statements) {
        const auto& replacement_ptr = replacements_.find(old_statement);
        if (replacement_ptr != replacements_.end()) {
            new_statements.insert(new_statements.end(),
                                  replacement_ptr->second.begin(),
                                  replacement_ptr->second.end());

        } else if (to_be_removed_ == nullptr ||
                   to_be_removed_->find(&(*old_statement)) == to_be_removed_->end()) {
            new_statements.emplace_back(std::move(old_statement));
        }
    }


    if (dump_at_the_end) {
        pre_solve_statements_.emplace_back_all_statements(new_statements);
        tmp_statements_.emplace_back_all_statements(new_statements);
        solutions_.emplace_back_all_statements(new_statements);
    }

    node.set_statements(std::move(new_statements));
}

void SympyReplaceSolutionsVisitor::visit_diff_eq_expression(ast::DiffEqExpression& node) {
    const auto& statement = std::static_pointer_cast<ast::Statement>(
        node.get_parent()->get_shared_ptr());

    // do not visit if already marked
    if (replacements_.find(statement) != replacements_.end()) {
        return;
    }

    switch (policy_) {
    case ReplacePolicy::VALUE: {
        const auto dependencies = statement_dependencies(node.get_expression()->get_lhs(),
                                                         node.get_expression()->get_rhs());
        const auto& key = dependencies.first;
        const auto& vars = dependencies.second;

        if (solutions_.is_var_assigned_here(key)) {
            ast::StatementVector new_statements;

            pre_solve_statements_.emplace_back_all_statements(new_statements);
            tmp_statements_.emplace_back_all_statements(new_statements);
            solutions_.try_emplace_back_statement(new_statements, key);

            replacements_.emplace(statement, new_statements);
        }
        break;
    }
    case ReplacePolicy::GREEDY: {
        if (!solutions_.is_all_untagged()) {
            ast::StatementVector new_statements;

            pre_solve_statements_.emplace_back_all_statements(new_statements);
            tmp_statements_.emplace_back_all_statements(new_statements);
            solutions_.emplace_back_next_statement(new_statements);

            replacements_.emplace(statement, new_statements);
        }
        break;
    }
    }
}

void SympyReplaceSolutionsVisitor::visit_lin_equation(ast::LinEquation& node) {
    const auto& statement = std::static_pointer_cast<ast::Statement>(
        node.get_parent()->get_shared_ptr());

    const auto dependencies = statement_dependencies(node.get_left_linxpression(),
                                                     node.get_linxpression());

    // do not visit if already marked
    if (replacements_.find(statement) != replacements_.end()) {
        return;
    }

    switch (policy_) {
    case ReplacePolicy::VALUE: {
        const auto dependencies = statement_dependencies(node.get_left_linxpression(),
                                                         node.get_linxpression());
        const auto& key = dependencies.first;
        const auto& vars = dependencies.second;

        if (solutions_.is_var_assigned_here(key)) {
            ast::StatementVector new_statements;

            pre_solve_statements_.emplace_back_all_statements(new_statements);
            tmp_statements_.emplace_back_all_statements(new_statements);
            solutions_.try_emplace_back_statement(new_statements, key);

            replacements_.emplace(statement, new_statements);
        }
        break;
    }
    case ReplacePolicy::GREEDY: {
        if (!solutions_.is_all_untagged()) {
            ast::StatementVector new_statements;

            pre_solve_statements_.emplace_back_all_statements(new_statements);
            tmp_statements_.emplace_back_all_statements(new_statements);
            solutions_.emplace_back_next_statement(new_statements);

            replacements_.emplace(statement, new_statements);
        }
        break;
    }
    }
}

void SympyReplaceSolutionsVisitor::visit_binary_expression(ast::BinaryExpression& node) {
    if (node.get_op().get_value() == ast::BinaryOp::BOP_ASSIGN) {
        const auto& lhs = node.get_lhs();
        if (lhs->is_var_name()) {
            const auto& var =
                std::static_pointer_cast<ast::VarName>(lhs)->get_name()->get_node_name();
            pre_solve_statements_.tag_statement(var);
            tmp_statements_.tag_statement(var);
        }
    }
}


//// TODO add warnings!
std::pair<std::string, std::unordered_set<std::string>>
SympyReplaceSolutionsVisitor::statement_dependencies(const std::shared_ptr<ast::Expression>& lhs,
                                                     const std::shared_ptr<ast::Expression>& rhs) {
    std::string key;
    std::unordered_set<std::string> out;

    if (!lhs->is_var_name()) {
        return {key, out};
    }

    key = to_nmodl(lhs);
    visitor::AstLookupVisitor lookup_visitor;
    lookup_visitor.lookup(*rhs, ast::AstNodeType::VAR_NAME);
    auto rhs_nodes = lookup_visitor.get_nodes();
    std::for_each(rhs_nodes.begin(),
                  rhs_nodes.end(),
                  [&out](const std::shared_ptr<ast::Ast>& node) { out.emplace(to_nmodl(node)); });


    return {key, out};
}

SympyReplaceSolutionsVisitor::SolutionSorter::SolutionSorter(
    const std::vector<std::string>::const_iterator& statements_str_beg,
    const std::vector<std::string>::const_iterator& statements_str_end)
    : statements_(create_statements(statements_str_beg, statements_str_end))
    , tags_(statements_.size(), true) {
    build_maps();
}


void SympyReplaceSolutionsVisitor::SolutionSorter::build_maps() {
    for (size_t ii = 0; ii < statements_.size(); ++ii) {
        const auto& statement = statements_[ii];

        if (statement->is_expression_statement()) {
            const auto& e_statement =
                std::static_pointer_cast<ast::ExpressionStatement>(statement)->get_expression();
            if (e_statement->is_binary_expression()) {
                const auto& bin_exp = std::static_pointer_cast<ast::BinaryExpression>(e_statement);
                const auto& dependencies =
                    SympyReplaceSolutionsVisitor::statement_dependencies(bin_exp->get_lhs(),
                                                                         bin_exp->get_rhs());

                const auto& key = dependencies.first;
                const auto& vars = dependencies.second;
                if (!key.empty()) {
                    var2statement_.emplace(key, ii);
                    for (const auto& var: vars) {
                        const auto& var_already_inserted = dependency_map_.find(var);
                        if (var_already_inserted != dependency_map_.end()) {
                            dependency_map_[key].insert(var_already_inserted->second.begin(),
                                                        var_already_inserted->second.end());
                            for (const auto& root_var: var_already_inserted->second) {
                                var2dependants_[root_var].insert(ii);
                            }
                        } else {
                            dependency_map_[key].insert(var);
                            var2dependants_[var].insert(ii);
                        }
                    }
                }
            }
        }
    }
}

bool SympyReplaceSolutionsVisitor::SolutionSorter::try_emplace_back_statement(
    ast::StatementVector& new_statements,
    const std::string& var) {
    auto ptr = var2statement_.find(var);
    bool emplaced = false;
    if (ptr != var2statement_.end()) {
        const auto ii = ptr->second;
        new_statements.emplace_back(statements_[ii]->clone());
        tags_[ii] = false;
        emplaced = true;
    }
    return emplaced;
}

bool SympyReplaceSolutionsVisitor::SolutionSorter::emplace_back_next_statement(
    ast::StatementVector& new_statements) {
    const size_t next_statement_ii = std::find(tags_.begin(), tags_.end(), true) - tags_.begin();
    bool emplaced = false;
    if (next_statement_ii < statements_.size()) {
        new_statements.emplace_back(statements_[next_statement_ii]->clone());
        tags_[next_statement_ii] = false;
        emplaced = true;
    }
    return emplaced;
}

size_t SympyReplaceSolutionsVisitor::SolutionSorter::emplace_back_all_statements(
    ast::StatementVector& new_statements) {
    size_t n = 0;
    for (size_t ii = 0; ii < statements_.size(); ++ii) {
        if (tags_[ii]) {
            new_statements.emplace_back(statements_[ii]->clone());
            tags_[ii] = false;
            ++n;
        }
    }
    return n;
}

size_t SympyReplaceSolutionsVisitor::SolutionSorter::tag_statement(const std::string& var) {
    auto ptr = var2dependants_.find(var);
    size_t n = 0;
    if (ptr != var2dependants_.end()) {
        for (const auto ii: ptr->second) {
            tags_[ii] = true;
            ++n;
        }
    }
    return n;
}

void SympyReplaceSolutionsVisitor::SolutionSorter::tag_all_statements() {
    tags_ = std::vector<bool>(tags_.size(), true);
}

// TODO remove
void SympyReplaceSolutionsVisitor::SolutionSorter::print() {
    std::cout << "---" << std::endl;
    for (const auto& eq: var2statement_) {
        const auto ii = eq.second;
        std::cout << eq.first << " | " << to_nmodl(statements_[ii]) << " | " << tags_[ii]
                  << std::endl;
    }
    std::cout << "---" << std::endl;
    for (const auto& p: var2dependants_) {
        auto var = p.first;
        std::cout << var << " | ";
        for (auto ii: p.second) {
            std::cout << ii << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << "-- dep map ---" << std::endl;
    for (const auto& p: dependency_map_) {
        auto var = p.first;
        std::cout << var << " | ";
        for (auto ii: p.second) {
            std::cout << ii << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << "---" << std::endl;
}


}  // namespace visitor
}  // namespace nmodl
