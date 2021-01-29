/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "visitors/sympy_replace_solutions_visitor.hpp"
#include "visitors/lookup_visitor.hpp"


#include "ast/all.hpp"
#include "utils/logger.hpp"
#include "visitors/visitor_utils.hpp"

namespace nmodl {
namespace visitor {


void SympyReplaceSolutionsVisitor::InterleavesCounter::new_equation(const bool is_in_system) {
    n_ += (!in_system_ && is_in_system);
    in_system_ = is_in_system;
}

SympyReplaceSolutionsVisitor::SympyReplaceSolutionsVisitor(
    const std::vector<std::string>& pre_solve_statements,
    const std::vector<std::string>& solutions,
    const std::unordered_set<ast::Statement*>& to_be_removed,
    const ReplacePolicy policy,
    const size_t n_next_equations)
    : pre_solve_statements_(pre_solve_statements.begin(), pre_solve_statements.end(), 2)
    , to_be_removed_(&to_be_removed)
    , policy_(policy)
    , n_next_equations_(n_next_equations)
    , replaced_statements_begin_(-1)
    , replaced_statements_end_(-1) {
    const auto ss_tmp_delimeter =
        std::find_if(solutions.begin(), solutions.end(), [](const std::string& statement) {
            return statement.substr(0, 3) != "tmp";
        });
    tmp_statements_ = SolutionSorter(solutions.begin(), ss_tmp_delimeter, -1);
    solutions_ = SolutionSorter(ss_tmp_delimeter, solutions.end(), -1);

    replacements_.clear();
    is_statement_block_root_ = true;
}


void SympyReplaceSolutionsVisitor::visit_statement_block(ast::StatementBlock& node) {
    const bool is_root = is_statement_block_root_;
    is_statement_block_root_ = false;

    if (is_root) {
        logger->debug("SympyReplaceSolutionsVisitor :: visit statements. Matching policy: {}",
                      (policy_ == ReplacePolicy::VALUE ? "VALUE" : "GREEDY"));
        interleaves_counter_ = InterleavesCounter();
        node.visit_children(*this);

        if (!solutions_.tags_.empty() && policy_ == ReplacePolicy::VALUE) {
            logger->debug(
                "SympyReplaceSolutionsVisitor :: not all solutions were replaced. Policy: GREEDY");
            interleaves_counter_ = InterleavesCounter();
            policy_ = ReplacePolicy::GREEDY;
            node.visit_children(*this);

            if (interleaves_counter_.n() > 0) {
                logger->warn(
                    "SympyReplaceSolutionsVisitor :: Found ambiguous system of equations "
                    "interleaved with {} assignment statements. I do not know what equations go "
                    "before and what "
                    "equations go after the assignment statements. Either put all the equations "
                    "that need to be solved "
                    "in the form: x = f(...) and with distinct variable assignments or do not "
                    "interleave the system with assignments.",
                    interleaves_counter_.n());
            }
        }
    } else {
        node.visit_children(*this);
    }

    const auto& old_statements = node.get_statements();

    ast::StatementVector new_statements;
    new_statements.reserve(2 * old_statements.size());
    for (const auto& old_statement: old_statements) {
        const auto& replacement_ptr = replacements_.find(old_statement);
        if (replacement_ptr != replacements_.end()) {
            if (replaced_statements_begin_ == -1) {
                replaced_statements_begin_ = new_statements.size();
            }

            new_statements.insert(new_statements.end(),
                                  replacement_ptr->second.begin(),
                                  replacement_ptr->second.end());

            replaced_statements_end_ = new_statements.size();

            logger->debug("SympyReplaceSolutionsVisitor :: erasing {}", to_nmodl(old_statement));
            for (const auto& replacement: replacement_ptr->second) {
                logger->debug("SympyReplaceSolutionsVisitor :: adding {}", to_nmodl(replacement));
            }
        } else if (to_be_removed_ == nullptr ||
                   to_be_removed_->find(&(*old_statement)) == to_be_removed_->end()) {
            new_statements.emplace_back(std::move(old_statement));
        } else {
            logger->debug("SympyReplaceSolutionsVisitor :: erasing {}", to_nmodl(old_statement));
        }
    }

    if (is_root) {
        if (!solutions_.tags_.empty()) {
            throw std::runtime_error(
                "Not all solutions were replaced! Probably something went wrong.");
        }

        if (replaced_statements_begin_ == -1) {
            replaced_statements_begin_ = new_statements.size();
        }
        if (replaced_statements_end_ == -1) {
            replaced_statements_end_ = new_statements.size();
        }
    }

    node.set_statements(std::move(new_statements));
}

void SympyReplaceSolutionsVisitor::try_replace_tagged_statement(
    const ast::Node& node,
    const std::shared_ptr<ast::Expression>& get_lhs(const ast::Node& node),
    const std::shared_ptr<ast::Expression>& get_rhs(const ast::Node& node)) {
    interleaves_counter_.new_equation(true);

    const auto& statement = std::static_pointer_cast<ast::Statement>(
        node.get_parent()->get_shared_ptr());

    // do not visit if already marked
    if (replacements_.find(statement) != replacements_.end()) {
        return;
    }

    switch (policy_) {
    case ReplacePolicy::VALUE: {
        const auto dependencies = statement_dependencies(get_lhs(node), get_rhs(node));
        const auto& key = dependencies.first;
        const auto& vars = dependencies.second;

        if (solutions_.is_var_assigned_here(key)) {
            ast::StatementVector new_statements;

            pre_solve_statements_.emplace_back_all_tagged_statements(new_statements);
            tmp_statements_.emplace_back_all_tagged_statements(new_statements);
            solutions_.try_emplace_back_tagged_statement(new_statements, key);

            replacements_.emplace(statement, new_statements);
        }
        break;
    }
    case ReplacePolicy::GREEDY: {
        ast::StatementVector new_statements;

        pre_solve_statements_.emplace_back_all_tagged_statements(new_statements);
        tmp_statements_.emplace_back_all_tagged_statements(new_statements);
        solutions_.emplace_back_next_tagged_statements(new_statements, n_next_equations_);

        replacements_.emplace(statement, new_statements);

        break;
    }
    }
}


void SympyReplaceSolutionsVisitor::visit_diff_eq_expression(ast::DiffEqExpression& node) {
    auto get_lhs = [](const ast::Node& node) -> const std::shared_ptr<ast::Expression>& {
        return static_cast<const ast::DiffEqExpression&>(node).get_expression()->get_lhs();
    };

    auto get_rhs = [](const ast::Node& node) -> const std::shared_ptr<ast::Expression>& {
        return static_cast<const ast::DiffEqExpression&>(node).get_expression()->get_rhs();
    };

    try_replace_tagged_statement(node, get_lhs, get_rhs);
}

void SympyReplaceSolutionsVisitor::visit_lin_equation(ast::LinEquation& node) {
    auto get_lhs = [](const ast::Node& node) -> const std::shared_ptr<ast::Expression>& {
        return static_cast<const ast::LinEquation&>(node).get_left_linxpression();
    };

    auto get_rhs = [](const ast::Node& node) -> const std::shared_ptr<ast::Expression>& {
        return static_cast<const ast::LinEquation&>(node).get_left_linxpression();
    };

    try_replace_tagged_statement(node, get_lhs, get_rhs);
}


void SympyReplaceSolutionsVisitor::visit_non_lin_equation(ast::NonLinEquation& node) {
    auto get_lhs = [](const ast::Node& node) -> const std::shared_ptr<ast::Expression>& {
        return static_cast<const ast::NonLinEquation&>(node).get_lhs();
    };

    auto get_rhs = [](const ast::Node& node) -> const std::shared_ptr<ast::Expression>& {
        return static_cast<const ast::NonLinEquation&>(node).get_rhs();
    };

    try_replace_tagged_statement(node, get_lhs, get_rhs);
}


void SympyReplaceSolutionsVisitor::visit_binary_expression(ast::BinaryExpression& node) {
    if (node.get_op().get_value() == ast::BinaryOp::BOP_ASSIGN && node.get_lhs()->is_var_name()) {
        interleaves_counter_.new_equation(false);

        const auto& var =
            std::static_pointer_cast<ast::VarName>(node.get_lhs())->get_name()->get_node_name();
        pre_solve_statements_.tag_dependant_statements(var);
        tmp_statements_.tag_dependant_statements(var);
    }
}


SympyReplaceSolutionsVisitor::SolutionSorter::SolutionSorter(
    const std::vector<std::string>::const_iterator& statements_str_beg,
    const std::vector<std::string>::const_iterator& statements_str_end,
    const int error_on_n_flushes)
    : statements_(create_statements(statements_str_beg, statements_str_end))
    , error_on_n_flushes_(error_on_n_flushes) {
    tag_all_statements();
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
                const auto& dependencies = statement_dependencies(bin_exp->get_lhs(),
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

bool SympyReplaceSolutionsVisitor::SolutionSorter::try_emplace_back_tagged_statement(
    ast::StatementVector& new_statements,
    const std::string& var) {
    auto ptr = var2statement_.find(var);
    bool emplaced = false;
    if (ptr != var2statement_.end()) {
        const auto ii = ptr->second;
        const auto tag_ptr = tags_.find(ii);
        if (tag_ptr != tags_.end()) {
            new_statements.emplace_back(statements_[ii]->clone());
            tags_.erase(tag_ptr);
            emplaced = true;
        }
    }
    return emplaced;
}

size_t SympyReplaceSolutionsVisitor::SolutionSorter::emplace_back_next_tagged_statements(
    ast::StatementVector& new_statements,
    const size_t n_next_statements) {
    size_t counter = 0;
    for (size_t next_statement_ii = 0;
         next_statement_ii < statements_.size() && counter < n_next_statements;
         ++next_statement_ii) {
        const auto tag_ptr = tags_.find(next_statement_ii);
        if (tag_ptr != tags_.end()) {
            new_statements.emplace_back(statements_[next_statement_ii]->clone());
            tags_.erase(tag_ptr);
            ++counter;
        }
    }
    return counter;
}

size_t SympyReplaceSolutionsVisitor::SolutionSorter::emplace_back_all_tagged_statements(
    ast::StatementVector& new_statements) {
    for (const auto ii: tags_) {
        new_statements.emplace_back(statements_[ii]->clone());
        logger->debug("SympyReplaceSolutionsVisitor :: adding {}", to_nmodl(statements_[ii]));
    }

    n_flushes_ += (!tags_.empty());
    if (error_on_n_flushes_ > 0 && n_flushes_ >= error_on_n_flushes_) {
        throw std::runtime_error(
            "SympyReplaceSolutionsVisitor :: State variable assignment(s) interleaved in system "
            "of "
            "equations/differential equations. It is not allowed due to possible numerical "
            "instability and undefined "
            "behavior. Erase the assignment statement(s) or move them before/after the"
            " set of equations/differential equations.");
    }

    const auto n_replacements = tags_.size();

    tags_.clear();

    return n_replacements;
}

size_t SympyReplaceSolutionsVisitor::SolutionSorter::tag_dependant_statements(
    const std::string& var) {
    auto ptr = var2dependants_.find(var);
    size_t n = 0;
    if (ptr != var2dependants_.end()) {
        for (const auto ii: ptr->second) {
            tags_.insert(ii);
            ++n;
        }
    }
    return n;
}

void SympyReplaceSolutionsVisitor::SolutionSorter::tag_all_statements() {
    for (size_t i = 0; i < statements_.size(); ++i) {
        tags_.insert(i);
    }
}


}  // namespace visitor
}  // namespace nmodl
