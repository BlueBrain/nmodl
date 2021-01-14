/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::SympyReplaceSolutionsVisitor
 */

#include "visitors/ast_visitor.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class SympyReplaceSolutionsVisitor
 * \brief TODO
 */
class SympyReplaceSolutionsVisitor: public AstVisitor {
  private:
    enum class ReplacePolicy {
        VALUE = 0,   //!< Replace statements matching by lhs varName
        GREEDY = 1,  //!< Replace statements greedily
    };

  public:
    /// Default constructor
    SympyReplaceSolutionsVisitor() = delete;

    SympyReplaceSolutionsVisitor(const std::vector<std::string>& pre_solve_statements,
                                 const std::vector<std::string>& solutions,
                                 const std::unordered_set<ast::Statement*>& to_be_removed);

    static std::pair<std::string, std::unordered_set<std::string>> statement_dependencies(
        const std::shared_ptr<ast::Expression>& lhs,
        const std::shared_ptr<ast::Expression>& rhs);

    void visit_statement_block(ast::StatementBlock& node) override;
    void visit_diff_eq_expression(ast::DiffEqExpression& node) override;
    void visit_lin_equation(ast::LinEquation& node) override;
    void visit_binary_expression(ast::BinaryExpression& node) override;


  private:
    struct SolutionSorter {
        SolutionSorter() {}

        SolutionSorter(const std::vector<std::string>::const_iterator& statements_str_beg,
                       const std::vector<std::string>::const_iterator& statements_str_end);

        /**
         * Here we construct a map variable -> affected equations. In other words this map tells me
         * what equations need to be updated when I change a particular variable. To do that we
         * build a a graph of dependencies var -> vars and in the mean time we reduce it to the root
         * variables. This is ensured by the fact that the tmp variables are sorted so that the next
         * tmp variable may depend on the previous one. Since it is a relation of equivalence (if an
         * equation depends on a variable, it needs to be updated if the variable changes), we build
         * the two maps at the same time.
         *
         * An example:
         *
         *  - tmp0 = x + a
         *  - tmp1 = tmp0 + b
         *  - tmp2 = y
         *
         * dependency_map_ should be (the order of the equation is unimportant since we are building
         * a map):
         *
         * - tmp0 : x, a
         * - tmp1 : x, a, b
         * - tmp2 : y
         *
         * and the var2statement_ map should be (the order of the following equations is unimportant
         * since we are building a map. The number represents the index of the original equations):
         *
         * - x : 0, 1
         * - y : 2
         * - a : 0, 1
         * - b : 1
         */
        void build_maps();

        inline bool is_var_assigned_here(const std::string& var) const {
            return var2statement_.find(var) != var2statement_.end();
        }

        inline bool is_all_untagged() const {
            return std::find(tags_.begin(), tags_.end(), true) == tags_.end();
        }

        bool try_emplace_back_statement(ast::StatementVector& new_statements,
                                        const std::string& var);

        bool emplace_back_next_statement(ast::StatementVector& new_statements);

        size_t emplace_back_all_statements(ast::StatementVector& new_statements);

        size_t tag_statement(const std::string& var);

        void tag_all_statements();

        // TODO remove
        void print();

        /// var -> (depends on) vars
        std::unordered_map<std::string, std::unordered_set<std::string>> dependency_map_;
        /// var -> (statements that depend on) statements
        std::unordered_map<std::string, std::set<size_t>> var2dependants_;
        /// var -> statement that sets that var
        std::unordered_map<std::string, size_t> var2statement_;
        std::vector<std::shared_ptr<ast::Statement>> statements_;
        std::vector<bool> tags_;
    };

    SolutionSorter pre_solve_statements_;
    SolutionSorter tmp_statements_;
    SolutionSorter solutions_;
    std::unordered_map<std::shared_ptr<ast::Statement>, ast::StatementVector> replacements_;
    bool is_statement_block_root_ = true;
    ReplacePolicy policy_;
    const std::unordered_set<ast::Statement*>* to_be_removed_;
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
