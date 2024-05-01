/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::SympyReplaceSolutionsVisitor
 */

#include "visitors/ast_visitor.hpp"

#include <set>
#include <string>
#include <unordered_set>

namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */


/**
 * \class SympyReplaceSolutionsVisitor
 * \brief Replace statements in \p node with pre_solve_statements, tmp_statements, and solutions
 *
 * The goal is to replace statements with solutions in place. In this way we can allow (to
 * some extent) the use of control flow blocks and assignments. \ref pre_solve_statements are added
 * in front of the replaced statements in case their variable needs updating. \ref
 StatementDispenser
 * keeps track of what needs updating. Let's start with some nomenclature:
 *
 * - statement: a line in the .mod file. It can be a diff_eq_expression, binary_expression, or
 * linEquation
 * - old_Statement: line in the staementBlock that must be replaced with the solution
 * - solution_statements/ new_statement: a nmodl-statement (always binary expression) provided by
 sympy that
 * assigns a variable
 * - pre_solve_Statements: statements that update the variables (i.e. x = old_x)
 * - tmp_statements: assignment of temporary variables in the solution generated by sympy in case
 * --cse. (i.e. \f$ tmp = f (...) \f$
 *
 * We employ a multi-step approach:
 *
 * - try to replace the old_statements (not binary_expressions) and in "assignment form: \f$ x =
 * f(...) \f$" with the corresponding solution matching by variable (i.e. x in \f$ x = f(...) \f$)
 * - try to replace the old_Statements with a greedy approach. When we find a
 * diff_eq_expression/linEquation that needs replacing we take the next solution that was not yet
 * used
 * - add all the remaining solutions at the end
 *
 * Let's finish with an example (that are usually better than blabbling around).
 *
 * Imagine we have this derivative block in the ast (before SympyReplaceSolutionsVisitor passes):
 *
 * \code{.mod}
 * DERIVATIVE d {
 *     LOCAL a, old_x, old_y, old_z, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7
 *     b = 1
 *     x' = x + y + a + b
 *     if ( x == 0) {
 *         a = a + 1
 *         # x = x + 1 // this would be an error. Explained later
 *     }
 *     y' = x + y + a
 *     z' = y + a
 *     x = x + 1
 * }
 * \endcode
 *
 * where SympySolverVisitor already added variables in the LOCAL declaration.
 *
 * Sympy solver visitor also provides:
 *
 * - pre-solve statements:
 *
 * \code{.mod}
 * old_x = x
 * old_y = y
 * old_z = z
 * \endcode
 *
 * - tmp statements:
 *
 * \code{.mod}
 * tmp0 = 2.0*dt
 * tmp1 = 1.0/(tmp0-1.0)
 * tmp2 = pow(dt, 2)
 * tmp3 = b*tmp2
 * tmp4 = dt*old_x
 * tmp5 = a*dt
 * tmp6 = dt*old_y
 * tmp7 = tmp5+tmp6
 * \endcode
 *
 * - solutions:
 *
 * \code{.mod}
 * x = -tmp1*(b*dt+old_x-tmp3-tmp4+tmp7)
 * y = -tmp1*(old_y+tmp3+tmp4+tmp5-tmp6)
 * z = -tmp1*(-a*tmp2+b*pow(dt, 3)+old_x*tmp2-old_y*tmp2-old_z*tmp0+old_z+tmp7)
 * \endcode
 *
 * SympySolveVisitor works in this way:
 *
 * \code{.mod}
 * DERIVATIVE d {                                                                   // nothing to do
 *
 *     LOCAL a, old_x, old_y, old_z, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7 // nothing to do
 *
 *     b = 1                                                     // initial statement, nothing to do
 *
 *     x' = x + y + a + b   ->   old_x = x                    // before printing this solution let's
 *                               old_y = y                         // flush the pre solve statements
 *                               old_z = z                             // mark down that we did this
 *
 *                               tmp0 = 2.0*dt                   // we also flush the tmp statements
 *                               tmp1 = 1.0/(tmp0-1.0)
 *                               tmp2 = pow(dt, 2)
 *                               tmp3 = b*tmp2
 *                               tmp4 = dt*old_x
 *                               tmp5 = a*dt
 *                               tmp6 = dt*old_y
 *                               tmp7 = tmp5+tmp6
 *
 *                               x = -tmp1*(b*dt+old_x-tmp3-tmp4+tmp7)      // finally, the solution
 *
 *     if ( x == 0) {                                                               // nothing to do
 *         a = a + 1                        // mark down the tmp statements and pre solve statements
 *                                           // that contain 'a' in the rhs as in need for an update
 *
 *         // x = x + 1                       // the same as before but for 'x'. In particular a pre
 *                                              // solve statement is marked for updating. This will
 *                                                             // produce an error later in the code
 *     }                                                                            // nothing to do
 *
 *     y' = x + y + a       ->       // old_x = x    // here, if 'x = x + 1' were not commented, the
 *                                                  // code would try to print this line an throw an
 *                                              // error since the pre solve statements were already
 *                                                                                   // printed once
 *                                   tmp5 = a*dt  // flush the tmp statements that need updating. In
 *                                   tmp7 = tmp5+tmp6    // our example, all the tmp statements that
 *                                                             // directly or indirectly depend on a
 *                                     // for performance, we print only the ones that need updating
 *
 *     z' = y + a   ->  z = -tmp1*(-a*tmp2+b*pow(dt, 3)+old_x*tmp2-old_y*tmp2-old_z*tmp0+old_z+tmp7)
 *                             // nothing is marked for updating (among pre solve statements and tmp
 *                                                           // statements): just print the solution
 *
 *     x = x + 1                                                                    // nothing to do
 * }                                                                                // nothing to do
 * \endcode
 *
 * Last notes:
 *
 * For linEquations or NonLinEquations association of the solution with a particular statement could
 * be impossible. For example \f$ ~ x + y = 0 \f$ does not have a simple variable in the lhs. Thus,
 * an association with a particular solution statement is not possible. Thus we do 2 runs where we
 * first match everything we can by value and then we associate everything we can in a greedy way.
 *
 * For large system of equations the code sets up the J matrix and F vector to be sent to eigen for
 * the Newton method which will solve a bunch of J x = F for each time step). In this case it is
 always safe to
 * replace greedy because sympy does not sort the equations in the matrix/vector. In addition, cse
 is disabled by
 * default. Thus, there is a 1:1 correspondence of an equation of the original mod file and a row of
 the matrix and
 * an element of F. So if we have:
 *
 * \code{.mod}
 * LINEAR lin {
 *     ~ x = ...
 *     a = a + 1
 *     ~ y = ...
 *     ~ z = ...
 *     ~ w = ...
 * }
 * \endcode
 *
 * We get the vector F and matrix J
 *
 * \code
 * F = [0,     J = [0, 4, 8,  12,
 *      1,          1, 5, 9,  13,
 *      2,          2, 6, 10, 14,
 *      3]          3, 7, 11, 15]
 * \endcode
 *
 * Where the numbers indicate their column-wise index. The solution replacement becomes:
 *
 * \code
 * ~ x = ...  -> F[0] = ...
 *               J[0] = ...
 *               J[4] = ...
 *               J[8] = ...
 *               J[12] = ...
 * a = a + 1
 * ~ y = ...  -> ...
 * \endcode
 *
 */
class SympyReplaceSolutionsVisitor: public AstVisitor {
  public:
    enum class ReplacePolicy {
        VALUE = 0,   //!< Replace statements matching by lhs varName
        GREEDY = 1,  //!< Replace statements greedily
    };
    /// Empty ctor
    SympyReplaceSolutionsVisitor() = delete;

    /// Default constructor
    SympyReplaceSolutionsVisitor(const std::vector<std::string>& pre_solve_statements,
                                 const std::vector<std::string>& solutions,
                                 const std::unordered_set<ast::Statement*>& to_be_removed,
                                 const ReplacePolicy policy,
                                 size_t n_next_equations,
                                 const std::string& tmp_unique_prefix);

    /// idx (in the new statementVector) of the first statement that was added. -1 if nothing was
    /// added
    inline int replaced_statements_begin() const {
        return replaced_statements_range.first;
    }
    /// idx (in the new statementVector) of the last statement that was added. -1 if nothing was
    /// added
    inline int replaced_statements_end() const {
        return replaced_statements_range.second;
    }

    void visit_statement_block(ast::StatementBlock& node) override;
    void visit_diff_eq_expression(ast::DiffEqExpression& node) override;
    void visit_lin_equation(ast::LinEquation& node) override;
    void visit_non_lin_equation(ast::NonLinEquation& node) override;
    void visit_binary_expression(ast::BinaryExpression& node) override;


  private:
    /** \brief Try to replace a statement
     *
     * \param node it can be Diff_Eq_Expression/LinEquation/NonLinEquation
     * \param get_lhs method with witch we may get the lhs (in case we need it)
     * \param get_rhs method with witch we may get the rhs (in case we need it)
     */
    void try_replace_tagged_statement(
        const ast::Node& node,
        std::shared_ptr<ast::Expression> get_lhs(const ast::Node& node),
        std::shared_ptr<ast::Expression> get_rhs(const ast::Node& node));

    /**
     * \struct InterleavesCounter
     * \brief Count interleaves of assignment statement inside the system of equations
     *
     * Example:
     *
     * \code
     * \\ not in the system, n = 0, is_in_system = false
     * ~ x + y = 0 \\ system, in_system switch false -> true, n = 1
     * ~ y = a + 1 \\ system, no switch, nothing to do
     * a = ... \\ no system, in_system switch true -> false, nothing to do
     * ~ z = x + y + z \\ system, in_system switch false -> true, n = 2
     * \endcode
     *
     * Number of interleaves: n-1 = 1
     */
    struct InterleavesCounter {
        /// Count interleaves defined as a switch false -> true for \ref in_system
        void new_equation(const bool is_in_system);

        /// Number of interleaves. We need to remove the first activation of the switch except if
        /// there were no switches
        inline size_t n() const {
            return n_interleaves == 0 ? 0 : n_interleaves - 1;
        }

      private:
        /**
         * \brief Number of interleaves of assignment statements in between equations of the system
         * of equations
         *
         * This is equivalent to the number of switches false -> true of \ref in_system minus the
         * very first one (if the system exists).
         */
        size_t n_interleaves = 0;

        /// Bool that keeps track if just wrote an equation of the system of equations (true) or not
        /// (false)
        bool in_system = false;
    };


    /**
     * \struct StatementDispenser
     * \brief Sorts and maps statements to variables keeping track of what needs updating
     *
     * This is a multi-purpose object that:
     *
     * - keeps track of what was already updated
     * - decides what statements need updating in case there was a variable assignment (i.e. \f$ a =
     * 3 \f$)
     * - builds the statements from a vector of strings
     *
     */
    struct StatementDispenser {
        /// Empty ctor
        StatementDispenser() = default;

        /// Standard ctor
        StatementDispenser(const std::vector<std::string>::const_iterator& statements_str_beg,
                           const std::vector<std::string>::const_iterator& statements_str_end,
                           const int error_on_n_flushes);

        /// Construct the maps \ref var2dependants, \ref var2statement and \ref dependency_map
        /// for easy access and classification of the statements
        void build_maps();

        /// Check if one of the statements assigns this variable (i.e. \f$ x' = f(x, y, x) \f$) and
        /// is still tagged
        inline bool is_var_assigned_here(const std::string& var) const {
            const auto it = var2statement.find(var);
            return it != var2statement.end() && tags.find(it->second) != tags.end();
        }

        /**
         * \brief Look for \p var in \ref var2statement and emplace back that statement in \p
         * new_statements
         *
         * If there is no \p var key in \ref var2statement, return false
         */
        bool try_emplace_back_tagged_statement(ast::StatementVector& new_statements,
                                               const std::string& var);


        /// Emplace back the next \p n_next_statements solutions in \ref statements that is marked
        /// for updating in \ref tags
        size_t emplace_back_next_tagged_statements(ast::StatementVector& new_statements,
                                                   const size_t n_next_statements);

        /// Emplace back all the statements that are marked for updating in \ref tags
        size_t emplace_back_all_tagged_statements(ast::StatementVector& new_statements);

        /**
         * \brief Tag all the statements that depend on \p var for updating
         *
         * This is necessary when an assignment has invalidated this variable
         */
        size_t tag_dependant_statements(const std::string& var);

        /// Mark that all the statements need updating (probably unused)
        void tag_all_statements();

        /**
         * \brief x (key) : f(a, b, c, ...) (values)
         *
         * Given a certain variable (map key) we get all the (root) variables on which this variable
         * depends on (values)
         *
         * For example, imagine we have these assignments:
         *
         * \code
         * tmp = b
         * x = a + tmp + exp(a)
         * \endcode
         *
         * \ref dependency_map is:
         *
         * - tmp : b
         * - x : a, b
         *
         */
        std::unordered_map<std::string, std::unordered_set<std::string>> dependency_map;

        /**
         * \brief a (key) : f(..., a, ...), g(..., a, ...), h(..., a, ...), ... (values)
         *
         * This the "reverse" of \ref dependency_map. Given a certain variable it provides
         * the statements that depend on it. It is a set because we want to print them in
         * order and we do not want duplicates. The value is the index in \ref statements or \ref
         * tags
         *
         * For example:
         *
         * \code
         * tmp = b // statement 0
         * x = a + tmp + exp(a) // statement 1
         * \endcode
         *
         * \ref var2dependants is:
         *
         * - a : 1
         * - b : 0, 1
         *
         */
        std::unordered_map<std::string, std::set<size_t>> var2dependants;

        /**
         * \brief a (key) : a = f(...) (value)
         *
         * Given a certain variable we get the statement where that variable is defined
         *
         * For example:
         *
         * \code{.mod}
         * tmp = b // statement 0
         * x = a + tmp + exp(a) // statement 1
         * \endcode
         *
         * \ref var2dependants is:
         *
         * - tmp : 0
         * - x : 1
         *
         */
        std::unordered_map<std::string, size_t> var2statement;

        /// Vector of statements
        std::vector<std::shared_ptr<ast::Statement>> statements;

        /**
         * \brief Keeps track of what statements need updating
         *
         * The elements of this set are the indexes of the \ref statements vector that need
         * updating. It is a set because we need to be able to easily find them by value and we need
         * them ordered to pick "the next one"
         */
        std::set<size_t> tags;

        /**
         * \brief Max number of times a statement was printed using an \ref
         * emplace_back_all_tagged_statements command
         *
         * This is useful to check if, during updates, a variable was assigned.
         *
         * For example:
         *
         * \code{.mod}
         * x' = a
         * x = a + 1
         * y' = b
         * \endcode
         *
         * In this sequence of statements \f$ x \f$ was assigned within variable updates. This
         * sequence of statements could lead to instability/wrong results for derivimplicit methods.
         * Better to prevent this entirely. It can still be assigned at the end/beginning
         */
        size_t n_flushes = 0;

        /// Emit error when \ref n_flushes reaches this number. -1 disables the error entirely
        int error_on_n_flushes;
    };

    /// Update state variable statements (i.e. \f$old_x = x \f$)
    StatementDispenser pre_solve_statements;

    /// tmp statements that appear with --cse (i.e. \f$tmp0 = a \f$)
    StatementDispenser tmp_statements;

    /// solutions that we want to replace
    StatementDispenser solution_statements;

    /**
     * \brief Replacements found by the visitor
     *
     * The keys are the old_statements that need replacing with the new ones (the
     * value). Since there are \ref pre_solve_statements and \ref tmp_statements; it is in general
     * a replacement of 1 : n statements
     */
    std::unordered_map<std::shared_ptr<ast::Statement>, ast::StatementVector> replacements;

    /// Used to notify to visit_statement_block was called by the user (or another visitor) or
    /// re-called in a nested block
    bool is_top_level_statement_block = true;

    /// Replacement policy used by the various visitors
    ReplacePolicy policy;

    /// Number of solutions that match each old_statement with the greedy policy
    size_t n_next_equations;

    /// group of old statements that need replacing
    const std::unordered_set<ast::Statement*>* to_be_removed;

    /// counts how many times the solution statements are interleaved with assignment expressions
    InterleavesCounter interleaves_counter;

    /// {begin index, end index} of the added statements. -1 means that it is invalid
    std::pair<int, int> replaced_statements_range = {-1, -1};
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
