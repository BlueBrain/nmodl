/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <memory>
#include <string>

#include "ast/ast_decl.hpp"
#include "lexer/modtoken.hpp"
#include "symtab/symbol_table.hpp"
#include "visitors/visitor.hpp"

namespace nmodl {

/// Abstract Syntax Tree (AST) related implementations
namespace ast {

/**
 * @defgroup ast AST Infrastructure
 * @brief All AST related implementation details
 *
 * @defgroup ast_prop AST Properties
 * @ingroup ast
 * @brief Properties used with different members of AST classes
 * @{
 */

/**
 * \brief enum type for binary operators in NMODL
 *
 * NMODL support different binary operators and ast::BinaryOp
 * type is used to store their value in the AST.
 */
typedef enum {
    BOP_ADDITION,        ///< \+
    BOP_SUBTRACTION,     ///< --
    BOP_MULTIPLICATION,  ///< \c *
    BOP_DIVISION,        ///< \/
    BOP_POWER,           ///< ^
    BOP_AND,             ///< &&
    BOP_OR,              ///< ||
    BOP_GREATER,         ///< >
    BOP_LESS,            ///< <
    BOP_GREATER_EQUAL,   ///< >=
    BOP_LESS_EQUAL,      ///< <=
    BOP_ASSIGN,          ///< =
    BOP_NOT_EQUAL,       ///< !=
    BOP_EXACT_EQUAL      ///< ==
} BinaryOp;

/**
 * \brief string representation of \link ast::BinaryOp
 *
 * When AST is converted back to NMODL or C code, \link ast::BinaryOpNames
 * is used to lookup the corresponding symbol for the operator.
 */
static const std::string BinaryOpNames[] =
    {"+", "-", "*", "/", "^", "&&", "||", ">", "<", ">=", "<=", "=", "!=", "=="};

/// enum type for unary operators
typedef enum { UOP_NOT, UOP_NEGATION } UnaryOp;

/// string representation of \link ast::UnaryOp
static const std::string UnaryOpNames[] = {"!", "-"};

/// enum type for partial equation types
typedef enum { PEQ_FIRST, PEQ_LAST } FirstLastType;

/// string representation of \link ast::FirstLastType
static const std::string FirstLastTypeNames[] = {"FIRST", "LAST"};

/// enum type for partial equation types
typedef enum { PUT_QUEUE, GET_QUEUE } QueueType;

/// string representation of \link ast::QueueType
static const std::string QueueTypeNames[] = {"PUTQ", "GETQ"};

/// enum type to distinguish BEFORE or AFTER blocks
typedef enum { BATYPE_BREAKPOINT, BATYPE_SOLVE, BATYPE_INITIAL, BATYPE_STEP } BAType;

/// string representation of \link ast::BAType
static const std::string BATypeNames[] = {"BREAKPOINT", "SOLVE", "INITIAL", "STEP"};

/// enum type used for UNIT_ON or UNIT_OFF state
typedef enum { UNIT_ON, UNIT_OFF } UnitStateType;

/// string representation of \link ast::UnitStateType
static const std::string UnitStateTypeNames[] = {"UNITSON", "UNITSOFF"};

/// enum type used for Reaction statement
typedef enum { LTMINUSGT, LTLT, MINUSGT } ReactionOp;

/// string representation of \link ast::ReactionOp
static const std::string ReactionOpNames[] = {"<->", "<<", "->"};

/** @} */  // end of ast_prop


/**
 * @defgroup ast_class AST Classes
 * @ingroup ast
 * @brief Classes for implementing Abstract Syntax Tree (AST)
 * @{
 */
/**
 * \brief Base class for all Abstract Syntax Tree node types
 *
 * Every node in the Abstract Syntax Tree is inherited from base class
 * ast::AST. This class provides base properties and pure virtual
 * functions that must be implemented by base classes. We inherit from
 * std::enable_shared_from_this to get a valid shared_ptr to this pointer.
 */
struct AST: public std::enable_shared_from_this<AST> {
    /// \name Pure Virtual Functions
    /// \{

    /**
     * \brief Return type (ast::AstNodeType) of ast node
     *
     * Every node in the ast has a type defined in ast::AstNodeType.
     * This type is can be used to check/compare node types.
     */
    virtual AstNodeType get_node_type() = 0;

    /**
     * \brief Return type (ast::AstNodeType) of ast node as std::string
     *
     * Every node in the ast has a type defined in ast::AstNodeType.
     * This type name can be returned as a std::string for printing
     * ast to text/json form.
     *
     * @return name of the node type as a string
     */
    virtual std::string get_node_type_name() = 0;

    /**
     * \brief Accept (or visit) the current AST node using current visitor
     *
     * Instead of visiting children of AST node, like AST::visit_children,
     * accept allows to visit the current node itself using the concrete
     * visitor provided.
     *
     * @param v Concrete visitor that will be used to recursively visit children
     *
     * \note Below is an example of `accept` method implementation which shows how
     *       visitor method corresponding to ast::IndexedName node is called allowing
     *       to visit the node itself in the visitor.
     *
     * \code{.cpp}
     *   void IndexedName::accept(Visitor* v) override {
     *       v->visit_indexed_name(this);
     *   }
     * \endcode
     */
    virtual void accept(Visitor* v) = 0;

    /**
     * \brief Visit children i.e. member of current AST node using current visitor
     *
     * Different nodes in the AST have different members (i.e. children). This method
     * recursively visits children using provided concrete visitor.
     *
     * @param v Concrete visitor that will be used to recursively visit children
     *
     * \note Below is an example of `visit_children` method implementation which shows
     *       ast::IndexedName node children are visited instead of node itself.
     *
     * \code{.cpp}
     * void IndexedName::visit_children(Visitor* v) {
     *    name->accept(v);
     *    length->accept(v);
     * }
     * \endcode
     */
    virtual void visit_children(Visitor* v) = 0;

    /// \}

    virtual std::string get_node_name() {
        throw std::logic_error("get_node_name() not implemented");
    }

    virtual AST* clone() {
        throw std::logic_error("clone() not implemented");
    }

    /* @todo: revisit, adding quickly for symtab */
    virtual ModToken* get_token() { /*std::cout << "\n ERROR: get_token not implemented!";*/
        return nullptr;
    }

    virtual void set_symbol_table(symtab::SymbolTable* newsymtab) {
        throw std::runtime_error("set_symbol_table() not implemented");
    }

    virtual symtab::SymbolTable* get_symbol_table() {
        throw std::runtime_error("get_symbol_table() not implemented");
    }

    virtual std::shared_ptr<StatementBlock> get_statement_block() {
        throw std::runtime_error("get_statement_block not implemented");
    }

    // implemented in Number sub classes
    virtual void negate() {
        throw std::runtime_error("negate() not implemented");
    }

    // implemented in Identifier sub classes
    virtual void set_name(std::string /*name*/) {
        throw std::runtime_error("set_name() not implemented");
    }

    virtual ~AST() {}

    virtual std::shared_ptr<AST> get_shared_ptr() {
        return std::static_pointer_cast<AST>(shared_from_this());
    }

    virtual bool is_ast() {
        return true;
    }

    virtual bool is_statement() {
        return false;
    }

    virtual bool is_expression() {
        return false;
    }

    virtual bool is_block() {
        return false;
    }

    virtual bool is_identifier() {
        return false;
    }

    virtual bool is_number() {
        return false;
    }

    virtual bool is_string() {
        return false;
    }

    virtual bool is_integer() {
        return false;
    }

    virtual bool is_float() {
        return false;
    }

    virtual bool is_double() {
        return false;
    }

    virtual bool is_boolean() {
        return false;
    }

    virtual bool is_name() {
        return false;
    }

    virtual bool is_prime_name() {
        return false;
    }

    virtual bool is_var_name() {
        return false;
    }

    virtual bool is_indexed_name() {
        return false;
    }

    virtual bool is_argument() {
        return false;
    }

    virtual bool is_react_var_name() {
        return false;
    }

    virtual bool is_read_ion_var() {
        return false;
    }

    virtual bool is_write_ion_var() {
        return false;
    }

    virtual bool is_nonspecific_cur_var() {
        return false;
    }

    virtual bool is_electrode_cur_var() {
        return false;
    }

    virtual bool is_section_var() {
        return false;
    }

    virtual bool is_range_var() {
        return false;
    }

    virtual bool is_global_var() {
        return false;
    }

    virtual bool is_pointer_var() {
        return false;
    }

    virtual bool is_bbcore_pointer_var() {
        return false;
    }

    virtual bool is_extern_var() {
        return false;
    }

    virtual bool is_threadsafe_var() {
        return false;
    }

    virtual bool is_param_block() {
        return false;
    }

    virtual bool is_step_block() {
        return false;
    }

    virtual bool is_independent_block() {
        return false;
    }

    virtual bool is_dependent_block() {
        return false;
    }

    virtual bool is_state_block() {
        return false;
    }

    virtual bool is_plot_block() {
        return false;
    }

    virtual bool is_initial_block() {
        return false;
    }

    virtual bool is_constructor_block() {
        return false;
    }

    virtual bool is_destructor_block() {
        return false;
    }

    virtual bool is_statement_block() {
        return false;
    }

    virtual bool is_derivative_block() {
        return false;
    }

    virtual bool is_linear_block() {
        return false;
    }

    virtual bool is_non_linear_block() {
        return false;
    }

    virtual bool is_discrete_block() {
        return false;
    }

    virtual bool is_partial_block() {
        return false;
    }

    virtual bool is_function_table_block() {
        return false;
    }

    virtual bool is_function_block() {
        return false;
    }

    virtual bool is_eigen_newton_solver_block() {
        return false;
    }

    virtual bool is_nrn_state_block() {
        return false;
    }

    virtual bool is_solution_expression() {
        return false;
    }

    virtual bool is_derivimplicit_callback() {
        return false;
    }

    virtual bool is_eigen_linear_solver_block() {
        return false;
    }

    virtual bool is_procedure_block() {
        return false;
    }

    virtual bool is_net_receive_block() {
        return false;
    }

    virtual bool is_solve_block() {
        return false;
    }

    virtual bool is_breakpoint_block() {
        return false;
    }

    virtual bool is_terminal_block() {
        return false;
    }

    virtual bool is_before_block() {
        return false;
    }

    virtual bool is_after_block() {
        return false;
    }

    virtual bool is_ba_block() {
        return false;
    }

    virtual bool is_for_netcon() {
        return false;
    }

    virtual bool is_kinetic_block() {
        return false;
    }

    virtual bool is_match_block() {
        return false;
    }

    virtual bool is_unit_block() {
        return false;
    }

    virtual bool is_constant_block() {
        return false;
    }

    virtual bool is_neuron_block() {
        return false;
    }

    virtual bool is_unit() {
        return false;
    }

    virtual bool is_double_unit() {
        return false;
    }

    virtual bool is_local_var() {
        return false;
    }

    virtual bool is_limits() {
        return false;
    }

    virtual bool is_number_range() {
        return false;
    }

    virtual bool is_plot_var() {
        return false;
    }

    virtual bool is_binary_operator() {
        return false;
    }

    virtual bool is_wrapped_expression() {
        return false;
    }

    virtual bool is_paren_expression() {
        return false;
    }

    virtual bool is_unary_operator() {
        return false;
    }

    virtual bool is_reaction_operator() {
        return false;
    }

    virtual bool is_binary_expression() {
        return false;
    }

    virtual bool is_unary_expression() {
        return false;
    }

    virtual bool is_non_lin_equation() {
        return false;
    }

    virtual bool is_lin_equation() {
        return false;
    }

    virtual bool is_function_call() {
        return false;
    }

    virtual bool is_first_last_type_index() {
        return false;
    }

    virtual bool is_watch() {
        return false;
    }

    virtual bool is_queue_expression_type() {
        return false;
    }

    virtual bool is_match() {
        return false;
    }

    virtual bool is_ba_block_type() {
        return false;
    }

    virtual bool is_unit_def() {
        return false;
    }

    virtual bool is_factor_def() {
        return false;
    }

    virtual bool is_valence() {
        return false;
    }

    virtual bool is_unit_state() {
        return false;
    }

    virtual bool is_local_list_statement() {
        return false;
    }

    virtual bool is_model() {
        return false;
    }

    virtual bool is_define() {
        return false;
    }

    virtual bool is_include() {
        return false;
    }

    virtual bool is_param_assign() {
        return false;
    }

    virtual bool is_stepped() {
        return false;
    }

    virtual bool is_independent_def() {
        return false;
    }

    virtual bool is_dependent_def() {
        return false;
    }

    virtual bool is_plot_declaration() {
        return false;
    }

    virtual bool is_conductance_hint() {
        return false;
    }

    virtual bool is_expression_statement() {
        return false;
    }

    virtual bool is_protect_statement() {
        return false;
    }

    virtual bool is_from_statement() {
        return false;
    }

    virtual bool is_for_all_statement() {
        return false;
    }

    virtual bool is_while_statement() {
        return false;
    }

    virtual bool is_if_statement() {
        return false;
    }

    virtual bool is_else_if_statement() {
        return false;
    }

    virtual bool is_else_statement() {
        return false;
    }

    virtual bool is_partial_equation() {
        return false;
    }

    virtual bool is_partial_boundary() {
        return false;
    }

    virtual bool is_watch_statement() {
        return false;
    }

    virtual bool is_mutex_lock() {
        return false;
    }

    virtual bool is_mutex_unlock() {
        return false;
    }

    virtual bool is_reset() {
        return false;
    }

    virtual bool is_sens() {
        return false;
    }

    virtual bool is_conserve() {
        return false;
    }

    virtual bool is_compartment() {
        return false;
    }

    virtual bool is_lon_difuse() {
        return false;
    }

    virtual bool is_reaction_statement() {
        return false;
    }

    virtual bool is_lag_statement() {
        return false;
    }

    virtual bool is_queue_statement() {
        return false;
    }

    virtual bool is_constant_statement() {
        return false;
    }

    virtual bool is_table_statement() {
        return false;
    }

    virtual bool is_suffix() {
        return false;
    }

    virtual bool is_useion() {
        return false;
    }

    /// \todo : how is this different from is_nonspecific_cur_var ?
    virtual bool is_nonspecific() {
        return false;
    }

    virtual bool is_elctrode_current() {
        return false;
    }

    virtual bool is_section() {
        return false;
    }

    virtual bool is_range() {
        return false;
    }

    virtual bool is_global() {
        return false;
    }

    /// \todo : how is this different from is_pointer_var ?
    virtual bool is_pointer() {
        return false;
    }

    virtual bool is_bbcore_ptr() {
        return false;
    }

    virtual bool is_external() {
        return false;
    }

    virtual bool is_thread_safe() {
        return false;
    }

    virtual bool is_verbatim() {
        return false;
    }

    virtual bool is_line_comment() {
        return false;
    }

    virtual bool is_block_comment() {
        return false;
    }

    virtual bool is_node() {
        return false;
    }

    virtual bool is_program() {
        return false;
    }

    virtual bool is_constant_var() {
        return false;
    }

    virtual bool is_diff_eq_expression() {
        return false;
    }
};

/** @} */  // end of ast_class

}  // namespace ast
}  // namespace nmodl