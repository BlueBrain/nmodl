/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \dir
 * \brief AST Implementation
 *
 * \file
 * \brief Implementation of AST base class and it's properties
 */

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
 * @defgroup ast AST Implementation
 * @brief All AST related implementation details
 *
 * @defgroup ast_prop AST Properties
 * @ingroup ast
 * @brief Properties and types used with of AST classes
 * @{
 */

/**
 * \brief enum Type for binary operators in NMODL
 *
 * NMODL support different binary operators and this
 * type is used to store their value in the AST.
 *
 * \note `+=` and `-=` are not supported by NMODL. They
 * are added for code generation purpose.
 */
typedef enum {
    BOP_ADDITION = 0,    ///< \+
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
    BOP_EXACT_EQUAL,     ///< ==
    BOP_ADD_ASSIGN,      ///< \+=
    BOP_SUB_ASSIGN       ///< \-=
} BinaryOp;

/**
 * \brief string representation of ast::BinaryOp
 *
 * When AST is converted back to NMODL or C code, ast::BinaryOpNames
 * is used to lookup the corresponding symbol for the operator.
 */
static const std::string BinaryOpNames[] =
    {"+", "-", "*", "/", "^", "&&", "||", ">", "<", ">=", "<=", "=", "!=", "==", "+=", "-="};

/// enum type for unary operators
typedef enum { UOP_NOT, UOP_NEGATION } UnaryOp;

/// string representation of ast::UnaryOp
static const std::string UnaryOpNames[] = {"!", "-"};

/// enum type for partial equation types
typedef enum { PEQ_FIRST, PEQ_LAST } FirstLastType;

/// string representation of ast::FirstLastType
static const std::string FirstLastTypeNames[] = {"FIRST", "LAST"};

/// enum type for queue types
typedef enum { PUT_QUEUE, GET_QUEUE } QueueType;

/// string representation of ast::QueueType
static const std::string QueueTypeNames[] = {"PUTQ", "GETQ"};

/// enum type to distinguish BEFORE or AFTER blocks
typedef enum { BATYPE_BREAKPOINT, BATYPE_SOLVE, BATYPE_INITIAL, BATYPE_STEP } BAType;

/// string representation of ast::BAType
static const std::string BATypeNames[] = {"BREAKPOINT", "SOLVE", "INITIAL", "STEP"};

/// enum type used for UNIT_ON or UNIT_OFF state
typedef enum { UNIT_ON, UNIT_OFF } UnitStateType;

/// string representation of ast::UnitStateType
static const std::string UnitStateTypeNames[] = {"UNITSON", "UNITSOFF"};

/// enum type used for Reaction statement
typedef enum { LTMINUSGT, LTLT, MINUSGT } ReactionOp;

/// string representation of ast::ReactionOp
static const std::string ReactionOpNames[] = {"<->", "<<", "->"};

/// convert operator in std::string form to ast::BinaryOp
static inline BinaryOp string_to_binaryop(const std::string& op) {
    auto it = std::find(std::begin(BinaryOpNames), std::end(BinaryOpNames), op);
    if (it == std::end(BinaryOpNames)) {
        throw std::runtime_error("Error in string_to_binaryop, can not find " + op);
    }
    int pos = std::distance(std::begin(BinaryOpNames), it);
    return static_cast<BinaryOp>(pos);
}
/** @} */  // end of ast_prop

}  // namespace ast
}  // namespace nmodl
