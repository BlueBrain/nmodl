/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/
#pragma once

namespace nmodl {
namespace docstring {

static const char* binary_op_enum = R"(
    Enum type for binary operators in NMODL

    NMODL support different binary operators and this
    type is used to store their value in the AST. See
    nmodl::ast::Ast for details.
)";

static const char* ast_nodetype_enum = R"(
    Enum type for every AST node type

    Every node in the ast has associated type represented by
    this enum class. See nmodl::ast::AstNodeType for details.

)";

static const char* ast_class = R"(
    Base class for all Abstract Syntax Tree node types

    Every node in the Abstract Syntax Tree is inherited from base class
    Ast. This class provides base properties and functions that are implemented
    by base classes.
)";

static const char* accept_method = R"(
    Accept (or visit) the current AST node using current visitor

    Instead of visiting children of AST node, like Ast::visit_children,
    accept allows to visit the current node itself using the concrete
    visitor provided.

    Args:
        v (Visitor):  Concrete visitor that will be used to recursively visit node
)";

static const char* visit_children_method = R"(
    Visit children i.e. member of current AST node using provided visitor

    Different nodes in the AST have different members (i.e. children). This method
    recursively visits children using provided concrete visitor.

    Args:
        v (Visitor):  Concrete visitor that will be used to recursively visit node
)";

static const char* get_node_type_method = R"(
    Return type (ast.AstNodeType) of the ast node
)";

static const char* get_node_type_name_method = R"(
    Return type (ast.AstNodeType) of the ast node as string
)";

static const char* get_node_name_method = R"(
    Return name of the node

    Some ast nodes have a member designated as node name. For example,
    in case of ast.FunctionCall, name of the function is returned as a node
    name. Note that this is different from ast node type name and not every
    ast node has name.
)";

static const char* get_nmodl_name_method = R"(
    Return nmodl statement of the node

    Some ast nodes have a member designated as nmodl name. For example,
    in case of "NEURON { }" the statement of NMODL which is stored as nmodl
    name is "NEURON". This function is only implemented by node types that
    have a nmodl statement.
)";


static const char* clone_method = R"(
    Create a copy of the AST node
)";

static const char* get_token_method = R"(
    Return associated token for the AST node
)";

static const char* get_symbol_table_method = R"(
    Return associated symbol table for the AST node

    Certain ast nodes (e.g. inherited from ast.Block) have associated
    symbol table. These nodes have nmodl.symtab.SymbolTable as member
    and it can be accessed using this method.
)";

static const char* get_statement_block_method = R"(
    Return associated statement block for the AST node

    Top level block nodes encloses all statements in the ast::StatementBlock.
    For example, ast.BreakpointBlock has all statements in curly brace (`{ }`)
    stored in ast.StatementBlock :

    BREAKPOINT {
        SOLVE states METHOD cnexp
        gNaTs2_t = gNaTs2_tbar*m*m*m*h
        ina = gNaTs2_t*(v-ena)
    }

    This method return enclosing statement block.
)";

static const char* negate_method = R"(
    Negate the value of AST node
)";

static const char* set_name_method = R"(
    Set name for the AST node
)";

static const char* is_ast_method = R"(
    Check if current node is of type ast.Ast
)";

static const char* eval_method = R"(
    Return value of the ast node
)";

}  // namespace docstring
}  // namespace nmodl
