/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ast/ast.hpp"
#include "lexer/modtoken.hpp"
#include "symtab/symbol_table.hpp"


using namespace nmodl;
using namespace ast;


/**
 *
 * @defgroup ast_python AST Python Classes
 * @ingroup ast
 * @brief AST classes for Python bindings
 * @{
 */

/**
 * \brief Class mirroring nmodl::ast::AST for Python bindings
 *
 * \details \copydetails nmodl::ast::AST
 *
 * The goal of this class is to only interface nmodl::ast::AST with
 * the Python world using `pybind11`.
 */
struct PyAST: public AST {

    void visit_children(Visitor* v) override {
        PYBIND11_OVERLOAD_PURE(void,            /// Return type
                               AST,             /// Parent class
                               visit_children,  /// Name of function in C++ (must match Python name)
                               v                /// Argument(s)
        );
    }

    void accept(Visitor* v) override {
        PYBIND11_OVERLOAD_PURE(void, AST, accept, v);
    }


    AST* clone() override {
        PYBIND11_OVERLOAD(AST*, AST, clone, );
    }

    AstNodeType get_node_type() override {
        PYBIND11_OVERLOAD_PURE(AstNodeType,    // Return type
                               AST,            // Parent class
                               get_node_type,  // Name of function in C++ (must match Python name)
                                               // No argument (trailing ,)
        );
    }

    std::string get_node_type_name() override {
        PYBIND11_OVERLOAD_PURE(std::string, AST, get_node_type_name, );
    }

    std::string get_node_name() override {
        PYBIND11_OVERLOAD(std::string, AST, get_node_name, );
    }

    std::shared_ptr<AST> get_shared_ptr() override {
        PYBIND11_OVERLOAD(std::shared_ptr<AST>, AST, get_shared_ptr, );
    }

    ModToken* get_token() override {
        PYBIND11_OVERLOAD(ModToken*, AST, get_token, );
    }

    symtab::SymbolTable* get_symbol_table() override {
        PYBIND11_OVERLOAD(symtab::SymbolTable*, AST, get_symbol_table, );
    }

    std::shared_ptr<StatementBlock> get_statement_block() override {
        PYBIND11_OVERLOAD(std::shared_ptr<StatementBlock>, AST, get_statement_block, );
    }

    void set_symbol_table(symtab::SymbolTable* newsymtab) override {
        PYBIND11_OVERLOAD(void, AST, set_symbol_table, newsymtab);
    }

    void set_name(std::string name) override {
        PYBIND11_OVERLOAD(void, AST, set_name, name);
    }

    void negate() override {
        PYBIND11_OVERLOAD(void, AST, negate, );
    }

    bool is_ast() override {
        PYBIND11_OVERLOAD(bool, AST, is_ast, );
    }

    {% for node in nodes %}

    bool is_{{node.class_name | snake_case}}() override {
        PYBIND11_OVERLOAD(bool, AST, is_{{node.class_name | snake_case}}, );
    }

    {% endfor %}
};

/** @} */  // end of ast_python