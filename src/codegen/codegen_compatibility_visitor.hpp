/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::codegen::CodegenCompatibilityVisitor
 */

#include <set>

#include "ast/ast.hpp"
#include "codegen_naming.hpp"
#include "symtab/symbol_table.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace codegen {

using namespace ast;

/**
 * @addtogroup codegen_backends
 * @{
 */

/**
 * \class CodegenCompatibilityVisitor
 * \brief %Visitor for printing compatibility issues of the mod file
 */
class CodegenCompatibilityVisitor: public visitor::AstVisitor {
    /// Array of all the ast::AstNodeType that are unhandled
    /// by the NMODL \c C++ code generator
    std::vector<ast::AstNodeType> unhandled_ast_types = {AstNodeType::SOLVE_BLOCK,
                                                         AstNodeType::TERMINAL_BLOCK,
                                                         AstNodeType::PARTIAL_BLOCK,
                                                         AstNodeType::MATCH_BLOCK,
                                                         AstNodeType::BEFORE_BLOCK,
                                                         AstNodeType::AFTER_BLOCK,
                                                         AstNodeType::CONSTANT_BLOCK,
                                                         AstNodeType::CONSTRUCTOR_BLOCK,
                                                         AstNodeType::DESTRUCTOR_BLOCK,
                                                         AstNodeType::DISCRETE_BLOCK,
                                                         AstNodeType::FUNCTION_TABLE_BLOCK,
                                                         AstNodeType::INDEPENDENT_BLOCK,
                                                         AstNodeType::GLOBAL_VAR,
                                                         AstNodeType::POINTER_VAR,
                                                         AstNodeType::BBCORE_POINTER_VAR};

    /// Set of handled solvers by the NMODL \c C++ code generator
    const std::set<std::string> handled_solvers{codegen::naming::CNEXP_METHOD,
                                                codegen::naming::EULER_METHOD,
                                                codegen::naming::DERIVIMPLICIT_METHOD,
                                                codegen::naming::SPARSE_METHOD};

    /// Vector that stores all the ast::Node that are unhandled
    /// by the NMODL \c C++ code generator
    std::vector<std::shared_ptr<ast::Ast>> unhandled_ast_nodes;

  public:
    /// \name Ctor & dtor
    /// \{

    /// Default CodegenCompatibilityVisitor constructor
    CodegenCompatibilityVisitor() = default;

    /// \}

    /// Function that searches the ast::Ast for nodes that
    /// are incompatible with NMODL \c C++ code generator
    bool find_unhandled_ast_nodes(Ast* node);

    /// Takes as parameter an ast::SolveBlock, searches if the
    /// method used for solving is supported and if it is not
    /// it returns a relative error message
    std::string return_error_if_solve_method_is_unhandled(ast::SolveBlock* solve_block_ast_node);

    /// Takes as parameter an ast::Ast node and returns a relative
    /// error with the name, the type and the location of the
    /// unhandled statement
    ///
    /// \tparam T Type of node parameter in the ast::Ast
    template <typename T>
    std::string return_error_with_name(const std::shared_ptr<ast::Ast>& ast_node);

    /// Takes as parameter an ast::Ast node and returns a relative
    /// error with the type and the location of the unhandled
    /// statement
    ///
    /// \tparam T Type of node parameter in the ast::Ast
    template <typename T>
    std::string return_error_without_name(const std::shared_ptr<ast::Ast>& ast_node);

    /// Takes as parameter the ast::Ast to read the symbol table
    /// and an ast::GlobarVar node and returns relative error if a
    /// variable that is writen in the mod file is defined as
    /// GLOBAL instead of RANGE
    std::string return_error_global_var(Ast* node, ast::GlobalVar* global_var);

    /// Takes as parameter an ast::PointerVar and returns a
    /// relative error with the name and the location of the
    /// pointer, as well as a suggestion to define it as
    /// BBCOREPOINTER
    std::string return_error_pointer(ast::PointerVar* pointer_var);

    /// Takes as parameter the ast::Ast and checks if the
    /// functions "bbcore_read" and "bbcore_write" are defined
    /// in any of the ast::Ast VERBATIM blocks. The function is
    /// called if there is a BBCORE_POINTER defined in the mod
    /// file
    std::string return_error_if_no_bbcore_read_write(Ast* node);
};

/** @} */  // end of codegen_backends

}  // namespace codegen
}  // namespace nmodl
