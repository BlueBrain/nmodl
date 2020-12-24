/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \dir
 * \brief LLVM based code generation backend implementation for CoreNEURON
 *
 * \file
 * \brief \copybrief nmodl::codegen::CodegenLLVMVisitor
 */

#include <ostream>
#include <string>

#include "utils/logger.hpp"
#include "visitors/ast_visitor.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace nmodl {
namespace codegen {

/**
 * @defgroup llvm LLVM Based Code Generation Implementation
 * @brief Implementations of LLVM based code generation
 *
 * @defgroup llvm_backends LLVM Codegen Backend
 * @ingroup llvm
 * @brief Code generation backends for NMODL AST to LLVM IR
 * @{
 */

/**
 * \class CodegenLLVMVisitor
 * \brief %Visitor for transforming NMODL AST to LLVM IR
 */
class CodegenLLVMVisitor: public visitor::ConstAstVisitor {
    // Name of mod file (without .mod suffix)
    std::string mod_filename;

    // Output directory for code generation
    std::string output_dir;

  private:
    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();

    std::unique_ptr<llvm::Module> module = std::make_unique<llvm::Module>(mod_filename, *context);

    llvm::IRBuilder<> builder;

    // Stack to hold visited values
    std::vector<llvm::Value*> values;

    // Pointer to the local symbol table.
    llvm::ValueSymbolTable* local_named_values = nullptr;

  public:
    /**
     * \brief Constructs the LLVM code generator visitor
     *
     * This constructor instantiates an NMODL LLVM code generator. This is
     * just template to work with initial implementation.
     */
    CodegenLLVMVisitor(const std::string& mod_filename, const std::string& output_dir)
        : mod_filename(mod_filename)
        , output_dir(output_dir)
        , builder(*context) {}

    /**
     * Visit nmodl function or procedure
     * \param node the AST node representing the function or procedure in NMODL
     */
    void visit_procedure_or_function(const ast::Block& node);

    // Visitors
    void visit_binary_expression(const ast::BinaryExpression& node) override;
    void visit_boolean(const ast::Boolean& node) override;
    void visit_double(const ast::Double& node) override;
    void visit_function_block(const ast::FunctionBlock& node) override;
    void visit_integer(const ast::Integer& node) override;
    void visit_local_list_statement(const ast::LocalListStatement& node) override;
    void visit_procedure_block(const ast::ProcedureBlock& node) override;
    void visit_program(const ast::Program& node) override;
    void visit_unary_expression(const ast::UnaryExpression& node) override;
    void visit_var_name(const ast::VarName& node) override;

    // TODO: use custom printer here
    std::string print_module() const {
        std::string str;
        llvm::raw_string_ostream os(str);
        os << *module;
        os.flush();
        return str;
    }
};

/** \} */  // end of llvm_backends

}  // namespace codegen
}  // namespace nmodl
