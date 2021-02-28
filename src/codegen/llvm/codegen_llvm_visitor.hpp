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

#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/logger.hpp"
#include "visitors/ast_visitor.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

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

    InstanceVarHelper instance_var_helper;

    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();

    std::unique_ptr<llvm::Module> module = std::make_unique<llvm::Module>(mod_filename, *context);

    llvm::IRBuilder<> builder;

    llvm::legacy::FunctionPassManager fpm;

    // Stack to hold visited values
    std::vector<llvm::Value*> values;

    // Pointer to the current function.
    llvm::Function* current_func = nullptr;

    // Pointer to AST symbol table.
    symtab::SymbolTable* sym_tab;

    // Run optimisation passes if true.
    bool opt_passes;

    // Use 32-bit floating-point type if true. Otherwise, use deafult 64-bit.
    bool use_single_precision;

    // explicit vectorisation width
    int vector_width;

    // LLVM mechanism struct
    llvm::StructType* llvm_struct;

    /**
     *\brief Run LLVM optimisation passes on generated IR
     *
     * LLVM provides number of optimisation passes that can be run on the generated IR.
     * Here we run common optimisation LLVM passes that benefits code optimisation.
     */
    void run_llvm_opt_passes();

  public:
    /**
     * \brief Constructs the LLVM code generator visitor
     *
     * This constructor instantiates an NMODL LLVM code generator. This is
     * just template to work with initial implementation.
     */
    CodegenLLVMVisitor(const std::string& mod_filename,
                       const std::string& output_dir,
                       bool opt_passes,
                       int vector_width = 1,
                       bool use_single_precision = false)
        : mod_filename(mod_filename)
        , output_dir(output_dir)
        , opt_passes(opt_passes)
        , use_single_precision(use_single_precision)
        , vector_width(vector_width)
        , builder(*context)
        , fpm(module.get()) {}

    /**
     * Checks if array index specified by the given IndexedName is within bounds
     * \param node IndexedName representing array
     * \return     \c true if the index is within bounds
     */
    bool check_array_bounds(const ast::IndexedName& node, unsigned index);

    /**
     * Generates LLVM code for the given IndexedName
     * \param node IndexedName NMODL AST node
     * \return LLVM code generated for this AST node
     */
    llvm::Value* codegen_indexed_name(const ast::IndexedName& node);

    /**
     * Returns GEP instruction to 1D array
     * \param name 1D array name
     * \param index element index
     * \return GEP instruction value
     */
    llvm::Value* create_gep(const std::string& name, unsigned index);

    /**
     * Returns array index or length from given IndexedName
     * \param node IndexedName representing array
     * \return array index or length
     */
    unsigned get_array_index_or_length(const ast::IndexedName& node);

    /**
     * Returns LLVM type for the given CodegenVarType node
     * \param node CodegenVarType
     * \return LLVM type
     */
    llvm::Type* get_codegen_var_type(const ast::CodegenVarType& node);

    /**
     * Returns 64-bit or 32-bit LLVM floating type
     * \return     \c LLVM floating point type according to `use_single_precision` flag
     */
    llvm::Type* get_default_fp_type();

    /**
     * Returns pointer to 64-bit or 32-bit LLVM floating type
     * \return     \c LLVM pointer to floating point type according to `use_single_precision` flag
     */
    llvm::Type* get_default_fp_ptr_type();

    /**
     * Create a function call to an external method
     * \param name external method name
     * \param arguments expressions passed as arguments to the given external method
     */
    void create_external_method_call(const std::string& name,
                                     const ast::ExpressionVector& arguments);

    /**
     * Create a function call to NMODL function or procedure in the same mod file
     * \param func LLVM function corresponding ti this call
     * \param name function name
     * \param arguments expressions passed as arguments to the function call
     */
    void create_function_call(llvm::Function* func,
                              const std::string& name,
                              const ast::ExpressionVector& arguments);
    /**
     * Create a function call to printf function
     * \param arguments expressions passed as arguments to the printf call
     */
    void create_printf_call(const ast::ExpressionVector& arguments);

    /**
     * Emit function or procedure declaration in LLVM given the node
     *
     * \param node the AST node representing the function or procedure in NMODL
     */
    void emit_procedure_or_function_declaration(const ast::CodegenFunction& node);

    /**
     * Return module pointer
     * \return LLVM IR module pointer
     */
    std::unique_ptr<llvm::Module> get_module() {
        return std::move(module);
    }

    /**
     * Lookup the given name in the current function's symbol table
     * \return LLVM value
     */
    llvm::Value* lookup(const std::string& name);

    /**
     * Fills values vector with processed NMODL function call arguments
     * \param arguments expression vector
     * \param arg_values vector of LLVM IR values to fill
     */
    void pack_function_call_arguments(const ast::ExpressionVector& arguments,
                                      std::vector<llvm::Value*>& arg_values);

    /**
     * Visit nmodl arithmetic binary operator
     * \param lhs LLVM value of evaluated lhs expression
     * \param rhs LLVM value of evaluated rhs expression
     * \param op the AST binary operator (ADD, DIV, MUL, SUB)
     * \return LLVM IR value result
     */
    llvm::Value* visit_arithmetic_bin_op(llvm::Value* lhs, llvm::Value* rhs, unsigned op);

    /**
     * Visit nmodl assignment operator (ASSIGN)
     * \param node the AST node representing the binary expression in NMODL
     * \param rhs LLVM value of evaluated rhs expression
     */
    void visit_assign_op(const ast::BinaryExpression& node, llvm::Value* rhs);

    /**
     * Visit nmodl logical binary operator
     * \param lhs LLVM value of evaluated lhs expression
     * \param rhs LLVM value of evaluated rhs expression
     * \param op the AST binary operator (AND, OR)
     * \return LLVM IR value result
     */
    llvm::Value* visit_logical_bin_op(llvm::Value* lhs, llvm::Value* rhs, unsigned op);

    /**
     * Visit nmodl comparison binary operator
     * \param lhs LLVM value of evaluated lhs expression
     * \param rhs LLVM value of evaluated rhs expression
     * \param op the AST binary operator (EXACT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL,
     * NOT_EQUAL) \return LLVM IR value result
     */
    llvm::Value* visit_comparison_bin_op(llvm::Value* lhs, llvm::Value* rhs, unsigned op);


    // Visitors
    void visit_binary_expression(const ast::BinaryExpression& node) override;
    void visit_boolean(const ast::Boolean& node) override;
    void visit_statement_block(const ast::StatementBlock& node) override;
    void visit_codegen_function(const ast::CodegenFunction& node) override;
    void visit_codegen_return_statement(const ast::CodegenReturnStatement& node) override;
    void visit_codegen_var_list_statement(const ast::CodegenVarListStatement& node) override;
    void visit_double(const ast::Double& node) override;
    void visit_function_block(const ast::FunctionBlock& node) override;
    void visit_function_call(const ast::FunctionCall& node) override;
    void visit_if_statement(const ast::IfStatement& node) override;
    void visit_integer(const ast::Integer& node) override;
    void visit_procedure_block(const ast::ProcedureBlock& node) override;
    void visit_program(const ast::Program& node) override;
    void visit_unary_expression(const ast::UnaryExpression& node) override;
    void visit_var_name(const ast::VarName& node) override;
    void visit_codegen_instance_var(const ast::CodegenInstanceVar& node) override;
    void visit_instance_struct(const ast::InstanceStruct& node) override;
    void visit_while_statement(const ast::WhileStatement& node) override;

    // \todo: move this to debug mode (e.g. -v option or --dump-ir)
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
