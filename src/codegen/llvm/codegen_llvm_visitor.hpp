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
#include "codegen/llvm/llvm_debug_builder.hpp"
#include "codegen/llvm/llvm_ir_builder.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/logger.hpp"
#include "visitors/ast_visitor.hpp"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DIBuilder.h"
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

/// A map to query vector library by its string value.
static const std::map<std::string, llvm::TargetLibraryInfoImpl::VectorLibrary> veclib_map = {
    {"Accelerate", llvm::TargetLibraryInfoImpl::Accelerate},
#ifndef LLVM_VERSION_LESS_THAN_13
    {"libmvec", llvm::TargetLibraryInfoImpl::LIBMVEC_X86},
#endif
    {"MASSV", llvm::TargetLibraryInfoImpl::MASSV},
    {"SVML", llvm::TargetLibraryInfoImpl::SVML},
    {"none", llvm::TargetLibraryInfoImpl::NoLibrary}};

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

    // LLVM IR builder.
    IRBuilder ir_builder;

    // Debug information builder.
    DebugBuilder debug_builder;

    // Add debug information to the module.
    bool add_debug_information;

    // Pass manager for optimisation passes that are used for target code generation.
    llvm::legacy::FunctionPassManager codegen_pm;

    // Vector library used for maths functions.
    llvm::TargetLibraryInfoImpl::VectorLibrary vector_library;

    // Pass manager for optimisation passes that are run on IR and are not related to target.
    llvm::legacy::FunctionPassManager opt_pm;

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

    // Explicit vectorisation width.
    int vector_width;

    // The name of induction variable used in the kernel functions.
    std::string kernel_id;

    // A flag to indicate that the code is generated for the kernel.
    bool is_kernel_code = false;

    /**
     *\brief Run LLVM optimisation passes on generated IR
     *
     * LLVM provides number of optimisation passes that can be run on the generated IR.
     * Here we run common optimisation LLVM passes that benefits code optimisation.
     */
    void run_ir_opt_passes();

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
                       bool use_single_precision = false,
                       int vector_width = 1,
                       std::string vec_lib = "none",
                       bool add_debug_information = false)
        : mod_filename(mod_filename)
        , output_dir(output_dir)
        , opt_passes(opt_passes)
        , use_single_precision(use_single_precision)
        , vector_width(vector_width)
        , vector_library(veclib_map.at(vec_lib))
        , add_debug_information(add_debug_information)
        , ir_builder(*context, use_single_precision, vector_width)
        , debug_builder(*module)
        , codegen_pm(module.get())
        , opt_pm(module.get()) {}


    /**
     * Generates LLVM code for the given IndexedName
     * \param node IndexedName NMODL AST node
     * \return LLVM code generated for this AST node
     */
    llvm::Value* codegen_indexed_name(const ast::IndexedName& node);

    /**
     * Generates LLVM code for the given Instance variable
     * \param node CodegenInstanceVar NMODL AST node
     * \return LLVM code generated for this AST node
     */
    llvm::Value* codegen_instance_var(const ast::CodegenInstanceVar& node);

    /**
     * Returns array index from given IndexedName
     * \param node IndexedName representing array
     * \return array index
     */
    llvm::Value* get_array_index(const ast::IndexedName& node);

    /**
     * Returns array length from given IndexedName
     * \param node IndexedName representing array
     * \return array length
     */
    int get_array_length(const ast::IndexedName& node);

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
     * Returns a pointer to LLVM struct type
     * \return LLVM pointer type
     */
    llvm::Type* get_instance_struct_type();

    /**
     * Returns a LLVM value corresponding to the VarName node
     * \return LLVM value
     */
    llvm::Value* get_variable_ptr(const ast::VarName& node);

    /**
     * Returns shared_ptr to generated ast::InstanceStruct
     * \return std::shared_ptr<ast::InstanceStruct>
     */
    std::shared_ptr<ast::InstanceStruct> get_instance_struct_ptr();

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
     * Return InstanceVarHelper
     * \return InstanceVarHelper
     */
    InstanceVarHelper get_instance_var_helper() {
        return instance_var_helper;
    }

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
     * Visit nmodl assignment operator (ASSIGN)
     * \param node the AST node representing the binary expression in NMODL
     * \param rhs LLVM value of evaluated rhs expression
     */
    void visit_assign_op(const ast::BinaryExpression& node, llvm::Value* rhs);

    // Visitors
    void visit_binary_expression(const ast::BinaryExpression& node) override;
    void visit_boolean(const ast::Boolean& node) override;
    void visit_statement_block(const ast::StatementBlock& node) override;
    void visit_codegen_for_statement(const ast::CodegenForStatement& node) override;
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
    void visit_while_statement(const ast::WhileStatement& node) override;

    /**
     * Dumps the generated LLVM IR module to string.
     */
    std::string dump_module() const {
        std::string str;
        llvm::raw_string_ostream os(str);
        os << *module;
        os.flush();
        return str;
    }

    /**
     * Fills the container with the names of kernel functions from the MOD file.
     */
    void find_kernel_names(std::vector<std::string>& container);

    /**
     * Wraps all kernel function calls into wrapper functions that use void* to pass the data to the
     * kernel.
     */
    void wrap_kernel_functions();

  private:
    /// Accepts the given AST node and returns the processed value.
    llvm::Value* accept_and_get(const std::shared_ptr<ast::Node>& node);
};

/** \} */  // end of llvm_backends

}  // namespace codegen
}  // namespace nmodl
