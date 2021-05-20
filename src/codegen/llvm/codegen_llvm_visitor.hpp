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
#if LLVM_VERSION_MAJOR >= 13
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
    /// Name of mod file (without .mod suffix).
    std::string mod_filename;

    /// Output directory for code generation.
    std::string output_dir;

  private:
    /// Underlying LLVM context.
    std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();

    /// Underlying LLVM module.
    std::unique_ptr<llvm::Module> module = std::make_unique<llvm::Module>(mod_filename, *context);

    /// LLVM IR builder.
    IRBuilder ir_builder;

    /// Debug information builder.
    DebugBuilder debug_builder;

    /// Add debug information to the module.
    bool add_debug_information;

    /// Pointer to AST symbol table.
    symtab::SymbolTable* sym_tab;

    /// Instance variable helper.
    InstanceVarHelper instance_var_helper;

    /// Run optimisation passes if true.
    bool opt_passes;

    /// Pass manager for optimisation passes that are run on IR and are not related to target.
    llvm::legacy::FunctionPassManager opt_pm;

    /// Pass manager for optimisation passes that are used for target code generation.
    llvm::legacy::FunctionPassManager codegen_pm;

    /// Vector library used for maths functions.
    llvm::TargetLibraryInfoImpl::VectorLibrary vector_library;

    /// Explicit vectorisation width.
    int vector_width;

  public:
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
        , vector_width(vector_width)
        , vector_library(veclib_map.at(vec_lib))
        , add_debug_information(add_debug_information)
        , ir_builder(*context, use_single_precision, vector_width)
        , debug_builder(*module)
        , codegen_pm(module.get())
        , opt_pm(module.get()) {}

    /// Dumps the generated LLVM IR module to string.
    std::string dump_module() const {
        std::string str;
        llvm::raw_string_ostream os(str);
        os << *module;
        os.flush();
        return str;
    }

    /// Fills the container with the names of kernel functions from the MOD file.
    void find_kernel_names(std::vector<std::string>& container);

    /// Returns underlying module.
    std::unique_ptr<llvm::Module> get_module() {
        return std::move(module);
    }

    /// Returns shared_ptr to generated ast::InstanceStruct.
    std::shared_ptr<ast::InstanceStruct> get_instance_struct_ptr() {
        return instance_var_helper.instance;
    }

    /// Returns InstanceVarHelper for the given MOD file.
    InstanceVarHelper get_instance_var_helper() {
        return instance_var_helper;
    }

    /// Returns vector width
    int get_vector_width() const {
        return vector_width;
    }

    // Visitors.
    void visit_binary_expression(const ast::BinaryExpression& node) override;
    void visit_boolean(const ast::Boolean& node) override;
    void visit_codegen_atomic_statement(const ast::CodegenAtomicStatement& node) override;
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
    void visit_statement_block(const ast::StatementBlock& node) override;
    void visit_unary_expression(const ast::UnaryExpression& node) override;
    void visit_var_name(const ast::VarName& node) override;
    void visit_while_statement(const ast::WhileStatement& node) override;

    /// Wraps all kernel function calls into wrapper functions that use `void*` to pass the data to
    /// the kernel.
    void wrap_kernel_functions();

  private:
    /// Accepts the given AST node and returns the processed value.
    llvm::Value* accept_and_get(const std::shared_ptr<ast::Node>& node);

    /// Creates a call to an external function (e.g pow, exp, etc.)
    void create_external_function_call(const std::string& name,
                                       const ast::ExpressionVector& arguments);

    /// Creates a call to NMODL function or procedure in the same MOD file.
    void create_function_call(llvm::Function* func,
                              const std::string& name,
                              const ast::ExpressionVector& arguments);

    /// Fills values vector with processed NMODL function call arguments.
    void create_function_call_arguments(const ast::ExpressionVector& arguments,
                                        ValueVector& arg_values);

    /// Creates the function declaration for the given AST node.
    void create_function_declaration(const ast::CodegenFunction& node);

    /// Creates a call to `printf` function.
    void create_printf_call(const ast::ExpressionVector& arguments);

    /// Creates a vectorized version of the LLVM IR for the simple control flow statement.
    void create_vectorized_control_flow_block(const ast::IfStatement& node);

    /// Returns LLVM type for the given CodegenVarType AST node.
    llvm::Type* get_codegen_var_type(const ast::CodegenVarType& node);

    /// Returns the index value from the IndexedName AST node.
    llvm::Value* get_index(const ast::IndexedName& node);

    /// Returns an instance struct type.
    llvm::Type* get_instance_struct_type();

    /// Returns the number of elements in the array specified by the IndexedName AST node.
    int get_num_elements(const ast::IndexedName& node);

    /// Returns whether the function is an NMODL compute kernel.
    bool is_kernel_function(const std::string& function_name);

    /// If the value to store is specified, writes it to the instance. Otherwise, returns the
    /// instance variable.
    llvm::Value* read_from_or_write_to_instance(const ast::CodegenInstanceVar& node,
                                                llvm::Value* maybe_value_to_store = nullptr);

    /// Reads the given variable and returns the processed value.
    llvm::Value* read_variable(const ast::VarName& node);


    /// Run multiple LLVM optimisation passes on generated IR.
    /// TODO: this can be moved to a dedicated file or deprecated.
    void run_ir_opt_passes();

    //// Writes the value to the given variable.
    void write_to_variable(const ast::VarName& node, llvm::Value* value);
};

/** \} */  // end of llvm_backends

}  // namespace codegen
}  // namespace nmodl
