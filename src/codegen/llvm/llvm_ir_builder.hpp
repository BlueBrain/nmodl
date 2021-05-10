/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"
#include "symtab/symbol_table.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

namespace nmodl {
namespace codegen {

/// Floating point bit widths.
static constexpr const unsigned single_precision= 32;
static constexpr const unsigned double_precision= 64;

/// Some typedefs.
using ConstantVector = std::vector<llvm::Constant*>;
using TypeVector = std::vector<llvm::Type*>;
using ValueVector = std::vector<llvm::Value*>;

/**
 * \class IRBuilder
 * \brief A helper class to generate LLVM IR for NMODL AST.
 */
class IRBuilder {
  public:
    /// Underlying LLVM IR builder.
    llvm::IRBuilder<> builder;

    /// Stack to hold visited and processed values.
    ValueVector value_stack;

    /// Pointer to the current function for which the code is generated.
    llvm::Function* current_function;

  private:
    /// Symbol table of the NMODL AST.
    symtab::SymbolTable* symbol_table;

    /// Flag to indicate that the generated IR should be vectorized.
    bool vectorize;

    /// A helper to query the instance variable information.
    InstanceVarHelper* instance_var_helper;

    /// Precision of the floating-point numbers (32 or 64 bit).
    unsigned fp_precision;

    /// The vector width if generating vectorized code.
    unsigned vector_width;

    /// The name of induction variable used in kernel loops.
    std::string kernel_id;

  public:
    IRBuilder(llvm::LLVMContext& context,
              bool use_single_precision = false,
              unsigned vector_width = 1)
        : builder(context)
        , symbol_table(nullptr)
        , current_function(nullptr)
        , vectorize(false)
        , instance_var_helper(nullptr)
        , fp_precision(use_single_precision ? single_precision : double_precision)
        , vector_width(vector_width)
        , kernel_id("") {}

    /// Initializes the builder with the symbol table, kernel id and instance variable info.
    void initialize(symtab::SymbolTable& symbol_table, std::string& kernel_id, InstanceVarHelper& instance_var_helper);

    /// Generates LLVM IR for the given binary operator.
    void create_binary_op(llvm::Value* lhs, llvm::Value* rhs, ast::BinaryOp op);

    /// Returns an inbounds GEP instruction to one-dimensional array `var_name`.
    llvm::Value* create_inbounds_gep(const std::string& var_name, llvm::Value* index);

    /// Generates LLVM IR for the given unary operator.
    void create_unary_op(llvm::Value* value, ast::UnaryOp op);

    /// Returns array length from given IndexedName.
    int get_array_length(const ast::IndexedName& node);

    /// Returns LLVM boolean (1-bit integer) constant.
    llvm::Value* get_bool_constant(int value);

    /// Returns LLVM floating-point constant.
    llvm::Value* get_fp_constant(const std::string& value);

    /// Returns LLVM vector with `vector_width` floating-point values.
    llvm::Value* get_fp_vector_constant(const std::string& value);

    /// Returns LLVM 32-bit integer constant.
    llvm::Value* get_i32_constant(int value);

    /// Returns LLVM vector with `vector_width` 32-bit integer values.
    llvm::Value* get_i32_vector_constant(int value);

    /// Lookups the value by  its name in the current function's symbol table.
    llvm::Value* lookup_value(const std::string& value_name);

    /// Pops the last visited value from the value stack.
    llvm::Value* pop_last_value();

  private:

    /// Creates a boolean (1-bit integer) type.
    llvm::Type* get_boolean_type();

    /// Creates a 32-bit integer type.
    llvm::Type* get_i32_type();

    /// Creates a pointer to 32-bit integer type.
    llvm::Type* get_i32_ptr_type();

    /// Creates a 64-bit integer type.
    llvm::Type* get_i64_type();

    /// Creates a floating-point type.
    llvm::Type* get_fp_type();

    /// Creates a pointer to floating-point type.
    llvm::Type* get_fp_ptr_type();

    /// Creates a void type.
    llvm::Type* get_void_type();

    /// Creates an instance struct type.
    llvm::Type* get_instance_struct_type();
};
}  // namespace codegen
}  // namespace nmodl
