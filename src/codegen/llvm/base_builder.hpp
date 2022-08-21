/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

// Floating point bit widths.
static constexpr const unsigned single_precision = 32;
static constexpr const unsigned double_precision = 64;

// Some typedefs.
using ConstantVector = std::vector<llvm::Constant*>;
using TypeVector = std::vector<llvm::Type*>;
using ValueVector = std::vector<llvm::Value*>;

/// Transforms the fast math flags provided to the builder into LLVM's representation.
static llvm::FastMathFlags transform_to_fmf(std::vector<std::string>& flags) {
    static const std::map<std::string, void (llvm::FastMathFlags::*)(bool)> set_flag = {
        {"nnan", &llvm::FastMathFlags::setNoNaNs},
        {"ninf", &llvm::FastMathFlags::setNoInfs},
        {"nsz", &llvm::FastMathFlags::setNoSignedZeros},
        {"contract", &llvm::FastMathFlags::setAllowContract},
        {"afn", &llvm::FastMathFlags::setApproxFunc},
        {"reassoc", &llvm::FastMathFlags::setAllowReassoc},
        {"fast", &llvm::FastMathFlags::setFast}};
    llvm::FastMathFlags fmf;
    for (const auto& flag: flags) {
        (fmf.*(set_flag.at(flag)))(true);
    }
    return fmf;
}

namespace nmodl {
namespace codegen {

/**
 * \class BaseBuilder
 * \brief A base class to generate LLVM IR for NMODL AST. Assumes a CPU target
 * by default.
 */
class BaseBuilder {
  protected:
    /// Underlying LLVM IR builder.
    llvm::IRBuilder<> builder;

    /// Stack to hold visited and processed values.
    ValueVector value_stack;

    /// Pointer to the current function for which the code is generated.
    llvm::Function* current_function;

    /// Insertion point for `alloca` instructions.
    llvm::Instruction* alloca_ip;

    /// If single-precision floating point numbers are used. 
    bool single_precision;

    /// Fast math flags for floating-point IR instructions.
    std::vector<std::string> fast_math_flags;

    BaseBuilder(llvm::LLVMContext& context,
                bool single_precision,
                std::vector<std::string> fast_math_flags = {})
        : builder(context)
        , current_function(nullptr)
        , alloca_ip(nullptr)
        , single_precision(single_precision) {}

  public:
    /// Creates a new instance of a base builder.
    void create(llvm::LLVMContext& context,
              bool single_precision,
              std::vector<std::string> fast_math_flags) {
        // TODO: create class!
        // if (!fast_math_flags.empty())
        //     builder.setFastMathFlags(transform_to_fmf(fast_math_flags));
    }

    /*************************************************************************/
    /*                       Value processing utilities                      */
    /*************************************************************************/

    /// Lookups the value by its name in the current function's symbol table.
    llvm::Value* lookup_value(const std::string& value_name);

    /// Pops the last value from the stack.
    llvm::Value* pop_last_value();

    /*************************************************************************/
    /*                            Type utilities                             */
    /*************************************************************************/

    /// Creates a boolean type.
    llvm::Type* get_boolean_type();

    /// Creates a 32-bit integer type.
    llvm::Type* get_i32_type();

    /// Creates a 64-bit integer type.
    llvm::Type* get_i64_type();

    /// Creates a floating-point type.
    llvm::Type* get_fp_type();

    /// Creates a void type.
    llvm::Type* get_void_type();

    /// Creates a pointer to 8-bit integer type.
    llvm::Type* get_i8_ptr_type();

    /// Creates a pointer to 32-bit integer type.
    llvm::Type* get_i32_ptr_type();

    /// Creates a pointer to floating-point type.
    llvm::Type* get_fp_ptr_type();

    /// Creates a pointer to struct type with the given name and given members.
    llvm::Type* get_struct_ptr_type(const std::string& struct_type_name, TypeVector& member_types);

    /*************************************************************************/
    /*                           Function utilities                          */
    /*************************************************************************/

    /// Returns the name of the function for which LLVM IR is generated.
    std::string get_current_function_name();

    /// Sets the current function for which LLVM IR is generated.
    void set_function(llvm::Function* function);

    /// Clears the stack of the values and unsets the current function.
    void unset_function();

    /// Generates LLVM IR for a call to the function.
    void generate_function_call(llvm::Function* callee,
                                ValueVector& arguments,
                                bool with_result = true);
    
    /// Generates an intrinsic that corresponds to the given name.
    void generate_intrinsic(const std::string& name,
                            ValueVector& argument_values,
                            TypeVector& argument_types);

    /// Generates LLVM IR to allocate the arguments of the function on the stack.
    void allocate_function_arguments(llvm::Function* function,
                                     const ast::CodegenVarWithTypeVector& nmodl_arguments);

    /// Generates LLVM IR to allocate the arguments of the NMODL compute kernel
    /// on the stack, bitcasting void* pointer to mechanism struct pointer.
    void allocate_and_wrap_kernel_arguments(llvm::Function* function,
                                            const ast::CodegenVarWithTypeVector& nmodl_arguments,
                                            llvm::Type* mechanism_ptr);

    /*************************************************************************/
    /*                          Basic block utilities                        */
    /*************************************************************************/

    /// Returns the current basic block.
    llvm::BasicBlock* get_current_block();

    /// Sets builder's insertion point to the given block.
    void set_insertion_point(llvm::BasicBlock* block);

    /// Creates a basic block and set the builder's insertion point to it.
    llvm::BasicBlock* create_block_and_set_insertion_point(llvm::Function* function,
                                                           llvm::BasicBlock* insert_before,
                                                           std::string name = "");

    /// Creates LLVM IR conditional branch instruction.
    llvm::BranchInst* create_cond_br(llvm::Value* condition,
                                     llvm::BasicBlock* true_block,
                                     llvm::BasicBlock* false_block);

    /// Generates LLVM IR for unconditional branch.
    void generate_br(llvm::BasicBlock* block);

    /// Generates LLVM IR for unconditional branch and sets the insertion point to this block.
    void generate_br_and_set_insertion_point(llvm::BasicBlock* block);

    /*************************************************************************/
    /*                         Instruction utilities                         */
    /*************************************************************************/

    /// Creates LLVM IR alloca instruction.
    llvm::Value* create_alloca(const std::string& name, llvm::Type* type);

    /// Creates LLVM IR alloca instruction for an array.
    llvm::Value* create_array_alloca(const std::string& name, llvm::Type* element_type, int num_elements);

    /// Creates LLVM IR global string value.
    llvm::Value* create_global_string(const ast::String& node);

    /// Creates LLVM IR inbounds GEP instruction for the given name and returns the calculated address.
    llvm::Value* create_inbounds_gep(const std::string& variable_name, llvm::Value* index);

    /// Creates LLVM IR return instructions.
    void create_return(llvm::Value* return_value = nullptr);

    /// Creates LLVM IR to get the address of the struct's value at given offset. Returns the
    /// calculated address.
    llvm::Value* create_struct_field_ptr(llvm::Value* struct_variable, int offset);

    /// Extracts binary operator (+ or -) from atomic update (+= or =-).
    ast::BinaryOp into_atomic_op(ast::BinaryOp op);

    /// Transforms an integer value into an index.
    llvm::Value* into_index(llvm::Value* value);

    /*************************************************************************/
    /*                           Constant utilities                          */
    /*************************************************************************/

  protected:
    /// Returns a scalar constant of the provided type.
    template <typename C, typename V>
    llvm::Value* scalar_constant(llvm::Type* type, V value);

    /// Returns a vector constant of the provided type.
    template <typename C, typename V>
    llvm::Value* vector_constant(llvm::Type* type, V value, unsigned num_elements);

    /*************************************************************************/
    /*                     Virtual generation methods                        */
    /*************************************************************************/

    /// Generates LLVM IR to handle atomic updates, e.g. *ptr += rhs.
    virtual void generate_atomic_statement(llvm::Value* ptr, llvm::Value* rhs, ast::BinaryOp op);

    /// Generates LLVM IR for the binary operator.
    virtual void generate_binary_op(llvm::Value* lhs, llvm::Value* rhs, ast::BinaryOp op);

    /// Generates LLVM IR for the unary operator.
    virtual void generate_unary_op(llvm::Value* value, ast::UnaryOp op);

    /// Generates LLVM IR for boolean constants.
    virtual void generate_boolean_constant(int value);

    /// Generates LLVM IR for integer constants.
    virtual void generate_i32_constant(int value);

    /// Generates LLVM IR for floating-point constants.
    virtual void generate_fp_constant(const std::string& value);

    /// Generates LLVM IR for directed data read.
    virtual void generate_load_direct(llvm::Value* ptr);

    /// Generates LLVM IR for indirected data read.
    virtual void generate_load_indirect(llvm::Value* ptr);

    /// Generates LLVM IR for directed data write.
    virtual void generate_store_direct(llvm::Value* ptr, llvm::Value* value);

    /// Generates LLVM IR for indirected data write.
    virtual void generate_store_indirect(llvm::Value* ptr, llvm::Value* value);

    /// Generates LLVM IR for brodcasting a value, if necessary.
    virtual void try_generate_broadcast(llvm::Value* ptr);

    /// Generates LLVM IR for loop initialization.
    virtual void generate_loop_start();

    /// Generates LLVM IR for loop increment.
    virtual void generate_loop_increment();

    /// Generates LLVM IR for loop termination.
    /// TODO: should take an AST node!
    virtual void generate_loop_end();
};

}  // namespace codegen
}  // namespace nmodl
