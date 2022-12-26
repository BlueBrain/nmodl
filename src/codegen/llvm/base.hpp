/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "ast/all.hpp"
#include "codegen/codegen_naming.hpp"
#include "codegen/llvm/target_platform.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

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

    /// Underlying platform for this builder.
    /// TODO: change this when we have other builders. 
    Platform platform;

    bool vectorize;

    /// Mask to predicate generated IR. Unless the IR is vectorzied, this is kept as a nullptr.
    llvm::Value* mask;

    /// Fast math flags for floating-point IR instructions.
    std::vector<std::string> fast_math_flags;

  public:
    BaseBuilder(llvm::LLVMContext& context,
                Platform& platform,
                std::vector<std::string> fast_math_flags = {})
        : builder(context)
        , current_function(nullptr)
        , alloca_ip(nullptr)
        , platform(platform)
        , mask(nullptr)
        , vectorize(false) {
        if (!fast_math_flags.empty())
          builder.setFastMathFlags(transform_to_fmf(fast_math_flags));
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
                                                           llvm::BasicBlock* insert_before = nullptr,
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

    /// Creates LLVM IR instruction for the given binary operator (+, -).
    llvm::Value* create_atomic_op(llvm::Value* ptr, llvm::Value* update, ast::BinaryOp op);

    /// Creates LLVM IR bitcast instruction.
    llvm::Value* create_bitcast(llvm::Value* value, llvm::Type* type);

    /// Creates LLVM IR global string value.
    llvm::Value* create_global_string(const ast::String& node);

    /// Creates LLVM IR inbounds GEP instruction for the given name and returns the calculated address.
    llvm::Value* create_inbounds_gep(const std::string& variable_name, llvm::Value* index);

    /// Creates LLVM IR inbounds GEP instruction for the given variable and index. Returns calculated address.
    llvm::Value* create_inbounds_gep(llvm::Value* variable, llvm::Value* index);

    /// Creates LLVM IR to load the value specified by its name and returns it.
    llvm::Value* create_load_direct(const std::string& name, bool masked = false);

    /// Creates LLVM IR to load the value from the pointer and returns it.
    llvm::Value* create_load_direct(llvm::Value* ptr, bool masked = false);

    /// Creates LLVM IR to get the address of the struct's value at given offset. Returns the
    /// calculated address.
    llvm::Value* create_struct_field_ptr(llvm::Value* struct_variable, int offset);

    /// Creates LLVM IR to store the value to the location specified by the name.
    void create_store_direct(const std::string& name, llvm::Value* value, bool masked = false);

    /// Creates LLVM IR to store the value to the location specified by the pointer.
    void create_store_direct(llvm::Value* ptr, llvm::Value* value, bool masked = false);

    /// Extracts binary operator (+ or -) from atomic update (+= or =-).
    ast::BinaryOp into_atomic_op(ast::BinaryOp op);

    /// Transforms an integer value into an index.
    llvm::Value* into_index(llvm::Value* value);

    /*************************************************************************/
    /*                           Constant utilities                          */
    /*************************************************************************/

    /// Returns a scalar constant of the provided type.
    template <typename C, typename V>
    llvm::Value* scalar_constant(llvm::Type* type, V value);

    /// Returns a vector constant of the provided type.
    template <typename C, typename V>
    llvm::Value* vector_constant(llvm::Type* type, V value, unsigned num_elements);

    /*************************************************************************/
    /*                     Virtual generation methods                        */
    /*************************************************************************/

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

    /// Generates LLVM IR return instructions.
    virtual void generate_return(llvm::Value* return_value = nullptr);

    /// Generates IR to replicate the value if vectorizing the code.
    virtual void generate_broadcast(llvm::Value* value);

    /*************************************************************************/
    /*                       Virtual helper methods                          */
    /*************************************************************************/

    /// Sets the value to be the mask for vector code generation. If the builder
    /// does not emit vector code, throws an error.
    virtual void set_mask(llvm::Value* mask);

    /// Clears the mask for vector code generation. If the builder does not emit
    /// vector code, throws an error.
    virtual void unset_mask();

    /// Explicitly forces the builder to produce IR for compute parts. For
    /// example, this includes vectorizing the code in the FOR loop.
    virtual void start_generating_ir_for_compute();

    /// Explicitly forces the builder to produce IR for non-compute parts.
    virtual void stop_generating_ir_for_compute();

    /// Indicates whether the builder generates vector LLVM IR.
    virtual bool generating_vector_ir();

    /// Indicates whether the builder generates predicated vector LLVM IR.
    virtual bool generating_masked_vector_ir();

    /// Inverts the mask for vector code generation by xoring it.
    virtual void invert_mask();

    // TODO: These two methods are native for GPUBuilder!

    /// Creates an expression of the form: blockDim.x * gridDim.x
    void create_grid_stride();

    /// Creates an expression of the form: blockIdx.x * blockDim.x + threadIdx.x
    void create_thread_id();

    /// Generates IR that loads the elements of the array even during vectorization. If the value is
    /// specified, then it is stored to the array at the given index.
    llvm::Value* load_to_or_store_from_array(const std::string& id_name,
                                             llvm::Value* id_value,
                                             llvm::Value* array,
                                             llvm::Value* maybe_value_to_store = nullptr);

    // TODO: These three methods are native for SIMDBuilder!
    /// Creates a vector splat of starting addresses of the given member.
    llvm::Value* create_member_addresses(llvm::Value* member_ptr);

    /// Creates IR for calculating offest to member values. For more context, see
    /// `visit_codegen_atomic_statement` in LLVM visitor.
    llvm::Value* create_member_offsets(llvm::Value* start, llvm::Value* indices);

    /// Creates IR to perform scalar updates to instance member based on `ptrs_arr` for every
    /// element in a vector by
    ///     member[*ptrs_arr[i]] = member[*ptrs_arr[i]] op rhs.
    /// Returns condition (i1 value) to break out of atomic update loop.
    llvm::Value* create_atomic_loop(llvm::Value* ptrs_arr, llvm::Value* rhs, ast::BinaryOp op);
};

}  // namespace codegen
}  // namespace nmodl
