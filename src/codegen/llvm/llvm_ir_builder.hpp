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
static constexpr const unsigned single_precision = 32;
static constexpr const unsigned double_precision = 64;

/// Some typedefs.
using ConstantVector = std::vector<llvm::Constant*>;
using MetadataVector = std::vector<llvm::Metadata*>;
using TypeVector = std::vector<llvm::Type*>;
using ValueVector = std::vector<llvm::Value*>;

/**
 * \class IRBuilder
 * \brief A helper class to generate LLVM IR for NMODL AST.
 */
class IRBuilder {
  private:
    /// Underlying LLVM IR builder.
    llvm::IRBuilder<> builder;

    /// Stack to hold visited and processed values.
    ValueVector value_stack;

    /// Pointer to the current function for which the code is generated.
    llvm::Function* current_function;

    /// Symbol table of the NMODL AST.
    symtab::SymbolTable* symbol_table;

    /// Insertion point for `alloca` instructions.
    llvm::Instruction* alloca_ip;

    /// Flag to indicate that the generated IR should be vectorized.
    bool vectorize;

    /// Precision of the floating-point numbers (32 or 64 bit).
    unsigned fp_precision;

    /// The vector width used for the vectorized code.
    unsigned vector_width;

    /// Instance struct fields do not alias.
    bool assume_noalias;

    /// Masked value used to predicate vector instructions.
    llvm::Value* mask;

    /// The name of induction variable used in kernel loops.
    std::string kernel_id;

    /// Fast math flags for floating-point IR instructions.
    std::vector<std::string> fast_math_flags;

  public:
    IRBuilder(llvm::LLVMContext& context,
              bool use_single_precision = false,
              unsigned vector_width = 1,
              std::vector<std::string> fast_math_flags = {},
              bool assume_noalias = true)
        : builder(context)
        , symbol_table(nullptr)
        , current_function(nullptr)
        , vectorize(false)
        , alloca_ip(nullptr)
        , fp_precision(use_single_precision ? single_precision : double_precision)
        , vector_width(vector_width)
        , mask(nullptr)
        , kernel_id("")
        , fast_math_flags(fast_math_flags)
        , assume_noalias(assume_noalias) {}

    /// Transforms the fast math flags provided to the builder into LLVM's representation.
    llvm::FastMathFlags transform_to_fmf(std::vector<std::string>& flags) {
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

    /// Initializes the builder with the symbol table and the kernel induction variable id.
    void initialize(symtab::SymbolTable& symbol_table, std::string& kernel_id) {
        if (!fast_math_flags.empty())
            builder.setFastMathFlags(transform_to_fmf(fast_math_flags));
        this->symbol_table = &symbol_table;
        this->kernel_id = kernel_id;
    }

    /// Explicitly sets the builder to produce scalar IR.
    void generate_scalar_ir() {
        vectorize = false;
    }

    /// Indicates whether the builder generates vectorized IR.
    bool vectorizing() {
        return vectorize;
    }

    /// Explicitly sets the builder to produce vectorized IR.
    void generate_vector_ir() {
        vectorize = true;
    }

    /// Sets the current function for which LLVM IR is generated.
    void set_function(llvm::Function* function) {
        current_function = function;
    }

    /// Clears the stack of the values and unsets the current function.
    void clear_function() {
        value_stack.clear();
        current_function = nullptr;
        alloca_ip = nullptr;
    }

    /// Sets the value to be the mask for vector code generation.
    void set_mask(llvm::Value* value) {
        mask = value;
    }

    /// Clears the mask for vector code generation.
    void clear_mask() {
        mask = nullptr;
    }

    /// Indicates whether the vectorized IR is predicated.
    bool generates_predicated_ir() {
        return vectorize && mask;
    }

    /// Generates LLVM IR to allocate the arguments of the function on the stack.
    void allocate_function_arguments(llvm::Function* function,
                                     const ast::CodegenVarWithTypeVector& nmodl_arguments);

    llvm::Value* create_alloca(const std::string& name, llvm::Type* type);

    /// Generates IR for allocating an array.
    void create_array_alloca(const std::string& name, llvm::Type* element_type, int num_elements);

    /// Generates LLVM IR for the given binary operator.
    void create_binary_op(llvm::Value* lhs, llvm::Value* rhs, ast::BinaryOp op);

    /// Generates LLVM IR for the bitcast instruction.
    llvm::Value* create_bitcast(llvm::Value* value, llvm::Type* dst_type);

    /// Create a basic block and set the builder's insertion point to it.
    llvm::BasicBlock* create_block_and_set_insertion_point(
        llvm::Function* function,
        llvm::BasicBlock* insert_before = nullptr,
        std::string name = "");

    /// Generates LLVM IR for unconditional branch.
    void create_br(llvm::BasicBlock* block);

    /// Generates LLVM IR for unconditional branch and sets the insertion point to this block.
    void create_br_and_set_insertion_point(llvm::BasicBlock* block);

    /// Generates LLVM IR for conditional branch.
    llvm::BranchInst* create_cond_br(llvm::Value* condition,
                                     llvm::BasicBlock* true_block,
                                     llvm::BasicBlock* false_block);

    /// Generates LLVM IR for the boolean constant.
    void create_boolean_constant(int value);

    /// Generates LLVM IR for the floating-point constant.
    void create_fp_constant(const std::string& value);

    /// Generates LLVM IR for a call to the function.
    void create_function_call(llvm::Function* callee,
                              ValueVector& arguments,
                              bool use_result = true);

    /// Generates LLVM IR for the string value.
    llvm::Value* create_global_string(const ast::String& node);

    /// Generates LLVM IR to transform the value into an index by possibly sign-extending it.
    llvm::Value* create_index(llvm::Value* value);

    /// Generates an intrinsic that corresponds to the given name.
    void create_intrinsic(const std::string& name,
                          ValueVector& argument_values,
                          TypeVector& argument_types);

    /// Generates LLVM IR for the integer constant.
    void create_i32_constant(int value);

    /// Generates LLVM IR to load the value specified by its name and returns it.
    llvm::Value* create_load(const std::string& name, bool masked = false);

    /// Generates LLVM IR to load the value from the pointer and returns it.
    llvm::Value* create_load(llvm::Value* ptr, bool masked = false);

    /// Generates LLVM IR to load the element at the specified index from the given array name and
    /// returns it.
    llvm::Value* create_load_from_array(const std::string& name, llvm::Value* index);

    /// Generates LLVM IR to store the value to the location specified by the name.
    void create_store(const std::string& name, llvm::Value* value, bool masked = false);

    /// Generates LLVM IR to store the value to the location specified by the pointer.
    void create_store(llvm::Value* ptr, llvm::Value* value, bool masked = false);

    /// Generates LLVM IR to store the value to the array element, where array is specified by the
    /// name.
    void create_store_to_array(const std::string& name, llvm::Value* index, llvm::Value* value);

    /// Generates LLVM IR return instructions.
    void create_return(llvm::Value* return_value = nullptr);

    /// Generates IR for allocating a scalar or vector variable.
    void create_scalar_or_vector_alloca(const std::string& name,
                                        llvm::Type* element_or_scalar_type);

    /// Generates LLVM IR for the given unary operator.
    void create_unary_op(llvm::Value* value, ast::UnaryOp op);

    /// Creates a boolean (1-bit integer) type.
    llvm::Type* get_boolean_type();

    /// Returns current basic block.
    llvm::BasicBlock* get_current_block();

    /// Returns the name of the function for which LLVM IR is generated.
    std::string get_current_function_name();

    /// Creates a pointer to 8-bit integer type.
    llvm::Type* get_i8_ptr_type();

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

    /// Generates LLVM IR to get the address of the struct's member at given index. Returns the
    /// calculated value.
    llvm::Value* get_struct_member_ptr(llvm::Value* struct_variable, int member_index);

    /// Creates a pointer to struct type with the given name and given members.
    llvm::Type* get_struct_ptr_type(const std::string& struct_type_name, TypeVector& member_types);

    /// Inverts the mask for vector code generation by xoring it.
    void invert_mask();

    /// Generates IR that loads the elements of the array even during vectorization. If the value is
    /// specified, then it is stored to the array at the given index.
    llvm::Value* load_to_or_store_from_array(const std::string& id_name,
                                             llvm::Value* id_value,
                                             llvm::Value* array,
                                             llvm::Value* maybe_value_to_store = nullptr);

    /// Lookups the value by  its name in the current function's symbol table.
    llvm::Value* lookup_value(const std::string& value_name);

    /// Generates IR to replicate the value if vectorizing the code.
    void maybe_replicate_value(llvm::Value* value);

    /// Sets builder's insertion point to the given block.
    void set_insertion_point(llvm::BasicBlock* block);

    /// Sets the necessary attributes for the kernel and its arguments.
    void set_kernel_attributes();

    /// Sets the loop metadata for the given branch from the loop.
    void set_loop_metadata(llvm::BranchInst* branch);

    /// Pops the last visited value from the value stack.
    llvm::Value* pop_last_value();

  private:
    /// Generates an inbounds GEP instruction for the given name and returns calculated address.
    llvm::Value* create_inbounds_gep(const std::string& variable_name, llvm::Value* index);

    /// Generates an inbounds GEP instruction for the given value and returns calculated address.
    llvm::Value* create_inbounds_gep(llvm::Value* variable, llvm::Value* index);

    /// Returns a scalar constant of the provided type.
    template <typename C, typename V>
    llvm::Value* get_scalar_constant(llvm::Type* type, V value);

    /// Returns a vector constant of the provided type.
    template <typename C, typename V>
    llvm::Value* get_vector_constant(llvm::Type* type, V value);
};

}  // namespace codegen
}  // namespace nmodl
