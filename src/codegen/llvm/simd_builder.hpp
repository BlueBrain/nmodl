/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "codegen/llvm/base_builder.hpp"
#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"

namespace nmodl {
namespace codegen {

/**
 * \class SIMDBuilder
 * \brief A class to generate LLVM IR for NMODL AST targeting SIMD platforms.
 */
class SIMDBuilder: public BaseBuilder {
  private:
    /// Vectorization width.
    int vector_width;

    /// Flag to indicate that the generated IR should be vectorized. This flag
    /// is used to catch cases when non-SIMD code needs to be emitted.
    bool vectorize;

    /// Masked value used to predicate vector instructions.
    llvm::Value* mask;

  protected:
    SIMDBuilder(llvm::LLVMContext& context,
                bool single_precision,
                int vector_width,
                std::vector<std::string> fast_math_flags = {})
        : BaseBuilder(context, single_precision, fast_math_flags)
        , vector_width(vector_width) {}

    /*************************************************************************/
    /*                     Virtual generation methods                        */
    /*************************************************************************/

  public:
    /// Generates LLVM IR to handle atomic updates, e.g. *ptr += rhs.
    void generate_atomic_statement(llvm::Value* lhs, llvm::Value* rhs, ast::BinaryOp op) override;

    /// Generates LLVM IR for boolean constants.
    void generate_boolean_constant(int value) override;

    /// Generates LLVM IR for integer constants.
    void generate_i32_constant(int value) override;

    /// Generates LLVM IR for floating-point constants.
    void generate_fp_constant(const std::string& value) override;

    /// Generates LLVM IR for directed data read.
    void generate_load_direct(llvm::Value* ptr) override;

    /// Generates LLVM IR for indirected data read.
    void generate_load_indirect(llvm::Value* ptr) override;

    /// Generates LLVM IR for directed data write.
    void generate_store_direct(llvm::Value* ptr, llvm::Value* value) override;

    /// Generates LLVM IR for indirected data write.
    void generate_store_indirect(llvm::Value* ptr, llvm::Value* value) override;

    /// Generates LLVM IR for brodcasting a value, if necessary.
    void try_generate_broadcast(llvm::Value* ptr) override;

    /// Generates LLVM IR for loop increment.
    void generate_loop_increment() override;

    /// Generates LLVM IR for loop termination.
    /// TODO: should take an AST node!
    void generate_loop_end() override;

    /*************************************************************************/
    /*                               Helpers                                 */
    /*************************************************************************/

  private:
    /// Inverts the mask by xoring it.
    void invert_mask();

    /// Creates a vector of (replicated) starting addresses of the data.
    llvm::Value* replicate_data_ptr(llvm::Value* data_ptr);

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
