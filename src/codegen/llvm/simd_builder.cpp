/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/simd_builder.hpp"
#include "ast/all.hpp"

namespace nmodl {
namespace codegen {

/****************************************************************************************/
/*                           Generation virtual functions                               */
/****************************************************************************************/

void SIMDBuilder::generate_atomic_statement(llvm::Value* ptr, llvm::Value* rhs, ast::BinaryOp op) {
    // To handle atomic update over vectors, we will a scalar loop. The overall
    // structure will be:
    //
    //  +---------------------------+
    //  | <for body code>           |
    //  | <some initialisation>     |
    //  | br %atomic                |
    //  +---------------------------+
    //                |
    //                V
    //  +-----------------------------+
    //  | <atomic update code>        |
    //  | %cmp = ...                  |<------+
    //  | cond_br %cmp, %atomic, %rem |       |
    //  +-----------------------------+       |
    //      |                 |               |
    //      |                 +---------------+
    //      V
    //  +---------------------------+
    //  | <for body remaining code> |
    //  |                           |
    //  +---------------------------+

    // Step 1: Create a vector of (replicated) starting addresses of the given member.
    llvm::Value* start = replicate_data_ptr(ptr);

    // Step 2: Create a vector alloca that will store addresses of data values. Also,
    // create an array of these addresses (as pointers).
    llvm::Type* vi64_type = llvm::FixedVectorType::get(get_i64_type(), vector_width);
    llvm::Type* array_type = llvm::ArrayType::get(get_fp_ptr_type(), vector_width);

    llvm::Value* ptrs_vec = create_alloca(/*name=*/"ptrs", vi64_type);
    llvm::Value* ptrs_arr = builder.CreateBitCast(ptrs_vec,  llvm::PointerType::get(array_type, /*AddressSpace=*/0));

    // Step 3: Calculate offsets of the values in the member by:
    //     offset = start + (index * sizeof(fp_type))
    // Store this vector to a temporary for later reuse.
    llvm::Value* offsets; // = ir_builder.create_member_offsets(start, i64_index);
    generate_store_direct(ptrs_vec, offsets);

    // Step 4: Create a new block that  will be used for atomic code generation.
    llvm::BasicBlock* body_bb = get_current_block();
    llvm::BasicBlock* cond_bb = body_bb->getNextNode();
    llvm::Function* func = body_bb->getParent();
    llvm::BasicBlock* atomic_bb =
        llvm::BasicBlock::Create(*builder.getContext(), /*Name=*/"atomic.update", func, cond_bb);
    llvm::BasicBlock* remaining_body_bb =
        llvm::BasicBlock::Create(*context, /*Name=*/"for.body.remaining", func, cond_bb);
    create_br_and_set_insertion_point(atomic_bb);

    // Step 5: Generate code for the atomic update: go through each element in the vector
    // performing the computation.
    llvm::Value* cmp = create_atomic_loop(ptrs_arr, rhs, op);

    // Create branch to close the loop and restore the insertion point.
    create_cond_br(cmp, remaining_body_bb, atomic_bb);
    set_insertion_point(remaining_body_bb);
}

void SIMDBuilder::generate_boolean_constant(int value) {
    if (vectorize) {
        value_stack.push_back(vector_constant<llvm::ConstantInt>(get_boolean_type(), value, vector_width));
    } else {
        value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_boolean_type(), value));
    }
}

void SIMDBuilder::generate_i32_constant(int value) {
    if (vectorize) {
        value_stack.push_back(vector_constant<llvm::ConstantInt>(get_i32_type(), value, vector_width));
    } else {
        value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_i32_type(), value));
    }
}

void SIMDBuilder::generate_fp_constant(const std::string& value) {
    if (vectorize) {
        value_stack.push_back(vector_constant<llvm::ConstantFP>(get_fp_type(), value, vector_width));
    } else {
        value_stack.push_back(scalar_constant<llvm::ConstantFP>(get_fp_type(), value));
    }
}

void SIMDBuilder::generate_load_direct(llvm::Value* ptr) {
    llvm::Type* loaded_type = ptr->getType()->getPointerElementType();

    // Check if the generated IR is vectorized and masked.
    if (vectorize && mask) {
        builder.CreateMaskedLoad(loaded_type, ptr, llvm::Align(), mask);
    }

    llvm::Value* loaded = builder.CreateLoad(loaded_type, ptr);
    value_stack.push_back(loaded);
}

void SIMDBuilder::generate_load_indirect(llvm::Value* ptr) {
    // Construct the loaded vector type.
    auto* ptrs = llvm::cast<llvm::VectorType>(ptr->getType());
    llvm::ElementCount element_count = ptrs->getElementCount();
    llvm::Type* element_type = ptrs->getElementType()->getPointerElementType();
    llvm::Type* loaded_type = llvm::VectorType::get(element_type, element_count);

    builder.CreateMaskedGather(loaded_type, ptr, llvm::Align(), mask);
}

void SIMDBuilder::generate_store_direct(llvm::Value* ptr, llvm::Value* value) {
    // Check if the generated IR is vectorized and masked.
    if (vectorize && mask) {
        builder.CreateMaskedStore(value, ptr, llvm::Align(), mask);
        return;
    }
    builder.CreateStore(value, ptr);
}

void SIMDBuilder::generate_store_indirect(llvm::Value* ptr, llvm::Value* value) {
    builder.CreateMaskedScatter(value, ptr, llvm::Align(), mask);
}

void SIMDBuilder::try_generate_broadcast(llvm::Value* value) {
    // If the value should not be vectorised, or it is already a vector, add it to the stack.
    if (!vectorize || value->getType()->isVectorTy()) {
        value_stack.push_back(value);
    } else {
        // Otherwise, we generate vectorized code, so replicate the value to form a vector.
        llvm::Value* vector_value = builder.CreateVectorSplat(vector_width, value);
        value_stack.push_back(vector_value);
    }
}

void SIMDBuilder::generate_loop_increment() {
    generate_i32_constant(vector_width);
}

void SIMDBuilder::generate_loop_end() {
    // TODO: fix this when an AST node is added.
}

/****************************************************************************************/
/*                                     Helpers                                          */
/****************************************************************************************/

void SIMDBuilder::invert_mask() {
    if (!mask)
        throw std::runtime_error("Error: mask is not set\n");

    // Create the vector with all `true` values.
    generate_boolean_constant(1);
    llvm::Value* one = pop_last_value();

    mask = builder.CreateXor(mask, one);
}

llvm::Value* SIMDBuilder::replicate_data_ptr(llvm::Value* data_ptr) {
    llvm::Module* m = builder.GetInsertBlock()->getParent()->getParent();

    // Treat this data address as integer values.
    llvm::Type* int_ptr_type = m->getDataLayout().getIntPtrType(builder.getContext());
    llvm::Value* ptr_to_int = builder.CreatePtrToInt(data_ptr, int_ptr_type);

    // Create a vector that has an address at 0.
    llvm::Type* vector_type = llvm::FixedVectorType::get(int_ptr_type, vector_width);
    llvm::Value* zero = scalar_constant<llvm::ConstantInt>(get_i32_type(), 0);
    llvm::Value* tmp =
        builder.CreateInsertElement(llvm::UndefValue::get(vector_type), ptr_to_int, zero);

    // Finally, use `shufflevector` with zeroinitializer to replicate the 0th element.
    llvm::Value* select = llvm::Constant::getNullValue(vector_type);
    return builder.CreateShuffleVector(tmp, llvm::UndefValue::get(vector_type), select);
}

llvm::Value* SIMDBuilder::create_member_offsets(llvm::Value* start, llvm::Value* indices) {
    llvm::Value* factor = get_vector_constant<llvm::ConstantInt>(get_i64_type(),
                                                                 platform.get_precision() / 8);
    llvm::Value* offset = builder.CreateMul(indices, factor);
    return builder.CreateAdd(start, offset);
}

llvm::Value* SIMDBuilder::create_atomic_loop(llvm::Value* ptrs_arr,
                                           llvm::Value* rhs,
                                           ast::BinaryOp op) {
    const int vector_width = platform.get_instruction_width();
    llvm::BasicBlock* curr = get_current_block();
    llvm::BasicBlock* prev = curr->getPrevNode();
    llvm::BasicBlock* next = curr->getNextNode();

    // Some constant values.
    llvm::Value* false_value = get_scalar_constant<llvm::ConstantInt>(get_boolean_type(), 0);
    llvm::Value* zero = get_scalar_constant<llvm::ConstantInt>(get_i64_type(), 0);
    llvm::Value* one = get_scalar_constant<llvm::ConstantInt>(get_i64_type(), 1);
    llvm::Value* minus_one = get_scalar_constant<llvm::ConstantInt>(get_i64_type(), -1);

    // First, we create a PHI node that holds the mask of active vector elements.
    llvm::PHINode* mask = builder.CreatePHI(get_i64_type(), /*NumReservedValues=*/2);

    // Intially, all elements are active.
    llvm::Value* init_value = get_scalar_constant<llvm::ConstantInt>(get_i64_type(),
                                                                     ~((~0) << vector_width));

    // Find the index of the next active element and update the mask. This can be easily computed
    // with:
    //     index    = cttz(mask)
    //     new_mask = mask & ((1 << index) ^ -1)
    llvm::Value* index =
        builder.CreateIntrinsic(llvm::Intrinsic::cttz, {get_i64_type()}, {mask, false_value});
    llvm::Value* new_mask = builder.CreateShl(one, index);
    new_mask = builder.CreateXor(new_mask, minus_one);
    new_mask = builder.CreateAnd(mask, new_mask);

    // Update PHI with appropriate values.
    mask->addIncoming(init_value, prev);
    mask->addIncoming(new_mask, curr);

    // Get the pointer to the current value, the value itself and the update.b
    llvm::Value* gep =
        builder.CreateGEP(ptrs_arr->getType()->getPointerElementType(), ptrs_arr, {zero, index});
    llvm::Value* ptr = create_load(gep);
    llvm::Value* source = create_load(ptr);
    llvm::Value* update = builder.CreateExtractElement(rhs, index);

    // Perform the update and store the result back.
    //     source = *ptr
    //     *ptr = source + update
    create_binary_op(source, update, op);
    llvm::Value* result = pop_last_value();
    create_store(ptr, result);

    // Return condition to break out of atomic update loop.
    return builder.CreateICmpEQ(new_mask, zero);
}

}  // namespace codegen
}  // namespace nmodl
