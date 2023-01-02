/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/llvm_ir_builder.hpp"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ValueSymbolTable.h"

#include "llvm/IR/IntrinsicsNVPTX.h"

namespace nmodl {
namespace codegen {

/****************************************************************************************/
/*                         Value processing utilities                                   */
/****************************************************************************************/

llvm::Value* BaseBuilder::lookup_value(const std::string& value_name) {
    auto value = current_function->getValueSymbolTable()->lookup(value_name);
    if (!value)
        throw std::runtime_error("Error: variable " + value_name + " is not in the scope\n");
    return value;
}

llvm::Value* BaseBuilder::pop_last_value() {
    // Check if the stack is empty.
    if (value_stack.empty())
        throw std::runtime_error("Error: popping a value from the empty stack\n");

    // Return the last added value and delete it from the stack.
    llvm::Value* last = value_stack.back();
    value_stack.pop_back();
    return last;
}

/****************************************************************************************/
/*                               Type utilities                                         */
/****************************************************************************************/

llvm::Type* BaseBuilder::get_boolean_type() {
    return llvm::Type::getInt1Ty(builder.getContext());
}

llvm::Type* BaseBuilder::get_i32_type() {
    return llvm::Type::getInt32Ty(builder.getContext());
}

llvm::Type* BaseBuilder::get_i64_type() {
    return llvm::Type::getInt64Ty(builder.getContext());
}

llvm::Type* BaseBuilder::get_fp_type() {
    if (platform.is_single_precision())
        return llvm::Type::getFloatTy(builder.getContext());
    return llvm::Type::getDoubleTy(builder.getContext());
}

llvm::Type* BaseBuilder::get_void_type() {
    return llvm::Type::getVoidTy(builder.getContext());
}

llvm::Type* BaseBuilder::get_i8_ptr_type() {
    return llvm::Type::getInt8PtrTy(builder.getContext());
}

llvm::Type* BaseBuilder::get_i32_ptr_type() {
    return llvm::Type::getInt32PtrTy(builder.getContext());
}

llvm::Type* BaseBuilder::get_fp_ptr_type() {
    if (platform.is_single_precision())
        return llvm::Type::getFloatPtrTy(builder.getContext());
    return llvm::Type::getDoublePtrTy(builder.getContext());
}

llvm::Type* BaseBuilder::get_struct_ptr_type(const std::string& struct_type_name,
                                             TypeVector& member_types) {
    llvm::StructType* llvm_struct_type = llvm::StructType::getTypeByName(builder.getContext(),
                                                                         struct_type_name);

    if (!llvm_struct_type) {
        llvm_struct_type = llvm::StructType::create(builder.getContext(), struct_type_name);
        llvm_struct_type->setBody(member_types);
    }

    return llvm::PointerType::get(llvm_struct_type, /*AddressSpace=*/0);
}

/****************************************************************************************/
/*                                  Function utilities                                  */
/****************************************************************************************/

std::string BaseBuilder::get_current_function_name() {
    return current_function->getName().str();
}

void BaseBuilder::set_function(llvm::Function* function) {
    current_function = function;
}

void BaseBuilder::unset_function() {
    value_stack.clear();
    current_function = nullptr;
    alloca_ip = nullptr;
}

void BaseBuilder::generate_function_call(llvm::Function* callee,
                                         ValueVector& arguments,
                                         bool with_result) {
    llvm::Value* call_instruction = builder.CreateCall(callee, arguments);
    if (with_result)
        value_stack.push_back(call_instruction);
}

void BaseBuilder::generate_intrinsic(const std::string& name,
                                     ValueVector& argument_values,
                                     TypeVector& argument_types) {
    // Process 'pow' call separately.
    if (name == "pow") {
        llvm::Value* pow_intrinsic = builder.CreateIntrinsic(llvm::Intrinsic::pow,
                                                             {argument_types.front()},
                                                             argument_values);
        value_stack.push_back(pow_intrinsic);
        return;
    }

    // Create other intrinsics.
    unsigned intrinsic_id = llvm::StringSwitch<llvm::Intrinsic::ID>(name)
                                .Case("ceil", llvm::Intrinsic::ceil)
                                .Case("cos", llvm::Intrinsic::cos)
                                .Case("exp", llvm::Intrinsic::exp)
                                .Case("fabs", llvm::Intrinsic::fabs)
                                .Case("floor", llvm::Intrinsic::floor)
                                .Case("log", llvm::Intrinsic::log)
                                .Case("log10", llvm::Intrinsic::log10)
                                .Case("sin", llvm::Intrinsic::sin)
                                .Case("sqrt", llvm::Intrinsic::sqrt)
                                .Default(llvm::Intrinsic::not_intrinsic);
    if (intrinsic_id) {
        llvm::Value* intrinsic =
            builder.CreateIntrinsic(intrinsic_id, argument_types, argument_values);
        value_stack.push_back(intrinsic);
    } else {
        throw std::runtime_error("Error: calls to " + name + " are not valid or not supported\n");
    }
}

void BaseBuilder::allocate_function_arguments(llvm::Function* function,
                                              const ast::CodegenVarWithTypeVector& nmodl_arguments) {
    unsigned i = 0;
    for (auto& arg: function->args()) {
        std::string arg_name = nmodl_arguments[i++].get()->get_node_name();
        llvm::Type* arg_type = arg.getType();
        llvm::Value* alloca = create_alloca(arg_name, arg_type);
        arg.setName(arg_name);
        builder.CreateStore(&arg, alloca);
    }
}

void BaseBuilder::allocate_and_wrap_kernel_arguments(llvm::Function* function,
                                                     const ast::CodegenVarWithTypeVector& nmodl_arguments,
                                                     llvm::Type* struct_type) {
    // In theory, this should never happen but let's guard anyway.
    if (nmodl_arguments.size() != 1) {
        throw std::runtime_error("Error: NMODL computer kernel must have a single argument\n");
    }

    // Bitcast void* pointer provided as compute kernel argument to mechanism data type.
    llvm::Value* data_ptr = builder.CreateBitCast(function->getArg(0), struct_type);

    std::string arg_name = nmodl_arguments[0].get()->get_node_name();
    llvm::Value* alloca = create_alloca(arg_name, struct_type);
    builder.CreateStore(data_ptr, alloca);
}

/****************************************************************************************/
/*                                 Basic block utilities                                */
/****************************************************************************************/

llvm::BasicBlock* BaseBuilder::get_current_block() {
    return builder.GetInsertBlock();
}

void BaseBuilder::set_insertion_point(llvm::BasicBlock* block) {
    builder.SetInsertPoint(block);
}

llvm::BasicBlock* BaseBuilder::create_block_and_set_insertion_point(llvm::Function* function,
                                                                    llvm::BasicBlock* insert_before,
                                                                    std::string name) {
    llvm::BasicBlock* block =
        llvm::BasicBlock::Create(builder.getContext(), name, function, insert_before);
    builder.SetInsertPoint(block);
    return block;
}

llvm::BranchInst* BaseBuilder::create_cond_br(llvm::Value* condition,
                                              llvm::BasicBlock* true_block,
                                              llvm::BasicBlock* false_block) {
    return builder.CreateCondBr(condition, true_block, false_block);
}

void BaseBuilder::generate_br(llvm::BasicBlock* block) {
    builder.CreateBr(block);
}

void BaseBuilder::generate_br_and_set_insertion_point(llvm::BasicBlock* block) {
    builder.CreateBr(block);
    builder.SetInsertPoint(block);
}

/****************************************************************************************/
/*                                Instruction utilities                                 */
/****************************************************************************************/

llvm::Value* BaseBuilder::create_alloca(const std::string& name, llvm::Type* type) {
    // If insertion point for `alloca` instructions is not set, then create the instruction in the
    // entry block and set it to be the insertion point.
    if (!alloca_ip) {
        // Get the entry block and insert the `alloca` instruction there.
        llvm::BasicBlock* current_block = builder.GetInsertBlock();
        llvm::BasicBlock& entry_block = current_block->getParent()->getEntryBlock();
        builder.SetInsertPoint(&entry_block);
        llvm::Value* alloca = builder.CreateAlloca(type, /*ArraySize=*/nullptr, name);

        // Set the `alloca` instruction insertion point and restore the insertion point for the next
        // set of instructions.
        alloca_ip = llvm::cast<llvm::AllocaInst>(alloca);
        builder.SetInsertPoint(current_block);
        return alloca;
    }

    // Create `alloca` instruction.
    llvm::BasicBlock* alloca_block = alloca_ip->getParent();
    const auto& data_layout = alloca_block->getModule()->getDataLayout();
    auto* alloca = new llvm::AllocaInst(type,
                                        data_layout.getAllocaAddrSpace(),
                                        /*ArraySize=*/nullptr,
                                        data_layout.getPrefTypeAlign(type),
                                        name);

    // Insert `alloca` at the specified insertion point and reset it for the next instructions.
    alloca_block->getInstList().insertAfter(alloca_ip->getIterator(), alloca);
    alloca_ip = alloca;
    return alloca;
}

llvm::Value* BaseBuilder::create_array_alloca(const std::string& name,
                                      llvm::Type* element_type,
                                      int num_elements) {
    llvm::Type* array_type = llvm::ArrayType::get(element_type, num_elements);
    return create_alloca(name, array_type);
}

llvm::Value* BaseBuilder::create_atomic_op(llvm::Value* ptr, llvm::Value* update, ast::BinaryOp op) {
    if (op == ast::BinaryOp::BOP_SUBTRACTION) {
        update = builder.CreateFNeg(update);
    }
    return builder.CreateAtomicRMW(llvm::AtomicRMWInst::FAdd,
                                   ptr,
                                   update,
                                   llvm::MaybeAlign(),
                                   llvm::AtomicOrdering::SequentiallyConsistent);
}

llvm::Value* BaseBuilder::create_bitcast(llvm::Value* value, llvm::Type* type) {
    return builder.CreateBitCast(value, type);
}

llvm::Value* BaseBuilder::create_global_string(const ast::String& node) {
    return builder.CreateGlobalStringPtr(node.get_value());
}

llvm::Value* BaseBuilder::create_inbounds_gep(const std::string& var_name, llvm::Value* index) {
    llvm::Value* variable_ptr = lookup_value(var_name);

    // Since we index through the pointer, we need an extra 0 index in the indices list for GEP.
    ValueVector indices{llvm::ConstantInt::get(get_i64_type(), 0), index};
    llvm::Type* variable_type = variable_ptr->getType()->getPointerElementType();
    return builder.CreateInBoundsGEP(variable_type, variable_ptr, indices);
}

llvm::Value* BaseBuilder::create_inbounds_gep(llvm::Value* variable, llvm::Value* index) {
    ValueVector indices{index};
    llvm::Type* variable_type = variable->getType()->getPointerElementType();
    return builder.CreateInBoundsGEP(variable_type, variable, indices);
}

llvm::Value* BaseBuilder::create_load_direct(const std::string& name, bool masked) {
    llvm::Value* ptr = lookup_value(name);
    llvm::Type* loaded_type = ptr->getType()->getPointerElementType();

    // Check if the generated IR is vectorized and masked.
    llvm::Value* loaded;
    if (masked) {
        loaded = builder.CreateMaskedLoad(loaded_type, ptr, llvm::Align(), mask);
    } else {
        loaded = builder.CreateLoad(loaded_type, ptr);
    }
    value_stack.push_back(loaded);
    return loaded;
}

llvm::Value* BaseBuilder::create_load_direct(llvm::Value* ptr, bool masked) {
    llvm::Type* loaded_type = ptr->getType()->getPointerElementType();

    // Check if the generated IR is vectorized and masked.
    llvm::Value* loaded;
    if (masked) {
        loaded = builder.CreateMaskedLoad(loaded_type, ptr, llvm::Align(), mask);
    } else {
        loaded = builder.CreateLoad(loaded_type, ptr);
    }
    value_stack.push_back(loaded);
    return loaded;
}

llvm::Value* BaseBuilder::create_struct_field_ptr(llvm::Value* struct_variable, int offset) {
    ValueVector indices;
    indices.push_back(llvm::ConstantInt::get(get_i32_type(), 0));
    indices.push_back(llvm::ConstantInt::get(get_i32_type(), offset));

    llvm::Type* type = struct_variable->getType()->getPointerElementType();
    return builder.CreateInBoundsGEP(type, struct_variable, indices);
}

ast::BinaryOp BaseBuilder::into_atomic_op(ast::BinaryOp op) {
    switch (op) {
    case ast::BinaryOp::BOP_SUB_ASSIGN:
        return ast::BinaryOp::BOP_SUBTRACTION;
    case ast::BinaryOp::BOP_ADD_ASSIGN:
        return ast::BinaryOp::BOP_ADDITION;
    default:
        throw std::runtime_error("Error: only atomic addition and subtraction is supported\n");
    }
}

void BaseBuilder::create_store_direct(const std::string& name, llvm::Value* value, bool masked) {
    llvm::Value* ptr = lookup_value(name);

    // Check if the generated IR is vectorized and masked.
    if (masked) {
        builder.CreateMaskedStore(value, ptr, llvm::Align(), mask);
        return;
    }
    builder.CreateStore(value, ptr);
}

void BaseBuilder::create_store_direct(llvm::Value* ptr, llvm::Value* value, bool masked) {
    // Check if the generated IR is vectorized and masked.
    if (masked) {
        builder.CreateMaskedStore(value, ptr, llvm::Align(), mask);
        return;
    }
    builder.CreateStore(value, ptr);
}

llvm::Value* BaseBuilder::into_index(llvm::Value* value) {
    // First, check if the passed value is an integer.
    llvm::Type* value_type = value->getType();
    if (!value_type->isIntOrIntVectorTy())
        throw std::runtime_error("Error: only integer indexing is supported\n");

    // Conventionally, in LLVM array indices are 64 bit.
    llvm::Type* index_type = get_i64_type();

    // If value is a scalar.
    if (auto integer_type = llvm::dyn_cast<llvm::IntegerType>(value_type)) {
        if (integer_type->getBitWidth() == index_type->getIntegerBitWidth())
            return value;
        return builder.CreateSExtOrTrunc(value, index_type);
    }

    // If value is a vector.
    const auto& vector_type = llvm::cast<llvm::FixedVectorType>(value_type);
    const auto& element_type = llvm::cast<llvm::IntegerType>(vector_type->getElementType());
    if (element_type->getBitWidth() == index_type->getIntegerBitWidth())
        return value;
    unsigned num_elements = vector_type->getNumElements();
    return builder.CreateSExtOrTrunc(value, llvm::FixedVectorType::get(index_type, num_elements));
}

/****************************************************************************************/
/*                                 Constant utilities                                   */
/****************************************************************************************/

template <typename C, typename V>
llvm::Value* BaseBuilder::scalar_constant(llvm::Type* type, V value) {
    return C::get(type, value);
}

template <typename C, typename V>
llvm::Value* BaseBuilder::vector_constant(llvm::Type* type, V value, unsigned num_elements) {
    ConstantVector constants;
    for (unsigned i = 0; i < num_elements; ++i) {
        const auto& element = C::get(type, value);
        constants.push_back(element);
    }
    return llvm::ConstantVector::get(constants);
}

/****************************************************************************************/
/*                           Generation virtual functions                               */
/****************************************************************************************/

void BaseBuilder::generate_binary_op(llvm::Value* lhs, llvm::Value* rhs, ast::BinaryOp op) {
    // Check that both lhs and rhs have the same types.
    if (lhs->getType() != rhs->getType())
        throw std::runtime_error(
            "Error: lhs and rhs of the binary operator have different types\n");

    llvm::Value* result;
    switch (op) {
#define DISPATCH(binary_op, fp_instruction, integer_instruction) \
    case binary_op:                                              \
        if (lhs->getType()->isIntOrIntVectorTy())                \
            result = integer_instruction(lhs, rhs);              \
        else                                                     \
            result = fp_instruction(lhs, rhs);                   \
        break;

        // Arithmetic instructions.
        DISPATCH(ast::BinaryOp::BOP_ADDITION, builder.CreateFAdd, builder.CreateAdd);
        DISPATCH(ast::BinaryOp::BOP_DIVISION, builder.CreateFDiv, builder.CreateSDiv);
        DISPATCH(ast::BinaryOp::BOP_MULTIPLICATION, builder.CreateFMul, builder.CreateMul);
        DISPATCH(ast::BinaryOp::BOP_SUBTRACTION, builder.CreateFSub, builder.CreateSub);

        // Comparison instructions.
        DISPATCH(ast::BinaryOp::BOP_EXACT_EQUAL, builder.CreateFCmpOEQ, builder.CreateICmpEQ);
        DISPATCH(ast::BinaryOp::BOP_GREATER, builder.CreateFCmpOGT, builder.CreateICmpSGT);
        DISPATCH(ast::BinaryOp::BOP_GREATER_EQUAL, builder.CreateFCmpOGE, builder.CreateICmpSGE);
        DISPATCH(ast::BinaryOp::BOP_LESS, builder.CreateFCmpOLT, builder.CreateICmpSLT);
        DISPATCH(ast::BinaryOp::BOP_LESS_EQUAL, builder.CreateFCmpOLE, builder.CreateICmpSLE);
        DISPATCH(ast::BinaryOp::BOP_NOT_EQUAL, builder.CreateFCmpONE, builder.CreateICmpNE);

#undef DISPATCH

    // Separately replace ^ with the `pow` intrinsic.
    case ast::BinaryOp::BOP_POWER:
        result = builder.CreateIntrinsic(llvm::Intrinsic::pow, {lhs->getType()}, {lhs, rhs});
        break;

    // Logical instructions.
    case ast::BinaryOp::BOP_AND:
        result = builder.CreateAnd(lhs, rhs);
        break;
    case ast::BinaryOp::BOP_OR:
        result = builder.CreateOr(lhs, rhs);
        break;

    default:
        throw std::runtime_error("Error: unsupported binary operator\n");
    }
    value_stack.push_back(result);
}

void BaseBuilder::generate_unary_op(llvm::Value* value, ast::UnaryOp op) {
    if (op == ast::UOP_NEGATION) {
        value_stack.push_back(builder.CreateFNeg(value));
    } else if (op == ast::UOP_NOT) {
        value_stack.push_back(builder.CreateNot(value));
    } else {
        throw std::runtime_error("Error: unsupported unary operator\n");
    }
}

void BaseBuilder::generate_boolean_constant(int value) {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_boolean_type(), value));
    } else if (platform.is_cpu_with_simd()) {
        if (vectorize) {
            value_stack.push_back(vector_constant<llvm::ConstantInt>(get_boolean_type(), value, platform.get_instruction_width()));
        } else {
            value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_boolean_type(), value));
        }
    } else {
        value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_boolean_type(), value));
    }
}

void BaseBuilder::generate_i32_constant(int value) {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_i32_type(), value));
    } else if (platform.is_cpu_with_simd()) {
        if (vectorize) {
            value_stack.push_back(vector_constant<llvm::ConstantInt>(get_i32_type(), value, platform.get_instruction_width()));
        } else {
            value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_i32_type(), value));
        }
    } else {
        value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_i32_type(), value));
    }
}

void BaseBuilder::generate_fp_constant(const std::string& value) {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        value_stack.push_back(scalar_constant<llvm::ConstantFP>(get_fp_type(), value));
    } else if (platform.is_cpu_with_simd()) {
        if (vectorize) {
            value_stack.push_back(vector_constant<llvm::ConstantFP>(get_fp_type(), value, platform.get_instruction_width()));
        } else {
            value_stack.push_back(scalar_constant<llvm::ConstantFP>(get_fp_type(), value));
        }
    } else {
        value_stack.push_back(scalar_constant<llvm::ConstantFP>(get_fp_type(), value));
    }
}

void BaseBuilder::generate_return(llvm::Value* return_value) {
    if (return_value)
        builder.CreateRet(return_value);
    else
        builder.CreateRetVoid();
}

void BaseBuilder::generate_broadcast(llvm::Value* value) {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        value_stack.push_back(value);
    } else if (platform.is_cpu_with_simd()) {
        if (!vectorize || value->getType()->isVectorTy()) {
            value_stack.push_back(value);
        } else {
            // Otherwise, we generate vectorized code inside the loop, so replicate the value to form a
            // vector.
            int vector_width = platform.get_instruction_width();
            llvm::Value* vector_value = builder.CreateVectorSplat(vector_width, value);
            value_stack.push_back(vector_value);
        }
    } else {
        value_stack.push_back(value);
    }
}

/****************************************************************************************/
/*                             Helper virtual functions                                 */
/****************************************************************************************/

void BaseBuilder::set_mask(llvm::Value* value) {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        throw std::runtime_error("Error: cannot set mask in GPUBuilder\n");
    } else if (platform.is_cpu_with_simd()) {
        mask = value;
    } else {
        throw std::runtime_error("Error: cannot set mask in BaseBuilder\n");
    }
}

void BaseBuilder::unset_mask() {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        throw std::runtime_error("Error: cannot unset mask in GPUBuilder\n");
    } else if (platform.is_cpu_with_simd()) {
        mask = nullptr;
    } else {
        throw std::runtime_error("Error: cannot unset mask in BaseBuilder\n");
    }
}

void BaseBuilder::start_generating_ir_for_compute() {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        // Do nothing.
    } else if (platform.is_cpu_with_simd()) {
        vectorize = true;
    } else {
        // Do nothing.
    }
}

void BaseBuilder::stop_generating_ir_for_compute() {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        // Do nothing.
    } else if (platform.is_cpu_with_simd()) {
        vectorize = false;
    } else {
        // Do nothing.
    }
}

bool BaseBuilder::generating_vector_ir() {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        return false;
    } else if (platform.is_cpu_with_simd()) {
        return vectorize;
    } else {
        return false;
    }
}

bool BaseBuilder::generating_masked_vector_ir() {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        return false;
    } else if (platform.is_cpu_with_simd()) {
        return vectorize && mask;
    } else {
        return false;
    }
}

void BaseBuilder::invert_mask() {
    // TODO: split 3 cases into 3 builders.
    if (platform.is_gpu()) {
        throw std::runtime_error("Error: cannot invert mask in GPUBuilder\n");
    } else if (platform.is_cpu_with_simd()) {
        if (!mask)
            throw std::runtime_error("Error: mask is not set\n");

        // Create the vector with all `true` values.
        generate_boolean_constant(1);
        llvm::Value* one = pop_last_value();
        mask = builder.CreateXor(mask, one);
    } else {
        throw std::runtime_error("Error: cannot invert mask in BaseBuilder\n");
    }
}

llvm::Value* BaseBuilder::create_member_addresses(llvm::Value* member_ptr) {
    llvm::Module* m = builder.GetInsertBlock()->getParent()->getParent();

    // Treat this member address as integer value.
    llvm::Type* int_ptr_type = m->getDataLayout().getIntPtrType(builder.getContext());
    llvm::Value* ptr_to_int = builder.CreatePtrToInt(member_ptr, int_ptr_type);

    // Create a vector that has address at 0.
    llvm::Type* vector_type = llvm::FixedVectorType::get(int_ptr_type,
                                                         platform.get_instruction_width());
    llvm::Value* zero = scalar_constant<llvm::ConstantInt>(get_i32_type(), 0);
    llvm::Value* tmp =
        builder.CreateInsertElement(llvm::UndefValue::get(vector_type), ptr_to_int, zero);

    // Finally, use `shufflevector` with zeroinitializer to replicate the 0th element.
    llvm::Value* select = llvm::Constant::getNullValue(vector_type);
    return builder.CreateShuffleVector(tmp, llvm::UndefValue::get(vector_type), select);
}

llvm::Value* BaseBuilder::create_member_offsets(llvm::Value* start, llvm::Value* indices) {
    llvm::Value* factor = vector_constant<llvm::ConstantInt>(get_i64_type(),
                                                             platform.get_precision() / 8,
                                                             platform.get_instruction_width());
    llvm::Value* offset = builder.CreateMul(indices, factor);
    return builder.CreateAdd(start, offset);
}

llvm::Value* BaseBuilder::create_atomic_loop(llvm::Value* ptrs_arr,
                                           llvm::Value* rhs,
                                           ast::BinaryOp op) {
    const int vector_width = platform.get_instruction_width();
    llvm::BasicBlock* curr = get_current_block();
    llvm::BasicBlock* prev = curr->getPrevNode();
    llvm::BasicBlock* next = curr->getNextNode();

    // Some constant values.
    llvm::Value* false_value = scalar_constant<llvm::ConstantInt>(get_boolean_type(), 0);
    llvm::Value* zero = scalar_constant<llvm::ConstantInt>(get_i64_type(), 0);
    llvm::Value* one = scalar_constant<llvm::ConstantInt>(get_i64_type(), 1);
    llvm::Value* minus_one = scalar_constant<llvm::ConstantInt>(get_i64_type(), -1);

    // First, we create a PHI node that holds the mask of active vector elements.
    llvm::PHINode* mask = builder.CreatePHI(get_i64_type(), /*NumReservedValues=*/2);

    // Intially, all elements are active.
    llvm::Value* init_value = scalar_constant<llvm::ConstantInt>(get_i64_type(), ~((-1u) << vector_width));

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
    llvm::Value* ptr = create_load_direct(gep);
    llvm::Value* source = create_load_direct(ptr);
    llvm::Value* update = builder.CreateExtractElement(rhs, index);

    // Perform the update and store the result back.
    //     source = *ptr
    //     *ptr = source + update
    generate_binary_op(source, update, op);
    llvm::Value* result = pop_last_value();
    create_store_direct(ptr, result);

    // Return condition to break out of atomic update loop.
    return builder.CreateICmpEQ(new_mask, zero);
}

llvm::Value* BaseBuilder::load_to_or_store_from_array(const std::string& id_name,
                                                    llvm::Value* id_value,
                                                    llvm::Value* array,
                                                    llvm::Value* maybe_value_to_store) {
    // First, calculate the address of the element in the array.
    llvm::Value* element_ptr = create_inbounds_gep(array, id_value);

    // Find out if the vector code is generated.
    bool generating_vector_ir = platform.is_cpu_with_simd() && vectorize;

    // If the vector code is generated, we need to distinguish between two cases. If the array is
    // indexed indirectly (i.e. not by an induction variable `kernel_id`), create gather/scatter
    // instructions.
    if (id_name != naming::INDUCTION_VAR && generating_vector_ir) {
        if (maybe_value_to_store) {
            return builder.CreateMaskedScatter(maybe_value_to_store,
                                               element_ptr,
                                               llvm::Align(),
                                               mask);
        } else {
            // Construct the loaded vector type.
            auto* ptrs = llvm::cast<llvm::VectorType>(element_ptr->getType());
            llvm::ElementCount element_count = ptrs->getElementCount();
            llvm::Type* element_type = ptrs->getElementType()->getPointerElementType();
            llvm::Type* loaded_type = llvm::VectorType::get(element_type, element_count);

            return builder.CreateMaskedGather(loaded_type, element_ptr, llvm::Align(), mask);
        }
    }

    llvm::Value* ptr;
    if (generating_vector_ir) {
        // If direct indexing is used during the vectorization, we simply bitcast the scalar pointer
        // to a vector pointer
        llvm::Type* vector_type = llvm::PointerType::get(
            llvm::FixedVectorType::get(element_ptr->getType()->getPointerElementType(),
                                       platform.get_instruction_width()),
            /*AddressSpace=*/0);
        ptr = builder.CreateBitCast(element_ptr, vector_type);
    } else {
        // Otherwise, scalar code is generated and hence return the element pointer.
        ptr = element_ptr;
    }

    if (maybe_value_to_store) {
        create_store_direct(ptr, maybe_value_to_store, /*masked=*/mask && generating_vector_ir);
        return nullptr;
    } else {
        return create_load_direct(ptr, /*masked=*/mask && generating_vector_ir);
    }
}

void BaseBuilder::create_grid_stride() {
    llvm::Module* m = builder.GetInsertBlock()->getParent()->getParent();
    auto create_call = [&](llvm::Intrinsic::ID id) {
        llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(m, id);
        return builder.CreateCall(intrinsic, {});
    };

    llvm::Value* block_dim = create_call(llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);
    llvm::Value* grid_dim = create_call(llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
    llvm::Value* stride = builder.CreateMul(block_dim, grid_dim);

    value_stack.push_back(stride);
}

void BaseBuilder::create_thread_id() {
    llvm::Module* m = builder.GetInsertBlock()->getParent()->getParent();
    auto create_call = [&](llvm::Intrinsic::ID id) {
        llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(m, id);
        return builder.CreateCall(intrinsic, {});
    };

    // For now, this function only supports NVPTX backend, however it can be easily
    // adjusted to generate thread id variable for any other platform.
    llvm::Value* block_id = create_call(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
    llvm::Value* block_dim = create_call(llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);
    llvm::Value* tmp = builder.CreateMul(block_id, block_dim);

    llvm::Value* tid = create_call(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
    llvm::Value* id = builder.CreateAdd(tmp, tid);

    value_stack.push_back(id);
}

}  // namespace codegen
}  // namespace nmodl
