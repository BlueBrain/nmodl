/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/llvm_ir_builder.hpp"
#include "ast/all.hpp"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/ValueSymbolTable.h"

namespace nmodl {
namespace codegen {


/****************************************************************************************/
/*                            LLVM type utilities                                       */
/****************************************************************************************/

llvm::Type* IRBuilder::get_boolean_type() {
    return llvm::Type::getInt1Ty(builder.getContext());
}

llvm::Type* IRBuilder::get_i8_ptr_type() {
    return llvm::Type::getInt8PtrTy(builder.getContext());
}

llvm::Type* IRBuilder::get_i32_type() {
    return llvm::Type::getInt32Ty(builder.getContext());
}

llvm::Type* IRBuilder::get_i32_ptr_type() {
    return llvm::Type::getInt32PtrTy(builder.getContext());
}

llvm::Type* IRBuilder::get_i64_type() {
    return llvm::Type::getInt64Ty(builder.getContext());
}

llvm::Type* IRBuilder::get_fp_type() {
    if (platform.is_single_precision())
        return llvm::Type::getFloatTy(builder.getContext());
    return llvm::Type::getDoubleTy(builder.getContext());
}

llvm::Type* IRBuilder::get_fp_ptr_type() {
    if (platform.is_single_precision())
        return llvm::Type::getFloatPtrTy(builder.getContext());
    return llvm::Type::getDoublePtrTy(builder.getContext());
}

llvm::Type* IRBuilder::get_void_type() {
    return llvm::Type::getVoidTy(builder.getContext());
}

llvm::Type* IRBuilder::get_struct_ptr_type(const std::string& struct_type_name,
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
/*                            LLVM value utilities                                      */
/****************************************************************************************/

llvm::Value* IRBuilder::lookup_value(const std::string& value_name) {
    auto value = current_function->getValueSymbolTable()->lookup(value_name);
    if (!value)
        throw std::runtime_error("Error: variable " + value_name + " is not in the scope\n");
    return value;
}

llvm::Value* IRBuilder::pop_last_value() {
    // Check if the stack is empty.
    if (value_stack.empty())
        throw std::runtime_error("Error: popping a value from the empty stack\n");

    // Return the last added value and delete it from the stack.
    llvm::Value* last = value_stack.back();
    value_stack.pop_back();
    return last;
}

/****************************************************************************************/
/*                            LLVM constants utilities                                  */
/****************************************************************************************/

void IRBuilder::create_boolean_constant(int value) {
    if (platform.is_cpu_with_simd() && vectorize) {
        value_stack.push_back(get_vector_constant<llvm::ConstantInt>(get_boolean_type(), value));
    } else {
        value_stack.push_back(get_scalar_constant<llvm::ConstantInt>(get_boolean_type(), value));
    }
}

void IRBuilder::create_fp_constant(const std::string& value) {
    if (platform.is_cpu_with_simd() && vectorize) {
        value_stack.push_back(get_vector_constant<llvm::ConstantFP>(get_fp_type(), value));
    } else {
        value_stack.push_back(get_scalar_constant<llvm::ConstantFP>(get_fp_type(), value));
    }
}

llvm::Value* IRBuilder::create_global_string(const ast::String& node) {
    return builder.CreateGlobalStringPtr(node.get_value());
}

void IRBuilder::create_i32_constant(int value) {
    if (platform.is_cpu_with_simd() && vectorize) {
        value_stack.push_back(get_vector_constant<llvm::ConstantInt>(get_i32_type(), value));
    } else {
        value_stack.push_back(get_scalar_constant<llvm::ConstantInt>(get_i32_type(), value));
    }
}

template <typename C, typename V>
llvm::Value* IRBuilder::get_scalar_constant(llvm::Type* type, V value) {
    return C::get(type, value);
}

template <typename C, typename V>
llvm::Value* IRBuilder::get_vector_constant(llvm::Type* type, V value) {
    int vector_width = platform.get_instruction_width();

    ConstantVector constants;
    for (unsigned i = 0; i < vector_width; ++i) {
        const auto& element = C::get(type, value);
        constants.push_back(element);
    }
    return llvm::ConstantVector::get(constants);
}

/****************************************************************************************/
/*                              LLVM function utilities                                 */
/****************************************************************************************/

void IRBuilder::allocate_function_arguments(llvm::Function* function,
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

void IRBuilder::allocate_and_wrap_kernel_arguments(
    llvm::Function* function,
    const ast::CodegenVarWithTypeVector& nmodl_arguments,
    llvm::Type* struct_type) {
    // In theory, this should never happen but let's guard anyway.
    if (nmodl_arguments.size() != 1) {
        throw std::runtime_error("Error: NMODL computer kernel must have a single argument\n");
    }

    // Bitcast void* pointer provided as compute kernel argument to mechanism data type.
    llvm::Value* data_ptr = create_bitcast(function->getArg(0), struct_type);

    std::string arg_name = nmodl_arguments[0].get()->get_node_name();
    llvm::Value* alloca = create_alloca(arg_name, struct_type);
    builder.CreateStore(data_ptr, alloca);
}

std::string IRBuilder::get_current_function_name() {
    return current_function->getName().str();
}

void IRBuilder::create_function_call(llvm::Function* callee,
                                     ValueVector& arguments,
                                     bool use_result) {
    llvm::Value* call_instruction = builder.CreateCall(callee, arguments);
    if (use_result)
        value_stack.push_back(call_instruction);
}

void IRBuilder::create_intrinsic(const std::string& name,
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

/****************************************************************************************/
/*                             LLVM instruction utilities                               */
/****************************************************************************************/

llvm::Value* IRBuilder::create_alloca(const std::string& name, llvm::Type* type) {
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

void IRBuilder::create_array_alloca(const std::string& name,
                                    llvm::Type* element_type,
                                    int num_elements) {
    llvm::Type* array_type = llvm::ArrayType::get(element_type, num_elements);
    create_alloca(name, array_type);
}

ast::BinaryOp IRBuilder::extract_atomic_op(ast::BinaryOp op) {
    switch (op) {
    case ast::BinaryOp::BOP_SUB_ASSIGN:
        return ast::BinaryOp::BOP_SUBTRACTION;
    case ast::BinaryOp::BOP_ADD_ASSIGN:
        return ast::BinaryOp::BOP_ADDITION;
    default:
        throw std::runtime_error("Error: only atomic addition and subtraction is supported\n");
    }
}

void IRBuilder::create_atomic_op(llvm::Value* ptr, llvm::Value* update, ast::BinaryOp op) {
    if (op == ast::BinaryOp::BOP_SUBTRACTION) {
        update = builder.CreateFNeg(update);
    }
    builder.CreateAtomicRMW(llvm::AtomicRMWInst::FAdd,
                            ptr,
                            update,
                            llvm::MaybeAlign(),
                            llvm::AtomicOrdering::SequentiallyConsistent);
}

llvm::Value* IRBuilder::create_member_addresses(llvm::Value* member_ptr) {
    llvm::Module* m = builder.GetInsertBlock()->getParent()->getParent();

    // Treat this member address as integer value.
    llvm::Type* int_ptr_type = m->getDataLayout().getIntPtrType(builder.getContext());
    llvm::Value* ptr_to_int = builder.CreatePtrToInt(member_ptr, int_ptr_type);

    // Create a vector that has address at 0.
    llvm::Type* vector_type = llvm::FixedVectorType::get(int_ptr_type,
                                                         platform.get_instruction_width());
    llvm::Value* zero = get_scalar_constant<llvm::ConstantInt>(get_i32_type(), 0);
    llvm::Value* tmp =
        builder.CreateInsertElement(llvm::UndefValue::get(vector_type), ptr_to_int, zero);

    // Finally, use `shufflevector` with zeroinitializer to replicate the 0th element.
    llvm::Value* select = llvm::Constant::getNullValue(vector_type);
    return builder.CreateShuffleVector(tmp, llvm::UndefValue::get(vector_type), select);
}

llvm::Value* IRBuilder::create_member_offsets(llvm::Value* start, llvm::Value* indices) {
    llvm::Value* factor = get_vector_constant<llvm::ConstantInt>(get_i64_type(),
                                                                 platform.get_precision() / 8);
    llvm::Value* offset = builder.CreateMul(indices, factor);
    return builder.CreateAdd(start, offset);
}

llvm::Value* IRBuilder::create_atomic_loop(llvm::Value* ptrs_arr,
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

void IRBuilder::create_binary_op(llvm::Value* lhs, llvm::Value* rhs, ast::BinaryOp op) {
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

llvm::Value* IRBuilder::create_bitcast(llvm::Value* value, llvm::Type* dst_type) {
    return builder.CreateBitCast(value, dst_type);
}

llvm::Value* IRBuilder::create_inbounds_gep(const std::string& var_name, llvm::Value* index) {
    llvm::Value* variable_ptr = lookup_value(var_name);

    // Since we index through the pointer, we need an extra 0 index in the indices list for GEP.
    ValueVector indices{llvm::ConstantInt::get(get_i64_type(), 0), index};
    llvm::Type* variable_type = variable_ptr->getType()->getPointerElementType();
    return builder.CreateInBoundsGEP(variable_type, variable_ptr, indices);
}

llvm::Value* IRBuilder::create_inbounds_gep(llvm::Value* variable, llvm::Value* index) {
    ValueVector indices{index};
    llvm::Type* variable_type = variable->getType()->getPointerElementType();
    return builder.CreateInBoundsGEP(variable_type, variable, indices);
}

llvm::Value* IRBuilder::create_index(llvm::Value* value) {
    // Check if index is a double. While it is possible to use casting from double to integer
    // values, we choose not to support these cases.
    llvm::Type* value_type = value->getType();
    if (!value_type->isIntOrIntVectorTy())
        throw std::runtime_error("Error: only integer indexing is supported\n");

    // Conventionally, in LLVM array indices are 64 bit.
    llvm::Type* i64_type = get_i64_type();
    if (auto index_type = llvm::dyn_cast<llvm::IntegerType>(value_type)) {
        if (index_type->getBitWidth() == i64_type->getIntegerBitWidth())
            return value;
        return builder.CreateSExtOrTrunc(value, i64_type);
    }

    const auto& vector_type = llvm::cast<llvm::FixedVectorType>(value_type);
    const auto& element_type = llvm::cast<llvm::IntegerType>(vector_type->getElementType());
    if (element_type->getBitWidth() == i64_type->getIntegerBitWidth())
        return value;
    int vector_width = platform.get_instruction_width();
    return builder.CreateSExtOrTrunc(value, llvm::FixedVectorType::get(i64_type, vector_width));
}

llvm::Value* IRBuilder::create_load(const std::string& name, bool masked) {
    llvm::Value* ptr = lookup_value(name);
    llvm::Type* loaded_type = ptr->getType()->getPointerElementType();

    // Check if the generated IR is vectorized and masked.
    if (masked) {
        builder.CreateMaskedLoad(loaded_type, ptr, llvm::Align(), mask);
    }
    llvm::Value* loaded = builder.CreateLoad(loaded_type, ptr);
    value_stack.push_back(loaded);
    return loaded;
}

llvm::Value* IRBuilder::create_load(llvm::Value* ptr, bool masked) {
    llvm::Type* loaded_type = ptr->getType()->getPointerElementType();

    // Check if the generated IR is vectorized and masked.
    if (masked) {
        builder.CreateMaskedLoad(loaded_type, ptr, llvm::Align(), mask);
    }

    llvm::Value* loaded = builder.CreateLoad(loaded_type, ptr);
    value_stack.push_back(loaded);
    return loaded;
}

llvm::Value* IRBuilder::create_load_from_array(const std::string& name, llvm::Value* index) {
    llvm::Value* element_ptr = create_inbounds_gep(name, index);
    return create_load(element_ptr);
}

void IRBuilder::create_store(const std::string& name, llvm::Value* value, bool masked) {
    llvm::Value* ptr = lookup_value(name);

    // Check if the generated IR is vectorized and masked.
    if (masked) {
        builder.CreateMaskedStore(value, ptr, llvm::Align(), mask);
        return;
    }
    builder.CreateStore(value, ptr);
}

void IRBuilder::create_store(llvm::Value* ptr, llvm::Value* value, bool masked) {
    // Check if the generated IR is vectorized and masked.
    if (masked) {
        builder.CreateMaskedStore(value, ptr, llvm::Align(), mask);
        return;
    }
    builder.CreateStore(value, ptr);
}

void IRBuilder::create_store_to_array(const std::string& name,
                                      llvm::Value* index,
                                      llvm::Value* value) {
    llvm::Value* element_ptr = create_inbounds_gep(name, index);
    create_store(element_ptr, value);
}

void IRBuilder::create_return(llvm::Value* return_value) {
    if (return_value)
        builder.CreateRet(return_value);
    else
        builder.CreateRetVoid();
}

void IRBuilder::create_scalar_or_vector_alloca(const std::string& name,
                                               llvm::Type* element_or_scalar_type) {
    // Even if generating vectorised code, some variables still need to be scalar. Particularly, the
    // induction variable "id" and remainder loop variables (that start with "epilogue" prefix).
    llvm::Type* type;
    if (platform.is_cpu_with_simd() && vectorize && name != kernel_id &&
        name.rfind("epilogue", 0)) {
        int vector_width = platform.get_instruction_width();
        type = llvm::FixedVectorType::get(element_or_scalar_type, vector_width);
    } else {
        type = element_or_scalar_type;
    }
    create_alloca(name, type);
}

void IRBuilder::create_unary_op(llvm::Value* value, ast::UnaryOp op) {
    if (op == ast::UOP_NEGATION) {
        value_stack.push_back(builder.CreateFNeg(value));
    } else if (op == ast::UOP_NOT) {
        value_stack.push_back(builder.CreateNot(value));
    } else {
        throw std::runtime_error("Error: unsupported unary operator\n");
    }
}

llvm::Value* IRBuilder::get_struct_member_ptr(llvm::Value* struct_variable, int member_index) {
    ValueVector indices;
    indices.push_back(llvm::ConstantInt::get(get_i32_type(), 0));
    indices.push_back(llvm::ConstantInt::get(get_i32_type(), member_index));

    llvm::Type* type = struct_variable->getType()->getPointerElementType();
    return builder.CreateInBoundsGEP(type, struct_variable, indices);
}

void IRBuilder::invert_mask() {
    if (!mask)
        throw std::runtime_error("Error: mask is not set\n");

    // Create the vector with all `true` values.
    create_boolean_constant(1);
    llvm::Value* one = pop_last_value();

    mask = builder.CreateXor(mask, one);
}

llvm::Value* IRBuilder::load_to_or_store_from_array(const std::string& id_name,
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
    if (id_name != kernel_id && generating_vector_ir) {
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
        create_store(ptr, maybe_value_to_store, /*masked=*/mask && generating_vector_ir);
        return nullptr;
    } else {
        return create_load(ptr, /*masked=*/mask && generating_vector_ir);
    }
}

void IRBuilder::maybe_replicate_value(llvm::Value* value) {
    // If the value should not be vectorised, or it is already a vector, add it to the stack.
    if (!vectorize || !platform.is_cpu_with_simd() || value->getType()->isVectorTy()) {
        value_stack.push_back(value);
    } else {
        // Otherwise, we generate vectorized code inside the loop, so replicate the value to form a
        // vector.
        int vector_width = platform.get_instruction_width();
        llvm::Value* vector_value = builder.CreateVectorSplat(vector_width, value);
        value_stack.push_back(vector_value);
    }
}

void IRBuilder::create_grid_stride() {
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

void IRBuilder::create_thread_id() {
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


/****************************************************************************************/
/*                                 LLVM block utilities                                 */
/****************************************************************************************/

llvm::BasicBlock* IRBuilder::create_block_and_set_insertion_point(llvm::Function* function,
                                                                  llvm::BasicBlock* insert_before,
                                                                  std::string name) {
    llvm::BasicBlock* block =
        llvm::BasicBlock::Create(builder.getContext(), name, function, insert_before);
    builder.SetInsertPoint(block);
    return block;
}

void IRBuilder::create_br(llvm::BasicBlock* block) {
    builder.CreateBr(block);
}

void IRBuilder::create_br_and_set_insertion_point(llvm::BasicBlock* block) {
    builder.CreateBr(block);
    builder.SetInsertPoint(block);
}

llvm::BranchInst* IRBuilder::create_cond_br(llvm::Value* condition,
                                            llvm::BasicBlock* true_block,
                                            llvm::BasicBlock* false_block) {
    return builder.CreateCondBr(condition, true_block, false_block);
}

llvm::BasicBlock* IRBuilder::get_current_block() {
    return builder.GetInsertBlock();
}

void IRBuilder::set_insertion_point(llvm::BasicBlock* block) {
    builder.SetInsertPoint(block);
}

}  // namespace codegen
}  // namespace nmodl
