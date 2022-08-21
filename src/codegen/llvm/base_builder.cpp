/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/base_builder.hpp"
#include "ast/all.hpp"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ValueSymbolTable.h"

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
    if (single_precision)
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
    if (single_precision)
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

llvm::Value* BaseBuilder::create_global_string(const ast::String& node) {
    return builder.CreateGlobalStringPtr(node.get_value());
}

llvm::Value* BaseBuilder::create_inbounds_gep(const std::string& variable_name,
                                              llvm::Value* index) {
    llvm::Value* variable_ptr = lookup_value(variable_name);

    // Since we index through the pointer, we need an extra 0 index in the indices list for GEP.
    ValueVector indices{llvm::ConstantInt::get(get_i64_type(), 0), index};
    llvm::Type* variable_type = variable_ptr->getType()->getPointerElementType();
    return builder.CreateInBoundsGEP(variable_type, variable_ptr, indices);
}

void BaseBuilder::create_return(llvm::Value* return_value) {
    if (return_value)
        builder.CreateRet(return_value);
    else
        builder.CreateRetVoid();
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

void BaseBuilder::generate_atomic_statement(llvm::Value* ptr, llvm::Value* rhs, ast::BinaryOp op) {
    // TODO: rewrite this!
}

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
    value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_boolean_type(), value));
}

void BaseBuilder::generate_i32_constant(int value) {
    value_stack.push_back(scalar_constant<llvm::ConstantInt>(get_i32_type(), value));
}

void BaseBuilder::generate_fp_constant(const std::string& value) {
    value_stack.push_back(scalar_constant<llvm::ConstantFP>(get_fp_type(), value));
}

void BaseBuilder::generate_load_direct(llvm::Value* ptr) {
    llvm::Type* loaded_type = ptr->getType()->getPointerElementType();
    llvm::Value* loaded = builder.CreateLoad(loaded_type, ptr);
    value_stack.push_back(loaded);
}

void BaseBuilder::generate_load_indirect(llvm::Value* ptr) {
    llvm::Type* loaded_type = ptr->getType()->getPointerElementType();
    llvm::Value* loaded = builder.CreateLoad(loaded_type, ptr);
    value_stack.push_back(loaded);
}

void BaseBuilder::generate_store_direct(llvm::Value* ptr, llvm::Value* value) {
    builder.CreateStore(value, ptr);
}

void BaseBuilder::generate_store_indirect(llvm::Value* ptr, llvm::Value* value) {
    builder.CreateStore(value, ptr);
}

void BaseBuilder::try_generate_broadcast(llvm::Value* value) {
    // No need to broadcast anything!
    value_stack.push_back(value);
}

void BaseBuilder::generate_loop_start() {
    generate_i32_constant(0);
}

void BaseBuilder::generate_loop_increment() {
    generate_i32_constant(1);
}

void BaseBuilder::generate_loop_end() {
    // TODO: fix this when an AST node is added.
}

}  // namespace codegen
}  // namespace nmodl
