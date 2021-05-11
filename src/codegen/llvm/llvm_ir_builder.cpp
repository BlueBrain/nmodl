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
    if (fp_precision == single_precision)
        return llvm::Type::getFloatTy(builder.getContext());
    return llvm::Type::getDoubleTy(builder.getContext());
}

llvm::Type* IRBuilder::get_fp_ptr_type() {
    if (fp_precision == single_precision)
        return llvm::Type::getFloatPtrTy(builder.getContext());
    return llvm::Type::getDoublePtrTy(builder.getContext());
}

llvm::Type* IRBuilder::get_void_type() {
    return llvm::Type::getVoidTy(builder.getContext());
}

llvm::Type* IRBuilder::get_struct_type(const std::string& struct_type_name,
                                       TypeVector member_types) {
    llvm::StructType* llvm_struct_type = llvm::StructType::create(builder.getContext(),
                                                                  struct_type_name);
    llvm_struct_type->setBody(member_types);
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
    value_stack.push_back(get_vector_constant<llvm::ConstantInt>(get_boolean_type(), value));
}

void IRBuilder::create_fp_constant(const std::string& value) {
    if (vector_width > 1 && vectorize) {
        value_stack.push_back(get_vector_constant<llvm::ConstantFP>(get_fp_type(), value));
    } else {
        value_stack.push_back(get_scalar_constant<llvm::ConstantFP>(get_fp_type(), value));
    }
}

void IRBuilder::create_i32_constant(int value) {
    if (vector_width > 1 && vectorize) {
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
        llvm::Value* alloca = builder.CreateAlloca(arg_type, /*ArraySize=*/nullptr, arg_name);
        arg.setName(arg_name);
        builder.CreateStore(&arg, alloca);
    }
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
    unsigned intrinsic_id = llvm::StringSwitch<llvm::Intrinsic::ID>(name)
                                .Case("exp", llvm::Intrinsic::exp)
                                .Case("pow", llvm::Intrinsic::pow)
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
    return builder.CreateSExtOrTrunc(value, llvm::FixedVectorType::get(i64_type, vector_width));
}

llvm::Value* IRBuilder::create_inbounds_gep(const std::string& var_name, llvm::Value* index) {
    ValueVector indices{llvm::ConstantInt::get(get_i64_type(), 0), index};
    return builder.CreateInBoundsGEP(lookup_value(var_name), indices);
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

int IRBuilder::get_array_length(const ast::IndexedName& node) {
    // First, verify if the length is an integer value.
    const auto& integer = std::dynamic_pointer_cast<ast::Integer>(node.get_length());
    if (!integer)
        throw std::runtime_error("Error: only integer length is supported\n");

    // Check if the length value is a constant.
    if (!integer->get_macro())
        return integer->get_value();

    // Otherwise, the length is taken from the macro.
    const auto& macro = symbol_table->lookup(integer->get_macro()->get_node_name());
    return static_cast<int>(*macro->get_value());
}

/****************************************************************************************/
/*                                 LLVM block utilities                                 */
/****************************************************************************************/

llvm::BasicBlock* IRBuilder::get_current_bb() {
    return builder.GetInsertBlock();
}

}  // namespace codegen
}  // namespace nmodl
