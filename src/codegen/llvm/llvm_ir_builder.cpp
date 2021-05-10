/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/llvm_ir_builder.hpp"
#include "ast/all.hpp"

#include "llvm/IR/Function.h"
#include "llvm/IR/ValueSymbolTable.h"

namespace nmodl {
namespace codegen {

static constexpr const char instance_struct_type_name[] = "__instance_var__type";
static constexpr const char printf_name[] = "printf";

/****************************************************************************************/
/*                            LLVM type utilities                                       */
/****************************************************************************************/

llvm::Type* IRBuilder::get_boolean_type() {
    return llvm::Type::getInt1Ty(builder.getContext());
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

llvm::Type* IRBuilder::get_instance_struct_type() {
    TypeVector member_types;
    for (const auto& variable: instance_var_helper->instance->get_codegen_vars()) {
        // First, get information about member's type and whether it is a pointer.
        auto nmodl_type = variable->get_type()->get_type();
        auto is_pointer = variable->get_is_pointer();

        // Create the corresponding LLVM type.
        switch (nmodl_type) {
        case ast::AstNodeType::INTEGER:
            member_types.push_back(is_pointer ? get_i32_ptr_type() : get_i32_type());
            break;
        case ast::AstNodeType::DOUBLE:
            member_types.push_back(is_pointer ? get_fp_type() : get_fp_ptr_type());
            break;
        default:
            throw std::runtime_error("Error: unsupported type encountered in instance struct\n");
        }
    }

    // Create the struct type with the given members.
    llvm::StructType* llvm_struct_type =
        llvm::StructType::create(builder.getContext(),
                                 /*mod_filename + */ instance_struct_type_name);
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
/*                            LLVM Constants utilities                                  */
/****************************************************************************************/

llvm::Value* IRBuilder::get_constant_fp_vector(const std::string& value) {
    ConstantVector constants;
    for (unsigned i = 0; i < vector_width; ++i) {
        const auto& element = llvm::ConstantFP::get(get_fp_type(), value);
        constants.push_back(element);
    }
    return llvm::ConstantVector::get(constants);
}

llvm::Value* IRBuilder::get_constant_i32_vector(int value) {
    ConstantVector constants;
    for (unsigned i = 0; i < vector_width; ++i) {
        const auto& element = llvm::ConstantInt::get(get_i32_type(), value);
        constants.push_back(element);
    }
    return llvm::ConstantVector::get(constants);
}

/****************************************************************************************/
/*                              LLVM function utilities                                 */
/****************************************************************************************/

//void IRBuilder::create_printf_call(const ast::ExpressionVector& arguments) {
    // First, create printf declaration or insert it if it does not exit.
//    llvm::Function* printf = module->getFunction(printf_name);
//    if (!printf) {
//        llvm::Type* ptr_type = llvm::Type::getInt8PtrTy(*context);
//        llvm::Type* i32_type = llvm::Type::getInt32Ty(*context);
//        llvm::FunctionType* printf_type =
//                llvm::FunctionType::get(i32_type, ptr_type, /*isVarArg=*/true);
//
//        printf =
//                llvm::Function::Create(printf_type, llvm::Function::ExternalLinkage, name, *module);
//    }
//
//    // Create a call instruction.
//    std::vector<llvm::Value*> argument_values;
//    argument_values.reserve(arguments.size());
//    pack_function_call_arguments(arguments, argument_values);
//    ir_builder.CreateCall(printf, argument_values);
//}

//void IRBuilder::create_external_function_call(const std::string& name, const ast::ExpressionVector& arguments) {
//    // Process `printf` calls separately.
//    if (name == printf_name) {
//        create_printf_call(arguments);
//        return;
//    }
//
//    // Create argument value and type vectors.
//    ValueVector argument_values;
//    TypeVector argument_types;
//    for (const auto& arg: arguments) {
//        arg->accept(visitor);
//        llvm::Value* value = pop_last_value();
//        argument_types.push_back(value->getType());
//        argument_values.push_back(value);
//    }
//
//#define DISPATCH(function_name, intrinsic)                                       \
//    if (name == (function_name)) {                                               \
//        llvm::Value* result =                                                    \
//            builder.CreateIntrinsic(intrinsic, argument_types, argument_values); \
//        value_stack.push_back(result);                                           \
//        return;                                                                  \
//    }
//
//    DISPATCH("exp", llvm::Intrinsic::exp);
//    DISPATCH("pow", llvm::Intrinsic::pow);
//#undef DISPATCH
//
//    throw std::runtime_error("Error: function call to" + name + " is not currently supported\n");
//}
//
//void IRBuilder::create_function_call_arguments(const ast::ExpressionVector &arguments, ValueVector &values) {
//    for (const auto& arg: arguments) {
//        if (arg->is_string()) {
//            // If the argument is a string, create a global i8* variable with it.
//            const auto& string_arg = std::dynamic_pointer_cast<ast::String>(arg);
//            llvm::Value* str = builder.CreateGlobalStringPtr(string_arg->get_value());
//            values.push_back(str);
//        } else {
//            // Otherwise, visit the expression and add the processed value.
//            arg->accept(visitor);
//            values.push_back(pop_last_value());
//        }
//    }
//}

/****************************************************************************************/
/*                             LLVM instruction utilities                               */
/****************************************************************************************/

void IRBuilder::create_bin_op(llvm::Value* lhs, llvm::Value* rhs, ast::BinaryOp op) {
    // Check that both lhs and rhs have the same types.
    if (lhs->getType() != rhs->getType())
        throw std::runtime_error("Error: lhs and rhs of the binary operator have different types\n");

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
        case  ast::BinaryOp::BOP_OR:
            result = builder.CreateOr(lhs, rhs);
            break;

        default:
            throw std::runtime_error("Error: unsupported binary operator\n");
    }
    value_stack.push_back(result);
}

llvm::Value* IRBuilder::create_inbounds_gep(const std::string& var_name, llvm::Value* index) {
    ValueVector indices{llvm::ConstantInt::get(get_i64_type(), 0), index};
    return builder.CreateInBoundsGEP(lookup_value(var_name), indices);
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
}  // namespace codegen
}  // namespace nmodl
