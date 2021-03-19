/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/codegen_llvm_visitor.hpp"

#include "ast/all.hpp"
#include "visitors/rename_visitor.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueSymbolTable.h"

namespace nmodl {
namespace codegen {


static constexpr const char instance_struct_type_name[] = "__instance_var__type";

// The prefix is used to create a vectorised id that can be used as index to GEPs. However, for
// simple aligned vector loads and stores vector id is not needed. This is because we can bitcast
// the pointer to the vector pointer! \todo: Consider removing this.
static constexpr const char kernel_id_prefix[] = "__vec_";


/****************************************************************************************/
/*                            Helper routines                                           */
/****************************************************************************************/

static bool is_supported_statement(const ast::Statement& statement) {
    return statement.is_codegen_var_list_statement() || statement.is_expression_statement() ||
           statement.is_codegen_for_statement() || statement.is_codegen_return_statement() ||
           statement.is_if_statement() || statement.is_while_statement();
}

llvm::Value* CodegenLLVMVisitor::create_gep(const std::string& name, llvm::Value* index) {
    llvm::Type* index_type = llvm::Type::getInt64Ty(*context);
    std::vector<llvm::Value*> indices;
    indices.push_back(llvm::ConstantInt::get(index_type, 0));
    indices.push_back(index);

    return builder.CreateInBoundsGEP(lookup(name), indices);
}

llvm::Value* CodegenLLVMVisitor::codegen_indexed_name(const ast::IndexedName& node) {
    llvm::Value* index = get_array_index(node);
    return create_gep(node.get_node_name(), index);
}

llvm::Value* CodegenLLVMVisitor::codegen_instance_var(const ast::CodegenInstanceVar& node) {
    const auto& member_node = node.get_member_var();
    const auto& instance_name = node.get_instance_var()->get_node_name();
    const auto& member_name = member_node->get_node_name();

    if (!instance_var_helper.is_an_instance_variable(member_name))
        throw std::runtime_error("Error: " + member_name + " is not a member of the instance!");

    // Load the instance struct given its name from the ValueSymbolTable.
    llvm::Value* instance_ptr = builder.CreateLoad(lookup(instance_name));

    // Create a GEP instruction to get a pointer to the member.
    int member_index = instance_var_helper.get_variable_index(member_name);
    llvm::Type* index_type = llvm::Type::getInt32Ty(*context);

    std::vector<llvm::Value*> indices;
    indices.push_back(llvm::ConstantInt::get(index_type, 0));
    indices.push_back(llvm::ConstantInt::get(index_type, member_index));
    llvm::Value* member_ptr = builder.CreateInBoundsGEP(instance_ptr, indices);

    // Get the member AST node from the instance AST node, for which we proceed with the code
    // generation. If the member is scalar, return the pointer to it straight away.
    auto codegen_var_with_type = instance_var_helper.get_variable(member_name);
    if (!codegen_var_with_type->get_is_pointer()) {
        return member_ptr;
    }

    // Otherwise, the codegen variable is a pointer, and the member AST node must be an IndexedName.
    auto member_var_name = std::dynamic_pointer_cast<ast::VarName>(member_node);
    if (!member_var_name->get_name()->is_indexed_name())
        throw std::runtime_error("Error: " + member_name + " is not an IndexedName!");

    // Proceed to creating a GEP instruction to get the pointer to the member's element.
    auto member_indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(
        member_var_name->get_name());
    llvm::Value* i64_index = get_array_index(*member_indexed_name);


    // Create a indices vector for GEP to return the pointer to the element at the specified index.
    std::vector<llvm::Value*> member_indices;
    member_indices.push_back(i64_index);

    // The codegen variable type is always a scalar, so we need to transform it to a pointer. Then
    // load the member which would be indexed later.
    llvm::Type* type = get_codegen_var_type(*codegen_var_with_type->get_type());
    llvm::Value* instance_member =
        builder.CreateLoad(llvm::PointerType::get(type, /*AddressSpace=*/0), member_ptr);


    // If the code is vectorised, then bitcast to a vector pointer.
    if (is_kernel_code && vector_width > 1) {
        llvm::Type* vector_type =
            llvm::PointerType::get(llvm::FixedVectorType::get(type, vector_width),
                                   /*AddressSpace=*/0);
        llvm::Value* instance_member_bitcasted = builder.CreateBitCast(instance_member,
                                                                       vector_type);
        return builder.CreateInBoundsGEP(instance_member_bitcasted, member_indices);
    }

    return builder.CreateInBoundsGEP(instance_member, member_indices);
}

llvm::Value* CodegenLLVMVisitor::get_array_index(const ast::IndexedName& node) {
    // Process the index expression. It can either be a Name node:
    //    k[id]     // id is an integer
    // or an integer expression.
    llvm::Value* index_value;
    if (node.get_length()->is_name()) {
        llvm::Value* ptr = lookup(node.get_length()->get_node_name());
        index_value = builder.CreateLoad(ptr);
    } else {
        node.get_length()->accept(*this);
        index_value = values.back();
        values.pop_back();
    }

    // Check if index is a double. While it is possible to use casting from double to integer
    // values, we choose not to support these cases.
    if (!index_value->getType()->isIntOrIntVectorTy())
        throw std::runtime_error("Error: only integer indexing is supported!");

    // Conventionally, in LLVM array indices are 64 bit.
    auto index_type = llvm::cast<llvm::IntegerType>(index_value->getType());
    llvm::Type* i64_type = llvm::Type::getInt64Ty(*context);
    if (index_type->getBitWidth() == i64_type->getIntegerBitWidth())
        return index_value;

    return builder.CreateSExtOrTrunc(index_value, i64_type);
}

int CodegenLLVMVisitor::get_array_length(const ast::IndexedName& node) {
    auto integer = std::dynamic_pointer_cast<ast::Integer>(node.get_length());
    if (!integer)
        throw std::runtime_error("Error: only integer length is supported!");

    // Check if integer value is taken from a macro.
    if (!integer->get_macro())
        return integer->get_value();
    const auto& macro = sym_tab->lookup(integer->get_macro()->get_node_name());
    return static_cast<int>(*macro->get_value());
}

llvm::Type* CodegenLLVMVisitor::get_codegen_var_type(const ast::CodegenVarType& node) {
    switch (node.get_type()) {
    case ast::AstNodeType::BOOLEAN:
        return llvm::Type::getInt1Ty(*context);
    case ast::AstNodeType::DOUBLE:
        return get_default_fp_type();
    case ast::AstNodeType::INSTANCE_STRUCT:
        return get_instance_struct_type();
    case ast::AstNodeType::INTEGER:
        return llvm::Type::getInt32Ty(*context);
    case ast::AstNodeType::VOID:
        return llvm::Type::getVoidTy(*context);
    // TODO :: George/Ioannis : Here we have to also return INSTANCE_STRUCT type
    //         as it is used as an argument to nrn_state function
    default:
        throw std::runtime_error("Error: expecting a type in CodegenVarType node\n");
    }
}

llvm::Value* CodegenLLVMVisitor::get_constant_int_vector(int value) {
    llvm::Type* i32_type = llvm::Type::getInt32Ty(*context);
    std::vector<llvm::Constant*> constants;
    for (unsigned i = 0; i < vector_width; ++i) {
        const auto& element = llvm::ConstantInt::get(i32_type, value);
        constants.push_back(element);
    }
    return llvm::ConstantVector::get(constants);
}

llvm::Value* CodegenLLVMVisitor::get_constant_fp_vector(const std::string& value) {
    llvm::Type* fp_type = get_default_fp_type();
    std::vector<llvm::Constant*> constants;
    for (unsigned i = 0; i < vector_width; ++i) {
        const auto& element = llvm::ConstantFP::get(fp_type, value);
        constants.push_back(element);
    }
    return llvm::ConstantVector::get(constants);
}

llvm::Type* CodegenLLVMVisitor::get_default_fp_type() {
    if (use_single_precision)
        return llvm::Type::getFloatTy(*context);
    return llvm::Type::getDoubleTy(*context);
}

llvm::Type* CodegenLLVMVisitor::get_default_fp_ptr_type() {
    if (use_single_precision)
        return llvm::Type::getFloatPtrTy(*context);
    return llvm::Type::getDoublePtrTy(*context);
}

llvm::Type* CodegenLLVMVisitor::get_instance_struct_type() {
    std::vector<llvm::Type*> members;
    for (const auto& variable: instance_var_helper.instance->get_codegen_vars()) {
        auto is_pointer = variable->get_is_pointer();
        auto nmodl_type = variable->get_type()->get_type();

        llvm::Type* i32_type = llvm::Type::getInt32Ty(*context);
        llvm::Type* i32ptr_type = llvm::Type::getInt32PtrTy(*context);

        switch (nmodl_type) {
#define DISPATCH(type, llvm_ptr_type, llvm_type)                       \
    case type:                                                         \
        members.push_back(is_pointer ? (llvm_ptr_type) : (llvm_type)); \
        break;

            DISPATCH(ast::AstNodeType::DOUBLE, get_default_fp_ptr_type(), get_default_fp_type());
            DISPATCH(ast::AstNodeType::INTEGER, i32ptr_type, i32_type);

#undef DISPATCH
        default:
            throw std::runtime_error("Error: unsupported type found in instance struct");
        }
    }

    llvm::StructType* llvm_struct_type =
        llvm::StructType::create(*context, mod_filename + instance_struct_type_name);
    llvm_struct_type->setBody(members);
    return llvm::PointerType::get(llvm_struct_type, /*AddressSpace=*/0);
}

llvm::Value* CodegenLLVMVisitor::get_variable_ptr(const ast::VarName& node) {
    const auto& identifier = node.get_name();
    if (!identifier->is_name() && !identifier->is_indexed_name() &&
        !identifier->is_codegen_instance_var()) {
        throw std::runtime_error("Error: Unsupported variable type - " + node.get_node_name());
    }

    llvm::Value* ptr;
    if (identifier->is_name())
        ptr = lookup(node.get_node_name());

    if (identifier->is_indexed_name()) {
        auto indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(identifier);
        ptr = codegen_indexed_name(*indexed_name);
    }

    if (identifier->is_codegen_instance_var()) {
        auto instance_var = std::dynamic_pointer_cast<ast::CodegenInstanceVar>(identifier);
        ptr = codegen_instance_var(*instance_var);
    }
    return ptr;
}

std::shared_ptr<ast::InstanceStruct> CodegenLLVMVisitor::get_instance_struct_ptr() {
    return instance_var_helper.instance;
}

void CodegenLLVMVisitor::run_llvm_opt_passes() {
    /// run some common optimisation passes that are commonly suggested
    fpm.add(llvm::createInstructionCombiningPass());
    fpm.add(llvm::createReassociatePass());
    fpm.add(llvm::createGVNPass());
    fpm.add(llvm::createCFGSimplificationPass());

    /// initialize pass manager
    fpm.doInitialization();

    /// iterate over all functions and run the optimisation passes
    auto& functions = module->getFunctionList();
    for (auto& function: functions) {
        llvm::verifyFunction(function);
        fpm.run(function);
    }
}

void CodegenLLVMVisitor::create_external_method_call(const std::string& name,
                                                     const ast::ExpressionVector& arguments) {
    if (name == "printf") {
        create_printf_call(arguments);
        return;
    }

    std::vector<llvm::Value*> argument_values;
    std::vector<llvm::Type*> argument_types;
    for (const auto& arg: arguments) {
        arg->accept(*this);
        llvm::Value* value = values.back();
        llvm::Type* type = value->getType();
        values.pop_back();
        argument_types.push_back(type);
        argument_values.push_back(value);
    }

#define DISPATCH(method_name, intrinsic)                                                           \
    if (name == (method_name)) {                                                                   \
        llvm::Value* result = builder.CreateIntrinsic(intrinsic, argument_types, argument_values); \
        values.push_back(result);                                                                  \
        return;                                                                                    \
    }

    DISPATCH("exp", llvm::Intrinsic::exp);
    DISPATCH("pow", llvm::Intrinsic::pow);
#undef DISPATCH

    throw std::runtime_error("Error: External method" + name + " is not currently supported");
}

void CodegenLLVMVisitor::create_function_call(llvm::Function* func,
                                              const std::string& name,
                                              const ast::ExpressionVector& arguments) {
    // Check that function is called with the expected number of arguments.
    if (!func->isVarArg() && arguments.size() != func->arg_size()) {
        throw std::runtime_error("Error: Incorrect number of arguments passed");
    }

    // Pack function call arguments to vector and create a call instruction.
    std::vector<llvm::Value*> argument_values;
    argument_values.reserve(arguments.size());
    pack_function_call_arguments(arguments, argument_values);
    llvm::Value* call = builder.CreateCall(func, argument_values);
    values.push_back(call);
}

void CodegenLLVMVisitor::create_printf_call(const ast::ExpressionVector& arguments) {
    // First, create printf declaration or insert it if it does not exit.
    std::string name = "printf";
    llvm::Function* printf = module->getFunction(name);
    if (!printf) {
        llvm::Type* ptr_type = llvm::Type::getInt8PtrTy(*context);
        llvm::Type* i32_type = llvm::Type::getInt32Ty(*context);
        llvm::FunctionType* printf_type =
            llvm::FunctionType::get(i32_type, ptr_type, /*isVarArg=*/true);

        printf =
            llvm::Function::Create(printf_type, llvm::Function::ExternalLinkage, name, *module);
    }

    // Create a call instruction.
    std::vector<llvm::Value*> argument_values;
    argument_values.reserve(arguments.size());
    pack_function_call_arguments(arguments, argument_values);
    builder.CreateCall(printf, argument_values);
}

void CodegenLLVMVisitor::emit_procedure_or_function_declaration(const ast::CodegenFunction& node) {
    const auto& name = node.get_node_name();
    const auto& arguments = node.get_arguments();

    // Procedure or function parameters are doubles by default.
    std::vector<llvm::Type*> arg_types;
    for (size_t i = 0; i < arguments.size(); ++i)
        arg_types.push_back(get_codegen_var_type(*arguments[i]->get_type()));

    llvm::Type* return_type = get_codegen_var_type(*node.get_return_type());

    // Create a function that is automatically inserted into module's symbol table.
    llvm::Function::Create(llvm::FunctionType::get(return_type, arg_types, /*isVarArg=*/false),
                           llvm::Function::ExternalLinkage,
                           name,
                           *module);
}

llvm::Value* CodegenLLVMVisitor::lookup(const std::string& name) {
    auto val = current_func->getValueSymbolTable()->lookup(name);
    if (!val)
        throw std::runtime_error("Error: variable " + name + " is not in scope\n");
    return val;
}

void CodegenLLVMVisitor::pack_function_call_arguments(const ast::ExpressionVector& arguments,
                                                      std::vector<llvm::Value*>& arg_values) {
    for (const auto& arg: arguments) {
        if (arg->is_string()) {
            // If the argument is a string, create a global i8* variable with it.
            auto string_arg = std::dynamic_pointer_cast<ast::String>(arg);
            llvm::Value* str = builder.CreateGlobalStringPtr(string_arg->get_value());
            arg_values.push_back(str);
        } else {
            arg->accept(*this);
            llvm::Value* value = values.back();
            values.pop_back();
            arg_values.push_back(value);
        }
    }
}

llvm::Value* CodegenLLVMVisitor::visit_arithmetic_bin_op(llvm::Value* lhs,
                                                         llvm::Value* rhs,
                                                         unsigned op) {
    const auto& bin_op = static_cast<ast::BinaryOp>(op);
    llvm::Type* lhs_type = lhs->getType();
    llvm::Value* result;

    switch (bin_op) {
#define DISPATCH(binary_op, llvm_fp_op, llvm_int_op) \
    case binary_op:                                  \
        if (lhs_type->isIntOrIntVectorTy())          \
            result = llvm_int_op(lhs, rhs);          \
        else                                         \
            result = llvm_fp_op(lhs, rhs);           \
        return result;

        DISPATCH(ast::BinaryOp::BOP_ADDITION, builder.CreateFAdd, builder.CreateAdd);
        DISPATCH(ast::BinaryOp::BOP_DIVISION, builder.CreateFDiv, builder.CreateSDiv);
        DISPATCH(ast::BinaryOp::BOP_MULTIPLICATION, builder.CreateFMul, builder.CreateMul);
        DISPATCH(ast::BinaryOp::BOP_SUBTRACTION, builder.CreateFSub, builder.CreateSub);

#undef DISPATCH

    default:
        return nullptr;
    }
}

void CodegenLLVMVisitor::visit_assign_op(const ast::BinaryExpression& node, llvm::Value* rhs) {
    auto var = dynamic_cast<ast::VarName*>(node.get_lhs().get());
    if (!var)
        throw std::runtime_error("Error: only VarName assignment is supported!");

    llvm::Value* ptr = get_variable_ptr(*var);
    builder.CreateStore(rhs, ptr);
}

llvm::Value* CodegenLLVMVisitor::visit_logical_bin_op(llvm::Value* lhs,
                                                      llvm::Value* rhs,
                                                      unsigned op) {
    const auto& bin_op = static_cast<ast::BinaryOp>(op);
    return bin_op == ast::BinaryOp::BOP_AND ? builder.CreateAnd(lhs, rhs)
                                            : builder.CreateOr(lhs, rhs);
}

llvm::Value* CodegenLLVMVisitor::visit_comparison_bin_op(llvm::Value* lhs,
                                                         llvm::Value* rhs,
                                                         unsigned op) {
    const auto& bin_op = static_cast<ast::BinaryOp>(op);
    llvm::Type* lhs_type = lhs->getType();
    llvm::Value* result;

    switch (bin_op) {
#define DISPATCH(binary_op, i_llvm_op, f_llvm_op)            \
    case binary_op:                                          \
        if (lhs_type->isDoubleTy() || lhs_type->isFloatTy()) \
            result = f_llvm_op(lhs, rhs);                    \
        else                                                 \
            result = i_llvm_op(lhs, rhs);                    \
        return result;

        DISPATCH(ast::BinaryOp::BOP_EXACT_EQUAL, builder.CreateICmpEQ, builder.CreateFCmpOEQ);
        DISPATCH(ast::BinaryOp::BOP_GREATER, builder.CreateICmpSGT, builder.CreateFCmpOGT);
        DISPATCH(ast::BinaryOp::BOP_GREATER_EQUAL, builder.CreateICmpSGE, builder.CreateFCmpOGE);
        DISPATCH(ast::BinaryOp::BOP_LESS, builder.CreateICmpSLT, builder.CreateFCmpOLT);
        DISPATCH(ast::BinaryOp::BOP_LESS_EQUAL, builder.CreateICmpSLE, builder.CreateFCmpOLE);
        DISPATCH(ast::BinaryOp::BOP_NOT_EQUAL, builder.CreateICmpNE, builder.CreateFCmpONE);

#undef DISPATCH

    default:
        return nullptr;
    }
}

/****************************************************************************************/
/*                            Overloaded visitor routines                               */
/****************************************************************************************/


void CodegenLLVMVisitor::visit_binary_expression(const ast::BinaryExpression& node) {
    const auto& op = node.get_op().get_value();

    // Process rhs first, since lhs is handled differently for assignment and binary
    // operators.
    node.get_rhs()->accept(*this);
    llvm::Value* rhs = values.back();
    values.pop_back();
    if (op == ast::BinaryOp::BOP_ASSIGN) {
        visit_assign_op(node, rhs);
        return;
    }

    node.get_lhs()->accept(*this);
    llvm::Value* lhs = values.back();
    values.pop_back();

    llvm::Value* result;
    switch (op) {
    case ast::BOP_ADDITION:
    case ast::BOP_DIVISION:
    case ast::BOP_MULTIPLICATION:
    case ast::BOP_SUBTRACTION:
        result = visit_arithmetic_bin_op(lhs, rhs, op);
        break;
    case ast::BOP_AND:
    case ast::BOP_OR:
        result = visit_logical_bin_op(lhs, rhs, op);
        break;
    case ast::BOP_EXACT_EQUAL:
    case ast::BOP_GREATER:
    case ast::BOP_GREATER_EQUAL:
    case ast::BOP_LESS:
    case ast::BOP_LESS_EQUAL:
    case ast::BOP_NOT_EQUAL:
        result = visit_comparison_bin_op(lhs, rhs, op);
        break;
    default:
        throw std::runtime_error("Error: binary operator is not supported\n");
    }

    values.push_back(result);
}

void CodegenLLVMVisitor::visit_statement_block(const ast::StatementBlock& node) {
    const auto& statements = node.get_statements();
    for (const auto& statement: statements) {
        if (is_supported_statement(*statement))
            statement->accept(*this);
    }
}

void CodegenLLVMVisitor::visit_boolean(const ast::Boolean& node) {
    const auto& constant = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*context),
                                                  node.get_value());
    values.push_back(constant);
}

// Generating FOR loop in LLVM IR creates the following structure:
//
//  +---------------------------+
//  | <code before for loop>    |
//  | <for loop initialisation> |
//  | br %cond                  |
//  +---------------------------+
//                |
//                V
//  +-----------------------------+
//  | <condition code>            |
//  | %cond = ...                 |<------+
//  | cond_br %cond, %body, %exit |       |
//  +-----------------------------+       |
//      |                 |               |
//      |                 V               |
//      |     +------------------------+  |
//      |     | <body code>            |  |
//      |     | br %inc                |  |
//      |     +------------------------+  |
//      |                 |               |
//      |                 V               |
//      |     +------------------------+  |
//      |     | <increment code>       |  |
//      |      | br %cond              |  |
//      |     +------------------------+  |
//      |                 |               |
//      |                 +---------------+
//      V
//  +---------------------------+
//  | <code after for loop>     |
//  +---------------------------+
void CodegenLLVMVisitor::visit_codegen_for_statement(const ast::CodegenForStatement& node) {
    // Get the current and the next blocks within the function.
    llvm::BasicBlock* curr_block = builder.GetInsertBlock();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Create the basic blocks for FOR loop.
    llvm::BasicBlock* for_cond =
        llvm::BasicBlock::Create(*context, /*Name=*/"for.cond", func, next);
    llvm::BasicBlock* for_body =
        llvm::BasicBlock::Create(*context, /*Name=*/"for.body", func, next);
    llvm::BasicBlock* for_inc = llvm::BasicBlock::Create(*context, /*Name=*/"for.inc", func, next);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(*context, /*Name=*/"for.exit", func, next);

    // First, initialise the loop in the same basic block. This block is optional.
    if (node.get_initialization()) {
        node.get_initialization()->accept(*this);
    }

    // If the loop is to be vectorised, create a separate vector induction variable.
    // \todo: See the comment for `kernel_id_prefix`.
    if (vector_width > 1) {
        // First, create a vector type and alloca for it.
        llvm::Type* i32_type = llvm::Type::getInt32Ty(*context);
        llvm::Type* vec_type = llvm::FixedVectorType::get(i32_type, vector_width);
        llvm::Value* vec_alloca = builder.CreateAlloca(vec_type,
                                                       /*ArraySize=*/nullptr,
                                                       /*Name=*/kernel_id_prefix + kernel_id);

        // Then, store the initial value of <0, 1, ..., [W-1]> o the alloca pointer, where W is the
        // vector width.
        std::vector<llvm::Constant*> constants;
        for (unsigned i = 0; i < vector_width; ++i) {
            const auto& element = llvm::ConstantInt::get(i32_type, i);
            constants.push_back(element);
        }
        llvm::Value* vector_id = llvm::ConstantVector::get(constants);
        builder.CreateStore(vector_id, vec_alloca);
    }
    // Branch to condition basic block and insert condition code there.
    builder.CreateBr(for_cond);
    builder.SetInsertPoint(for_cond);
    node.get_condition()->accept(*this);

    // Extract the condition to decide whether to branch to the loop body or loop exit.
    llvm::Value* cond = values.back();
    values.pop_back();
    builder.CreateCondBr(cond, for_body, exit);

    // Generate code for the loop body and create the basic block for the increment.
    builder.SetInsertPoint(for_body);
    is_kernel_code = true;
    const auto& statement_block = node.get_statement_block();
    statement_block->accept(*this);
    is_kernel_code = false;
    builder.CreateBr(for_inc);

    // Process increment.
    builder.SetInsertPoint(for_inc);
    node.get_increment()->accept(*this);

    // If the code is vectorised, then increment the vector id by <W, W, ..., W> where W is the
    // vector width.
    // \todo: See the comment for `kernel_id_prefix`.
    if (vector_width > 1) {
        // First, create an increment vector.
        llvm::Value* vector_inc = get_constant_int_vector(vector_width);

        // Increment the kernel id elements by a constant vector width.
        llvm::Value* vector_id_ptr = lookup(kernel_id_prefix + kernel_id);
        llvm::Value* vector_id = builder.CreateLoad(vector_id_ptr);
        llvm::Value* incremented = builder.CreateAdd(vector_id, vector_inc);
        builder.CreateStore(incremented, vector_id_ptr);
    }

    // Create a branch to condition block, then generate exit code out of the loop.
    builder.CreateBr(for_cond);
    builder.SetInsertPoint(exit);
}


void CodegenLLVMVisitor::visit_codegen_function(const ast::CodegenFunction& node) {
    const auto& name = node.get_node_name();
    const auto& arguments = node.get_arguments();
    llvm::Function* func = module->getFunction(name);
    current_func = func;

    // Create the entry basic block of the function/procedure and point the local named values table
    // to the symbol table.
    llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, /*Name=*/"", func);
    builder.SetInsertPoint(body);

    // When processing a function, it returns a value named <function_name> in NMODL. Therefore, we
    // first run RenameVisitor to rename it into ret_<function_name>. This will aid in avoiding
    // symbolic conflicts.
    std::string return_var_name = "ret_" + name;
    const auto& block = node.get_statement_block();
    visitor::RenameVisitor v(name, return_var_name);
    block->accept(v);


    // Allocate parameters on the stack and add them to the symbol table.
    unsigned i = 0;
    for (auto& arg: func->args()) {
        std::string arg_name = arguments[i++].get()->get_node_name();
        llvm::Value* alloca = builder.CreateAlloca(arg.getType(), /*ArraySize=*/nullptr, arg_name);
        arg.setName(arg_name);
        builder.CreateStore(&arg, alloca);
    }

    // Process function or procedure body. The return statement is handled in a separate visitor.
    block->accept(*this);

    // If function has a void return type, add a terminator not handled by CodegenReturnVar.
    if (node.get_return_type()->get_type() == ast::AstNodeType::VOID)
        builder.CreateRetVoid();

    // Clear local values stack and remove the pointer to the local symbol table.
    values.clear();
    current_func = nullptr;
}

void CodegenLLVMVisitor::visit_codegen_return_statement(const ast::CodegenReturnStatement& node) {
    if (!node.get_statement()->is_name())
        throw std::runtime_error("Error: CodegenReturnStatement must contain a name node\n");

    std::string ret = "ret_" + current_func->getName().str();
    llvm::Value* ret_value = builder.CreateLoad(lookup(ret));
    builder.CreateRet(ret_value);
}

void CodegenLLVMVisitor::visit_codegen_var_list_statement(
    const ast::CodegenVarListStatement& node) {
    llvm::Type* scalar_var_type = get_codegen_var_type(*node.get_var_type());
    for (const auto& variable: node.get_variables()) {
        std::string name = variable->get_node_name();
        const auto& identifier = variable->get_name();
        // Local variable can be a scalar (Node AST class) or an array (IndexedName AST class). For
        // each case, create memory allocations with the corresponding LLVM type.
        llvm::Type* var_type;
        if (identifier->is_indexed_name()) {
            auto indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(identifier);
            int length = get_array_length(*indexed_name);
            var_type = llvm::ArrayType::get(scalar_var_type, length);
        } else if (identifier->is_name()) {
            // This case corresponds to a scalar local variable. Its type is double by default.
            var_type = scalar_var_type;
        } else {
            throw std::runtime_error("Error: Unsupported local variable type");
        }
        llvm::Value* alloca = builder.CreateAlloca(var_type, /*ArraySize=*/nullptr, name);
    }
}

void CodegenLLVMVisitor::visit_double(const ast::Double& node) {
    if (is_kernel_code && vector_width > 1) {
        values.push_back(get_constant_fp_vector(node.get_value()));
        return;
    }
    const auto& constant = llvm::ConstantFP::get(get_default_fp_type(), node.get_value());
    values.push_back(constant);
}

void CodegenLLVMVisitor::visit_function_block(const ast::FunctionBlock& node) {
    // do nothing. \todo: remove old function blocks from ast.
}

void CodegenLLVMVisitor::visit_function_call(const ast::FunctionCall& node) {
    const auto& name = node.get_node_name();
    auto func = module->getFunction(name);
    if (func) {
        create_function_call(func, name, node.get_arguments());
    } else {
        auto symbol = sym_tab->lookup(name);
        if (symbol && symbol->has_any_property(symtab::syminfo::NmodlType::extern_method)) {
            create_external_method_call(name, node.get_arguments());
        } else {
            throw std::runtime_error("Error: Unknown function name: " + name +
                                     ". (External functions references are not supported)");
        }
    }
}

void CodegenLLVMVisitor::visit_if_statement(const ast::IfStatement& node) {
    // Get the current and the next blocks within the function.
    llvm::BasicBlock* curr_block = builder.GetInsertBlock();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Add a true block and a merge block where the control flow merges.
    llvm::BasicBlock* true_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);

    // Add condition to the current block.
    node.get_condition()->accept(*this);
    llvm::Value* cond = values.back();
    values.pop_back();

    // Process the true block.
    builder.SetInsertPoint(true_block);
    node.get_statement_block()->accept(*this);
    builder.CreateBr(merge_block);

    // Save the merge block and proceed with codegen for `else if` statements.
    llvm::BasicBlock* exit = merge_block;
    for (const auto& else_if: node.get_elseifs()) {
        // Link the current block to the true and else blocks.
        llvm::BasicBlock* else_block =
            llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        builder.SetInsertPoint(curr_block);
        builder.CreateCondBr(cond, true_block, else_block);

        // Process else block.
        builder.SetInsertPoint(else_block);
        else_if->get_condition()->accept(*this);
        cond = values.back();
        values.pop_back();

        // Reassign true and merge blocks respectively. Note that the new merge block has to be
        // connected to the old merge block (tmp).
        true_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        llvm::BasicBlock* tmp = merge_block;
        merge_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        builder.SetInsertPoint(merge_block);
        builder.CreateBr(tmp);

        // Process true block.
        builder.SetInsertPoint(true_block);
        else_if->get_statement_block()->accept(*this);
        builder.CreateBr(merge_block);
        curr_block = else_block;
    }

    // Finally, generate code for `else` statement if it exists.
    const auto& elses = node.get_elses();
    llvm::BasicBlock* else_block;
    if (elses) {
        else_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        builder.SetInsertPoint(else_block);
        elses->get_statement_block()->accept(*this);
        builder.CreateBr(merge_block);
    } else {
        else_block = merge_block;
    }
    builder.SetInsertPoint(curr_block);
    builder.CreateCondBr(cond, true_block, else_block);
    builder.SetInsertPoint(exit);
}

void CodegenLLVMVisitor::visit_integer(const ast::Integer& node) {
    if (is_kernel_code && vector_width > 1) {
        values.push_back(get_constant_int_vector(node.get_value()));
        return;
    }
    const auto& constant = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context),
                                                  node.get_value());
    values.push_back(constant);
}

void CodegenLLVMVisitor::visit_program(const ast::Program& node) {
    // Before generating LLVM:
    //   - convert function and procedure blocks into CodegenFunctions
    //   - gather information about AST. For now, information about functions
    //     and procedures is used only.
    CodegenLLVMHelperVisitor v{vector_width};
    const auto& functions = v.get_codegen_functions(node);
    instance_var_helper = v.get_instance_var_helper();

    kernel_id = v.get_kernel_id();

    // For every function, generate its declaration. Thus, we can look up
    // `llvm::Function` in the symbol table in the module.
    for (const auto& func: functions) {
        emit_procedure_or_function_declaration(*func);
    }

    // Set the AST symbol table.
    sym_tab = node.get_symbol_table();

    // Proceed with code generation. Right now, we do not do
    //     node.visit_children(*this);
    // The reason is that the node may contain AST nodes for which the visitor functions have been
    // defined. In our implementation we assume that the code generation is happening within the
    // function scope. To avoid generating code outside of functions, visit only them for now.
    // \todo: Handle what is mentioned here.
    for (const auto& func: functions) {
        visit_codegen_function(*func);
    }

    if (opt_passes) {
        logger->info("Running LLVM optimisation passes");
        run_llvm_opt_passes();
    }

    // Keep this for easier development (maybe move to debug mode later).
    std::cout << print_module();
}

void CodegenLLVMVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    // do nothing. \todo: remove old procedures from ast.
}

void CodegenLLVMVisitor::visit_unary_expression(const ast::UnaryExpression& node) {
    ast::UnaryOp op = node.get_op().get_value();
    node.get_expression()->accept(*this);
    llvm::Value* value = values.back();
    values.pop_back();
    if (op == ast::UOP_NEGATION) {
        values.push_back(builder.CreateFNeg(value));
    } else if (op == ast::UOP_NOT) {
        values.push_back(builder.CreateNot(value));
    } else {
        throw std::runtime_error("Error: unsupported unary operator\n");
    }
}

void CodegenLLVMVisitor::visit_var_name(const ast::VarName& node) {
    llvm::Value* ptr = get_variable_ptr(node);

    // Finally, load the variable from the pointer value.
    llvm::Value* var = builder.CreateLoad(ptr);

    // If the vale should not be vectorised, or it is already a vector, add it to the stack.
    if (!is_kernel_code || vector_width <= 1 || var->getType()->isVectorTy()) {
        values.push_back(var);
        return;
    }

    // Otherwise, if we are generating vectorised inside the loop, replicate the value to form a
    // vector of `vector_width`.
    llvm::Value* vector_var = builder.CreateVectorSplat(vector_width, var);
    values.push_back(vector_var);
}

void CodegenLLVMVisitor::visit_while_statement(const ast::WhileStatement& node) {
    // Get the current and the next blocks within the function.
    llvm::BasicBlock* curr_block = builder.GetInsertBlock();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Add a header and the body blocks.
    llvm::BasicBlock* header = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);

    builder.CreateBr(header);
    builder.SetInsertPoint(header);

    // Generate code for condition and create branch to the body block.
    node.get_condition()->accept(*this);
    llvm::Value* condition = values.back();
    values.pop_back();
    builder.CreateCondBr(condition, body, exit);

    builder.SetInsertPoint(body);
    node.get_statement_block()->accept(*this);
    builder.CreateBr(header);

    builder.SetInsertPoint(exit);
}

}  // namespace codegen
}  // namespace nmodl
