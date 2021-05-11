/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/codegen_llvm_visitor.hpp"

#include "ast/all.hpp"
#include "visitors/rename_visitor.hpp"
#include "visitors/visitor_utils.hpp"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ToolOutputFile.h"

#ifndef LLVM_VERSION_LESS_THAN_13
#include "llvm/CodeGen/ReplaceWithVeclib.h"
#endif

namespace nmodl {
namespace codegen {


static constexpr const char instance_struct_type_name[] = "__instance_var__type";


/****************************************************************************************/
/*                            Helper routines                                           */
/****************************************************************************************/

/// A utility to check for supported Statement AST nodes.
static bool is_supported_statement(const ast::Statement& statement) {
    return statement.is_codegen_var_list_statement() || statement.is_expression_statement() ||
           statement.is_codegen_for_statement() || statement.is_codegen_return_statement() ||
           statement.is_if_statement() || statement.is_while_statement();
}

/// A utility to check of the kernel body can be vectorised.
static bool can_vectorise(const ast::CodegenForStatement& statement, symtab::SymbolTable* sym_tab) {
    // Check that function calls are made to external methods only.
    const auto& function_calls = collect_nodes(statement, {ast::AstNodeType::FUNCTION_CALL});
    for (const auto& call: function_calls) {
        const auto& name = call->get_node_name();
        auto symbol = sym_tab->lookup(name);
        if (symbol && !symbol->has_any_property(symtab::syminfo::NmodlType::extern_method))
            return false;
    }

    // Check there is no control flow in the kernel.
    const std::vector<ast::AstNodeType> unsupported_nodes = {ast::AstNodeType::IF_STATEMENT};
    const auto& collected = collect_nodes(statement, unsupported_nodes);

    return collected.empty();
}

llvm::Value* CodegenLLVMVisitor::codegen_indexed_name(const ast::IndexedName& node) {
    llvm::Value* index = get_array_index(node);
    return ir_builder.create_inbounds_gep(node.get_node_name(), index);
}

llvm::Value* CodegenLLVMVisitor::codegen_instance_var(const ast::CodegenInstanceVar& node) {
    const auto& member_node = node.get_member_var();
    const auto& instance_name = node.get_instance_var()->get_node_name();
    const auto& member_name = member_node->get_node_name();

    if (!instance_var_helper.is_an_instance_variable(member_name))
        throw std::runtime_error("Error: " + member_name + " is not a member of the instance!");

    // Load the instance struct given its name from the ValueSymbolTable.
    llvm::Value* instance_ptr = ir_builder.builder.CreateLoad(lookup(instance_name));

    // Create a GEP instruction to get a pointer to the member.
    int member_index = instance_var_helper.get_variable_index(member_name);
    llvm::Type* index_type = llvm::Type::getInt32Ty(*context);

    std::vector<llvm::Value*> indices;
    indices.push_back(llvm::ConstantInt::get(index_type, 0));
    indices.push_back(llvm::ConstantInt::get(index_type, member_index));
    llvm::Value* member_ptr = ir_builder.builder.CreateInBoundsGEP(instance_ptr, indices);

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

    if (!member_indexed_name->get_length()->is_name())
        throw std::runtime_error("Error: " + member_name + " must be indexed with a variable!");

    llvm::Value* i64_index = get_array_index(*member_indexed_name);

    // The codegen variable type is always a scalar, so we need to transform it to a pointer. Then
    // load the member which would be indexed later.
    llvm::Type* type = get_codegen_var_type(*codegen_var_with_type->get_type());
    llvm::Value* instance_member =
        ir_builder.builder.CreateLoad(llvm::PointerType::get(type, /*AddressSpace=*/0), member_ptr);

    // Check if the code is vectorised and the index is indirect.
    std::string id = member_indexed_name->get_length()->get_node_name();
    if (id != kernel_id && is_kernel_code && vector_width > 1) {
        // Calculate a vector of addresses via GEP instruction, and then created a gather to load
        // indirectly.
        llvm::Value* addresses = ir_builder.builder.CreateInBoundsGEP(instance_member, {i64_index});
        return ir_builder.builder.CreateMaskedGather(addresses, llvm::Align());
    }

    llvm::Value* member_addr = ir_builder.builder.CreateInBoundsGEP(instance_member, {i64_index});

    // If the code is vectorised, then bitcast to a vector pointer.
    if (is_kernel_code && vector_width > 1) {
        llvm::Type* vector_type =
            llvm::PointerType::get(llvm::FixedVectorType::get(type, vector_width),
                                   /*AddressSpace=*/0);
        return ir_builder.builder.CreateBitCast(member_addr, vector_type);
    }
    return member_addr;
}

llvm::Value* CodegenLLVMVisitor::get_array_index(const ast::IndexedName& node) {
    // Process the index expression. It can either be a Name node:
    //    k[id]     // id is an integer
    // or an integer expression.
    llvm::Value* index_value;
    if (node.get_length()->is_name()) {
        llvm::Value* ptr = lookup(node.get_length()->get_node_name());
        index_value = ir_builder.builder.CreateLoad(ptr);
    } else {
        index_value = accept_and_get(node.get_length());
    }

    // Check if index is a double. While it is possible to use casting from double to integer
    // values, we choose not to support these cases.
    if (!index_value->getType()->isIntOrIntVectorTy())
        throw std::runtime_error("Error: only integer indexing is supported!");

    // Conventionally, in LLVM array indices are 64 bit.
    llvm::Type* i64_type = llvm::Type::getInt64Ty(*context);
    if (auto index_type = llvm::dyn_cast<llvm::IntegerType>(index_value->getType())) {
        if (index_type->getBitWidth() == i64_type->getIntegerBitWidth())
            return index_value;
        return ir_builder.builder.CreateSExtOrTrunc(index_value, i64_type);
    }

    auto vector_type = llvm::cast<llvm::FixedVectorType>(index_value->getType());
    auto element_type = llvm::cast<llvm::IntegerType>(vector_type->getElementType());
    if (element_type->getBitWidth() == i64_type->getIntegerBitWidth())
        return index_value;
    return ir_builder.builder.CreateSExtOrTrunc(index_value,
                                                llvm::FixedVectorType::get(i64_type, vector_width));
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
        return ir_builder.get_boolean_type();
    case ast::AstNodeType::DOUBLE:
        return ir_builder.get_fp_type();
    case ast::AstNodeType::INSTANCE_STRUCT:
        return get_instance_struct_type();
    case ast::AstNodeType::INTEGER:
        return ir_builder.get_i32_type();
    case ast::AstNodeType::VOID:
        return ir_builder.get_void_type();
    default:
        throw std::runtime_error("Error: expecting a type in CodegenVarType node\n");
    }
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
    TypeVector member_types;
    for (const auto& variable: instance_var_helper.instance->get_codegen_vars()) {
        auto is_pointer = variable->get_is_pointer();
        auto nmodl_type = variable->get_type()->get_type();

        llvm::Type* i32_type = llvm::Type::getInt32Ty(*context);
        llvm::Type* i32ptr_type = llvm::Type::getInt32PtrTy(*context);

        switch (nmodl_type) {
#define DISPATCH(type, llvm_ptr_type, llvm_type)                            \
    case type:                                                              \
        member_types.push_back(is_pointer ? (llvm_ptr_type) : (llvm_type)); \
        break;

            DISPATCH(ast::AstNodeType::DOUBLE, get_default_fp_ptr_type(), get_default_fp_type());
            DISPATCH(ast::AstNodeType::INTEGER, i32ptr_type, i32_type);

#undef DISPATCH
        default:
            throw std::runtime_error("Error: unsupported type found in instance struct");
        }
    }

    return ir_builder.get_struct_type(mod_filename + instance_struct_type_name, member_types);
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

void CodegenLLVMVisitor::run_ir_opt_passes() {
    /// run some common optimisation passes that are commonly suggested
    opt_pm.add(llvm::createInstructionCombiningPass());
    opt_pm.add(llvm::createReassociatePass());
    opt_pm.add(llvm::createGVNPass());
    opt_pm.add(llvm::createCFGSimplificationPass());

    /// initialize pass manager
    opt_pm.doInitialization();

    /// iterate over all functions and run the optimisation passes
    auto& functions = module->getFunctionList();
    for (auto& function: functions) {
        llvm::verifyFunction(function);
        opt_pm.run(function);
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
        llvm::Value* value = accept_and_get(arg);
        llvm::Type* type = value->getType();
        argument_types.push_back(type);
        argument_values.push_back(value);
    }

#define DISPATCH(method_name, intrinsic)                                                    \
    if (name == (method_name)) {                                                            \
        llvm::Value* result =                                                               \
            ir_builder.builder.CreateIntrinsic(intrinsic, argument_types, argument_values); \
        ir_builder.value_stack.push_back(result);                                           \
        return;                                                                             \
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
    llvm::Value* call = ir_builder.builder.CreateCall(func, argument_values);
    ir_builder.value_stack.push_back(call);
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
    ir_builder.builder.CreateCall(printf, argument_values);
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
    auto func =
        llvm::Function::Create(llvm::FunctionType::get(return_type, arg_types, /*isVarArg=*/false),
                               llvm::Function::ExternalLinkage,
                               name,
                               *module);

    // Add function debug information, with location information if it exists.
    if (add_debug_information) {
        if (node.get_token()) {
            Location loc{node.get_token()->start_line(), node.get_token()->start_column()};
            debug_builder.add_function_debug_info(func, &loc);
        } else {
            debug_builder.add_function_debug_info(func);
        }
    }
}

llvm::Value* CodegenLLVMVisitor::lookup(const std::string& name) {
    auto val = ir_builder.current_function->getValueSymbolTable()->lookup(name);
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
            llvm::Value* str = ir_builder.builder.CreateGlobalStringPtr(string_arg->get_value());
            arg_values.push_back(str);
        } else {
            llvm::Value* value = accept_and_get(arg);
            arg_values.push_back(value);
        }
    }
}

void CodegenLLVMVisitor::visit_assign_op(const ast::BinaryExpression& node, llvm::Value* rhs) {
    auto var = dynamic_cast<ast::VarName*>(node.get_lhs().get());
    if (!var)
        throw std::runtime_error("Error: only VarName assignment is supported!");

    llvm::Value* ptr = get_variable_ptr(*var);
    ir_builder.builder.CreateStore(rhs, ptr);
}

/****************************************************************************************/
/*                            Overloaded visitor routines                               */
/****************************************************************************************/


void CodegenLLVMVisitor::visit_binary_expression(const ast::BinaryExpression& node) {
    const auto& op = node.get_op().get_value();

    // Process rhs first, since lhs is handled differently for assignment and binary
    // operators.
    llvm::Value* rhs = accept_and_get(node.get_rhs());
    if (op == ast::BinaryOp::BOP_ASSIGN) {
        visit_assign_op(node, rhs);
        return;
    }

    llvm::Value* lhs = accept_and_get(node.get_lhs());
    ir_builder.create_binary_op(lhs, rhs, op);
}

void CodegenLLVMVisitor::visit_statement_block(const ast::StatementBlock& node) {
    const auto& statements = node.get_statements();
    for (const auto& statement: statements) {
        if (is_supported_statement(*statement))
            statement->accept(*this);
    }
}

void CodegenLLVMVisitor::visit_boolean(const ast::Boolean& node) {
    ir_builder.create_boolean_constant(node.get_value());
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
    // Disable vector code generation for condition and increment blocks.
    is_kernel_code = false;
    ir_builder.stop_vectorization();

    // Get the current and the next blocks within the function.
    llvm::BasicBlock* curr_block = ir_builder.builder.GetInsertBlock();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Create the basic blocks for FOR loop.
    llvm::BasicBlock* for_cond =
        llvm::BasicBlock::Create(*context, /*Name=*/"for.cond", func, next);
    llvm::BasicBlock* for_body =
        llvm::BasicBlock::Create(*context, /*Name=*/"for.body", func, next);
    llvm::BasicBlock* for_inc = llvm::BasicBlock::Create(*context, /*Name=*/"for.inc", func, next);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(*context, /*Name=*/"for.exit", func, next);

    // Save the vector width.
    int tmp_vector_width = vector_width;

    // Check if the kernel can be vectorised. If not, generate scalar code.
    if (!can_vectorise(node, sym_tab)) {
        logger->info("Cannot vectorise the for loop in '" +
                     ir_builder.current_function->getName().str() + "'");
        logger->info("Generating scalar code...");
        vector_width = 1;
    }

    // First, initialise the loop in the same basic block. This block is optional. Also, reset
    // vector width to 1 if processing the remainder of the loop.
    if (node.get_initialization()) {
        node.get_initialization()->accept(*this);
    } else {
        vector_width = 1;
    }

    // Branch to condition basic block and insert condition code there.
    ir_builder.builder.CreateBr(for_cond);
    ir_builder.builder.SetInsertPoint(for_cond);

    // Extract the condition to decide whether to branch to the loop body or loop exit.
    llvm::Value* cond = accept_and_get(node.get_condition());
    ir_builder.builder.CreateCondBr(cond, for_body, exit);

    // Generate code for the loop body and create the basic block for the increment.
    ir_builder.builder.SetInsertPoint(for_body);
    is_kernel_code = true;
    ir_builder.start_vectorization();
    const auto& statement_block = node.get_statement_block();
    statement_block->accept(*this);
    is_kernel_code = false;
    ir_builder.stop_vectorization();
    ir_builder.builder.CreateBr(for_inc);

    // Process increment.
    ir_builder.builder.SetInsertPoint(for_inc);
    node.get_increment()->accept(*this);

    // Create a branch to condition block, then generate exit code out of the loop. Restore the
    // vector width.
    ir_builder.builder.CreateBr(for_cond);
    ir_builder.builder.SetInsertPoint(exit);
    vector_width = tmp_vector_width;
    is_kernel_code = true;
    ir_builder.start_vectorization();
}


void CodegenLLVMVisitor::visit_codegen_function(const ast::CodegenFunction& node) {
    const auto& name = node.get_node_name();
    const auto& arguments = node.get_arguments();
    llvm::Function* func = module->getFunction(name);
    ir_builder.current_function = func;

    // Create the entry basic block of the function/procedure and point the local named values table
    // to the symbol table.
    llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, /*Name=*/"", func);
    ir_builder.builder.SetInsertPoint(body);

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
        llvm::Type* arg_type = arg.getType();
        llvm::Value* alloca =
            ir_builder.builder.CreateAlloca(arg_type, /*ArraySize=*/nullptr, arg_name);
        arg.setName(arg_name);
        ir_builder.builder.CreateStore(&arg, alloca);
    }

    // Process function or procedure body. If the function is a compute kernel, then set the
    // corresponding flags. The return statement is handled in a separate visitor.
    bool has_void_ret_type = node.get_return_type()->get_type() == ast::AstNodeType::VOID;
    if (has_void_ret_type) {
        is_kernel_code = true;
        ir_builder.start_vectorization();
        block->accept(*this);
        is_kernel_code = false;
        ir_builder.stop_vectorization();
    } else {
        block->accept(*this);
    }

    // If function has a void return type, add a terminator not handled by CodegenReturnVar.
    if (has_void_ret_type)
        ir_builder.builder.CreateRetVoid();

    // Clear local values stack and remove the pointer to the local symbol table.
    ir_builder.value_stack.clear();
    ir_builder.current_function = nullptr;
}

void CodegenLLVMVisitor::visit_codegen_return_statement(const ast::CodegenReturnStatement& node) {
    if (!node.get_statement()->is_name())
        throw std::runtime_error("Error: CodegenReturnStatement must contain a name node\n");

    std::string ret = "ret_" + ir_builder.current_function->getName().str();
    llvm::Value* ret_value = ir_builder.builder.CreateLoad(lookup(ret));
    ir_builder.builder.CreateRet(ret_value);
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
            // This case corresponds to a scalar or vector local variable.
            const auto& identifier_name = identifier->get_node_name();

            // Even if generating vectorised code, some variables still need to be scalar.
            // Particularly, the induction variable "id" and remainder loop variables (that start
            // with "epilogue").
            if (is_kernel_code && vector_width > 1 && identifier_name != kernel_id &&
                identifier_name.rfind("epilogue", 0)) {
                var_type = llvm::FixedVectorType::get(scalar_var_type, vector_width);
            } else {
                var_type = scalar_var_type;
            }
        } else {
            throw std::runtime_error("Error: Unsupported local variable type");
        }
        ir_builder.builder.CreateAlloca(var_type, /*ArraySize=*/nullptr, name);
    }
}

void CodegenLLVMVisitor::visit_double(const ast::Double& node) {
    ir_builder.create_fp_constant(node.get_value());
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
    llvm::BasicBlock* curr_block = ir_builder.builder.GetInsertBlock();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Add a true block and a merge block where the control flow merges.
    llvm::BasicBlock* true_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);

    // Add condition to the current block.
    llvm::Value* cond = accept_and_get(node.get_condition());

    // Process the true block.
    ir_builder.builder.SetInsertPoint(true_block);
    node.get_statement_block()->accept(*this);
    ir_builder.builder.CreateBr(merge_block);

    // Save the merge block and proceed with codegen for `else if` statements.
    llvm::BasicBlock* exit = merge_block;
    for (const auto& else_if: node.get_elseifs()) {
        // Link the current block to the true and else blocks.
        llvm::BasicBlock* else_block =
            llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        ir_builder.builder.SetInsertPoint(curr_block);
        ir_builder.builder.CreateCondBr(cond, true_block, else_block);

        // Process else block.
        ir_builder.builder.SetInsertPoint(else_block);
        cond = accept_and_get(else_if->get_condition());

        // Reassign true and merge blocks respectively. Note that the new merge block has to be
        // connected to the old merge block (tmp).
        true_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        llvm::BasicBlock* tmp = merge_block;
        merge_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        ir_builder.builder.SetInsertPoint(merge_block);
        ir_builder.builder.CreateBr(tmp);

        // Process true block.
        ir_builder.builder.SetInsertPoint(true_block);
        else_if->get_statement_block()->accept(*this);
        ir_builder.builder.CreateBr(merge_block);
        curr_block = else_block;
    }

    // Finally, generate code for `else` statement if it exists.
    const auto& elses = node.get_elses();
    llvm::BasicBlock* else_block;
    if (elses) {
        else_block = llvm::BasicBlock::Create(*context, /*Name=*/"", func, merge_block);
        ir_builder.builder.SetInsertPoint(else_block);
        elses->get_statement_block()->accept(*this);
        ir_builder.builder.CreateBr(merge_block);
    } else {
        else_block = merge_block;
    }
    ir_builder.builder.SetInsertPoint(curr_block);
    ir_builder.builder.CreateCondBr(cond, true_block, else_block);
    ir_builder.builder.SetInsertPoint(exit);
}

void CodegenLLVMVisitor::visit_integer(const ast::Integer& node) {
    ir_builder.create_i32_constant(node.get_value());
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

    // Set the AST symbol table.
    sym_tab = node.get_symbol_table();

    ir_builder.initialize(*sym_tab, kernel_id);

    // Create compile unit if adding debug information to the module.
    if (add_debug_information) {
        debug_builder.create_compile_unit(*module, module->getModuleIdentifier(), output_dir);
    }

    // For every function, generate its declaration. Thus, we can look up
    // `llvm::Function` in the symbol table in the module.
    for (const auto& func: functions) {
        emit_procedure_or_function_declaration(*func);
    }

    // Proceed with code generation. Right now, we do not do
    //     node.visit_children(*this);
    // The reason is that the node may contain AST nodes for which the visitor functions have been
    // defined. In our implementation we assume that the code generation is happening within the
    // function scope. To avoid generating code outside of functions, visit only them for now.
    // \todo: Handle what is mentioned here.
    for (const auto& func: functions) {
        visit_codegen_function(*func);
    }

    // Finalize the debug information.
    if (add_debug_information) {
        debug_builder.finalize();
    }

    // Verify the generated LLVM IR module.
    std::string error;
    llvm::raw_string_ostream ostream(error);
    if (verifyModule(*module, &ostream)) {
        throw std::runtime_error("Error: incorrect IR has been generated!\n" + ostream.str());
    }

    if (opt_passes) {
        logger->info("Running LLVM optimisation passes");
        run_ir_opt_passes();
    }

    // Optionally, replace LLVM's maths intrinsics with vector library calls.
    if (vector_width > 1 && vector_library != llvm::TargetLibraryInfoImpl::NoLibrary) {
#ifdef LLVM_VERSION_LESS_THAN_13
        logger->warn(
            "This version of LLVM does not support replacement of LLVM intrinsics with vector "
            "library calls");
#else
        // First, get the target library information.
        llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
        llvm::TargetLibraryInfoImpl target_lib_info = llvm::TargetLibraryInfoImpl(triple);

        // Populate target library information with vectorisable functions. Since libmvec is
        // supported for x86_64 only, have a check to catch other architectures.
        if (vector_library != llvm::TargetLibraryInfoImpl::LIBMVEC_X86 ||
            (triple.isX86() && triple.isArch64Bit())) {
            target_lib_info.addVectorizableFunctionsFromVecLib(vector_library);
        }

        // Run the codegen optimisation passes that replace maths intrinsics.
        codegen_pm.add(new llvm::TargetLibraryInfoWrapperPass(target_lib_info));
        codegen_pm.add(new llvm::ReplaceWithVeclibLegacy);
        codegen_pm.doInitialization();
        for (auto& function: module->getFunctionList()) {
            if (!function.isDeclaration())
                codegen_pm.run(function);
        }
        codegen_pm.doFinalization();
#endif
    }

    // If the output directory is specified, save the IR to .ll file.
    // \todo: Consider saving the generated LLVM IR to bytecode (.bc) file instead.
    if (output_dir != ".") {
        std::error_code error_code;
        std::unique_ptr<llvm::ToolOutputFile> out = std::make_unique<llvm::ToolOutputFile>(
            output_dir + "/" + mod_filename + ".ll", error_code, llvm::sys::fs::OF_Text);
        if (error_code)
            throw std::runtime_error("Error: " + error_code.message());

        std::unique_ptr<llvm::AssemblyAnnotationWriter> annotator;
        module->print(out->os(), annotator.get());
        out->keep();
    }

    logger->debug("Dumping generated IR...\n" + dump_module());
}

void CodegenLLVMVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    // do nothing. \todo: remove old procedures from ast.
}

void CodegenLLVMVisitor::visit_unary_expression(const ast::UnaryExpression& node) {
    ast::UnaryOp op = node.get_op().get_value();
    llvm::Value* value = accept_and_get(node.get_expression());
    ir_builder.create_unary_op(value, op);
}

void CodegenLLVMVisitor::visit_var_name(const ast::VarName& node) {
    llvm::Value* ptr = get_variable_ptr(node);

    // Finally, load the variable from the pointer value unless it has already been loaded (e.g. via
    // gather instruction).
    llvm::Value* var = ptr->getType()->isPointerTy() ? ir_builder.builder.CreateLoad(ptr) : ptr;

    // If the value should not be vectorised, or it is already a vector, add it to the stack.
    if (!is_kernel_code || vector_width <= 1 || var->getType()->isVectorTy()) {
        ir_builder.value_stack.push_back(var);
        return;
    }

    // Otherwise, if we are generating vectorised inside the loop, replicate the value to form a
    // vector of `vector_width`.
    llvm::Value* vector_var = ir_builder.builder.CreateVectorSplat(vector_width, var);
    ir_builder.value_stack.push_back(vector_var);
}

void CodegenLLVMVisitor::visit_while_statement(const ast::WhileStatement& node) {
    // Get the current and the next blocks within the function.
    llvm::BasicBlock* curr_block = ir_builder.builder.GetInsertBlock();
    llvm::BasicBlock* next = curr_block->getNextNode();
    llvm::Function* func = curr_block->getParent();

    // Add a header and the body blocks.
    llvm::BasicBlock* header = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(*context, /*Name=*/"", func, next);

    ir_builder.builder.CreateBr(header);
    ir_builder.builder.SetInsertPoint(header);

    // Generate code for condition and create branch to the body block.
    llvm::Value* condition = accept_and_get(node.get_condition());
    ir_builder.builder.CreateCondBr(condition, body, exit);

    ir_builder.builder.SetInsertPoint(body);
    node.get_statement_block()->accept(*this);
    ir_builder.builder.CreateBr(header);

    ir_builder.builder.SetInsertPoint(exit);
}

void CodegenLLVMVisitor::find_kernel_names(std::vector<std::string>& container) {
    // By convention, only the kernel functions return void type.
    const auto& functions = module->getFunctionList();
    for (const auto& func: functions) {
        if (func.getReturnType()->isVoidTy()) {
            container.push_back(func.getName().str());
        }
    }
}

void CodegenLLVMVisitor::wrap_kernel_functions() {
    // First, identify all kernels.
    std::vector<std::string> kernel_names;
    find_kernel_names(kernel_names);

    for (const auto& kernel_name: kernel_names) {
        // Get the kernel function and the instance struct type.
        auto kernel = module->getFunction(kernel_name);
        if (!kernel)
            throw std::runtime_error("Kernel " + kernel_name + " is not found!");

        if (std::distance(kernel->args().begin(), kernel->args().end()) != 1)
            throw std::runtime_error("Kernel " + kernel_name + " must have a single argument!");

        auto instance_struct_ptr_type = llvm::dyn_cast<llvm::PointerType>(
            kernel->getArg(0)->getType());
        if (!instance_struct_ptr_type)
            throw std::runtime_error("Kernel " + kernel_name +
                                     " does not have an instance struct pointer argument!");

        // Create a wrapper void function that takes a void pointer as a single argument.
        llvm::Type* i32_type = llvm::Type::getInt32Ty(*context);
        llvm::Type* void_ptr_type = llvm::Type::getInt8PtrTy(*context);
        llvm::Function* wrapper_func = llvm::Function::Create(
            llvm::FunctionType::get(i32_type, {void_ptr_type}, /*isVarArg=*/false),
            llvm::Function::ExternalLinkage,
            "__" + kernel_name + "_wrapper",
            *module);

        // Optionally, add debug information for the wrapper function.
        if (add_debug_information) {
            debug_builder.add_function_debug_info(wrapper_func);
        }

        llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, /*Name=*/"", wrapper_func);
        ir_builder.builder.SetInsertPoint(body);

        // Proceed with bitcasting the void pointer to the struct pointer type, calling the kernel
        // and adding a terminator.
        llvm::Value* bitcasted = ir_builder.builder.CreateBitCast(wrapper_func->getArg(0),
                                                                  instance_struct_ptr_type);
        std::vector<llvm::Value*> args;
        args.push_back(bitcasted);
        ir_builder.builder.CreateCall(kernel, args);
        ir_builder.builder.CreateRet(llvm::ConstantInt::get(i32_type, 0));
    }
}

llvm::Value* CodegenLLVMVisitor::accept_and_get(const std::shared_ptr<ast::Node>& node) {
    node->accept(*this);
    return ir_builder.pop_last_value();
}

}  // namespace codegen
}  // namespace nmodl
