/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"

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


/****************************************************************************************/
/*                            Helper routines                                           */
/****************************************************************************************/

static bool is_supported_statement(const ast::Statement& statement) {
    return statement.is_codegen_var_list_statement() || statement.is_expression_statement() ||
           statement.is_codegen_return_statement() || statement.is_if_statement() ||
           statement.is_while_statement();
}

bool CodegenLLVMVisitor::check_array_bounds(const ast::IndexedName& node, unsigned index) {
    llvm::Type* array_type = lookup(node.get_node_name())->getType()->getPointerElementType();
    unsigned length = array_type->getArrayNumElements();
    return 0 <= index && index < length;
}

llvm::Value* CodegenLLVMVisitor::create_gep(const std::string& name, unsigned index) {
    llvm::Type* index_type = llvm::Type::getInt32Ty(*context);
    std::vector<llvm::Value*> indices;
    indices.push_back(llvm::ConstantInt::get(index_type, 0));
    indices.push_back(llvm::ConstantInt::get(index_type, index));

    return builder.CreateInBoundsGEP(lookup(name), indices);
}

llvm::Value* CodegenLLVMVisitor::codegen_indexed_name(const ast::IndexedName& node) {
    unsigned index = get_array_index_or_length(node);

    // Check if index is within array bounds.
    if (!check_array_bounds(node, index))
        throw std::runtime_error("Error: Index is out of bounds");

    return create_gep(node.get_node_name(), index);
}

unsigned CodegenLLVMVisitor::get_array_index_or_length(const ast::IndexedName& indexed_name) {
    auto integer = std::dynamic_pointer_cast<ast::Integer>(indexed_name.get_length());
    if (!integer)
        throw std::runtime_error("Error: expecting integer index or length");

    // Check if integer value is taken from a macro.
    if (!integer->get_macro())
        return integer->get_value();
    const auto& macro = sym_tab->lookup(integer->get_macro()->get_node_name());
    return static_cast<unsigned>(*macro->get_value());
}

llvm::Type* CodegenLLVMVisitor::get_codegen_var_type(const ast::CodegenVarType& node) {
    switch (node.get_type()) {
    case ast::AstNodeType::BOOLEAN:
        return llvm::Type::getInt1Ty(*context);
    case ast::AstNodeType::DOUBLE:
        return get_default_fp_type();
    case ast::AstNodeType::INTEGER:
        return llvm::Type::getInt32Ty(*context);
    case ast::AstNodeType::VOID:
        return llvm::Type::getVoidTy(*context);
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
    if (name == method_name) {                                                                     \
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
#define DISPATCH(binary_op, llvm_fp_op, llvm_int_op)         \
    case binary_op:                                          \
        if (lhs_type->isDoubleTy() || lhs_type->isFloatTy()) \
            result = llvm_fp_op(lhs, rhs);                   \
        else                                                 \
            result = llvm_int_op(lhs, rhs);                  \
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
    if (!var) {
        throw std::runtime_error("Error: only VarName assignment is currently supported.\n");
    }

    const auto& identifier = var->get_name();
    if (identifier->is_name()) {
        llvm::Value* alloca = lookup(var->get_node_name());
        builder.CreateStore(rhs, alloca);
    } else if (identifier->is_indexed_name()) {
        auto indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(identifier);
        builder.CreateStore(rhs, codegen_indexed_name(*indexed_name));
    } else {
        throw std::runtime_error("Error: Unsupported variable type");
    }
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
    if (node.is_void())
        builder.CreateRetVoid();

    // Clear local values stack and remove the pointer to the local symbol table.
    values.clear();
    current_func = nullptr;
}

void CodegenLLVMVisitor::visit_codegen_return_statement(const ast::CodegenReturnStatement& node) {
    if (!node.get_statement()->is_name())
        throw std::runtime_error("Error: CodegenReturnStatement must contain a name node\n");

    std::string ret = "ret_" + current_func->getName().str();
    llvm::Value* ret_value = builder.CreateLoad(current_func->getValueSymbolTable()->lookup(ret));
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
            unsigned length = get_array_index_or_length(*indexed_name);
            var_type = llvm::ArrayType::get(scalar_var_type, length);
        } else if (identifier->is_name()) {
            // This case corresponds to a scalar local variable. Its type is double by default.
            var_type = scalar_var_type;
        } else {
            throw std::runtime_error("Error: Unsupported local variable type");
        }
        llvm::Value* alloca = builder.CreateAlloca(var_type, /*ArraySize=*/nullptr, name);

        // Check if the variable we process is a procedure return variable (i.e. it has a name
        // "ret_<current_function_name>" and the function return type is integer). If so, initialise
        // it to 0.
        std::string ret_val_name = "ret_" + current_func->getName().str();
        if (name == ret_val_name && current_func->getReturnType()->isIntegerTy()) {
            llvm::Value* zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context), 0);
            builder.CreateStore(zero, alloca);
        }
    }
}

void CodegenLLVMVisitor::visit_double(const ast::Double& node) {
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

    // For every function, generate its declaration. Thus, we can look up
    // `llvm::Function` in the symbol table in the module.
    for (const auto& func: functions) {
        emit_procedure_or_function_declaration(*func);
    }

    // Set the AST symbol table.
    sym_tab = node.get_symbol_table();

    // Proceed with code generation.
    node.visit_children(*this);

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
    const auto& identifier = node.get_name();
    if (!identifier->is_name() && !identifier->is_indexed_name())
        throw std::runtime_error("Error: Unsupported variable type");

    llvm::Value* ptr;
    if (identifier->is_name())
        ptr = lookup(node.get_node_name());

    if (identifier->is_indexed_name()) {
        auto indexed_name = std::dynamic_pointer_cast<ast::IndexedName>(identifier);
        ptr = codegen_indexed_name(*indexed_name);
    }

    // Finally, load the variable from the pointer value.
    llvm::Value* var = builder.CreateLoad(ptr);
    values.push_back(var);
}

void CodegenLLVMVisitor::visit_instance_struct(const ast::InstanceStruct& node) {
    std::vector<llvm::Type*> members;
    for (const auto& variable: node.get_codegen_vars()) {
        members.push_back(get_default_fp_ptr_type());
    }

    llvm_struct = llvm::StructType::create(*context, mod_filename + "_Instance");
    llvm_struct->setBody(members);
    module->getOrInsertGlobal("inst", llvm_struct);
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
