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
#include "visitors/visitor_utils.hpp"

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


void CodegenLLVMVisitor::visit_procedure_or_function(const ast::Block& node) {
    const auto& name = node.get_node_name();
    const auto& parameters = node.get_parameters();

    // Procedure or function parameters are doubles by default.
    std::vector<llvm::Type*> arg_types;
    for (size_t i = 0; i < parameters.size(); ++i)
        arg_types.push_back(llvm::Type::getDoubleTy(*context));

    // If visiting a function, the return type is a double by default.
    llvm::Type* return_type = node.is_function_block() ? llvm::Type::getDoubleTy(*context)
                                                       : llvm::Type::getVoidTy(*context);

    llvm::Function* func =
        llvm::Function::Create(llvm::FunctionType::get(return_type, arg_types, /*isVarArg=*/false),
                               llvm::Function::ExternalLinkage,
                               name,
                               *module);

    // Create the entry basic block of the function/procedure and point the local named values table
    // to the symbol table.
    llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, /*Name=*/"", func);
    builder.SetInsertPoint(body);
    local_named_values = func->getValueSymbolTable();

    // When processing a function, it returns a value named <function_name> in NMODL. Therefore, we
    // first run RenameVisitor to rename it into ret_<function_name>. This will aid in avoiding
    // symbolic conflicts. Then, allocate the return variable on the local stack.
    std::string return_var_name = "ret_" + name;
    const auto& block = node.get_statement_block();
    if (node.is_function_block()) {
        visitor::RenameVisitor v(name, return_var_name);
        block->accept(v);
        builder.CreateAlloca(llvm::Type::getDoubleTy(*context),
                             /*ArraySize=*/nullptr,
                             return_var_name);
    }

    // Allocate parameters on the stack and add them to the symbol table.
    unsigned i = 0;
    for (auto& arg: func->args()) {
        std::string arg_name = parameters[i++].get()->get_node_name();
        llvm::Value* alloca = builder.CreateAlloca(arg.getType(), /*ArraySize=*/nullptr, arg_name);
        arg.setName(arg_name);
        builder.CreateStore(&arg, alloca);
    }

    // Process function or procedure body.
    const auto& statements = block->get_statements();
    for (const auto& statement: statements) {
        // \todo: Support other statement types.
        if (statement->is_local_list_statement() || statement->is_expression_statement())
            statement->accept(*this);
    }

    // Add the terminator. If visiting function, we need to return the value specified by
    // ret_<function_name>.
    if (node.is_function_block()) {
        llvm::Value* return_var = builder.CreateLoad(local_named_values->lookup(return_var_name));
        builder.CreateRet(return_var);
    } else {
        builder.CreateRetVoid();
    }

    // Clear local values stack and remove the pointer to the local symbol table.
    values.clear();
    local_named_values = nullptr;
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
        auto var = dynamic_cast<ast::VarName*>(node.get_lhs().get());
        if (!var) {
            throw std::runtime_error("Error: only VarName assignment is currently supported.\n");
        }
        llvm::Value* alloca = local_named_values->lookup(var->get_node_name());
        builder.CreateStore(rhs, alloca);
        return;
    }

    node.get_lhs()->accept(*this);
    llvm::Value* lhs = values.back();
    values.pop_back();
    llvm::Value* result;

    // \todo: Support other binary operators
    switch (op) {
#define DISPATCH(binary_op, llvm_op) \
    case binary_op:                  \
        result = llvm_op(lhs, rhs);  \
        values.push_back(result);    \
        break;

        DISPATCH(ast::BinaryOp::BOP_ADDITION, builder.CreateFAdd);
        DISPATCH(ast::BinaryOp::BOP_DIVISION, builder.CreateFDiv);
        DISPATCH(ast::BinaryOp::BOP_MULTIPLICATION, builder.CreateFMul);
        DISPATCH(ast::BinaryOp::BOP_SUBTRACTION, builder.CreateFSub);

#undef DISPATCH
    }
}

void CodegenLLVMVisitor::visit_boolean(const ast::Boolean& node) {
    const auto& constant = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*context),
                                                  node.get_value());
    values.push_back(constant);
}

void CodegenLLVMVisitor::visit_double(const ast::Double& node) {
    const auto& constant = llvm::ConstantFP::get(llvm::Type::getDoubleTy(*context),
                                                 node.get_value());
    values.push_back(constant);
}

void CodegenLLVMVisitor::visit_function_block(const ast::FunctionBlock& node) {
    visit_procedure_or_function(node);
}

void CodegenLLVMVisitor::visit_integer(const ast::Integer& node) {
    const auto& constant = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context),
                                                  node.get_value());
    values.push_back(constant);
}

void CodegenLLVMVisitor::visit_local_list_statement(const ast::LocalListStatement& node) {
    for (const auto& variable: node.get_variables()) {
        // LocalVar always stores a Name.
        auto name = variable->get_node_name();
        llvm::Type* var_type = llvm::Type::getDoubleTy(*context);
        llvm::Value* alloca = builder.CreateAlloca(var_type, /*ArraySize=*/nullptr, name);
    }
}

void CodegenLLVMVisitor::visit_program(const ast::Program& node) {
    node.visit_children(*this);

    if (opt_passes) {
        logger->info("Running LLVM optimisation passes");
        run_llvm_opt_passes();
    }

    // Keep this for easier development (maybe move to debug mode later).
    std::cout << print_module();

    CodegenLLVMHelperVisitor v;
    v.visit_program(const_cast<ast::Program&>(node));
}

void CodegenLLVMVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    visit_procedure_or_function(node);
}

void CodegenLLVMVisitor::visit_unary_expression(const ast::UnaryExpression& node) {
    ast::UnaryOp op = node.get_op().get_value();
    node.get_expression()->accept(*this);
    llvm::Value* value = values.back();
    values.pop_back();
    if (op == ast::UOP_NEGATION) {
        llvm::Value* result = builder.CreateFNeg(value);
        values.push_back(result);
    } else {
        // Support only `double` operators for now.
        throw std::runtime_error("Error: unsupported unary operator\n");
    }
}

void CodegenLLVMVisitor::visit_var_name(const ast::VarName& node) {
    llvm::Value* var = builder.CreateLoad(local_named_values->lookup(node.get_node_name()));
    values.push_back(var);
}

}  // namespace codegen
}  // namespace nmodl
