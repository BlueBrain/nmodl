/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "ast/all.hpp"
#include "visitors/visitor_utils.hpp"

#include "llvm/IR/BasicBlock.h"
#include <llvm/IR/Constants.h>
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueSymbolTable.h"

namespace nmodl {
namespace codegen {


/****************************************************************************************/
/*                            Overloaded visitor routines                               */
/****************************************************************************************/


void CodegenLLVMVisitor::visit_binary_expression(const ast::BinaryExpression& node) {
    ast::BinaryOp op = node.get_op().get_value();
    if (op == ast::BinaryOp::BOP_ASSIGN) {
        auto var = dynamic_cast<ast::VarName*>(node.get_lhs().get());
        node.get_rhs()->accept(*this);
        llvm::Value* rhs = values.back();
        values.pop_back();
        llvm::Value* alloca = namedValues[var->get_node_name()];
        builder.CreateStore(rhs, alloca);
    } else {
        // TODO: scale binary operators
        std::cout << "Error: not supported binary operator";
        abort();
    }
}

void CodegenLLVMVisitor::visit_double(const ast::Double &node) {
    llvm::Value* constant = llvm::ConstantFP::get(llvm::Type::getDoubleTy(*context), node.get_value());
    values.push_back(constant);
}

void CodegenLLVMVisitor::visit_local_list_statement(const ast::LocalListStatement &node) {
    for (const auto& variable : node.get_variables()) {
        auto name = variable->get_node_name();
        llvm::Type* var_type = llvm::Type::getDoubleTy(*context);
        llvm::Value* alloca = builder.CreateAlloca(var_type, /*ArraySize=*/nullptr, name);
        namedValues[name] = alloca;
    }
}

void CodegenLLVMVisitor::visit_program(const ast::Program& node) {
    node.visit_children(*this);
    // Keep this for easier development (maybe move to debug mode later)
    std::cout << print_module();
}

void CodegenLLVMVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    auto name = node.get_name()->get_value()->get_value();
    const auto& parameters = node.get_parameters();

    // The procedure parameters are doubles by default
    std::vector<llvm::Type*> arg_types;
    for (unsigned i = 0, e = parameters.size(); i < e; ++i)
        arg_types.push_back(llvm::Type::getDoubleTy(*context));
    llvm::Type* return_type = llvm::Type::getVoidTy(*context);

    llvm::Function* proc = llvm::Function::Create(
            llvm::FunctionType::get(return_type, arg_types, /*isVarArg=*/false),
            llvm::Function::ExternalLinkage,
            name, *module);

    llvm::BasicBlock* body = llvm::BasicBlock::Create(*context, /*Name=*/"", proc);;
    builder.SetInsertPoint(body);

    // First, allocate parameters on the stack and add them to the symbol table
    unsigned i = 0;
    for (auto &arg : proc->args()) {
        std::string arg_name = parameters[i++].get()->get_node_name();
        llvm::Value* alloca = builder.CreateAlloca(arg.getType(), /*ArraySize=*/nullptr, arg_name);
        arg.setName(arg_name);
        builder.CreateStore(&arg, alloca);
        namedValues[arg_name] = alloca;
    }

    auto statements = node.get_statement_block()->get_statements();
    for (const auto& statement: statements) {
        // TODO: support more statement types
        if (statement->is_local_list_statement() || statement->is_expression_statement() )
            statement->accept(*this);
    }

    values.clear();
}

void CodegenLLVMVisitor::visit_var_name(const ast::VarName &node) {
    llvm::Value* var = builder.CreateLoad(namedValues[node.get_node_name()]);
    values.push_back(var);
}

}  // namespace codegen
}  // namespace nmodl
