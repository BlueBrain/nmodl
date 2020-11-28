/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "ast/all.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace nmodl {
namespace codegen {


// LLVM code generator objects
using namespace llvm;
static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<IRBuilder<>> Builder;
static std::map<std::string, Value*> NamedValues;


void CodegenLLVMVisitor::visit_statement_block(const ast::StatementBlock& node) {
    logger->info("CodegenLLVMVisitor : visiting statement block");
    node.visit_children(*this);
    // TODO : code generation for new block scope
}

void CodegenLLVMVisitor::visit_procedure_block(const ast::ProcedureBlock& node) {
    logger->info("CodegenLLVMVisitor : visiting {} procedure", node.get_node_name());
    node.visit_children(*this);
    // TODO : code generation for procedure block
}

void CodegenLLVMVisitor::visit_program(const ast::Program& node) {
    node.visit_children(*this);
}

}  // namespace codegen
}  // namespace nmodl
