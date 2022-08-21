/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/gpu_builder.hpp"
#include "ast/all.hpp"

#include "llvm/IR/IntrinsicsNVPTX.h"

namespace nmodl {
namespace codegen {

/****************************************************************************************/
/*                           Generation virtual functions                               */
/****************************************************************************************/

void GPUBuilder::generate_atomic_statement(llvm::Value* ptr, llvm::Value* rhs, ast::BinaryOp op) {
    if (op == ast::BinaryOp::BOP_SUBTRACTION) {
        rhs = builder.CreateFNeg(rhs);
    }
    builder.CreateAtomicRMW(llvm::AtomicRMWInst::FAdd,
                            ptr,
                            rhs,
                            llvm::MaybeAlign(),
                            llvm::AtomicOrdering::SequentiallyConsistent);
}

void GPUBuilder::generate_loop_start() {
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

void GPUBuilder::generate_loop_increment() {
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

}  // namespace codegen
}  // namespace nmodl
