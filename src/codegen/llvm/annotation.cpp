/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/annotation.hpp"
#include "codegen/llvm/target_platform.hpp"

#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

static constexpr const char nmodl_annotations[] = "nmodl.annotations";
static constexpr const char nmodl_compute_kernel[] = "nmodl.compute-kernel";

namespace nmodl {
namespace custom {

void Annotator::add_nmodl_compute_kernel_annotation(llvm::Function& function) {
    llvm::LLVMContext& context = function.getContext();
    llvm::MDNode* node = llvm::MDNode::get(context, llvm::MDString::get(context, nmodl_compute_kernel));
    function.setMetadata(nmodl_annotations, node);
}

bool Annotator::has_nmodl_compute_kernel_annotation(llvm::Function& function) {
    if (!function.hasMetadata(nmodl_annotations))
        return false;
    
    llvm::MDNode* node = function.getMetadata(nmodl_annotations);
    std::string type = llvm::cast<llvm::MDString>(node->getOperand(0))->getString().str();
    return type == nmodl_compute_kernel;
}

void DefaultCPUAnnotator::annotate(llvm::Function& function) const {
    // By convention, the compute kernel does not free memory and does not
    // throw exceptions.
    function.setDoesNotFreeMemory();
    function.setDoesNotThrow();

    // We also want to specify that the pointers that instance struct holds
    // do not alias, unless specified otherwise. In order to do that, we
    // add a `noalias` attribute to the argument. As per Clang's
    // specification:
    //  > The `noalias` attribute indicates that the only memory accesses
    //  > inside function are loads and stores from objects pointed to by
    //  > its pointer-typed arguments, with arbitrary offsets.
    function.addParamAttr(0, llvm::Attribute::NoAlias);

    // Finally, specify that the mechanism data struct pointer does not
    // capture and is read-only. 
    function.addParamAttr(0, llvm::Attribute::NoCapture);
    function.addParamAttr(0, llvm::Attribute::ReadOnly);
}

void CUDAAnnotator::annotate(llvm::Function& function) const {    
    llvm::LLVMContext& context = function.getContext();
    llvm::Module* m = function.getParent();

    auto one = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 1);
    llvm::Metadata* metadata[] = {llvm::ValueAsMetadata::get(&function),
                                  llvm::MDString::get(context, "kernel"),
                                  llvm::ValueAsMetadata::get(one)};
    llvm::MDNode* node = llvm::MDNode::get(context, metadata);

    m->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(node);
}
}  // namespace custom
}  // namespace nmodl

using nmodl::custom::Annotator;
namespace llvm {

char AnnotationPass::ID = 0;

bool AnnotationPass::runOnModule(Module& module) {
    bool modified = false;

    for (auto& function: module.getFunctionList()) {
        if (!function.isDeclaration() &&
            Annotator::has_nmodl_compute_kernel_annotation(function)) {
            annotator->annotate(function);
            modified = true;
        }
    }

    return modified;
}

void AnnotationPass::getAnalysisUsage(AnalysisUsage& au) const {
    au.setPreservesCFG();
    au.addPreserved<ScalarEvolutionWrapperPass>();
    au.addPreserved<AAResultsWrapperPass>();
    au.addPreserved<LoopAccessLegacyAnalysis>();
    au.addPreserved<DemandedBitsWrapperPass>();
    au.addPreserved<OptimizationRemarkEmitterWrapperPass>();
    au.addPreserved<GlobalsAAWrapperPass>();
}
}  // namespace llvm
