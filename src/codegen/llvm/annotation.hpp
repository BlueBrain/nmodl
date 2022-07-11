/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace nmodl {
namespace custom {

/**
 * \class Annotator
 * \brief Base class that can be overriden to specify function annotations. 
 */
class Annotator {
  public:
    virtual void annotate(llvm::Function& function) const = 0;
    virtual ~Annotator() = default;

    /// Marks LLVM function as NMODL compute kernel. 
    static void add_nmodl_compute_kernel_annotation(llvm::Function& function);

    /// Returns true if LLVM function is marked as NMODL compute kernel. 
    static bool has_nmodl_compute_kernel_annotation(llvm::Function& function);
};

/**
 * \class DefaultAnnotator
 * \brief Specifies how LLVM IR functions for CPU platforms are annotated. Used
 * by default.
 */
class DefaultCPUAnnotator: public Annotator {
  public:
    void annotate(llvm::Function& function) const override;
};

/**
 * \class CUDAAnnotator
 * \brief Specifies how LLVM IR functions for CUDA platforms are annotated. This
 * includes marking functions with "kernel" or "device" attributes.
 */
class CUDAAnnotator: public Annotator {
  public:
    void annotate(llvm::Function& function) const override;
};
}  // namespace custom
}  // namespace nmodl

using nmodl::custom::Annotator;
namespace llvm {

/**
 * \class AnnotationPass
 * \brief LLVM module pass that annotates NMODL compute kernels.
 */
class AnnotationPass: public ModulePass {
  private:
    // Underlying annotator that is applied to each LLVM function.
    const Annotator* annotator;

  public:
    static char ID;

    AnnotationPass(Annotator* annotator)
        : ModulePass(ID)
        , annotator(annotator) {}

    bool runOnModule(Module& module) override;
};
}  // namespace llvm
