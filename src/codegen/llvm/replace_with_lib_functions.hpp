/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "codegen/llvm/target_platform.hpp"

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Host.h"

using Patterns = std::map<std::string, std::string>;

namespace nmodl {
namespace custom {

/**
 * \class Replacer
 * \brief Base class that can be overriden to specify how LLVM math intrinsics
 * are replaced.
 */
class Replacer {
  public:
    virtual Patterns patterns() const = 0;
    virtual ~Replacer() = default;
};

/**
 * \class DefaultCPUReplacer
 * \brief Specifies how LLVM IR math functions are replaced on CPUs by default.
 * Here we reuse LLVM's API so patterns() has no meaning and throws an error
 * instead! `DefaultCPUReplacer` threfore cannot be overriden.
 */
class DefaultCPUReplacer: public Replacer {
  private:
    std::string library_name;
  public:
    DefaultCPUReplacer(std::string library_name)
      : Replacer(), library_name(library_name) {}

    Patterns patterns() const final override;

    /// Returns the name of underlying library for which this default
    /// replacer is used.
    std::string get_library_name();
};

/**
 * \class CUDAReplacer
 * \brief Specifies replacement patterns for CUDA platforms.
 */
class CUDAReplacer: public Replacer {
  public:
    Patterns patterns() const override;
};
}  // namespace custom
}  // namespace nmodl

using nmodl::custom::Replacer;
namespace llvm {

/**
 * \class ReplacePass
 * \brief A module LLVM pass that replaces math intrinsics with
 * library calls.
 */
class ReplacePass: public ModulePass {
  private:
    // Underlying replacer that provides replacement patterns.
    const Replacer* replacer;

  public:
    static char ID;

    ReplacePass(Replacer* replacer)
        : ModulePass(ID)
        , replacer(replacer) {}

    bool runOnModule(Module& module) override;

    void getAnalysisUsage(AnalysisUsage& au) const override;

  private:
    /// Populates `tli` with vectorizable function definitions (hook for default replacements).
    void add_vectorizable_functions_from_vec_lib(TargetLibraryInfoImpl& tli, Triple& triple);

    /// Replaces call instruction with a new call from Replacer's patterns.
    bool replace_call(CallInst& call_inst);
};
}  // namespace llvm
