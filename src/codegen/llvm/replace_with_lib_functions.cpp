/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/replace_with_lib_functions.hpp"

#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/CodeGen/ReplaceWithVeclib.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"

namespace llvm {

char ReplaceMathFunctions::ID = 0;

bool ReplaceMathFunctions::runOnModule(Module& module) {
    legacy::FunctionPassManager fpm(&module);
    bool modified = false;

    // If the platform supports SIMD, replace math intrinsics with library
    // functions.
    if (platform->is_cpu_with_simd()) {

        // First, get the target library information and add vectorizable functions for the
        // specified vector library.
        Triple triple(sys::getDefaultTargetTriple());
        TargetLibraryInfoImpl tli = TargetLibraryInfoImpl(triple);
        add_vectorizable_functions_from_vec_lib(tli, triple);

        // Add passes that replace math intrinsics with calls.
        fpm.add(new TargetLibraryInfoWrapperPass(tli));
        fpm.add(new ReplaceWithVeclibLegacy);
    }

    // For CUDA GPUs, replace with calls to libdevice.
    if (platform->is_CUDA_gpu()) {
        fpm.add(new ReplaceWithLibdevice);
    }

    // Run passes.
    fpm.doInitialization();
    for (auto& function: module.getFunctionList()) {
        if (!function.isDeclaration())
            modified |= fpm.run(function);
    }
    fpm.doFinalization();

    return modified;
}

void
ReplaceMathFunctions::add_vectorizable_functions_from_vec_lib(TargetLibraryInfoImpl& tli,
                                                                 Triple& triple) {
    // Since LLVM does not support SLEEF as a vector library yet, process it separately.
    if (platform->get_math_library() == "SLEEF") {
// clang-format off
#define FIXED(w) ElementCount::getFixed(w)
// clang-format on
#define DISPATCH(func, vec_func, width) {func, vec_func, width},

        // Populate function definitions of only exp and pow (for now).
        const VecDesc aarch64_functions[] = {
            // clang-format off
            DISPATCH("llvm.exp.f32", "_ZGVnN4v_expf", FIXED(4))
            DISPATCH("llvm.exp.f64", "_ZGVnN2v_exp", FIXED(2))
            DISPATCH("llvm.pow.f32", "_ZGVnN4vv_powf", FIXED(4))
            DISPATCH("llvm.pow.f64", "_ZGVnN2vv_pow", FIXED(2))
            // clang-format on
        };
        const VecDesc x86_functions[] = {
            // clang-format off
            DISPATCH("llvm.exp.f64", "_ZGVbN2v_exp", FIXED(2))
            DISPATCH("llvm.exp.f64", "_ZGVdN4v_exp", FIXED(4))
            DISPATCH("llvm.exp.f64", "_ZGVeN8v_exp", FIXED(8))
            DISPATCH("llvm.pow.f64", "_ZGVbN2vv_pow", FIXED(2))
            DISPATCH("llvm.pow.f64", "_ZGVdN4vv_pow", FIXED(4))
            DISPATCH("llvm.pow.f64", "_ZGVeN8vv_pow", FIXED(8))
            // clang-format on
        };
#undef DISPATCH

        if (triple.isAArch64()) {
            tli.addVectorizableFunctions(aarch64_functions);
        }
        if (triple.isX86() && triple.isArch64Bit()) {
            tli.addVectorizableFunctions(x86_functions);
        }

    } else {
        // A map to query vector library by its string value.
        using VecLib = TargetLibraryInfoImpl::VectorLibrary;
        static const std::map<std::string, VecLib> llvm_supported_vector_libraries = {
            {"Accelerate", VecLib::Accelerate},
            {"libmvec", VecLib::LIBMVEC_X86},
            {"libsystem_m", VecLib ::DarwinLibSystemM},
            {"MASSV", VecLib::MASSV},
            {"none", VecLib::NoLibrary},
            {"SVML", VecLib::SVML}};

        const auto& library = llvm_supported_vector_libraries.find(platform->get_math_library());
        if (library == llvm_supported_vector_libraries.end())
            throw std::runtime_error("Error: unknown vector library - " + platform->get_math_library() + "\n");

        // Add vectorizable functions to the target library info.
        switch (library->second) {
        case VecLib::LIBMVEC_X86:
            if (!triple.isX86() || !triple.isArch64Bit())
                break;
        default:
            tli.addVectorizableFunctionsFromVecLib(library->second);
            break;
        }
    }
}

void ReplaceWithLibdevice::getAnalysisUsage(AnalysisUsage& au) const {
  au.setPreservesCFG();
  au.addPreserved<ScalarEvolutionWrapperPass>();
  au.addPreserved<AAResultsWrapperPass>();
  au.addPreserved<LoopAccessLegacyAnalysis>();
  au.addPreserved<DemandedBitsWrapperPass>();
  au.addPreserved<OptimizationRemarkEmitterWrapperPass>();
  au.addPreserved<GlobalsAAWrapperPass>();
}

bool ReplaceWithLibdevice::runOnFunction(Function& function) {
    bool modified = false;

    // Try to replace math intrinsics.
    std::vector<CallInst*> replaced_calls;
    for (auto& instruction: instructions(function)) {
        if (auto* call_inst = dyn_cast<CallInst>(&instruction)) {
            if (replace_call(*call_inst)) {
                replaced_calls.push_back(call_inst);
                modified = true;
            }
        }
    }

    // Remove calls to replaced intrinsics.
    for (auto* call_inst: replaced_calls) {
        call_inst->eraseFromParent();
    }

    return modified;
}

bool ReplaceWithLibdevice::replace_call(CallInst& call_inst) {
    Module* m = call_inst.getModule();
    Function* function = call_inst.getCalledFunction();

    // Replace math intrinsics only!
    auto id = function->getIntrinsicID();
    bool is_nvvm_intrinsic = id == Intrinsic::nvvm_read_ptx_sreg_ntid_x ||
            id == Intrinsic::nvvm_read_ptx_sreg_nctaid_x ||
            id == Intrinsic::nvvm_read_ptx_sreg_ctaid_x ||
            id == Intrinsic::nvvm_read_ptx_sreg_tid_x;
    if (id == Intrinsic::not_intrinsic || is_nvvm_intrinsic)
        return false;

    // Map of supported replacements. For now it is only exp.
    static const std::map<std::string, std::string> libdevice_name = {
            {"llvm.exp.f32", "__nv_expf"},
            {"llvm.exp.f64", "__nv_exp"}};

    // If replacement is not supported, abort.
    std::string old_name = function->getName().str();
    auto it = libdevice_name.find(old_name);
    if (it == libdevice_name.end())
        throw std::runtime_error("Error: replacements for " + old_name + " are not supported!\n");

    // Get (or create) libdevice function.
    Function* libdevice_func = m->getFunction(it->second);
    if (!libdevice_func) {
        libdevice_func = Function::Create(function->getFunctionType(),
                                   Function::ExternalLinkage, it->second, *m);
        libdevice_func->copyAttributesFrom(function);
    }

    // Create a call to libdevice function with the same operands.
    IRBuilder<> builder(&call_inst);
    std::vector<Value*> args(call_inst.arg_operands().begin(),
                             call_inst.arg_operands().end());
    SmallVector<OperandBundleDef, 1> op_bundles;
    call_inst.getOperandBundlesAsDefs(op_bundles);
    CallInst* new_call = builder.CreateCall(libdevice_func, args, op_bundles);

    // Replace all uses of old instruction with the new one. Also, copy
    // fast math flags if necessary.
    call_inst.replaceAllUsesWith(new_call);
    if (isa<FPMathOperator>(new_call)) {
        new_call->copyFastMathFlags(&call_inst);
    }

    return true;
}

char ReplaceWithLibdevice::ID = 0;
static RegisterPass<ReplaceWithLibdevice> X(
        "libdevice-replacement",
        "Pass replacing math functions with calls to libdevice",
        false,
        false);

}  // namespace llvm
