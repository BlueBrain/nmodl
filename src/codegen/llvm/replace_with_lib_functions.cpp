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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"

namespace nmodl {
namespace custom {

Patterns DefaultCPUReplacer::patterns() const {
    throw std::runtime_error("Error: DefaultCPUReplacer has no patterns and uses built-in LLVM passes instead.\n");
}

std::string DefaultCPUReplacer::get_library_name() {
    return this->library_name;
}

Patterns CUDAReplacer::patterns() const {
    return {
        {"llvm.exp.f32", "__nv_expf"},
        {"llvm.exp.f64", "__nv_exp"},
        {"llvm.pow.f32", "__nv_powf"},
        {"llvm.pow.f64", "__nv_pow"},
        {"llvm.log.f32", "__nv_logf"},
        {"llvm.log.f64", "__nv_log"},
        {"llvm.fabs.f32", "__nv_fabsf"},
        {"llvm.fabs.f64", "__nv_fabs"}
    };
}
}  // namespace custom
}  // namespace nmodl

using nmodl::custom::DefaultCPUReplacer;
namespace llvm {

char ReplacePass::ID = 0;

bool ReplacePass::runOnModule(Module& module) {
    bool modified = false;

    // If the platform supports SIMD, replace math intrinsics with library
    // functions.
    if (dynamic_cast<const DefaultCPUReplacer*>(replacer)) {
        legacy::FunctionPassManager fpm(&module);

        // First, get the target library information and add vectorizable functions for the
        // specified vector library.
        Triple triple(sys::getDefaultTargetTriple());
        TargetLibraryInfoImpl tli = TargetLibraryInfoImpl(triple);
        add_vectorizable_functions_from_vec_lib(tli, triple);

        // Add passes that replace math intrinsics with calls.
        fpm.add(new TargetLibraryInfoWrapperPass(tli));
        fpm.add(new ReplaceWithVeclibLegacy);

        // Run passes.
        fpm.doInitialization();
        for (auto& function: module.getFunctionList()) {
            if (!function.isDeclaration())
                modified |= fpm.run(function);
        }
        fpm.doFinalization();
    } else {
        // Otherwise, the replacer is not default and we need to apply patterns
        // from it to each function!
        for (auto& function: module.getFunctionList()) {
            if (!function.isDeclaration()) {
                // Try to replace a call instruction.
                std::vector<CallInst*> replaced_calls;
                for (auto& instruction: instructions(function)) {
                    if (auto* call_inst = dyn_cast<CallInst>(&instruction)) {
                        if (replace_call(*call_inst)) {
                            replaced_calls.push_back(call_inst);
                            modified = true;
                        }
                    }
                }

                // Remove calls to replaced functions.
                for (auto* call_inst: replaced_calls) {
                    call_inst->eraseFromParent();
                }
            }
        }
    }

    return modified;
}

void ReplacePass::add_vectorizable_functions_from_vec_lib(TargetLibraryInfoImpl& tli,
                                                          Triple& triple) {
    // Since LLVM does not support SLEEF as a vector library yet, process it separately.
    if (((DefaultCPUReplacer*)replacer)->get_library_name() == "SLEEF") {
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
            DISPATCH("llvm.log.f32", "_ZGVnN4v_logf", FIXED(4))
            DISPATCH("llvm.log.f64", "_ZGVnN2v_log", FIXED(2))
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
            DISPATCH("llvm.log.f64", "_ZGVbN2v_log", FIXED(2))
            DISPATCH("llvm.log.f64", "_ZGVdN4v_log", FIXED(4))
            DISPATCH("llvm.log.f64", "_ZGVeN8v_log", FIXED(8))
            // clang-format on
        };
#undef DISPATCH
#undef FIXED

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

        const auto& library = llvm_supported_vector_libraries.find(((DefaultCPUReplacer*)replacer)->get_library_name());
        if (library == llvm_supported_vector_libraries.end())
            throw std::runtime_error("Error: unknown vector library - " +
                                     ((DefaultCPUReplacer*)replacer)->get_library_name() + "\n");

        // Add vectorizable functions to the target library info.
        if (library->second != VecLib::LIBMVEC_X86 || (triple.isX86() && triple.isArch64Bit())) {
            tli.addVectorizableFunctionsFromVecLib(library->second);
        }
    }
}

bool ReplacePass::replace_call(CallInst& call_inst) {
    Module* m = call_inst.getModule();
    Function* function = call_inst.getCalledFunction();

    // Get supported replacement patterns.
    Patterns patterns = replacer->patterns();

    // Check if replacement is not supported.
    std::string old_name = function->getName().str();
    auto it = patterns.find(old_name);
    if (it == patterns.end())
        return false;

    // Get (or create) new function.
    Function* new_func = m->getFunction(it->second);
    if (!new_func) {
        new_func = Function::Create(function->getFunctionType(),
                                    Function::ExternalLinkage,
                                    it->second,
                                    *m);
        new_func->copyAttributesFrom(function);
    }

    // Create a call to libdevice function with the same operands.
    IRBuilder<> builder(&call_inst);
    std::vector<Value*> args(call_inst.arg_operands().begin(), call_inst.arg_operands().end());
    SmallVector<OperandBundleDef, 1> op_bundles;
    call_inst.getOperandBundlesAsDefs(op_bundles);
    CallInst* new_call = builder.CreateCall(new_func, args, op_bundles);

    // Replace all uses of old instruction with the new one. Also, copy
    // fast math flags if necessary.
    call_inst.replaceAllUsesWith(new_call);
    if (isa<FPMathOperator>(new_call)) {
        new_call->copyFastMathFlags(&call_inst);
    }

    return true;
}
}  // namespace llvm
