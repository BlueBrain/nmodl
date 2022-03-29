/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/replace_with_lib_functions.hpp"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/CodeGen/ReplaceWithVeclib.h"

namespace llvm {

char ReplaceMathFunctions::ID = 0;

bool ReplaceMathFunctions::runOnModule(Module &module) {
    bool modified = false;

    // If the platform supports SIMD, replace math intrinsics with library
    // functions.
    if (platform->is_cpu_with_simd()) {

        // First, get the target library information and add vectorizable functions for the
        // specified vector library.
        Triple triple(sys::getDefaultTargetTriple());
        TargetLibraryInfoImpl tli = TargetLibraryInfoImpl(triple);
        add_vectorizable_functions_from_vec_lib(tli, triple);

        // Run passes that replace math intrinsics.
        legacy::FunctionPassManager fpm(&module);
        fpm.add(new TargetLibraryInfoWrapperPass(tli));
        fpm.add(new ReplaceWithVeclibLegacy);
        fpm.doInitialization();
        for (auto& function: module.getFunctionList()) {
            if (!function.isDeclaration()) 
                modified |= fpm.run(function);
        }
        fpm.doFinalization();
    }

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

        // Populate function definitions of only exp and pow (for now)
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

}  // namespace llvm
