/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/


#pragma once

#include <string>
#include <utility>

#include "ast/program.hpp"
#include "config/config.h"
#include "utils/logger.hpp"

namespace nmodl {
namespace codegen {

struct CodeGenConfig {
    /// true if serial c code to be generated
    bool c_backend = true;

    /// true if c code with openmp to be generated
    bool omp_backend = false;

    /// true if ispc code to be generated
    bool ispc_backend = false;

    /// true if c code with openacc to be generated
    bool oacc_backend = false;

    /// true if cuda code to be generated
    bool cuda_backend = false;

    /// true if llvm code to be generated
    bool llvm_backend = false;

    /// true if sympy should be used for solving ODEs analytically
    bool sympy_analytic = false;

    /// true if Pade approximation to be used
    bool sympy_pade = false;

    /// true if CSE (temp variables) to be used
    bool sympy_cse = false;

    /// true if conductance keyword can be added to breakpoint
    bool sympy_conductance = false;

    /// true if inlining at nmodl level to be done
    bool nmodl_inline = false;

    /// true if unroll at nmodl level to be done
    bool nmodl_unroll = false;

    /// true if perform constant folding at nmodl level to be done
    bool nmodl_const_folding = false;

    /// true if range variables to be converted to local
    bool nmodl_localize = false;

    /// true if global variables to be converted to range
    bool nmodl_global_to_range = false;

    /// true if top level local variables to be converted to range
    bool nmodl_local_to_range = false;

    /// true if localize variables even if verbatim block is used
    bool localize_verbatim = false;

    /// true if local variables to be renamed
    bool local_rename = true;

    /// true if inline even if verbatim block exist
    bool verbatim_inline = false;

    /// true if verbatim blocks
    bool verbatim_rename = true;

    /// true if code generation is forced to happen even if there
    /// is any incompatibility
    bool force_codegen = false;

    /// true if we want to only check compatibility without generating code
    bool only_check_compatibility = false;

    /// true if ion variable copies should be avoided
    bool optimize_ionvar_copies_codegen = false;

    /// directory where code will be generated
    std::string output_dir  = ".";

    /// directory where intermediate file will be generated
    std::string scratch_dir = "tmp";

    /// directory where units lib file is located
    std::string units_dir = NrnUnitsLib::get_path();

    /// floating point data type
    std::string data_type = "double";

    /// true if ast should be converted to nmodl
    bool nmodl_ast = false;

    /// true if ast should be converted to json
    bool json_ast = false;

    /// true if performance stats should be converted to json
    bool json_perfstat = false;

#ifdef NMODL_LLVM_BACKEND
    /// generate llvm IR
    bool llvm_ir = false;

    /// use single precision floating-point types
    bool llvm_float_type = false;

    /// optimisation level for IR generation
    int llvm_opt_level_ir = 0;

    /// math library name
    std::string llvm_math_library = "none";

    /// disable debug information generation for the IR
    bool llvm_no_debug = false;

    /// fast math flags for LLVM backend
    std::vector<std::string> llvm_fast_math_flags;

    /// traget CPU platform name
    std::string llvm_cpu_name = "default";

    /// traget GPU platform name
    std::string llvm_gpu_name = "default";

    /// GPU target architecture
    std::string llvm_gpu_target_architecture = "sm_70";

    /// llvm vector width if generating code for CPUs
    int llvm_vector_width = 1;

    /// optimisation level for machine code generation
    int llvm_opt_level_codegen = 0;

    /// list of shared libraries to link against in JIT
    std::vector<std::string> shared_lib_paths;
#endif

};

class CodegenDriver {

  public:
    explicit CodegenDriver(CodeGenConfig  _cfg) :
        cfg(std::move(_cfg)) {}

    bool prepare_mod(nmodl::ast::Program& node);

  private:
    CodeGenConfig cfg;



    /// write ast to nmodl
    void ast_to_nmodl(ast::Program& ast, const std::string& filepath) const;
    void ast_to_json(ast::Program& ast, const std::string& filepath) const;
};

}
}
