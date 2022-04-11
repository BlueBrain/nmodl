/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "codegen/codegen_acc_visitor.hpp"
#include "codegen/codegen_c_visitor.hpp"
#include "codegen/codegen_cuda_visitor.hpp"
#include "codegen/codegen_ispc_visitor.hpp"
#include "codegen/codegen_omp_visitor.hpp"

#ifdef NMODL_LLVM_BACKEND
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "test/benchmark/llvm_benchmark.hpp"
#endif

#include "config/config.h"
#include "parser/nmodl_driver.hpp"
#include "pybind/pyembed.hpp"
#include "utils/common_utils.hpp"
#include "utils/logger.hpp"
#include "visitors/json_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "codegen/codegen_driver.hpp"

/**
 * \dir
 * \brief Main NMODL code generation program
 */

using namespace fmt::literals;
using namespace nmodl;
using namespace codegen;
using namespace visitor;
using nmodl::parser::NmodlDriver;

int main(int argc, const char* argv[]) {
    CLI::App app{
        "NMODL : Source-to-Source Code Generation Framework [{}]"_format(Version::to_string())};

    /// list of mod files to process
    std::vector<std::string> mod_files;

    /// true if debug logger statements should be shown
    std::string verbose("info");

    /// true if symbol table should be printed
    bool show_symtab(false);

    /// floating point data type
    std::string data_type("double");

#ifdef NMODL_LLVM_BACKEND
    /// run llvm benchmark
    bool llvm_benchmark(false);

    /// the size of the instance struct for the benchmark
    int instance_size = 10000;

    /// the number of repeated experiments for the benchmarking
    int num_experiments = 100;
#endif

    CodeGenConfig cfg;

    app.get_formatter()->column_width(40);
    app.set_help_all_flag("-H,--help-all", "Print this help message including all sub-commands");

    app.add_option("--verbose", verbose, "Verbosity of logger output", true)
        ->ignore_case()
        ->check(CLI::IsMember({"trace", "debug", "info", "warning", "error", "critical", "off"}));

    app.add_option("file", mod_files, "One or more MOD files to process")
        ->ignore_case()
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-o,--output", cfg.output_dir, "Directory for backend code output", true)
        ->ignore_case();
    app.add_option("--scratch", cfg.scratch_dir, "Directory for intermediate code output", true)
        ->ignore_case();
    app.add_option("--units", cfg.units_dir, "Directory of units lib file", true)->ignore_case();

    auto host_opt = app.add_subcommand("host", "HOST/CPU code backends")->ignore_case();
    host_opt->add_flag("--c", cfg.c_backend, "C/C++ backend ({})"_format(cfg.c_backend))->ignore_case();
    host_opt->add_flag("--omp", cfg.omp_backend, "C/C++ backend with OpenMP ({})"_format(cfg.omp_backend))
        ->ignore_case();
    host_opt->add_flag("--ispc", cfg.ispc_backend, "C/C++ backend with ISPC ({})"_format(cfg.ispc_backend))
        ->ignore_case();

    auto acc_opt = app.add_subcommand("acc", "Accelerator code backends")->ignore_case();
    acc_opt
        ->add_flag("--oacc", cfg.oacc_backend, "C/C++ backend with OpenACC ({})"_format(cfg.oacc_backend))
        ->ignore_case();
    acc_opt->add_flag("--cuda", cfg.cuda_backend, "C/C++ backend with CUDA ({})"_format(cfg.cuda_backend))
        ->ignore_case();

    // clang-format off
    auto sympy_opt = app.add_subcommand("sympy", "SymPy based analysis and optimizations")->ignore_case();
    sympy_opt->add_flag("--analytic",
        cfg.sympy_analytic,
        "Solve ODEs using SymPy analytic integration ({})"_format(cfg.sympy_analytic))->ignore_case();
    sympy_opt->add_flag("--pade",
        cfg.sympy_pade,
        "Pade approximation in SymPy analytic integration ({})"_format(cfg.sympy_pade))->ignore_case();
    sympy_opt->add_flag("--cse",
        cfg.sympy_cse,
        "CSE (Common Subexpression Elimination) in SymPy analytic integration ({})"_format(cfg.sympy_cse))->ignore_case();
    sympy_opt->add_flag("--conductance",
        cfg.sympy_conductance,
        "Add CONDUCTANCE keyword in BREAKPOINT ({})"_format(cfg.sympy_conductance))->ignore_case();

    auto passes_opt = app.add_subcommand("passes", "Analyse/Optimization passes")->ignore_case();
    passes_opt->add_flag("--inline",
        cfg.nmodl_inline,
        "Perform inlining at NMODL level ({})"_format(cfg.nmodl_inline))->ignore_case();
    passes_opt->add_flag("--unroll",
        cfg.nmodl_unroll,
        "Perform loop unroll at NMODL level ({})"_format(cfg.nmodl_unroll))->ignore_case();
    passes_opt->add_flag("--const-folding",
        cfg.nmodl_const_folding,
        "Perform constant folding at NMODL level ({})"_format(cfg.nmodl_const_folding))->ignore_case();
    passes_opt->add_flag("--localize",
        cfg.nmodl_localize,
        "Convert RANGE variables to LOCAL ({})"_format(cfg.nmodl_localize))->ignore_case();
    passes_opt->add_flag("--global-to-range",
         cfg.nmodl_global_to_range,
         "Convert GLOBAL variables to RANGE ({})"_format(cfg.nmodl_global_to_range))->ignore_case();
    passes_opt->add_flag("--local-to-range",
         cfg.nmodl_local_to_range,
         "Convert top level LOCAL variables to RANGE ({})"_format(cfg.nmodl_local_to_range))->ignore_case();
    passes_opt->add_flag("--localize-verbatim",
        cfg.localize_verbatim,
        "Convert RANGE variables to LOCAL even if verbatim block exist ({})"_format(cfg.localize_verbatim))->ignore_case();
    passes_opt->add_flag("--local-rename",
        cfg.local_rename,
        "Rename LOCAL variable if variable of same name exist in global scope ({})"_format(cfg.local_rename))->ignore_case();
    passes_opt->add_flag("--verbatim-inline",
        cfg.verbatim_inline,
        "Inline even if verbatim block exist ({})"_format(cfg.verbatim_inline))->ignore_case();
    passes_opt->add_flag("--verbatim-rename",
        cfg.verbatim_rename,
        "Rename variables in verbatim block ({})"_format(cfg.verbatim_rename))->ignore_case();
    passes_opt->add_flag("--json-ast",
        cfg.json_ast,
        "Write AST to JSON file ({})"_format(cfg.json_ast))->ignore_case();
    passes_opt->add_flag("--nmodl-ast",
        cfg.nmodl_ast,
        "Write AST to NMODL file ({})"_format(cfg.nmodl_ast))->ignore_case();
    passes_opt->add_flag("--json-perf",
        cfg.json_perfstat,
        "Write performance statistics to JSON file ({})"_format(cfg.json_perfstat))->ignore_case();
    passes_opt->add_flag("--show-symtab",
        show_symtab,
        "Write symbol table to stdout ({})"_format(show_symtab))->ignore_case();

    auto codegen_opt = app.add_subcommand("codegen", "Code generation options")->ignore_case();
    codegen_opt->add_option("--datatype",
        cfg.data_type,
        "Data type for floating point variables",
        true)->ignore_case()->check(CLI::IsMember({"float", "double"}));
    codegen_opt->add_flag("--force",
        cfg.force_codegen,
        "Force code generation even if there is any incompatibility");
    codegen_opt->add_flag("--only-check-compatibility",
                          cfg.only_check_compatibility,
                          "Check compatibility and return without generating code");
    codegen_opt->add_flag("--opt-ionvar-copy",
        cfg.optimize_ionvar_copies_codegen,
        "Optimize copies of ion variables ({})"_format(cfg.optimize_ionvar_copies_codegen))->ignore_case();

#ifdef NMODL_LLVM_BACKEND

    // LLVM IR code generation options.
    auto llvm_opt = app.add_subcommand("llvm", "LLVM code generation option")->ignore_case();
    auto llvm_ir_opt = llvm_opt->add_flag("--ir",
        cfg.llvm_ir,
        "Generate LLVM IR ({})"_format(cfg.llvm_ir))->ignore_case();
    llvm_ir_opt->required(true);
    llvm_opt->add_flag("--no-debug",
        cfg.llvm_no_debug,
        "Disable debug information ({})"_format(cfg.llvm_no_debug))->ignore_case();
    llvm_opt->add_option("--opt-level-ir",
        cfg.llvm_opt_level_ir,
        "LLVM IR optimisation level (O{})"_format(cfg.llvm_opt_level_ir))->ignore_case()->check(CLI::IsMember({"0", "1", "2", "3"}));
    llvm_opt->add_flag("--single-precision",
        cfg.llvm_float_type,
        "Use single precision floating-point types ({})"_format(cfg.llvm_float_type))->ignore_case();
    llvm_opt->add_option("--fmf",
        cfg.llvm_fast_math_flags,
        "Fast math flags for floating-point optimizations (none)")->check(CLI::IsMember({"afn", "arcp", "contract", "ninf", "nnan", "nsz", "reassoc", "fast"}));

    // Platform options for LLVM code generation.
    auto cpu_opt = app.add_subcommand("cpu", "LLVM CPU option")->ignore_case();
    cpu_opt->needs(llvm_opt);
    cpu_opt->add_option("--name",
        cfg.llvm_cpu_name,
        "Name of CPU platform to use")->ignore_case();
    auto simd_math_library_opt = cpu_opt->add_option("--math-library",
        cfg.llvm_math_library,
        "Math library for SIMD code generation ({})"_format(cfg.llvm_math_library));
    simd_math_library_opt->check(CLI::IsMember({"Accelerate", "libmvec", "libsystem_m", "MASSV", "SLEEF", "SVML", "none"}));
    cpu_opt->add_option("--vector-width",
        cfg.llvm_vector_width,
        "Explicit vectorization width for IR generation ({})"_format(cfg.llvm_vector_width))->ignore_case();

    auto gpu_opt = app.add_subcommand("gpu", "LLVM GPU option")->ignore_case();
    gpu_opt->needs(llvm_opt);
    gpu_opt->add_option("--name",
        cfg.llvm_gpu_name,
        "Name of GPU platform to use")->ignore_case();
    auto gpu_math_library_opt = gpu_opt->add_option("--math-library",
        cfg.llvm_math_library,
        "Math library for GPU code generation ({})"_format(cfg.llvm_math_library));
    gpu_math_library_opt->check(CLI::IsMember({"libdevice"}));

    // Allow only one platform at a time.
    cpu_opt->excludes(gpu_opt);
    gpu_opt->excludes(cpu_opt);

    // LLVM IR benchmark options.
    auto benchmark_opt = app.add_subcommand("benchmark", "LLVM benchmark option")->ignore_case();
    benchmark_opt->needs(llvm_opt);
    benchmark_opt->add_flag("--run",
                            llvm_benchmark,
                            "Run LLVM benchmark ({})"_format(llvm_benchmark))->ignore_case();
    benchmark_opt->add_option("--opt-level-codegen",
                              cfg.llvm_opt_level_codegen,
                              "Machine code optimisation level (O{})"_format(cfg.llvm_opt_level_codegen))->ignore_case()->check(CLI::IsMember({"0", "1", "2", "3"}));
    benchmark_opt->add_option("--libs", cfg.shared_lib_paths, "Shared libraries to link IR against")
            ->ignore_case()
            ->check(CLI::ExistingFile);
    benchmark_opt->add_option("--instance-size",
                       instance_size,
                       "Instance struct size ({})"_format(instance_size))->ignore_case();
    benchmark_opt->add_option("--repeat",
                              num_experiments,
                              "Number of experiments for benchmarking ({})"_format(num_experiments))->ignore_case();
#endif
    // clang-format on

    CLI11_PARSE(app, argc, argv);

    // if any of the other backends is used we force the C backend to be off.
    if (cfg.omp_backend || cfg.ispc_backend) {
        cfg.c_backend = false;
    }

    utils::make_path(cfg.output_dir);
    utils::make_path(cfg.scratch_dir);

    if (sympy_opt) {
        nmodl::pybind_wrappers::EmbeddedPythonLoader::get_instance()
            .api()
            ->initialize_interpreter();
    }

    logger->set_level(spdlog::level::from_str(verbose));



    for (const auto& file: mod_files) {
        logger->info("Processing {}", file);

        const auto modfile = utils::remove_extension(utils::base_name(file));

        /// create file path for nmodl file
        auto filepath = [cfg, modfile](const std::string& suffix, const std::string& ext) {
            static int count = 0;
            return "{}/{}.{}.{}.{}"_format(
                cfg.scratch_dir, modfile, std::to_string(count++), suffix, ext);
        };

        /// nmodl_driver object creates lexer and parser, just call parser method
        NmodlDriver nmodl_driver;

        /// parse mod file and construct ast
        const auto& ast = nmodl_driver.parse_file(file);

        auto cg_driver = CodegenDriver(cfg);
        auto success = cg_driver.prepare_mod(*ast);

        if (show_symtab) {
            logger->info("Printing symbol table");
            auto symtab = ast->get_model_symbol_table();
            symtab->print(std::cout);
        }

        if (cfg.only_check_compatibility) {
            return !success;
        }
        if (!success && !cfg.force_codegen) {
            return 1;
        }

        {
            if (cfg.ispc_backend) {
                logger->info("Running ISPC backend code generator");
                CodegenIspcVisitor visitor(modfile,
                                           cfg.output_dir,
                                           data_type,
                                           cfg.optimize_ionvar_copies_codegen);
                visitor.visit_program(*ast);
            }

            else if (cfg.oacc_backend) {
                logger->info("Running OpenACC backend code generator");
                CodegenAccVisitor visitor(modfile,
                                          cfg.output_dir,
                                          data_type,
                                          cfg.optimize_ionvar_copies_codegen);
                visitor.visit_program(*ast);
            }

            else if (cfg.omp_backend) {
                logger->info("Running OpenMP backend code generator");
                CodegenOmpVisitor visitor(modfile,
                                          cfg.output_dir,
                                          data_type,
                                          cfg.optimize_ionvar_copies_codegen);
                visitor.visit_program(*ast);
            }

            else if (cfg.c_backend) {
                logger->info("Running C backend code generator");
                CodegenCVisitor visitor(modfile,
                                        cfg.output_dir,
                                        data_type,
                                        cfg.optimize_ionvar_copies_codegen);
                visitor.visit_program(*ast);
            }

            if (cfg.cuda_backend) {
                logger->info("Running CUDA backend code generator");
                CodegenCudaVisitor visitor(modfile,
                                           cfg.output_dir,
                                           data_type,
                                           cfg.optimize_ionvar_copies_codegen);
                visitor.visit_program(*ast);
            }

#ifdef NMODL_LLVM_BACKEND
            if (cfg.llvm_ir || llvm_benchmark) {
              // If benchmarking, we want to optimize the IR with target
              // information and not in LLVM visitor.
              int llvm_opt_level = llvm_benchmark ? 0 : cfg.llvm_opt_level_ir;

              // Create platform abstraction.
              PlatformID pid = cfg.llvm_gpu_name == "default" ? PlatformID::CPU
                                                          : PlatformID::GPU;
              const std::string name =
                  cfg.llvm_gpu_name == "default" ? cfg.llvm_cpu_name : cfg.llvm_gpu_name;
              Platform platform(pid, name, cfg.llvm_math_library, cfg.llvm_float_type,
                                cfg.llvm_vector_width);

              logger->info("Running LLVM backend code generator");
              CodegenLLVMVisitor visitor(modfile, cfg.output_dir, platform,
                                         llvm_opt_level, !cfg.llvm_no_debug,
                                         cfg.llvm_fast_math_flags);
              visitor.visit_program(*ast);
              if (cfg.nmodl_ast) {
                  NmodlPrintVisitor(filepath("llvm", "mod")).visit_program(*ast);
                  logger->info("AST to NMODL transformation written to {}", filepath("llvm", "mod"));
              }
              if (cfg.json_ast) {
                  JSONVisitor(filepath("llvm", "json")).write(*ast);
                  logger->info("AST to JSON transformation written to {}", filepath("llvm", "json"));
              }

              if (llvm_benchmark) {
                // \todo integrate Platform class here
                if (cfg.llvm_gpu_name != "default") {
                  logger->warn("GPU benchmarking is not supported, targeting "
                               "CPU instead");
                }

                logger->info("Running LLVM benchmark");
                benchmark::LLVMBenchmark benchmark(
                    visitor, modfile, cfg.output_dir, cfg.shared_lib_paths,
                    num_experiments, instance_size, cfg.llvm_cpu_name,
                    cfg.llvm_opt_level_ir, cfg.llvm_opt_level_codegen);
                benchmark.run();
              }
            }
#endif
        }
    }

    if (sympy_opt) {
        nmodl::pybind_wrappers::EmbeddedPythonLoader::get_instance().api()->finalize_interpreter();
    }
}
