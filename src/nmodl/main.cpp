/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <sstream>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"

#include "ast/program.hpp"
#include "codegen/codegen_acc_visitor.hpp"
#include "codegen/codegen_c_visitor.hpp"
#include "codegen/codegen_compatibility_visitor.hpp"
#include "codegen/codegen_cuda_visitor.hpp"
#include "codegen/codegen_ispc_visitor.hpp"
#include "codegen/codegen_omp_visitor.hpp"
#include "config/config.h"
#include "parser/nmodl_driver.hpp"
#include "parser/unit_driver.hpp"
#include "pybind/pyembed.hpp"
#include "utils/common_utils.hpp"
#include "utils/logger.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/constant_folder_visitor.hpp"
#include "visitors/global_var_visitor.hpp"
#include "visitors/inline_visitor.hpp"
#include "visitors/ispc_rename_visitor.hpp"
#include "visitors/json_visitor.hpp"
#include "visitors/kinetic_block_visitor.hpp"
#include "visitors/local_var_rename_visitor.hpp"
#include "visitors/localize_visitor.hpp"
#include "visitors/loop_unroll_visitor.hpp"
#include "visitors/neuron_solve_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/perf_visitor.hpp"
#include "visitors/solve_block_visitor.hpp"
#include "visitors/steadystate_visitor.hpp"
#include "visitors/sympy_conductance_visitor.hpp"
#include "visitors/sympy_solver_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/units_visitor.hpp"
#include "visitors/verbatim_var_rename_visitor.hpp"
#include "visitors/verbatim_visitor.hpp"

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
    bool verbose(false);

    /// true if serial c code to be generated
    bool c_backend(true);

    /// true if c code with openmp to be generated
    bool omp_backend(false);

    /// true if ispc code to be generated
    bool ispc_backend(false);

    /// true if c code with openacc to be generated
    bool oacc_backend(false);

    /// true if cuda code to be generated
    bool cuda_backend(false);

    /// true if sympy should be used for solving ODEs analytically
    bool sympy_analytic(false);

    /// true if Pade approximation to be used
    bool sympy_pade(false);

    /// true if CSE (temp variables) to be used
    bool sympy_cse(false);

    /// true if conductance keyword can be added to breakpoint
    bool sympy_conductance(false);

    /// true if inlining at nmodl level to be done
    bool nmodl_inline(false);

    /// true if unroll at nmodl level to be done
    bool nmodl_unroll(false);

    /// true if perform constant folding at nmodl level to be done
    bool nmodl_const_folding(false);

    /// true if range variables to be converted to local
    bool nmodl_localize(false);

    /// true if global variables to be converted to range
    bool nmodl_global_to_range(false);

    /// true if localize variables even if verbatim block is used
    bool localize_verbatim(false);

    /// true if local variables to be renamed
    bool local_rename(false);

    /// true if inline even if verbatim block exist
    bool verbatim_inline(false);

    /// true if verbatim blocks
    bool verbatim_rename(true);

    /// true if code generation is forced to happed even if there
    /// is any incompatibility
    bool force_codegen(false);

    /// directory where code will be generated
    std::string output_dir(".");

    /// directory where intermediate file will be generated
    std::string scratch_dir("tmp");

    /// directory where units lib file is located
    std::string units_dir(NrnUnitsLib::get_path());

    /// true if ast should be converted to json
    bool json_ast(false);

    /// true if ast should be converted to nmodl
    bool nmodl_ast(false);

    /// true if performance stats should be converted to json
    bool json_perfstat(false);

    /// true if symbol table should be printed
    bool show_symtab(false);

    /// memory layout for code generation
    std::string layout("soa");

    /// floating point data type
    std::string data_type("double");

    app.get_formatter()->column_width(40);
    app.set_help_all_flag("-H,--help-all", "Print this help message including all sub-commands");

    app.add_flag("-v,--verbose", verbose, "Verbose logger output")->ignore_case();

    app.add_option("file", mod_files, "One or more MOD files to process")
        ->ignore_case()
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-o,--output", output_dir, "Directory for backend code output", true)
        ->ignore_case();
    app.add_option("--scratch", scratch_dir, "Directory for intermediate code output", true)
        ->ignore_case();
    app.add_option("--units", units_dir, "Directory of units lib file", true)->ignore_case();

    auto host_opt = app.add_subcommand("host", "HOST/CPU code backends")->ignore_case();
    host_opt->add_flag("--c", c_backend, "C/C++ backend ({})"_format(c_backend))->ignore_case();
    host_opt->add_flag("--omp", omp_backend, "C/C++ backend with OpenMP ({})"_format(omp_backend))
        ->ignore_case();
    host_opt->add_flag("--ispc", ispc_backend, "C/C++ backend with ISPC ({})"_format(ispc_backend))
        ->ignore_case();

    auto acc_opt = app.add_subcommand("acc", "Accelerator code backends")->ignore_case();
    acc_opt
        ->add_flag("--oacc", oacc_backend, "C/C++ backend with OpenACC ({})"_format(oacc_backend))
        ->ignore_case();
    acc_opt->add_flag("--cuda", cuda_backend, "C/C++ backend with CUDA ({})"_format(cuda_backend))
        ->ignore_case();

    // clang-format off
    auto sympy_opt = app.add_subcommand("sympy", "SymPy based analysis and optimizations")->ignore_case();
    sympy_opt->add_flag("--analytic",
        sympy_analytic,
        "Solve ODEs using SymPy analytic integration ({})"_format(sympy_analytic))->ignore_case();
    sympy_opt->add_flag("--pade",
        sympy_pade,
        "Pade approximation in SymPy analytic integration ({})"_format(sympy_pade))->ignore_case();
    sympy_opt->add_flag("--cse",
        sympy_cse,
        "CSE (Common Subexpression Elimination) in SymPy analytic integration ({})"_format(sympy_cse))->ignore_case();
    sympy_opt->add_flag("--conductance",
        sympy_conductance,
        "Add CONDUCTANCE keyword in BREAKPOINT ({})"_format(sympy_conductance))->ignore_case();

    auto passes_opt = app.add_subcommand("passes", "Analyse/Optimization passes")->ignore_case();
    passes_opt->add_flag("--inline",
        nmodl_inline,
        "Perform inlining at NMODL level ({})"_format(nmodl_inline))->ignore_case();
    passes_opt->add_flag("--unroll",
        nmodl_unroll,
        "Perform loop unroll at NMODL level ({})"_format(nmodl_unroll))->ignore_case();
    passes_opt->add_flag("--const-folding",
        nmodl_const_folding,
        "Perform constant folding at NMODL level ({})"_format(nmodl_const_folding))->ignore_case();
    passes_opt->add_flag("--localize",
        nmodl_localize,
        "Convert RANGE variables to LOCAL ({})"_format(nmodl_localize))->ignore_case();
    passes_opt->add_flag("--global-to-range",
         nmodl_global_to_range,
         "Convert GLOBAL variables to RANGE ({})"_format(nmodl_global_to_range))->ignore_case();
    passes_opt->add_flag("--localize-verbatim",
        localize_verbatim,
        "Convert RANGE variables to LOCAL even if verbatim block exist ({})"_format(localize_verbatim))->ignore_case();
    passes_opt->add_flag("--local-rename",
        local_rename,
        "Rename LOCAL variable if variable of same name exist in global scope ({})"_format(local_rename))->ignore_case();
    passes_opt->add_flag("--verbatim-inline",
        verbatim_inline,
        "Inline even if verbatim block exist ({})"_format(verbatim_inline))->ignore_case();
    passes_opt->add_flag("--verbatim-rename",
        verbatim_rename,
        "Rename variables in verbatim block ({})"_format(verbatim_rename))->ignore_case();
    passes_opt->add_flag("--json-ast",
        json_ast,
        "Write AST to JSON file ({})"_format(json_ast))->ignore_case();
    passes_opt->add_flag("--nmodl-ast",
        nmodl_ast,
        "Write AST to NMODL file ({})"_format(nmodl_ast))->ignore_case();
    passes_opt->add_flag("--json-perf",
        json_perfstat,
        "Write performance statistics to JSON file ({})"_format(json_perfstat))->ignore_case();
    passes_opt->add_flag("--show-symtab",
        show_symtab,
        "Write symbol table to stdout ({})"_format(show_symtab))->ignore_case();

    auto codegen_opt = app.add_subcommand("codegen", "Code generation options")->ignore_case();
    codegen_opt->add_option("--layout",
        layout,
        "Memory layout for code generation",
        true)->ignore_case()->check(CLI::IsMember({"aos", "soa"}));
    codegen_opt->add_option("--datatype",
        layout,
        "Data type for floating point variables",
        true)->ignore_case()->check(CLI::IsMember({"float", "double"}));
    codegen_opt->add_flag("--force",
        force_codegen,
        "Force code generation even if there is any incompatibility");

    // clang-format on

    CLI11_PARSE(app, argc, argv);

    // if any of the other backends is used we force the C backend to be off.
    if (omp_backend || ispc_backend) {
        c_backend = false;
    }

    utils::make_path(output_dir);
    utils::make_path(scratch_dir);

    if (sympy_opt) {
        nmodl::pybind_wrappers::EmbeddedPythonLoader::get_instance()
            .api()
            ->initialize_interpreter();
    }

    if (verbose) {
        logger->set_level(spdlog::level::debug);
    }

    /// write ast to nmodl
    const auto ast_to_nmodl = [nmodl_ast](ast::Program& ast, const std::string& filepath) {
        if (nmodl_ast) {
            NmodlPrintVisitor(filepath).visit_program(ast);
            logger->info("AST to NMODL transformation written to {}", filepath);
        }
    };

    for (const auto& file: mod_files) {
        logger->info("Processing {}", file);

        const auto modfile = utils::remove_extension(utils::base_name(file));

        /// create file path for nmodl file
        auto filepath = [scratch_dir, modfile](const std::string& suffix) {
            static int count = 0;
            return "{}/{}.{}.{}.mod"_format(scratch_dir, modfile, std::to_string(count++), suffix);
        };

        /// driver object creates lexer and parser, just call parser method
        NmodlDriver driver;

        /// parse mod file and construct ast
        const auto& ast = driver.parse_file(file);

        /// whether to update existing symbol table or create new
        /// one whenever we run symtab visitor.
        bool update_symtab = false;

        /// just visit the ast
        AstVisitor().visit_program(*ast);

        /// construct symbol table
        {
            logger->info("Running symtab visitor");
            SymtabVisitor(update_symtab).visit_program(*ast);
        }

        /// Rename variables that match ISPC compiler double constants
        if (ispc_backend) {
            IspcRenameVisitor(ast).visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            ast_to_nmodl(*ast, filepath("ispc_double_rename"));
        }

        /// GLOBAL to RANGE rename visitor
        if (nmodl_global_to_range) {
            // make sure to run perf visitor because code generator
            // looks for read/write counts const/non-const declaration
            PerfVisitor().visit_program(*ast);
            // make sure to run the GlobalToRange visitor after all the
            // reinitializations of Symtab
            logger->info("Running GlobalToRange visitor");
            GlobalToRangeVisitor(ast).visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            ast_to_nmodl(*ast, filepath("globaltorange"));
        }

        {
            // Compatibility Checking
            logger->info("Running code compatibility checker");
            // run perfvisitor to update read/wrie counts
            PerfVisitor().visit_program(*ast);
            // If there is an incompatible construct and code generation is not forced exit NMODL
            if (CodegenCompatibilityVisitor().find_unhandled_ast_nodes(*ast) && !force_codegen) {
                return 1;
            }
        }

        if (show_symtab) {
            logger->info("Printing symbol table");
            auto symtab = ast->get_model_symbol_table();
            symtab->print(std::cout);
        }

        ast_to_nmodl(*ast, filepath("ast"));

        if (json_ast) {
            logger->info("Writing AST into {}", file);
            auto file = scratch_dir + "/" + modfile + ".ast.json";
            JSONVisitor(file).visit_program(*ast);
        }

        if (verbatim_rename) {
            logger->info("Running verbatim rename visitor");
            VerbatimVarRenameVisitor().visit_program(*ast);
            ast_to_nmodl(*ast, filepath("verbatim_rename"));
        }

        if (nmodl_const_folding) {
            logger->info("Running nmodl constant folding visitor");
            ConstantFolderVisitor().visit_program(*ast);
            ast_to_nmodl(*ast, filepath("constfold"));
        }

        if (nmodl_unroll) {
            logger->info("Running nmodl loop unroll visitor");
            LoopUnrollVisitor().visit_program(*ast);
            ConstantFolderVisitor().visit_program(*ast);
            ast_to_nmodl(*ast, filepath("unroll"));
            SymtabVisitor(update_symtab).visit_program(*ast);
        }

        /// note that we can not symtab visitor in update mode as we
        /// replace kinetic block with derivative block of same name
        /// in global scope
        {
            logger->info("Running KINETIC block visitor");
            auto kineticBlockVisitor = KineticBlockVisitor();
            kineticBlockVisitor.visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            const auto filename = filepath("kinetic");
            ast_to_nmodl(*ast, filename);
            if (nmodl_ast && kineticBlockVisitor.get_conserve_statement_count()) {
                logger->warn(
                    "{} presents non-standard CONSERVE statements in DERIVATIVE blocks. Use it only for debugging/developing"_format(
                        filename));
            }
        }

        {
            logger->info("Running STEADYSTATE visitor");
            SteadystateVisitor().visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            ast_to_nmodl(*ast, filepath("steadystate"));
        }

        /// Parsing units fron "nrnunits.lib" and mod files
        {
            logger->info("Parsing Units");
            UnitsVisitor(units_dir).visit_program(*ast);
        }

        /// once we start modifying (especially removing) older constructs
        /// from ast then we should run symtab visitor in update mode so
        /// that old symbols (e.g. prime variables) are not lost
        update_symtab = true;

        if (nmodl_inline) {
            logger->info("Running nmodl inline visitor");
            InlineVisitor().visit_program(*ast);
            ast_to_nmodl(*ast, filepath("inline"));
        }

        if (local_rename) {
            logger->info("Running local variable rename visitor");
            LocalVarRenameVisitor().visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            ast_to_nmodl(*ast, filepath("local_rename"));
        }

        if (nmodl_localize) {
            // localize pass must follow rename pass to avoid conflict
            logger->info("Running localize visitor");
            LocalizeVisitor(localize_verbatim).visit_program(*ast);
            LocalVarRenameVisitor().visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            ast_to_nmodl(*ast, filepath("localize"));
        }

        if (sympy_conductance) {
            logger->info("Running sympy conductance visitor");
            SympyConductanceVisitor().visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            ast_to_nmodl(*ast, filepath("sympy_conductance"));
        }

        if (sympy_analytic) {
            logger->info("Running sympy solve visitor");
            SympySolverVisitor(sympy_pade, sympy_cse).visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            ast_to_nmodl(*ast, filepath("sympy_solve"));
        }

        {
            logger->info("Running cnexp visitor");
            NeuronSolveVisitor().visit_program(*ast);
            ast_to_nmodl(*ast, filepath("cnexp"));
        }

        {
            SolveBlockVisitor().visit_program(*ast);
            SymtabVisitor(update_symtab).visit_program(*ast);
            ast_to_nmodl(*ast, filepath("solveblock"));
        }

        if (json_perfstat) {
            auto file = scratch_dir + "/" + modfile + ".perf.json";
            logger->info("Writing performance statistics to {}", file);
            PerfVisitor(file).visit_program(*ast);
        }

        {
            // make sure to run perf visitor because code generator
            // looks for read/write counts const/non-const declaration
            PerfVisitor().visit_program(*ast);
        }

        {
            auto mem_layout = layout == "aos" ? codegen::LayoutType::aos : codegen::LayoutType::soa;


            if (ispc_backend) {
                logger->info("Running ISPC backend code generator");
                CodegenIspcVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(*ast);
            }

            else if (oacc_backend) {
                logger->info("Running OpenACC backend code generator");
                CodegenAccVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(*ast);
            }

            else if (omp_backend) {
                logger->info("Running OpenMP backend code generator");
                CodegenOmpVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(*ast);
            }

            else if (c_backend) {
                logger->info("Running C backend code generator");
                CodegenCVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(*ast);
            }

            if (cuda_backend) {
                logger->info("Running CUDA backend code generator");
                CodegenCudaVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(*ast);
            }
        }
    }

    if (sympy_opt) {
        nmodl::pybind_wrappers::EmbeddedPythonLoader::get_instance().api()->finalize_interpreter();
    }
}
