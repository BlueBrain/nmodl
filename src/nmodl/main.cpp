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
#include "codegen/codegen_acc_visitor.hpp"
#include "codegen/codegen_c_visitor.hpp"
#include "codegen/codegen_cuda_visitor.hpp"
#include "codegen/codegen_omp_visitor.hpp"
#include "parser/nmodl_driver.hpp"
#include "pybind11/embed.h"
#include "utils/common_utils.hpp"
#include "utils/logger.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/cnexp_solve_visitor.hpp"
#include "visitors/inline_visitor.hpp"
#include "visitors/json_visitor.hpp"
#include "visitors/local_var_rename_visitor.hpp"
#include "visitors/localize_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/perf_visitor.hpp"
#include "visitors/sympy_conductance_visitor.hpp"
#include "visitors/sympy_solver_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/verbatim_var_rename_visitor.hpp"
#include "visitors/verbatim_visitor.hpp"

using namespace nmodl;
using namespace codegen;
using nmodl::parser::NmodlDriver;

int main(int argc, const char* argv[]) {
    CLI::App app{"NMODL : Source-to-Source Code Generation Framework"};

    /// list of mod files to process
    std::vector<std::string> mod_files;

    /// true if debug logger statements should be shown
    bool verbose(false);

    /// true if serial c code to be generated
    bool c_backend(true);

    /// true if c code with openmp to be generated
    bool omp_backend(false);

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

    /// true if range variables to be converted to local
    bool localize(false);

    /// true if localize variables even if verbatim block is used
    bool localize_verbatim(false);

    /// true if local variables to be renamed
    bool local_rename(false);

    /// true if inline even if verbatim block exist
    bool verbatim_inline(false);

    /// true if verbatim blocks
    bool verbatim_rename(false);

    /// directory where code will be generated
    std::string output_dir(".");

    /// directory where intermediate file will be generated
    std::string scratch_dir("tmp");

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

    auto host_opt = app.add_subcommand("host", "HOST/CPU code backends")->ignore_case();
    host_opt->add_flag("--c", c_backend, "C/C++ backend")->ignore_case();
    host_opt->add_flag("--omp", omp_backend, "C/C++ backend with OpenMP")->ignore_case();

    auto acc_opt = app.add_subcommand("acc", "Accelerator code backends")->ignore_case();
    acc_opt->add_flag("--oacc", oacc_backend, "C/C++ backend with OpenACC")->ignore_case();
    acc_opt->add_flag("--cuda", cuda_backend, "C/C++ backend with CUDA")->ignore_case();

    // clang-format off
    auto sympy_opt = app.add_subcommand("sympy", "SymPy based analysis and optimizations")->ignore_case();
    sympy_opt->add_flag("--analytic", sympy_analytic, "Solve ODEs using SymPy analytic integration")->ignore_case();
    sympy_opt->add_flag("--pade", sympy_pade, "Pade approximation in SymPy analytic integration")->ignore_case();
    sympy_opt->add_flag("--cse", sympy_cse, "CSE (Common Sub Expressions) in SymPy analytic integration")->ignore_case();
    sympy_opt->add_flag("--conductance", sympy_conductance, "Add CONDUCTANCE keyword in BREAKPOINT")->ignore_case();

    auto passes_opt = app.add_subcommand("passes", "Analyse/Optimization passes")->ignore_case();
    passes_opt->add_flag("--inline", nmodl_inline, "Perform inlining at NMODL level")->ignore_case();
    passes_opt->add_flag("--localize", localize, "Convert RANGE variables to LOCAL")->ignore_case();
    passes_opt->add_flag("--localize-verbatim", localize_verbatim, "Convert RANGE variables to LOCAL even if verbatim block exist")->ignore_case();
    passes_opt->add_flag("--local-rename", local_rename, "Rename LOCAL variable if variable of same name exist in global scope")->ignore_case();
    passes_opt->add_flag("--verbatim-inline", verbatim_inline, "Inline even if verbatim block exist")->ignore_case();
    passes_opt->add_flag("--verbatim-rename", verbatim_rename, "Rename variables in verbatim block")->ignore_case();
    passes_opt->add_flag("--json-ast", json_ast, "Write AST to JSON file")->ignore_case();
    passes_opt->add_flag("--nmodl-ast", nmodl_ast, "Write AST to NMODL file")->ignore_case();
    passes_opt->add_flag("--json-perf", json_perfstat, "Write performance statistics to JSON file")->ignore_case();
    passes_opt->add_flag("--show-symtab", show_symtab, "Write symbol table to stdout")->ignore_case();

    auto codegen_opt = app.add_subcommand("codegen", "Code generation options")->ignore_case();
    codegen_opt->add_option("--layout", layout, "Memory layout for code generation", true)->ignore_case()->check(CLI::IsMember({"aos", "soa"}));
    codegen_opt->add_option("--datatype", layout, "Data type for floating point variables", true)->ignore_case()->check(CLI::IsMember({"float", "double"}));
    // clang-format on

    CLI11_PARSE(app, argc, argv);

    make_path(output_dir);
    make_path(scratch_dir);

    if (sympy_opt) {
        pybind11::initialize_interpreter();
    }

    if (verbose) {
        logger->set_level(spdlog::level::debug);
    }

    /// write ast to nmodl
    auto ast_to_nmodl = [&](ast::Program* ast, const std::string& filepath) {
        if (nmodl_ast) {
            NmodlPrintVisitor v(filepath);
            v.visit_program(ast);
            logger->info("AST to NMODL transformation written to {}", filepath);
        }
    };

    for (const auto& file: mod_files) {
        logger->info("Processing {}", file);

        auto modfile = remove_extension(base_name(file));

        /// create file path for nmodl file
        auto filepath = [&](std::string suffix) {
            static int count = 0;
            return scratch_dir + "/" + modfile + "." + std::to_string(count++) + "." + suffix +
                   ".mod";
        };

        /// driver object creates lexer and parser, just call parser method
        NmodlDriver driver;
        driver.parse_file(file);

        /// parse mod file and construct ast
        auto ast = driver.ast();

        /// just visit the astt
        {
            AstVisitor v;
            v.visit_program(ast.get());
        }

        /// construct symbol table
        {
            logger->info("Running symtab visitor");
            SymtabVisitor v(false);
            v.visit_program(ast.get());
        }

        if (show_symtab) {
            logger->info("Printing symbol table");
            std::stringstream stream;
            auto symtab = ast->get_model_symbol_table();
            symtab->print(stream);
            std::cout << stream.str();
        }

        ast_to_nmodl(ast.get(), filepath("ast"));

        if (json_ast) {
            logger->info("Writing AST into {}", file);
            auto file = scratch_dir + "/" + modfile + ".ast.json";
            JSONVisitor v(file);
            v.visit_program(ast.get());
            {
                SymtabVisitor v(false);
                v.visit_program(ast.get());
            }
        }

        if (verbatim_rename) {
            logger->info("Running verbatim rename visitor");
            VerbatimVarRenameVisitor v;
            v.visit_program(ast.get());
            ast_to_nmodl(ast.get(), filepath("verbatim_rename"));
        }

        if (sympy_conductance) {
            logger->info("Running sympy conductance visitor");
            SympyConductanceVisitor v1;
            SymtabVisitor v2(false);
            v1.visit_program(ast.get());
            v2.visit_program(ast.get());
            ast_to_nmodl(ast.get(), filepath("sympy_conductance"));
        }

        if (sympy_analytic) {
            logger->info("Running sympy solve visitor");
            SympySolverVisitor v(sympy_pade, sympy_cse);
            v.visit_program(ast.get());
            ast_to_nmodl(ast.get(), filepath("sympy_solve"));
        }

        {
            logger->info("Running cnexp visitor");
            CnexpSolveVisitor v;
            v.visit_program(ast.get());
            ast_to_nmodl(ast.get(), filepath("cnexp"));
        }

        if (nmodl_inline) {
            logger->info("Running nmodl inline visitor");
            InlineVisitor v;
            v.visit_program(ast.get());
            ast_to_nmodl(ast.get(), filepath("inline"));
        }

        if (local_rename) {
            logger->info("Running local variable rename visitor");
            LocalVarRenameVisitor v1;
            SymtabVisitor v2(true);
            v1.visit_program(ast.get());
            v2.visit_program(ast.get());
            ast_to_nmodl(ast.get(), filepath("local_rename"));
        }

        if (localize) {
            // localize pass must follow rename pass to avoid conflict
            logger->info("Running localize visitor");
            LocalizeVisitor v1(localize_verbatim);
            LocalVarRenameVisitor v2;
            SymtabVisitor v3(true);
            v1.visit_program(ast.get());
            v2.visit_program(ast.get());
            v3.visit_program(ast.get());
            ast_to_nmodl(ast.get(), filepath("localize"));
        }

        if (json_perfstat) {
            auto file = scratch_dir + "/" + modfile + ".perf.json";
            logger->info("Writing performance statistics to {}", file);
            PerfVisitor v(file);
            v.visit_program(ast.get());
        }

        {
            // make sure to run perf visitor because code generator
            // looks for read/write counts const/non-const declaration
            PerfVisitor v;
            v.visit_program(ast.get());
        }

        {
            auto mem_layout = layout == "aos" ? codegen::LayoutType::aos : codegen::LayoutType::soa;

            if (c_backend) {
                logger->info("Running C backend code generator");
                CodegenCVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(ast.get());
            }
            if (omp_backend) {
                logger->info("Running OpenMP backend code generator");
                CodegenOmpVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(ast.get());
            }
            if (oacc_backend) {
                logger->info("Running OpenACC backend code generator");
                CodegenAccVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(ast.get());
            }

            if (cuda_backend) {
                logger->info("Running CUDA backend code generator");
                CodegenCudaVisitor visitor(modfile, output_dir, mem_layout, data_type);
                visitor.visit_program(ast.get());
            }
        }
    }

    if (sympy_opt) {
        pybind11::finalize_interpreter();
    }
}
