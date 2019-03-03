/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <fstream>
#include <iostream>
#include <sstream>

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

using nmodl::codegen::LayoutType;


void ast_to_nmodl(ast::Program* ast, const std::string& filename) {
    NmodlPrintVisitor v(filename);
    v.visit_program(ast);
    logger->info("AST to NMODL transformation written to {}", filename);
}

int main(int argc, const char* argv[]) {
    CLI::App app{"NMODL : Source-to-Source Code Generation Framework"};

    app.get_formatter()->column_width(40);
    app.set_help_all_flag("-H,--help-all", "Print this help message including all sub-commands");

    std::vector<std::string> files;
    auto file_opt = app.add_subcommand("file", "List of mod files")->ignore_case();
    file_opt->add_option("file", files, "One or more MOD files to process")
        ->required()
        ->ignore_case()
        ->check(CLI::ExistingFile);

    // clang-format off
    auto host_opt = app.add_subcommand("host", "HOST/CPU code backends")->ignore_case();
    auto c_backend = host_opt->add_flag("--c", "C/C++ backend")->ignore_case();
    auto omp_backend = host_opt->add_flag("--omp", "C/C++ backend with OpenMP")->ignore_case();
    auto ispc_backend = host_opt->add_flag("--ispc", "C/C++ backend with ISPC")->ignore_case();

    auto acc_opt = app.add_subcommand("acc", "Accelerator code backends")->ignore_case();
    auto oacc_backend = acc_opt->add_flag("--oacc", "C/C++ backend with OpenACC")->ignore_case();
    auto cuda_backend = acc_opt->add_flag("--cuda", "C/C++ backend with CUDA")->ignore_case();

    auto sympy_opt = app.add_subcommand("sympy", "SymPy based analysis and optimizations")->ignore_case();
    auto sympy_solver = sympy_opt->add_flag("--analytic", "Solve ODEs using SymPy analytic integration")->ignore_case();
    auto sympy_pade = sympy_opt->add_flag("--pade", "Pade approximation in SymPy analytic integration")->ignore_case();
    auto sympy_conductance = sympy_opt->add_flag("--conductance", "Add CONDUCTANCE keyword in BREAKPOINT")->ignore_case();

    auto optz_opt = app.add_subcommand("passes", "Analyse/Optimization passes")->ignore_case();
    auto nmodl_inline = optz_opt->add_flag("--inline", "Perform inlining at NMODL level")->ignore_case();
    auto verbatim_inline = optz_opt->add_flag("--verbatim-inline", "Inline even if verbatim block exist")->ignore_case();
    auto localize = optz_opt->add_flag("--localize", "Convert RANGE variables to LOCAL")->ignore_case();
    auto localize_verbatim = optz_opt->add_flag("--localize-verbatim", "Convert RANGE variables to LOCAL even if verbatim block exist")->ignore_case();
    auto local_rename = optz_opt->add_flag("--local-rename", "Rename LOCAL variable if variable of same name exist in global scope")->ignore_case();
    auto verbatim_rename = optz_opt->add_flag("--verbatim-rename", "Rename variables in verbatim block")->ignore_case();

    std::string output_dir(".");
    std::string scratch_dir("tmp");
    auto output_opt = app.add_subcommand("output", "Code output")->ignore_case();
    output_opt->add_option("--dir", output_dir, "Directory for backend code output")->ignore_case();
    output_opt->add_option("--scratch", scratch_dir, "Directory for intermediate code output")->ignore_case();
    auto ast_json = output_opt->add_flag("--json-ast", "Write AST to JSON file")->ignore_case();
    auto ast_nmodl = output_opt->add_flag("--nmodl-ast", "Write AST to NMODL file")->ignore_case();
    auto perfstat_json = output_opt->add_flag("--json-perf", "Write performance statistics to JSON file")->ignore_case();
    auto show_symtab = output_opt->add_flag("--show-symtab", "Write symbol table to stdout")->ignore_case();

    std::string layout("soa");
    std::string data_type("double");
    auto codegen_opt = app.add_subcommand("codegen", "Code generation options")->ignore_case();
    codegen_opt->add_option("--layout", layout, "Memory layout for code generation")->check(CLI::IsMember({"aos", "soa"}));
    codegen_opt->add_option("--datatype", layout, "Data type for floating point variables")->check(CLI::IsMember({"float", "double"}));

    // clang-format on

    CLI11_PARSE(app, argc, argv);

    /// create output directories
    make_path(output_dir);
    make_path(scratch_dir);

    if (*sympy_opt) {
        pybind11::initialize_interpreter();
    }

    for (const auto& nmodl_file: files) {
        logger->info("Processing {}", nmodl_file);

        std::ifstream file(nmodl_file);
        auto mod = remove_extension(base_name(nmodl_file));

        /// driver object creates lexer and parser, just call parser method
        nmodl::parser::NmodlDriver driver;
        driver.parse_file(nmodl_file);

        /// shared_ptr to ast constructed from parsing nmodl file
        auto ast = driver.ast();

        {
            AstVisitor v;
            v.visit_program(ast.get());
        }

        {
            SymtabVisitor v(false);
            v.visit_program(ast.get());
        }

        if (*ast_nmodl) {
            ast_to_nmodl(ast.get(), scratch_dir + "/" + mod + ".nmodl.mod");
        }

        if (*verbatim_rename) {
            VerbatimVarRenameVisitor v;
            v.visit_program(ast.get());
            if (*ast_nmodl) {
                ast_to_nmodl(ast.get(), scratch_dir + "/" + mod + ".nmodl.verbrename.mod");
            }
        }

        if (*sympy_conductance) {
            SympyConductanceVisitor v;
            v.visit_program(ast.get());
            {
                SymtabVisitor v(false);
                v.visit_program(ast.get());
            }
            if (*ast_nmodl) {
                ast_to_nmodl(ast.get(), scratch_dir + "/" + mod + ".nmodl.conductance.mod");
            }
        }

        if (*sympy_solver) {
            bool pade = *sympy_pade;
            SympySolverVisitor v(pade);
            v.visit_program(ast.get());
        }

        {
            CnexpSolveVisitor v;
            v.visit_program(ast.get());
        }

        if (*ast_nmodl) {
            ast_to_nmodl(ast.get(), scratch_dir + "/" + mod + ".nmodl.cnexp.mod");
        }

        if (*nmodl_inline) {
            InlineVisitor v;
            v.visit_program(ast.get());
            if (*ast_nmodl) {
                ast_to_nmodl(ast.get(), scratch_dir + "/" + mod + ".nmodl.in.mod");
            }
        }

        if (*local_rename) {
            LocalVarRenameVisitor v;
            v.visit_program(ast.get());
            if (*ast_nmodl) {
                ast_to_nmodl(ast.get(), scratch_dir + "/" + mod + ".nmodl.locrename.mod");
            }
        }

        {
            SymtabVisitor v(true);
            v.visit_program(ast.get());
        }

        if (*localize) {
            // localize pass must be followed by renaming in order to avoid conflict
            // with global scope variables
            LocalizeVisitor v1(*localize_verbatim);
            v1.visit_program(ast.get());
            LocalVarRenameVisitor v2;
            v2.visit_program(ast.get());
            if (*ast_nmodl) {
                ast_to_nmodl(ast.get(), scratch_dir + "/" + mod + ".nmodl.localize.mod");
            }
        }

        {
            SymtabVisitor v(true);
            v.visit_program(ast.get());
        }

        if (*perfstat_json) {
            PerfVisitor v(scratch_dir + "/" + mod + ".perf.json");
            logger->info("Dumping performance statistics into JSON format");
            v.visit_program(ast.get());
        }

        if (*ast_json) {
            JSONVisitor v(scratch_dir + "/" + mod + ".ast.json");
            logger->info("Dumping AST state into JSON format");
            v.visit_program(ast.get());
        }

        if (*show_symtab) {
            logger->info("Printing symbol table");
            std::stringstream stream;
            auto symtab = ast->get_model_symbol_table();
            symtab->print(stream);
            std::cout << stream.str();
        }

        {
            // make sure to run perf visitor because code generator
            // looks for read/write counts const/non-const declaration
            PerfVisitor v;
            v.visit_program(ast.get());

            auto mem_layout = layout == "aos" ? LayoutType::aos : LayoutType::soa;

            if (*c_backend) {
                CodegenCVisitor visitor(mod, output_dir, mem_layout, data_type);
                visitor.visit_program(ast.get());
            }
            if (*omp_backend) {
                nmodl::codegen::CodegenOmpVisitor visitor(mod, output_dir, mem_layout, data_type);
                visitor.visit_program(ast.get());
            }
            if (*oacc_backend) {
                CodegenAccVisitor visitor(mod, output_dir, mem_layout, data_type);
                visitor.visit_program(ast.get());
            }

            if (*cuda_backend) {
                CodegenCudaVisitor visitor(mod, output_dir, mem_layout, data_type);
                visitor.visit_program(ast.get());
            }
        }
    }

    if (*sympy_opt) {
        pybind11::finalize_interpreter();
    }
}
