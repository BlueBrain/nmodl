/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <CLI/CLI.hpp>

#include "ast/program.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "parser/nmodl_driver.hpp"
#include "utils/logger.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;

int main(int argc, const char* argv[]) {
    CLI::App app{"NMODL LLVM Runner : Executes functions from a MOD file via LLVM IR code generation"};

    std::string filename;
    std::string entry_point;

    app.add_option("-f,--file,file", filename, "A single MOD file source")->required()->check(CLI::ExistingFile);
    app.add_option("-e,--entry-point,entry-point", entry_point, "An entry point function from the MOD file")->required();

    CLI11_PARSE(app, argc, argv);

    logger->info("Parsing MOD file to AST");
    parser::NmodlDriver driver;
    const auto& ast = driver.parse_file(filename);

    logger->info("Running Symtab Visitor");
    visitor::SymtabVisitor().visit_program(*ast);

    // Run LLVM visitor and generate LLVM module

    // Create JIT class (TODO)

    // Do smth like jit.run();

    return 0;
}
