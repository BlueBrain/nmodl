/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <CLI/CLI.hpp>

#include "ast/program.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "jit_driver.hpp"
#include "parser/nmodl_driver.hpp"
#include "utils/logger.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace runner;

int main(int argc, const char* argv[]) {
    CLI::App app{
        "NMODL LLVM Runner : Executes functions from a MOD file via LLVM IR code generation"};

    // Currently, only a sinle MOD file is supported, as well as an entry point with a double return
    // type. While returning a double value is a general case in NMODL,it will be nice to have a
    // more generic functionality. \todo: Add support for different return types (int, void).

    std::string filename;
    std::string entry_point_name = "main";

    app.add_option("-f,--file,file", filename, "A single MOD file source")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-e,--entry-point,entry-point",
                   entry_point_name,
                   "An entry point function from the MOD file");

    CLI11_PARSE(app, argc, argv);

    logger->info("Parsing MOD file to AST");
    parser::NmodlDriver driver;
    const auto& ast = driver.parse_file(filename);

    logger->info("Running Symtab Visitor");
    visitor::SymtabVisitor().visit_program(*ast);

    logger->info("Running LLVM Visitor");
    codegen::CodegenLLVMVisitor llvm_visitor(filename, /*output_dir=*/".", /*opt_passes=*/false);
    llvm_visitor.visit_program(*ast);

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    logger->info("Setting up JIT");
    std::unique_ptr<JITDriver> jit_runner = std::make_unique<JITDriver>(std::move(llvm_visitor.get_module()));
    jit_runner->init();
    jit_runner->execute(entry_point_name);

    return 0;
}
