/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>

#include "ast/program.hpp"
#include "parser/nmodl_driver.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/inline_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using nmodl::parser::NmodlDriver;

//=============================================================================
// Sample LLVM codegen test
//=============================================================================

std::string run_llvm_visitor(const std::string& text) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    InlineVisitor().visit_program(*ast);

    codegen::CodegenLLVMVisitor llvm_visitor("unknown", ".");
    llvm_visitor.visit_program(*ast);
    return llvm_visitor.print_module();
}

SCENARIO("Running LLVM Codegen", "[visitor][llvm]") {
    GIVEN("Simple procedure with local assignment") {
        std::string nmodl_text = R"(
            PROCEDURE one_arg(x) {
                LOCAL w
                w = x
            }
        )";

        THEN("Generated LLVM code") {
            std::string expected = "; ModuleID = 'unknown'\n"
                                   "source_filename = \"unknown\"\n"
                                   "\n"
                                   "define void @one_arg(double %x1) {\n"
                                   "  %x = alloca double, align 8\n"
                                   "  store double %x1, double* %x, align 8\n"
                                   "  %w = alloca double, align 8\n"
                                   "  %1 = load double, double* %x, align 8\n"
                                   "  store double %1, double* %w, align 8\n"
                                   "}\n"
                                   "";
            auto result = run_llvm_visitor(nmodl_text);
            REQUIRE(result == expected);
        }
    }
}