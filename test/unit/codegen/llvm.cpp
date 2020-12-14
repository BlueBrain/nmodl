/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>
#include <regex>

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
// Utility to get LLVM module as a string
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

//=============================================================================
// Procedure test
//=============================================================================

SCENARIO("Procedure", "[visitor][llvm]") {
    GIVEN("Empty procedure with no arguments") {
        std::string nmodl_text = R"(
            PROCEDURE empty() {}
        )";

        THEN("empty void function is produced") {
            std::smatch m;
            std::regex expected("define void @empty\\(\\) \\{\n"
                                "\\}"
            );
            std::string actual = run_llvm_visitor(nmodl_text);
            REQUIRE(std::regex_search(actual, m, expected));
        }
    }

    GIVEN("Empty procedure with arguments") {
        std::string nmodl_text = R"(
            PROCEDURE with_argument(x) {}
        )";

        THEN("empty void function is produced") {
            std::smatch m;
            std::regex expected("define void @with_argument\\(double %x1 \\) \\{\n"
                                "  %x = alloca double, align 8\n"
                                "  store double %x1, double\\* %x, align 8\n"
                                "\\}"
            );
            std::string actual = run_llvm_visitor(nmodl_text);
            std::regex_search(actual, m, expected);
            REQUIRE(std::regex_search(actual, m, expected));
        }
    }
}
