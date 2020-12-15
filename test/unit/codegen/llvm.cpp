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
// LocalList and LocalVar
//=============================================================================

SCENARIO("Local variable", "[visitor][llvm]") {
    GIVEN("Procedure with some local variables") {
        std::string nmodl_text = R"(
            PROCEDURE local() {
                LOCAL i, j
            }
        )";

        THEN("local variables are allocated on the stack") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check stack allocations for i and j
            std::regex i(R"(%i = alloca double)");
            std::regex j(R"(%j = alloca double)");
            REQUIRE(std::regex_search(module_string, m, i));
            REQUIRE(std::regex_search(module_string, m, j));
        }
    }
}

//=============================================================================
// ProcedureBlock
//=============================================================================

SCENARIO("Procedure", "[visitor][llvm]") {
    GIVEN("Empty procedure with no arguments") {
        std::string nmodl_text = R"(
            PROCEDURE empty() {}
        )";

        THEN("empty void function is produced") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check procedure has empty body
            std::regex procedure(R"(define void @empty\(\) \{\n\})");
            REQUIRE(std::regex_search(module_string, m, procedure));
        }
    }

    GIVEN("Empty procedure with arguments") {
        std::string nmodl_text = R"(
            PROCEDURE with_argument(x) {}
        )";

        THEN("void function is produced with arguments allocated on stack") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check procedure signature
            std::regex function_signature(R"(define void @with_argument\(double %x1\) \{)");
            REQUIRE(std::regex_search(module_string, m, function_signature));

            // Check that procedure arguments are allocated on the local stack
            std::regex alloca_instr(R"(%x = alloca double)");
            std::regex store_instr(R"(store double %x1, double\* %x)");
            REQUIRE(std::regex_search(module_string, m, alloca_instr));
            REQUIRE(std::regex_search(module_string, m, store_instr));
        }
    }
}
