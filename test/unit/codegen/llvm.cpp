/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>
#include <regex>

#include "ast/program.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "parser/nmodl_driver.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using nmodl::parser::NmodlDriver;

//=============================================================================
// Utility to get LLVM module as a string
//=============================================================================

std::string run_llvm_visitor(const std::string& text, bool opt = false) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);

    codegen::CodegenLLVMVisitor llvm_visitor("unknown", ".", opt);
    llvm_visitor.visit_program(*ast);
    return llvm_visitor.print_module();
}

//=============================================================================
// BinaryExpression and Double
//=============================================================================

SCENARIO("Binary expression", "[visitor][llvm]") {
    GIVEN("Procedure with addition of its arguments") {
        std::string nmodl_text = R"(
            PROCEDURE add(a, b) {
                LOCAL i
                i = a + b
            }
        )";

        THEN("variables are loaded and add instruction is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            std::regex rhs(R"(%1 = load double, double\* %b)");
            std::regex lhs(R"(%2 = load double, double\* %a)");
            std::regex res(R"(%3 = fadd double %2, %1)");

            // Check the values are loaded correctly and added
            REQUIRE(std::regex_search(module_string, m, rhs));
            REQUIRE(std::regex_search(module_string, m, lhs));
            REQUIRE(std::regex_search(module_string, m, res));
        }
    }

    GIVEN("Procedure with multiple binary operators") {
        std::string nmodl_text = R"(
            PROCEDURE multiple(a, b) {
                LOCAL i
                i = (a - b) / (a + b)
            }
        )";

        THEN("variables are processed from rhs first") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check rhs
            std::regex rr(R"(%1 = load double, double\* %b)");
            std::regex rl(R"(%2 = load double, double\* %a)");
            std::regex x(R"(%3 = fadd double %2, %1)");
            REQUIRE(std::regex_search(module_string, m, rr));
            REQUIRE(std::regex_search(module_string, m, rl));
            REQUIRE(std::regex_search(module_string, m, x));

            // Check lhs
            std::regex lr(R"(%4 = load double, double\* %b)");
            std::regex ll(R"(%5 = load double, double\* %a)");
            std::regex y(R"(%6 = fsub double %5, %4)");
            REQUIRE(std::regex_search(module_string, m, lr));
            REQUIRE(std::regex_search(module_string, m, ll));
            REQUIRE(std::regex_search(module_string, m, y));

            // Check result
            std::regex res(R"(%7 = fdiv double %6, %3)");
            REQUIRE(std::regex_search(module_string, m, res));
        }
    }

    GIVEN("Procedure with assignment") {
        std::string nmodl_text = R"(
            PROCEDURE assignment() {
                LOCAL i
                i = 2
            }
        )";

        THEN("double constant is stored into i") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check store immediate is created
            std::regex allocation(R"(%i = alloca double)");
            std::regex assignment(R"(store double 2.0*e\+00, double\* %i)");
            REQUIRE(std::regex_search(module_string, m, allocation));
            REQUIRE(std::regex_search(module_string, m, assignment));
        }
    }
}

//=============================================================================
// Define
//=============================================================================

SCENARIO("Define", "[visitor][llvm]") {
    GIVEN("Procedure with array variable of length specified by DEFINE") {
        std::string nmodl_text = R"(
            DEFINE N 100

            PROCEDURE foo() {
                LOCAL x[N]
            }
        )";

        THEN("macro is expanded and array is allocated") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check stack allocations for i and j
            std::regex array(R"(%x = alloca \[100 x double\])");
            REQUIRE(std::regex_search(module_string, m, array));
        }
    }
}

//=============================================================================
// FunctionBlock
//=============================================================================

SCENARIO("Function", "[visitor][llvm]") {
    GIVEN("Simple function with arguments") {
        std::string nmodl_text = R"(
            FUNCTION foo(x) {
               foo = x
            }
        )";

        THEN("function is produced with arguments allocated on stack and a return instruction") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check function signature. The return type should be the default double type.
            std::regex function_signature(R"(define double @foo\(double %x1\) \{)");
            REQUIRE(std::regex_search(module_string, m, function_signature));

            // Check that function arguments are allocated on the local stack.
            std::regex alloca_instr(R"(%x = alloca double)");
            std::regex store_instr(R"(store double %x1, double\* %x)");
            REQUIRE(std::regex_search(module_string, m, alloca_instr));
            REQUIRE(std::regex_search(module_string, m, store_instr));

            // Check the return variable has also been allocated.
            std::regex ret_instr(R"(%ret_foo = alloca double)");

            // Check that the return value has been loaded and passed to terminator.
            std::regex loaded(R"(%2 = load double, double\* %ret_foo)");
            std::regex terminator(R"(ret double %2)");
            REQUIRE(std::regex_search(module_string, m, loaded));
            REQUIRE(std::regex_search(module_string, m, terminator));
        }
    }
}

//=============================================================================
// FunctionCall
//=============================================================================

SCENARIO("Function call", "[visitor][llvm]") {
    GIVEN("A call to procedure") {
        std::string nmodl_text = R"(
            PROCEDURE bar() {}
            FUNCTION foo() {
                bar()
            }
        )";

        THEN("a void call instruction is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check for call instruction.
            std::regex call(R"(call void @bar\(\))");
            REQUIRE(std::regex_search(module_string, m, call));
        }
    }

    GIVEN("A call to function declared below the caller") {
        std::string nmodl_text = R"(
            FUNCTION foo(x) {
                foo = 4 * bar()
            }
            FUNCTION bar() {
                bar = 5
            }
        )";

        THEN("a correct call instruction is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check for call instruction.
            std::regex call(R"(%[0-9]+ = call double @bar\(\))");
            REQUIRE(std::regex_search(module_string, m, call));
        }
    }

    GIVEN("A call to function with arguments") {
        std::string nmodl_text = R"(
            FUNCTION foo(x, y) {
                foo = 4 * x - y
            }
            FUNCTION bar(i) {
                bar = foo(i, 4)
            }
        )";

        THEN("arguments are processed before the call and passed to call instruction") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check correct arguments.
            std::regex i(R"(%1 = load double, double\* %i)");
            std::regex call(R"(call double @foo\(double %1, double 4.000000e\+00\))");
            REQUIRE(std::regex_search(module_string, m, i));
            REQUIRE(std::regex_search(module_string, m, call));
        }
    }

    GIVEN("A call to external method") {
        std::string nmodl_text = R"(
            FUNCTION bar(i) {
                bar = exp(i)
            }
        )";

        THEN("LLVM intrinsic corresponding to this method is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check for intrinsic declaration.
            std::regex exp(R"(declare double @llvm\.exp\.f64\(double\))");
            REQUIRE(std::regex_search(module_string, m, exp));

            // Check the correct call is made.
            std::regex call(R"(call double @llvm\.exp\.f64\(double %[0-9]+\))");
            REQUIRE(std::regex_search(module_string, m, call));
        }
    }

    GIVEN("A call to function with the wrong number of arguments") {
        std::string nmodl_text = R"(
            FUNCTION foo(x, y) {
                foo = 4 * x - y
            }
            FUNCTION bar(i) {
                bar = foo(i)
            }
        )";

        THEN("a runtime error is thrown") {
            REQUIRE_THROWS_AS(run_llvm_visitor(nmodl_text), std::runtime_error);
        }
    }
}

//=============================================================================
// IndexedName
//=============================================================================

SCENARIO("Indexed name", "[visitor][llvm]") {
    GIVEN("Procedure with a local array variable") {
        std::string nmodl_text = R"(
            PROCEDURE foo() {
                LOCAL x[2]
            }
        )";

        THEN("array is allocated") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            std::regex array(R"(%x = alloca \[2 x double\])");
            REQUIRE(std::regex_search(module_string, m, array));
        }
    }

    GIVEN("Procedure with a local array assignment") {
        std::string nmodl_text = R"(
            PROCEDURE foo() {
                LOCAL x[2]
                x[1] = 3
            }
        )";

        THEN("element is stored to the array") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check GEP is created correctly to pint at array element.
            std::regex GEP(
                R"(%1 = getelementptr inbounds \[2 x double\], \[2 x double\]\* %x, i32 0, i32 1)");
            REQUIRE(std::regex_search(module_string, m, GEP));

            // Check the value is stored to the pointer.
            std::regex store(R"(store double 3.000000e\+00, double\* %1)");
            REQUIRE(std::regex_search(module_string, m, store));
        }
    }

    GIVEN("Procedure with a assignment of array element") {
        std::string nmodl_text = R"(
            PROCEDURE foo() {
                LOCAL x[2], y
                x[1] = 3
                y = x[1]
            }
        )";

        THEN("array element is stored to the variable") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check GEP is created correctly to pint at array element.
            std::regex GEP(
                R"(%2 = getelementptr inbounds \[2 x double\], \[2 x double\]\* %x, i32 0, i32 1)");
            REQUIRE(std::regex_search(module_string, m, GEP));

            // Check the value is loaded from the pointer.
            std::regex load(R"(%3 = load double, double\* %2)");
            REQUIRE(std::regex_search(module_string, m, load));

            // Check the value is stored to the the variable.
            std::regex store(R"(store double %3, double\* %y)");
            REQUIRE(std::regex_search(module_string, m, store));
        }
    }

    GIVEN("Array with out of bounds access") {
        std::string nmodl_text = R"(
            PROCEDURE foo() {
                LOCAL x[2]
                x[5] = 3
            }
        )";

        THEN("error is thrown") {
            REQUIRE_THROWS_AS(run_llvm_visitor(nmodl_text), std::runtime_error);
        }
    }
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

            // Check procedure has empty body with a void return.
            std::regex procedure(R"(define void @empty\(\) \{\n(\s)*ret void\n\})");
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

            // Check procedure signature.
            std::regex function_signature(R"(define void @with_argument\(double %x1\) \{)");
            REQUIRE(std::regex_search(module_string, m, function_signature));

            // Check that procedure arguments are allocated on the local stack.
            std::regex alloca_instr(R"(%x = alloca double)");
            std::regex store_instr(R"(store double %x1, double\* %x)");
            REQUIRE(std::regex_search(module_string, m, alloca_instr));
            REQUIRE(std::regex_search(module_string, m, store_instr));

            // Check terminator.
            std::regex terminator(R"(ret void)");
            REQUIRE(std::regex_search(module_string, m, terminator));
        }
    }
}

//=============================================================================
// UnaryExpression
//=============================================================================

SCENARIO("Unary expression", "[visitor][llvm]") {
    GIVEN("Procedure with negation") {
        std::string nmodl_text = R"(
            PROCEDURE negation(a) {
                LOCAL i
                i = -a
            }
        )";

        THEN("fneg instruction is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            std::regex allocation(R"(%1 = load double, double\* %a)");
            REQUIRE(std::regex_search(module_string, m, allocation));

            // llvm v9 and llvm v11 implementation for negation
            std::regex negation_v9(R"(%2 = fsub double -0.000000e\+00, %1)");
            std::regex negation_v11(R"(fneg double %1)");
            bool result = std::regex_search(module_string, m, negation_v9) ||
                          std::regex_search(module_string, m, negation_v11);
            REQUIRE(result == true);
        }
    }
}

//=============================================================================
// Optimization : dead code removal
//=============================================================================

SCENARIO("Dead code removal", "[visitor][llvm][opt]") {
    GIVEN("Procedure using local variables, without any side effects") {
        std::string nmodl_text = R"(
            PROCEDURE add(a, b) {
                LOCAL i
                i = a + b
            }
        )";

        THEN("with optimisation enabled, all ops are eliminated") {
            std::string module_string = run_llvm_visitor(nmodl_text, true);
            std::smatch m;

            // Check if the values are optimised out
            std::regex empty_proc(
                R"(define void @add\(double %a1, double %b2\) \{\n(\s)*ret void\n\})");
            REQUIRE(std::regex_search(module_string, m, empty_proc));
        }
    }
}
