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
#include "visitors/neuron_solve_visitor.hpp"
#include "visitors/solve_block_visitor.hpp"
#include "visitors/symtab_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using nmodl::parser::NmodlDriver;

//=============================================================================
// Utility to get LLVM module as a string
//=============================================================================

std::string run_llvm_visitor(const std::string& text,
                             bool opt = false,
                             bool use_single_precision = false,
                             int vector_width = 1) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    NeuronSolveVisitor().visit_program(*ast);
    SolveBlockVisitor().visit_program(*ast);

    codegen::CodegenLLVMVisitor llvm_visitor(/*mod_filename=*/"unknown",
                                             /*output_dir=*/".",
                                             opt,
                                             use_single_precision,
                                             vector_width);
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
            std::string module_string =
                run_llvm_visitor(nmodl_text, /*opt=*/false, /*use_single_precision=*/true);
            std::smatch m;

            std::regex rhs(R"(%1 = load float, float\* %b)");
            std::regex lhs(R"(%2 = load float, float\* %a)");
            std::regex res(R"(%3 = fadd float %2, %1)");

            // Check the float values are loaded correctly and added
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
// If/Else statements and comparison operators
//=============================================================================

SCENARIO("Comparison", "[visitor][llvm]") {
    GIVEN("Procedure with comparison operators") {
        std::string nmodl_text = R"(
            PROCEDURE foo(x) {
                if (x < 10) {

                } else if (x >= 10 && x <= 100) {

                } else if (x == 120) {

                } else if (!(x != 200)) {

                }
            }
        )";

        THEN("correct LLVM instructions are produced") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check less than.
            std::regex lt(R"(fcmp olt double %(.+), 1\.000000e\+01)");
            REQUIRE(std::regex_search(module_string, m, lt));

            // Check greater or equal than and logical and.
            std::regex ge(R"(fcmp ole double %(.+), 1\.000000e\+02)");
            std::regex logical_and(R"(and i1 %(.+), %(.+))");
            REQUIRE(std::regex_search(module_string, m, ge));
            REQUIRE(std::regex_search(module_string, m, logical_and));

            // Check equals.
            std::regex eq(R"(fcmp oeq double %(.+), 1\.200000e\+02)");
            REQUIRE(std::regex_search(module_string, m, eq));

            // Check not equals.
            std::regex ne(R"(fcmp one double %(.+), 2\.000000e\+02)");
            REQUIRE(std::regex_search(module_string, m, ne));
        }
    }
}

SCENARIO("If/Else", "[visitor][llvm]") {
    GIVEN("Function with only if statement") {
        std::string nmodl_text = R"(
            FUNCTION foo(y) {
                LOCAL x
                x = 100
                if (y == 20) {
                    x = 20
                }
                foo = x + y
            }
        )";

        THEN("correct LLVM instructions are produced") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            std::regex cond_br(
                "br i1 %2, label %3, label %4\n"
                "\n"
                "3:.*\n"
                "  store double 2\\.000000e\\+01, double\\* %x.*\n"
                "  br label %4\n"
                "\n"
                "4:");
            REQUIRE(std::regex_search(module_string, m, cond_br));
        }
    }

    GIVEN("Function with both if and else statements") {
        std::string nmodl_text = R"(
            FUNCTION sign(x) {
                LOCAL s
                if (x < 0) {
                    s = -1
                } else {
                    s = 1
                }
                sign = s
            }
        )";

        THEN("correct LLVM instructions are produced") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            std::regex if_else_br(
                "br i1 %2, label %3, label %4\n"
                "\n"
                "3:.*\n"
                "  store double -1\\.000000e\\+00, double\\* %s.*\n"
                "  br label %5\n"
                "\n"
                "4:.*\n"
                "  store double 1\\.000000e\\+00, double\\* %s.*\n"
                "  br label %5\n"
                "\n"
                "5:");
            REQUIRE(std::regex_search(module_string, m, if_else_br));
        }
    }

    GIVEN("Function with both if and else if statements") {
        std::string nmodl_text = R"(
            FUNCTION bar(x) {
                LOCAL s
                s = -1
                if (x <= 0) {
                    s = 0
                } else if (0 < x && x <= 1) {
                    s = 1
                }
                bar = s
            }
        )";

        THEN("correct LLVM instructions are produced") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            std::regex if_else_if(
                "br i1 %2, label %3, label %4\n"
                "\n"
                "3:.*\n"
                "  .*\n"
                "  br label %12\n"
                "\n"
                "4:.*\n"
                "  .*\n"
                "  .*\n"
                "  .*\n"
                "  .*\n"
                "  %.+ = and i1 %.+, %.+\n"
                "  br i1 %.+, label %10, label %11\n"
                "\n"
                "10:.*\n"
                "  .*\n"
                "  br label %11\n"
                "\n"
                "11:.*\n"
                "  br label %12\n"
                "\n"
                "12:");
            REQUIRE(std::regex_search(module_string, m, if_else_if));
        }
    }

    GIVEN("Function with if, else if anf else statements") {
        std::string nmodl_text = R"(
            FUNCTION bar(x) {
                LOCAL s
                if (x <= 0) {
                    s = 0
                } else if (0 < x && x <= 1) {
                    s = 1
                } else {
                    s = 100
                }
                bar = s
            }
        )";

        THEN("correct LLVM instructions are produced") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            std::regex if_else_if_else(
                "br i1 %2, label %3, label %4\n"
                "\n"
                "3:.*\n"
                "  .*\n"
                "  br label %13\n"
                "\n"
                "4:.*\n"
                "  .*\n"
                "  .*\n"
                "  .*\n"
                "  .*\n"
                "  %9 = and i1 %.+, %.+\n"
                "  br i1 %9, label %10, label %11\n"
                "\n"
                "10:.*\n"
                "  .*\n"
                "  br label %12\n"
                "\n"
                "11:.*\n"
                "  .*\n"
                "  br label %12\n"
                "\n"
                "12:.*\n"
                "  br label %13\n"
                "\n"
                "13:");
            REQUIRE(std::regex_search(module_string, m, if_else_if_else));
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
            std::regex function_signature(R"(define double @foo\(double %x[0-9].*\) \{)");
            REQUIRE(std::regex_search(module_string, m, function_signature));

            // Check that function arguments are allocated on the local stack.
            std::regex alloca_instr(R"(%x = alloca double)");
            std::regex store_instr(R"(store double %x[0-9].*, double\* %x)");
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

        THEN("an int call instruction is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check for call instruction.
            std::regex call(R"(call i32 @bar\(\))");
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

    GIVEN("A call to printf") {
        std::string nmodl_text = R"(
            PROCEDURE bar() {
                LOCAL i
                i = 0
                printf("foo")
                printf("bar %d", i)
            }
        )";

        THEN("printf is declared and global string values are created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check for global string values.
            std::regex str1(
                R"(@[0-9]+ = private unnamed_addr constant \[6 x i8\] c\"\\22foo\\22\\00\")");
            std::regex str2(
                R"(@[0-9]+ = private unnamed_addr constant \[9 x i8\] c\"\\22bar %d\\22\\00\")");
            REQUIRE(std::regex_search(module_string, m, str1));
            REQUIRE(std::regex_search(module_string, m, str2));

            // Check for printf declaration.
            std::regex declaration(R"(declare i32 @printf\(i8\*, \.\.\.\))");
            REQUIRE(std::regex_search(module_string, m, declaration));

            // Check the correct calls are made.
            std::regex call1(
                R"(call i32 \(i8\*, \.\.\.\) @printf\(i8\* getelementptr inbounds \(\[6 x i8\], \[6 x i8\]\* @[0-9]+, i32 0, i32 0\)\))");
            std::regex call2(
                R"(call i32 \(i8\*, \.\.\.\) @printf\(i8\* getelementptr inbounds \(\[9 x i8\], \[9 x i8\]\* @[0-9]+, i32 0, i32 0\), double %[0-9]+\))");
            REQUIRE(std::regex_search(module_string, m, call1));
            REQUIRE(std::regex_search(module_string, m, call2));
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
                x[10 - 10] = 1
                x[1] = 3
            }
        )";

        THEN("element is stored to the array") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check GEPs are created correctly to get the addresses of array elements.
            std::regex GEP1(
                R"(%1 = getelementptr inbounds \[2 x double\], \[2 x double\]\* %x, i64 0, i64 0)");
            std::regex GEP2(
                R"(%2 = getelementptr inbounds \[2 x double\], \[2 x double\]\* %x, i64 0, i64 1)");
            REQUIRE(std::regex_search(module_string, m, GEP1));
            REQUIRE(std::regex_search(module_string, m, GEP2));

            // Check the value is stored to the correct addresses.
            std::regex store1(R"(store double 1.000000e\+00, double\* %1)");
            std::regex store2(R"(store double 3.000000e\+00, double\* %2)");
            REQUIRE(std::regex_search(module_string, m, store1));
            REQUIRE(std::regex_search(module_string, m, store2));
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
                R"(%2 = getelementptr inbounds \[2 x double\], \[2 x double\]\* %x, i64 0, i64 1)");
            REQUIRE(std::regex_search(module_string, m, GEP));

            // Check the value is loaded from the pointer.
            std::regex load(R"(%3 = load double, double\* %2)");
            REQUIRE(std::regex_search(module_string, m, load));

            // Check the value is stored to the the variable.
            std::regex store(R"(store double %3, double\* %y)");
            REQUIRE(std::regex_search(module_string, m, store));
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

        THEN("a function returning 0 integer is produced") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check procedure has empty body with a dummy 0 allocation.
            std::regex signature(R"(define i32 @empty)");
            std::regex alloc(R"(%ret_empty = alloca i32)");
            std::regex store(R"(store i32 0, i32\* %ret_empty)");
            std::regex load(R"(%1 = load i32, i32\* %ret_empty)");
            std::regex ret(R"(ret i32 %1)");
            REQUIRE(std::regex_search(module_string, m, signature));
            REQUIRE(std::regex_search(module_string, m, alloc));
            REQUIRE(std::regex_search(module_string, m, store));
            REQUIRE(std::regex_search(module_string, m, ret));
        }
    }

    GIVEN("Empty procedure with arguments") {
        std::string nmodl_text = R"(
            PROCEDURE with_argument(x) {}
        )";

        THEN("int function is produced with arguments allocated on stack") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check procedure signature.
            std::regex function_signature(R"(define i32 @with_argument\(double %x[0-9].*\) \{)");
            REQUIRE(std::regex_search(module_string, m, function_signature));

            // Check dummy return.
            std::regex dummy_alloca(R"(%ret_with_argument = alloca i32)");
            std::regex dummy_store(R"(store i32 0, i32\* %ret_with_argument)");
            std::regex dummy_load(R"(%1 = load i32, i32\* %ret_with_argument)");
            std::regex ret(R"(ret i32 %1)");
            REQUIRE(std::regex_search(module_string, m, dummy_alloca));
            REQUIRE(std::regex_search(module_string, m, dummy_store));
            REQUIRE(std::regex_search(module_string, m, dummy_load));
            REQUIRE(std::regex_search(module_string, m, ret));

            // Check that procedure arguments are allocated on the local stack.
            std::regex alloca_instr(R"(%x = alloca double)");
            std::regex store_instr(R"(store double %x[0-9].*, double\* %x)");
            REQUIRE(std::regex_search(module_string, m, alloca_instr));
            REQUIRE(std::regex_search(module_string, m, store_instr));
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
// WhileStatement
//=============================================================================

SCENARIO("While", "[visitor][llvm]") {
    GIVEN("Procedure with a simple while loop") {
        std::string nmodl_text = R"(
            FUNCTION loop() {
                LOCAL i
                i = 0
                WHILE (i < 10) {
                    i = i + 1
                }
                loop = 0
            }
        )";

        THEN("correct loop is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            std::regex loop(
                "  br label %1\n"
                "\n"
                "1:.*\n"
                "  %2 = load double, double\\* %i.*\n"
                "  %3 = fcmp olt double %2, 1\\.000000e\\+01\n"
                "  br i1 %3, label %4, label %7\n"
                "\n"
                "4:.*\n"
                "  %5 = load double, double\\* %i.*\n"
                "  %6 = fadd double %5, 1\\.000000e\\+00\n"
                "  store double %6, double\\* %i.*\n"
                "  br label %1\n"
                "\n"
                "7:.*\n"
                "  store double 0\\.000000e\\+00, double\\* %ret_loop.*\n");
            // Check that 3 blocks are created: header, body and exit blocks. Also, there must be
            // a backedge from the body to the header.
            REQUIRE(std::regex_search(module_string, m, loop));
        }
    }
}

//=============================================================================
// State scalar kernel
//=============================================================================

SCENARIO("Scalar state kernel", "[visitor][llvm]") {
    GIVEN("A neuron state update") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX hh
                NONSPECIFIC_CURRENT il
                RANGE minf, mtau, gl, el
            }

            STATE {
                m
            }

            ASSIGNED {
                v (mV)
                minf
                mtau (ms)
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
                il = gl * (v - el)
            }

            DERIVATIVE states {
                    m = (minf-m) / mtau
            }
        )";

        THEN("a kernel with instance struct as an argument and a FOR loop is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check the struct type and the kernel declaration.
            std::regex struct_type(
                "%.*__instance_var__type = type \\{ double\\*, double\\*, double\\*, double\\*, "
                "double\\*, double\\*, double\\*, i32\\*, double, double, double, i32, i32 \\}");
            std::regex kernel_declaration(
                R"(define void @nrn_state_hh\(%.*__instance_var__type\* .*\))");
            REQUIRE(std::regex_search(module_string, m, struct_type));
            REQUIRE(std::regex_search(module_string, m, kernel_declaration));

            // Check for correct induction variable initialisation and a branch to condition block.
            std::regex alloca_instr(R"(%id = alloca i32)");
            std::regex br(R"(br label %for\.cond)");
            REQUIRE(std::regex_search(module_string, m, alloca_instr));
            REQUIRE(std::regex_search(module_string, m, br));

            // Check condition block: id < mech->node_count, and a conditional branch to loop body
            // or exit.
            std::regex condition(
                "  %.* = load %.*__instance_var__type\\*, %.*__instance_var__type\\*\\* %.*,.*\n"
                "  %.* = getelementptr inbounds %.*__instance_var__type, "
                "%.*__instance_var__type\\* "
                "%.*, i32 0, i32 [0-9]+\n"
                "  %.* = load i32, i32\\* %.*,.*\n"
                "  %.* = load i32, i32\\* %id,.*\n"
                "  %.* = icmp slt i32 %.*, %.*");
            std::regex cond_br(R"(br i1 %.*, label %for\.body, label %for\.exit)");
            REQUIRE(std::regex_search(module_string, m, condition));
            REQUIRE(std::regex_search(module_string, m, cond_br));

            // In the body block, `node_id` and voltage `v` are initialised with the data from the
            // struct. Check for variable allocations and correct loads from the struct with GEPs.
            std::regex initialisation(
                "for\\.body:.*\n"
                "  %node_id = alloca i32,.*\n"
                "  %v = alloca double,.*");
            std::regex load_from_struct(
                "  %.* = load %.*__instance_var__type\\*, %.*__instance_var__type\\*\\* %.*\n"
                "  %.* = getelementptr inbounds %.*__instance_var__type, "
                "%.*__instance_var__type\\* %.*, i32 0, i32 [0-9]+\n"
                "  %.* = load i32, i32\\* %id,.*\n"
                "  %.* = sext i32 %.* to i64\n"
                "  %.* = load (i32|double)\\*, (i32|double)\\*\\* %.*\n"
                "  %.* = getelementptr inbounds (i32|double), (i32|double)\\* %.*, i64 %.*\n"
                "  %.* = load (i32|double), (i32|double)\\* %.*");
            REQUIRE(std::regex_search(module_string, m, initialisation));
            REQUIRE(std::regex_search(module_string, m, load_from_struct));

            // Check induction variable is incremented in increment block.
            std::regex increment(
                "for.inc:.*\n"
                "  %.* = load i32, i32\\* %id,.*\n"
                "  %.* = add i32 %.*, 1\n"
                "  store i32 %.*, i32\\* %id,.*\n"
                "  br label %for\\.cond");
            REQUIRE(std::regex_search(module_string, m, increment));

            // Check exit block.
            std::regex exit(
                "for\\.exit:.*\n"
                "  ret void");
            REQUIRE(std::regex_search(module_string, m, exit));
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
                R"(define i32 @add\(double %a[0-9].*, double %b[0-9].*\) \{\n(\s)*ret i32 0\n\})");
            REQUIRE(std::regex_search(module_string, m, empty_proc));
        }
    }
}
