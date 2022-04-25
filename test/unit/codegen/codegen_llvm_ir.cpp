/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>
#include <regex>

#include "test/unit/utils/test_utils.hpp"

#include "ast/program.hpp"
#include "ast/statement_block.hpp"
#include "codegen/llvm/codegen_llvm_helper_visitor.hpp"
#include "codegen/llvm/codegen_llvm_visitor.hpp"
#include "parser/nmodl_driver.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/inline_visitor.hpp"
#include "visitors/neuron_solve_visitor.hpp"
#include "visitors/solve_block_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/visitor_utils.hpp"

using namespace nmodl;
using namespace codegen;
using namespace visitor;

using namespace test_utils;

using nmodl::parser::NmodlDriver;

//=============================================================================
// Utility to get LLVM module as a string
//=============================================================================

std::string run_gpu_llvm_visitor(const std::string& text,
                                 int opt_level = 0,
                                 bool use_single_precision = false,
                                 std::string math_library = "none",
                                 bool nmodl_inline = false) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    if (nmodl_inline) {
        InlineVisitor().visit_program(*ast);
    }
    NeuronSolveVisitor().visit_program(*ast);
    SolveBlockVisitor().visit_program(*ast);

    codegen::Platform gpu_platform(
        codegen::PlatformID::GPU, /*name=*/"nvptx64", math_library, use_single_precision, 1);
    codegen::CodegenLLVMVisitor llvm_visitor(
        /*mod_filename=*/"unknown",
        /*output_dir=*/".",
        gpu_platform,
        opt_level,
        /*add_debug_information=*/false);

    llvm_visitor.visit_program(*ast);
    return llvm_visitor.dump_module();
}

std::string run_llvm_visitor(const std::string& text,
                             int opt_level = 0,
                             bool use_single_precision = false,
                             int vector_width = 1,
                             std::string vec_lib = "none",
                             std::vector<std::string> fast_math_flags = {},
                             bool nmodl_inline = false) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    if (nmodl_inline) {
        InlineVisitor().visit_program(*ast);
    }
    NeuronSolveVisitor().visit_program(*ast);
    SolveBlockVisitor().visit_program(*ast);

    codegen::Platform cpu_platform(
        codegen::PlatformID::CPU, /*name=*/"default", vec_lib, use_single_precision, vector_width);
    codegen::CodegenLLVMVisitor llvm_visitor(
        /*mod_filename=*/"unknown",
        /*output_dir=*/".",
        cpu_platform,
        opt_level,
        /*add_debug_information=*/false,
        fast_math_flags);

    llvm_visitor.visit_program(*ast);
    return llvm_visitor.dump_module();
}

//=============================================================================
// Utility to get specific NMODL AST nodes
//=============================================================================

std::vector<std::shared_ptr<ast::Ast>> run_llvm_visitor_helper(
    const std::string& text,
    codegen::Platform& platform,
    const std::vector<ast::AstNodeType>& nodes_to_collect) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);

    SymtabVisitor().visit_program(*ast);
    SolveBlockVisitor().visit_program(*ast);
    CodegenLLVMHelperVisitor(platform).visit_program(*ast);

    const auto& nodes = collect_nodes(*ast, nodes_to_collect);

    return nodes;
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
                run_llvm_visitor(nmodl_text, /*opt_level=*/0, /*use_single_precision=*/true);
            std::smatch m;

            std::regex rhs(R"(%1 = load float, float\* %b)");
            std::regex lhs(R"(%2 = load float, float\* %a)");
            std::regex res(R"(%3 = fadd float %2, %1)");

            // Check the float values are loaded correctly and added.
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

            // Check rhs.
            std::regex rr(R"(%1 = load double, double\* %b)");
            std::regex rl(R"(%2 = load double, double\* %a)");
            std::regex x(R"(%3 = fadd double %2, %1)");
            REQUIRE(std::regex_search(module_string, m, rr));
            REQUIRE(std::regex_search(module_string, m, rl));
            REQUIRE(std::regex_search(module_string, m, x));

            // Check lhs.
            std::regex lr(R"(%4 = load double, double\* %b)");
            std::regex ll(R"(%5 = load double, double\* %a)");
            std::regex y(R"(%6 = fsub double %5, %4)");
            REQUIRE(std::regex_search(module_string, m, lr));
            REQUIRE(std::regex_search(module_string, m, ll));
            REQUIRE(std::regex_search(module_string, m, y));

            // Check result.
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

            // Check store immediate is created.
            std::regex allocation(R"(%i = alloca double)");
            std::regex assignment(R"(store double 2.0*e\+00, double\* %i)");
            REQUIRE(std::regex_search(module_string, m, allocation));
            REQUIRE(std::regex_search(module_string, m, assignment));
        }
    }

    GIVEN("Function with power operator") {
        std::string nmodl_text = R"(
            FUNCTION power() {
                LOCAL i, j
                i = 2
                j = 4
                power = i ^ j
            }
        )";

        THEN("'pow' intrinsic is created") {
            std::string module_string =
                run_llvm_visitor(nmodl_text, /*opt_level=*/0, /*use_single_precision=*/true);
            std::smatch m;

            // Check 'pow' intrinsic.
            std::regex declaration(R"(declare float @llvm\.pow\.f32\(float, float\))");
            std::regex pow(R"(call float @llvm\.pow\.f32\(float %.*, float %.*\))");
            REQUIRE(std::regex_search(module_string, m, declaration));
            REQUIRE(std::regex_search(module_string, m, pow));
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
            FUNCTION nmodl_ceil(x) {
                nmodl_ceil = ceil(x)
            }

            FUNCTION nmodl_cos(x) {
                nmodl_cos = cos(x)
            }

            FUNCTION nmodl_exp(x) {
                nmodl_exp = exp(x)
            }

            FUNCTION nmodl_fabs(x) {
                nmodl_fabs = fabs(x)
            }

            FUNCTION nmodl_floor(x) {
                nmodl_floor = floor(x)
            }

            FUNCTION nmodl_log(x) {
                nmodl_log = log(x)
            }

            FUNCTION nmodl_log10(x) {
                nmodl_log10 = log10(x)
            }

            FUNCTION nmodl_pow(x, y) {
                nmodl_pow = pow(x, y)
            }

            FUNCTION nmodl_sin(x) {
                nmodl_sin = sin(x)
            }

            FUNCTION nmodl_sqrt(x) {
                nmodl_sqrt = sqrt(x)
            }
        )";

        THEN("LLVM intrinsic corresponding to this method is created") {
            std::string module_string = run_llvm_visitor(nmodl_text);
            std::smatch m;

            // Check for intrinsic declarations.
            std::regex ceil(R"(declare double @llvm\.ceil\.f64\(double\))");
            std::regex cos(R"(declare double @llvm\.cos\.f64\(double\))");
            std::regex exp(R"(declare double @llvm\.exp\.f64\(double\))");
            std::regex fabs(R"(declare double @llvm\.fabs\.f64\(double\))");
            std::regex floor(R"(declare double @llvm\.floor\.f64\(double\))");
            std::regex log(R"(declare double @llvm\.log\.f64\(double\))");
            std::regex log10(R"(declare double @llvm\.log10\.f64\(double\))");
            std::regex pow(R"(declare double @llvm\.pow\.f64\(double, double\))");
            std::regex sin(R"(declare double @llvm\.sin\.f64\(double\))");
            std::regex sqrt(R"(declare double @llvm\.sqrt\.f64\(double\))");
            REQUIRE(std::regex_search(module_string, m, ceil));
            REQUIRE(std::regex_search(module_string, m, cos));
            REQUIRE(std::regex_search(module_string, m, exp));
            REQUIRE(std::regex_search(module_string, m, fabs));
            REQUIRE(std::regex_search(module_string, m, floor));
            REQUIRE(std::regex_search(module_string, m, log));
            REQUIRE(std::regex_search(module_string, m, log10));
            REQUIRE(std::regex_search(module_string, m, pow));
            REQUIRE(std::regex_search(module_string, m, sin));
            REQUIRE(std::regex_search(module_string, m, sqrt));

            // Check the correct call is made.
            std::regex ceil_call(R"(call double @llvm\.ceil\.f64\(double %[0-9]+\))");
            std::regex cos_call(R"(call double @llvm\.cos\.f64\(double %[0-9]+\))");
            std::regex exp_call(R"(call double @llvm\.exp\.f64\(double %[0-9]+\))");
            std::regex fabs_call(R"(call double @llvm\.fabs\.f64\(double %[0-9]+\))");
            std::regex floor_call(R"(call double @llvm\.floor\.f64\(double %[0-9]+\))");
            std::regex log_call(R"(call double @llvm\.log\.f64\(double %[0-9]+\))");
            std::regex log10_call(R"(call double @llvm\.log10\.f64\(double %[0-9]+\))");
            std::regex pow_call(R"(call double @llvm\.pow\.f64\(double %[0-9]+, double %[0-9]+\))");
            std::regex sin_call(R"(call double @llvm\.sin\.f64\(double %[0-9]+\))");
            std::regex sqrt_call(R"(call double @llvm\.sqrt\.f64\(double %[0-9]+\))");
            REQUIRE(std::regex_search(module_string, m, ceil_call));
            REQUIRE(std::regex_search(module_string, m, cos_call));
            REQUIRE(std::regex_search(module_string, m, exp_call));
            REQUIRE(std::regex_search(module_string, m, fabs_call));
            REQUIRE(std::regex_search(module_string, m, floor_call));
            REQUIRE(std::regex_search(module_string, m, log_call));
            REQUIRE(std::regex_search(module_string, m, log10_call));
            REQUIRE(std::regex_search(module_string, m, pow_call));
            REQUIRE(std::regex_search(module_string, m, sin_call));
            REQUIRE(std::regex_search(module_string, m, sqrt_call));
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
            REQUIRE(std::regex_search(module_string, m, load));
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

            PARAMETER {
                gl = .0003 (S/cm2)  <0,1e9>
                el = -54.3 (mV)
            }

            ASSIGNED {
                v (mV)
                minf
                mtau (ms)
                il (mA/cm2)
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

            // Check the struct type with correct attributes and the kernel declaration.
            std::regex struct_type(
                "%.*__instance_var__type = type \\{ double\\*, double\\*, double\\*, double\\*, "
                "double\\*, double\\*, double\\*, double\\*, double\\*, double\\*, i32\\*, double, "
                "double, double, i32, i32 \\}");
            std::regex kernel_declaration(
                R"(define void @nrn_state_hh\(%.*__instance_var__type.0\* noalias nocapture readonly .*\) #0)");
            REQUIRE(std::regex_search(module_string, m, struct_type));
            REQUIRE(std::regex_search(module_string, m, kernel_declaration));

            // Check kernel attributes.
            std::regex kernel_attributes(R"(attributes #0 = \{ nofree nounwind \})");
            REQUIRE(std::regex_search(module_string, m, kernel_attributes));

            // Check for correct variables initialisation and a branch to condition block.
            std::regex id_initialisation(R"(%id = alloca i32)");
            std::regex node_id_initialisation(R"(%node_id = alloca i32)");
            std::regex v_initialisation(R"(%v = alloca double)");
            std::regex br(R"(br label %for\.cond)");
            REQUIRE(std::regex_search(module_string, m, id_initialisation));
            REQUIRE(std::regex_search(module_string, m, node_id_initialisation));
            REQUIRE(std::regex_search(module_string, m, v_initialisation));
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

            // Check that loop metadata is attached to the scalar kernel.
            std::regex loop_metadata(R"(!llvm\.loop !0)");
            std::regex loop_metadata_self_reference(R"(!0 = distinct !\{!0, !1\})");
            std::regex loop_metadata_disable_vectorization(
                R"(!1 = !\{!\"llvm\.loop\.vectorize\.enable\", i1 false\})");
            REQUIRE(std::regex_search(module_string, m, loop_metadata));
            REQUIRE(std::regex_search(module_string, m, loop_metadata_self_reference));
            REQUIRE(std::regex_search(module_string, m, loop_metadata_disable_vectorization));

            // Check for correct loads from the struct with GEPs.
            std::regex load_from_struct(
                "  %.* = load %.*__instance_var__type\\*, %.*__instance_var__type\\*\\* %.*\n"
                "  %.* = getelementptr inbounds %.*__instance_var__type, "
                "%.*__instance_var__type\\* %.*, i32 0, i32 [0-9]+\n"
                "  %.* = load i32, i32\\* %id,.*\n"
                "  %.* = sext i32 %.* to i64\n"
                "  %.* = load (i32|double)\\*, (i32|double)\\*\\* %.*\n"
                "  %.* = getelementptr inbounds (i32|double), (i32|double)\\* %.*, i64 %.*\n"
                "  %.* = load (i32|double), (i32|double)\\* %.*");
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
                "for\\.exit[0-9]*:.*\n"
                "  ret void");
            REQUIRE(std::regex_search(module_string, m, exit));
        }
    }
}

//=============================================================================
// Gather for vectorised kernel
//=============================================================================

SCENARIO("Vectorised simple kernel", "[visitor][llvm]") {
    GIVEN("An indirect indexing of voltage") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX hh
                NONSPECIFIC_CURRENT i
            }

            STATE {}

            ASSIGNED {
                v (mV)
                i (mA/cm2)
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
                i = 2
            }

            DERIVATIVE states {}
        )";

        THEN("a gather instructions is created") {
            std::string module_string = run_llvm_visitor(nmodl_text,
                                                         /*opt_level=*/0,
                                                         /*use_single_precision=*/false,
                                                         /*vector_width=*/4);
            std::smatch m;

            // Check that no loop metadata is attached.
            std::regex loop_metadata(R"(!llvm\.loop !.*)");
            REQUIRE(!std::regex_search(module_string, m, loop_metadata));

            // Check gather intrinsic is correctly declared.
            std::regex declaration(
                R"(declare <4 x double> @llvm\.masked\.gather\.v4f64\.v4p0f64\(<4 x double\*>, i32 immarg, <4 x i1>, <4 x double>\) )");
            REQUIRE(std::regex_search(module_string, m, declaration));

            // Check that the indices vector is created correctly and extended to i64.
            std::regex index_load(R"(load <4 x i32>, <4 x i32>\* %node_id)");
            std::regex sext(R"(sext <4 x i32> %.* to <4 x i64>)");
            REQUIRE(std::regex_search(module_string, m, index_load));
            REQUIRE(std::regex_search(module_string, m, sext));

            // Check that the access to `voltage` is performed via gather instruction.
            //      v = mech->voltage[node_id]
            std::regex gather(
                "call <4 x double> @llvm\\.masked\\.gather\\.v4f64\\.v4p0f64\\("
                "<4 x double\\*> %.*, i32 1, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x "
                "double> undef\\)");
            REQUIRE(std::regex_search(module_string, m, gather));
        }
    }
}

//=============================================================================
// Scatter for vectorised kernel
//=============================================================================

SCENARIO("Vectorised simple kernel with ion writes", "[visitor][llvm]") {
    GIVEN("An indirect indexing of ca ion") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX hh
                USEION ca WRITE cai
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {}
        )";

        THEN("a scatter instructions is created") {
            std::string module_string = run_llvm_visitor(nmodl_text,
                                                         /*opt_level=*/0,
                                                         /*use_single_precision=*/false,
                                                         /*vector_width=*/4);
            std::smatch m;

            // Check scatter intrinsic is correctly declared.
            std::regex declaration(
                R"(declare void @llvm\.masked\.scatter\.v4f64\.v4p0f64\(<4 x double>, <4 x double\*>, i32 immarg, <4 x i1>\))");
            REQUIRE(std::regex_search(module_string, m, declaration));

            // Check that the indices vector is created correctly and extended to i64.
            std::regex index_load(R"(load <4 x i32>, <4 x i32>\* %ion_cai_id)");
            std::regex sext(R"(sext <4 x i32> %.* to <4 x i64>)");
            REQUIRE(std::regex_search(module_string, m, index_load));
            REQUIRE(std::regex_search(module_string, m, sext));

            // Check that store to `ion_cai` is performed via scatter instruction.
            //      ion_cai[ion_cai_id] = cai[id]
            std::regex scatter(
                "call void @llvm\\.masked\\.scatter\\.v4f64\\.v4p0f64\\(<4 x double> %.*, <4 x "
                "double\\*> %.*, i32 1, <4 x i1> <i1 true, i1 true, i1 true, i1 true>\\)");
            REQUIRE(std::regex_search(module_string, m, scatter));
        }
    }
}

//=============================================================================
// Vectorised kernel with simple control flow
//=============================================================================

SCENARIO("Vectorised simple kernel with control flow", "[visitor][llvm]") {
    GIVEN("A single if/else statement") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
            }

            STATE {
                y
            }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
                IF (y < 0) {
                    y = y + 7
                } ELSE {
                    y = v
                }
            }
        )";

        THEN("masked load and stores are created") {
            std::string module_string = run_llvm_visitor(nmodl_text,
                                                         /*opt_level=*/0,
                                                         /*use_single_precision=*/true,
                                                         /*vector_width=*/8);
            std::smatch m;

            // Check masked load/store intrinsics are correctly declared.
            std::regex masked_load(
                R"(declare <8 x float> @llvm\.masked\.load\.v8f32\.p0v8f32\(<8 x float>\*, i32 immarg, <8 x i1>, <8 x float>\))");
            std::regex masked_store(
                R"(declare void @llvm.masked\.store\.v8f32\.p0v8f32\(<8 x float>, <8 x float>\*, i32 immarg, <8 x i1>\))");
            REQUIRE(std::regex_search(module_string, m, masked_load));
            REQUIRE(std::regex_search(module_string, m, masked_store));

            // Check true direction instructions are predicated with mask.
            // IF (mech->y[id] < 0) {
            //     mech->y[id] = mech->y[id] + 7
            std::regex mask(R"(%30 = fcmp olt <8 x float> %.*, zeroinitializer)");
            std::regex true_load(
                R"(call <8 x float> @llvm\.masked\.load\.v8f32\.p0v8f32\(<8 x float>\* %.*, i32 1, <8 x i1> %30, <8 x float> undef\))");
            std::regex true_store(
                R"(call void @llvm\.masked\.store\.v8f32\.p0v8f32\(<8 x float> %.*, <8 x float>\* %.*, i32 1, <8 x i1> %30\))");
            REQUIRE(std::regex_search(module_string, m, mask));
            REQUIRE(std::regex_search(module_string, m, true_load));
            REQUIRE(std::regex_search(module_string, m, true_store));

            // Check false direction instructions are predicated with inverted mask.
            // } ELSE {
            //     mech->y[id] = v
            // }
            std::regex inverted_mask(
                R"(%47 = xor <8 x i1> %30, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)");
            std::regex false_load(
                R"(call <8 x float> @llvm\.masked\.load\.v8f32\.p0v8f32\(<8 x float>\* %v, i32 1, <8 x i1> %47, <8 x float> undef\))");
            std::regex false_store(
                R"(call void @llvm\.masked\.store\.v8f32\.p0v8f32\(<8 x float> %.*, <8 x float>\* %.*, i32 1, <8 x i1> %47\))");
        }
    }
}

//=============================================================================
// Derivative block : test optimization
//=============================================================================

SCENARIO("Scalar derivative block", "[visitor][llvm][derivative]") {
    GIVEN("After LLVM helper visitor transformations") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX hh
                NONSPECIFIC_CURRENT il
                RANGE minf, mtau
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
                il = 2
            }
            DERIVATIVE states {
                m = (minf-m)/mtau
            }
        )";

        std::string expected_state_loop = R"(
            for(id = 0; id<mech->node_count; id = id+1) {
                node_id = mech->node_index[id]
                v = mech->voltage[node_id]
                mech->m[id] = (mech->minf[id]-mech->m[id])/mech->mtau[id]
            })";

        THEN("a single scalar loops is constructed") {
            codegen::Platform default_platform;
            auto result = run_llvm_visitor_helper(nmodl_text,
                                                  default_platform,
                                                  {ast::AstNodeType::CODEGEN_FOR_STATEMENT});
            REQUIRE(result.size() == 2);

            auto main_state_loop = reindent_text(to_nmodl(result[1]));
            REQUIRE(main_state_loop == reindent_text(expected_state_loop));
        }
    }
}

SCENARIO("Vectorised derivative block", "[visitor][llvm][derivative]") {
    GIVEN("After LLVM helper visitor transformations") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX hh
                USEION na READ ena WRITE ina
                NONSPECIFIC_CURRENT il
                RANGE minf, mtau, gna, gnabar
            }
            STATE {
                m h
            }
            PARAMETER {
                gnabar = .12 (S/cm2) <0,1e9>
            }
            ASSIGNED {
                v (mV)
                minf
                mtau (ms)
                ena (mV)
                ina (mA/cm2)
                gna (S/cm2)
            }
            BREAKPOINT {
                SOLVE states METHOD cnexp
                gna = gnabar*m*m*m*h
                ina = gna*(v - ena)
            }
            DERIVATIVE states {
                m = (minf-m)/mtau
            }
        )";

        std::string expected_state_main_loop = R"(
            for(id = 0; id<mech->node_count-7; id = id+8) {
                node_id = mech->node_index[id]
                ena_id = mech->ion_ena_index[id]
                v = mech->voltage[node_id]
                mech->ena[id] = mech->ion_ena[ena_id]
                mech->m[id] = (mech->minf[id]-mech->m[id])/mech->mtau[id]
            })";

        std::string expected_state_epilogue_loop = R"(
            for(; id<mech->node_count; id = id+1) {
                epilogue_node_id = mech->node_index[id]
                epilogue_ena_id = mech->ion_ena_index[id]
                epilogue_v = mech->voltage[epilogue_node_id]
                mech->ena[id] = mech->ion_ena[epilogue_ena_id]
                mech->m[id] = (mech->minf[id]-mech->m[id])/mech->mtau[id]
            })";

        std::string expected_cur_main_loop = R"(
            for(id = 0; id<mech->node_count-7; id = id+8) {
                node_id = mech->node_index[id]
                ena_id = mech->ion_ena_index[id]
                ion_ina_id = mech->ion_ina_index[id]
                v = mech->voltage[node_id]
                mech->ena[id] = mech->ion_ena[ena_id]
                mech->gna[id] = mech->gnabar[id]*mech->m[id]*mech->m[id]*mech->m[id]*mech->h[id]
                mech->ina[id] = mech->gna[id]*(v-mech->ena[id])
                mech->ion_ina[ion_ina_id] = mech->ion_ina[ion_ina_id]+mech->ina[id]
            })";

        THEN("vector and epilogue scalar loops are constructed") {
            codegen::Platform simd_platform(/*use_single_precision=*/false,
                                            /*instruction_width=*/8);
            auto result = run_llvm_visitor_helper(nmodl_text,
                                                  simd_platform,
                                                  {ast::AstNodeType::CODEGEN_FOR_STATEMENT});
            REQUIRE(result.size() == 4);

            auto cur_main_loop = reindent_text(to_nmodl(result[0]));
            REQUIRE(cur_main_loop == reindent_text(expected_cur_main_loop));

            auto state_main_loop = reindent_text(to_nmodl(result[2]));
            REQUIRE(state_main_loop == reindent_text(expected_state_main_loop));

            auto state_epilogue_loop = reindent_text(to_nmodl(result[3]));
            REQUIRE(state_epilogue_loop == reindent_text(expected_state_epilogue_loop));
        }
    }
}

//=============================================================================
// Vector library calls.
//=============================================================================

SCENARIO("Vector library calls", "[visitor][llvm][vector_lib]") {
    GIVEN("A vector LLVM intrinsic") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX hh
                NONSPECIFIC_CURRENT il
            }
            STATE {
                m
            }
            ASSIGNED {
                v (mV)
                il (mA/cm2)
            }
            BREAKPOINT {
                SOLVE states METHOD cnexp
                il = 2
            }
            DERIVATIVE states {
                m = exp(m)
            }
        )";

        THEN("it is replaced with an appropriate vector library call") {
            std::smatch m;

            // Check exponential intrinsic is created.
            std::string no_library_module_str = run_llvm_visitor(nmodl_text,
                                                                 /*opt_level=*/0,
                                                                 /*use_single_precision=*/false,
                                                                 /*vector_width=*/2);
            std::regex exp_decl(R"(declare <2 x double> @llvm\.exp\.v2f64\(<2 x double>\))");
            std::regex exp_call(R"(call <2 x double> @llvm\.exp\.v2f64\(<2 x double> .*\))");
            REQUIRE(std::regex_search(no_library_module_str, m, exp_decl));
            REQUIRE(std::regex_search(no_library_module_str, m, exp_call));

            // Check exponential calls are replaced with calls to SVML library.
            std::string svml_library_module_str = run_llvm_visitor(nmodl_text,
                                                                   /*opt_level=*/0,
                                                                   /*use_single_precision=*/false,
                                                                   /*vector_width=*/2,
                                                                   /*vec_lib=*/"SVML");
            std::regex svml_exp_decl(R"(declare <2 x double> @__svml_exp2\(<2 x double>\))");
            std::regex svml_exp_call(R"(call <2 x double> @__svml_exp2\(<2 x double> .*\))");
            REQUIRE(std::regex_search(svml_library_module_str, m, svml_exp_decl));
            REQUIRE(std::regex_search(svml_library_module_str, m, svml_exp_call));
            REQUIRE(!std::regex_search(svml_library_module_str, m, exp_call));

            // Check that supported exponential calls are replaced with calls to MASSV library (i.e.
            // operating on vector of width 2).
            std::string massv2_library_module_str = run_llvm_visitor(nmodl_text,
                                                                     /*opt_level=*/0,
                                                                     /*use_single_precision=*/false,
                                                                     /*vector_width=*/2,
                                                                     /*vec_lib=*/"MASSV");
            std::regex massv2_exp_decl(R"(declare <2 x double> @__expd2\(<2 x double>\))");
            std::regex massv2_exp_call(R"(call <2 x double> @__expd2\(<2 x double> .*\))");
            REQUIRE(std::regex_search(massv2_library_module_str, m, massv2_exp_decl));
            REQUIRE(std::regex_search(massv2_library_module_str, m, massv2_exp_call));
            REQUIRE(!std::regex_search(massv2_library_module_str, m, exp_call));

            // Check no replacement for MASSV happens for non-supported vector widths.
            std::string massv4_library_module_str = run_llvm_visitor(nmodl_text,
                                                                     /*opt_level=*/0,
                                                                     /*use_single_precision=*/false,
                                                                     /*vector_width=*/4,
                                                                     /*vec_lib=*/"MASSV");
            std::regex exp4_call(R"(call <4 x double> @llvm\.exp\.v4f64\(<4 x double> .*\))");
            REQUIRE(std::regex_search(massv4_library_module_str, m, exp4_call));

            // Check correct replacement of @llvm.exp.v4f32 into @vexpf when using Accelerate.
            std::string accelerate_library_module_str =
                run_llvm_visitor(nmodl_text,
                                 /*opt_level=*/0,
                                 /*use_single_precision=*/true,
                                 /*vector_width=*/4,
                                 /*vec_lib=*/"Accelerate");
            std::regex accelerate_exp_decl(R"(declare <4 x float> @vexpf\(<4 x float>\))");
            std::regex accelerate_exp_call(R"(call <4 x float> @vexpf\(<4 x float> .*\))");
            std::regex fexp_call(R"(call <4 x float> @llvm\.exp\.v4f32\(<4 x float> .*\))");
            REQUIRE(std::regex_search(accelerate_library_module_str, m, accelerate_exp_decl));
            REQUIRE(std::regex_search(accelerate_library_module_str, m, accelerate_exp_call));
            REQUIRE(!std::regex_search(accelerate_library_module_str, m, fexp_call));

            // Check correct replacement of @llvm.exp.v2f64 into @_ZGV?N?v_exp when using SLEEF.
            std::string sleef_library_module_str = run_llvm_visitor(nmodl_text,
                                                                    /*opt_level=*/0,
                                                                    /*use_single_precision=*/false,
                                                                    /*vector_width=*/2,
                                                                    /*vec_lib=*/"SLEEF");
#if defined(__arm64__) || defined(__aarch64__)
            std::regex sleef_exp_decl(R"(declare <2 x double> @_ZGVnN2v_exp\(<2 x double>\))");
            std::regex sleef_exp_call(R"(call <2 x double> @_ZGVnN2v_exp\(<2 x double> .*\))");
#else
            std::regex sleef_exp_decl(R"(declare <2 x double> @_ZGVbN2v_exp\(<2 x double>\))");
            std::regex sleef_exp_call(R"(call <2 x double> @_ZGVbN2v_exp\(<2 x double> .*\))");
#endif
            REQUIRE(std::regex_search(sleef_library_module_str, m, sleef_exp_decl));
            REQUIRE(std::regex_search(sleef_library_module_str, m, sleef_exp_call));
            REQUIRE(!std::regex_search(sleef_library_module_str, m, fexp_call));

            // Check the replacements when using Darwin's libsystem_m.
            std::string libsystem_m_library_module_str =
                run_llvm_visitor(nmodl_text,
                                 /*opt_level=*/0,
                                 /*use_single_precision=*/true,
                                 /*vector_width=*/4,
                                 /*vec_lib=*/"libsystem_m");
            std::regex libsystem_m_exp_decl(R"(declare <4 x float> @_simd_exp_f4\(<4 x float>\))");
            std::regex libsystem_m_exp_call(R"(call <4 x float> @_simd_exp_f4\(<4 x float> .*\))");
            REQUIRE(std::regex_search(libsystem_m_library_module_str, m, libsystem_m_exp_decl));
            REQUIRE(std::regex_search(libsystem_m_library_module_str, m, libsystem_m_exp_call));
            REQUIRE(!std::regex_search(libsystem_m_library_module_str, m, fexp_call));
        }
    }
}

//=============================================================================
// Fast math flags
//=============================================================================

SCENARIO("Fast math flags", "[visitor][llvm]") {
    GIVEN("A function to produce fma and specified math flags") {
        std::string nmodl_text = R"(
            FUNCTION foo(a, b, c) {
                foo = (a * b) + c
            }
        )";

        THEN("instructions are generated with the flags set") {
            std::string module_string =
                run_llvm_visitor(nmodl_text,
                                 /*opt_level=*/3,
                                 /*use_single_precision=*/false,
                                 /*vector_width=*/1,
                                 /*vec_lib=*/"none",
                                 /*fast_math_flags=*/{"nnan", "contract", "afn"});
            std::smatch m;

            // Check flags for produced 'fmul' and 'fadd' instructions.
            std::regex fmul(R"(fmul nnan contract afn double %.*, %.*)");
            std::regex fadd(R"(fadd nnan contract afn double %.*, %.*)");
            REQUIRE(std::regex_search(module_string, m, fmul));
            REQUIRE(std::regex_search(module_string, m, fadd));
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
            std::string module_string = run_llvm_visitor(nmodl_text, /*opt_level=*/3);
            std::smatch m;

            // Check if the values are optimised out.
            std::regex empty_proc(
                R"(define i32 @add\(double %a[0-9].*, double %b[0-9].*\).*\{\n(\s)*ret i32 0\n\})");
            REQUIRE(std::regex_search(module_string, m, empty_proc));
        }
    }
}

//=============================================================================
// Inlining: remove inline code blocks
//=============================================================================

SCENARIO("Removal of inlined functions and procedures", "[visitor][llvm][inline]") {
    GIVEN("Simple breakpoint block calling a function and a procedure") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test_inline
                RANGE a, b, s
            }
            ASSIGNED {
                a
                b
                s
            }
            PROCEDURE test_add(a, b) {
                LOCAL i
                i = a + b
            }
            FUNCTION test_sub(a, b) {
                test_sub = a - b
            }
            BREAKPOINT {
                SOLVE states METHOD cnexp
            }
            DERIVATIVE states {
                a = 1
                b = 2
                test_add(a, b)
                s = test_sub(a, b)
            }
        )";

        THEN("when the code is inlined the procedure and function blocks are removed") {
            std::string module_string = run_llvm_visitor(nmodl_text,
                                                         /*opt_level=*/0,
                                                         /*use_single_precision=*/false,
                                                         /*vector_width=*/1,
                                                         /*vec_lib=*/"none",
                                                         /*fast_math_flags=*/{},
                                                         /*nmodl_inline=*/true);
            std::smatch m;

            // Check if the procedure and function declarations are removed
            std::regex add_proc(R"(define i32 @test_add\(double %a[0-9].*, double %b[0-9].*\))");
            REQUIRE(!std::regex_search(module_string, m, add_proc));
            std::regex sub_func(R"(define double @test_sub\(double %a[0-9].*, double %b[0-9].*\))");
            REQUIRE(!std::regex_search(module_string, m, sub_func));
        }
    }
}

//=============================================================================
// Basic GPU kernel AST generation
//=============================================================================

SCENARIO("GPU kernel body", "[visitor][llvm][gpu]") {
    GIVEN("For GPU platforms") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                RANGE x, y
            }

            ASSIGNED { x y }

            STATE { m }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
              m = y + 2
            }
        )";


        std::string expected_loop = R"(
            for(id = THREAD_ID; id<mech->node_count; id = id+GRID_STRIDE) {
                node_id = mech->node_index[id]
                v = mech->voltage[node_id]
                mech->m[id] = mech->y[id]+2
            })";

        THEN("a loop with GPU-specific AST nodes is constructed") {
            std::string name = "default";
            std::string math_library = "none";
            codegen::Platform gpu_platform(codegen::PlatformID::GPU, name, math_library);
            auto result = run_llvm_visitor_helper(nmodl_text,
                                                  gpu_platform,
                                                  {ast::AstNodeType::CODEGEN_FOR_STATEMENT});
            REQUIRE(result.size() == 1);

            auto loop = reindent_text(to_nmodl(result[0]));
            REQUIRE(loop == reindent_text(expected_loop));
        }
    }
}

//=============================================================================
// Basic NVVM/LLVM IR generation for GPU platforms
//=============================================================================

SCENARIO("GPU kernel body IR generation", "[visitor][llvm][gpu]") {
    GIVEN("For GPU platforms") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                RANGE x, y
            }

            ASSIGNED { x y }

            STATE { m }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
              m = y + 2
            }
        )";

        THEN("kernel annotations are added and thread id intrinsics generated") {
            std::string module_string = run_gpu_llvm_visitor(nmodl_text,
                                                             /*opt_level=*/0,
                                                             /*use_single_precision=*/false);
            std::smatch m;

            // Check kernel annotations are correclty created.
            std::regex annotations(R"(!nvvm\.annotations = !\{!0\})");
            std::regex kernel_data(
                R"(!0 = !\{void \(%.*__instance_var__type\*\)\* @nrn_state_.*, !\"kernel\", i32 1\})");
            REQUIRE(std::regex_search(module_string, m, annotations));
            REQUIRE(std::regex_search(module_string, m, kernel_data));

            // Check thread/block id/dim instrinsics are created.
            std::regex block_id(R"(call i32 @llvm\.nvvm\.read\.ptx\.sreg\.ctaid\.x\(\))");
            std::regex block_dim(R"(call i32 @llvm\.nvvm\.read\.ptx\.sreg\.ntid\.x\(\))");
            std::regex tid(R"(call i32 @llvm\.nvvm\.read\.ptx\.sreg\.tid\.x\(\))");
            std::regex grid_dim(R"(call i32 @llvm\.nvvm\.read\.ptx\.sreg\.nctaid\.x\(\))");
            REQUIRE(std::regex_search(module_string, m, block_id));
            REQUIRE(std::regex_search(module_string, m, block_dim));
            REQUIRE(std::regex_search(module_string, m, tid));
            REQUIRE(std::regex_search(module_string, m, grid_dim));
        }
    }

    GIVEN("When optimizing for GPU platforms") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                RANGE x, y
            }

            ASSIGNED { x y }

            STATE { m }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
              m = y + 2
            }
        )";

        THEN("address spaces are inferred and target information added") {
            std::string module_string = run_gpu_llvm_visitor(nmodl_text,
                                                             /*opt_level=*/3,
                                                             /*use_single_precision=*/false);
            std::smatch m;

            // Check target information.
            // TODO: this may change when more platforms are supported.
            std::regex data_layout(
                R"(target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\")");
            std::regex triple(R"(nvptx64-nvidia-cuda)");
            REQUIRE(std::regex_search(module_string, m, data_layout));
            REQUIRE(std::regex_search(module_string, m, triple));

            // Check for address space casts and address spaces in general when loading data.
            std::regex as_cast(
                R"(addrspacecast %.*__instance_var__type\* %.* to %.*__instance_var__type addrspace\(1\)\*)");
            std::regex gep_as1(
                R"(getelementptr inbounds %.*__instance_var__type, %.*__instance_var__type addrspace\(1\)\* %.*, i64 0, i32 .*)");
            std::regex load_as1(R"(load double\*, double\* addrspace\(1\)\* %.*)");
            REQUIRE(std::regex_search(module_string, m, as_cast));
            REQUIRE(std::regex_search(module_string, m, gep_as1));
            REQUIRE(std::regex_search(module_string, m, load_as1));
        }
    }

    GIVEN("When using math functions") {
        std::string nmodl_text = R"(
            NEURON {
                SUFFIX test
                RANGE x, y
            }

            ASSIGNED { x y }

            STATE { m }

            BREAKPOINT {
                SOLVE states METHOD cnexp
            }

            DERIVATIVE states {
              m = exp(y) + x ^ 2
            }
        )";

        THEN("calls to libdevice are created") {
            std::string module_string = run_gpu_llvm_visitor(nmodl_text,
                                                             /*opt_level=*/3,
                                                             /*use_single_precision=*/false,
                                                             /*math_library=*/"libdevice");
            std::smatch m;

            // Check if exp and pow intrinsics have been replaced.
            std::regex exp_declaration(R"(declare double @__nv_exp\(double\))");
            std::regex exp_new_call(R"(call double @__nv_exp\(double %.*\))");
            std::regex exp_old_call(R"(call double @llvm\.exp\.f64\(double %.*\))");
            std::regex pow_declaration(R"(declare double @__nv_pow\(double, double\))");
            std::regex pow_new_call(R"(call double @__nv_pow\(double %.*, double .*\))");
            std::regex pow_old_call(R"(call double @llvm\.pow\.f64\(double %.*, double .*\))");
            REQUIRE(std::regex_search(module_string, m, exp_declaration));
            REQUIRE(std::regex_search(module_string, m, exp_new_call));
            REQUIRE(!std::regex_search(module_string, m, exp_old_call));
            REQUIRE(std::regex_search(module_string, m, pow_declaration));
            REQUIRE(std::regex_search(module_string, m, pow_new_call));
            REQUIRE(!std::regex_search(module_string, m, pow_old_call));
        }
    }
}
