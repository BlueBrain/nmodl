/*************************************************************************
 * Copyright (C) 2019-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <catch/catch.hpp>

#include "codegen/codegen_c_visitor.hpp"
#include "codegen/codegen_ispc_visitor.hpp"
#include "codegen/codegen_utils.hpp"

using namespace nmodl;
using namespace visitor;
using namespace codegen;


SCENARIO("C codegen utility functions", "[codegen][util][c]") {
    GIVEN("Double constant as string") {
        std::string double_constant = "0.012345678901234567";

        THEN("Codegen C Visitor prints double with same precision") {
            auto nmodl_constant_result = codegen::utils::double_to_string<CodegenCVisitor>(
                double_constant);
            REQUIRE(nmodl_constant_result == double_constant);
        }
    }

    GIVEN("Integer constant as string") {
        std::string double_constant = "1";

        std::string codegen_output = "1.0";

        THEN("Codegen C Visitor prints integer as double number") {
            auto nmodl_constant_result = codegen::utils::double_to_string<CodegenCVisitor>(
                double_constant);
            REQUIRE(nmodl_constant_result == codegen_output);
        }
    }

    GIVEN("Float constant as string") {
        std::string float_constant = "0.01234567";

        THEN("Codegen C Visitor prints float with same precision") {
            auto nmodl_constant_result = codegen::utils::float_to_string<CodegenCVisitor>(
                float_constant);
            REQUIRE(nmodl_constant_result == float_constant);
        }
    }

    GIVEN("Float constant as string") {
        std::string float_constant = "1";

        std::string codegen_output = "1.0";

        THEN("Codegen C Visitor prints integer as double number") {
            auto nmodl_constant_result = codegen::utils::float_to_string<CodegenCVisitor>(
                float_constant);
            REQUIRE(nmodl_constant_result == codegen_output);
        }
    }

    GIVEN("Double constants in scientific notation as strings") {
        std::string double_constant_with_e = "1e+18";
        std::string double_constant_with_e_minus = "1e-18";
        std::string double_constant_with_E = "1E18";

        THEN("Codegen C Visitor prints doubles with scientific notation") {
            auto nmodl_constant_result = codegen::utils::double_to_string<CodegenCVisitor>(
                double_constant_with_e);
            REQUIRE(nmodl_constant_result == double_constant_with_e);
            nmodl_constant_result = codegen::utils::double_to_string<CodegenCVisitor>(
                double_constant_with_e_minus);
            REQUIRE(nmodl_constant_result == double_constant_with_e_minus);
            nmodl_constant_result = codegen::utils::double_to_string<CodegenCVisitor>(
                double_constant_with_E);
            REQUIRE(nmodl_constant_result == double_constant_with_E);
        }
    }

    GIVEN("Float constants in scientific notation as strings") {
        std::string float_constant_with_e = "1e+18";
        std::string float_constant_with_e_minus = "1e-18";
        std::string float_constant_with_E = "1E18";

        THEN("Codegen C Visitor prints doubles with scientific notation") {
            auto nmodl_constant_result = codegen::utils::float_to_string<CodegenCVisitor>(
                float_constant_with_e);
            REQUIRE(nmodl_constant_result == float_constant_with_e);
            nmodl_constant_result = codegen::utils::float_to_string<CodegenCVisitor>(
                float_constant_with_e_minus);
            REQUIRE(nmodl_constant_result == float_constant_with_e_minus);
            nmodl_constant_result = codegen::utils::float_to_string<CodegenCVisitor>(
                float_constant_with_E);
            REQUIRE(nmodl_constant_result == float_constant_with_E);
        }
    }
}

SCENARIO("ISPC codegen utility functions", "[codegen][util][ispc]") {
    GIVEN("Double constant as string") {
        std::string double_constant = "0.012345678901234567";
        std::string double_constant_with_front_decimal_point = ".012345678901234567";
        std::string double_constant_with_only_decimal_point = "123.";

        std::string codegen_output_long = "0.012345678901234567d";
        std::string codegen_output_only_decimal_point = "123.d";


        THEN("Codegen ISPC Visitor prints double with same precision") {
            auto nmodl_constant_result = codegen::utils::double_to_string<CodegenIspcVisitor>(
                double_constant);
            REQUIRE(nmodl_constant_result == codegen_output_long);
            nmodl_constant_result = codegen::utils::double_to_string<CodegenIspcVisitor>(
                double_constant_with_front_decimal_point);
            REQUIRE(nmodl_constant_result == codegen_output_long);
            nmodl_constant_result = codegen::utils::double_to_string<CodegenIspcVisitor>(
                double_constant_with_only_decimal_point);
            REQUIRE(nmodl_constant_result == codegen_output_only_decimal_point);
        }
    }

    GIVEN("Integer constant as string") {
        std::string double_constant = "1";

        std::string codegen_output = "1.0d";

        THEN("Codegen ISPC Visitor prints integer as double number") {
            auto nmodl_constant_result = codegen::utils::double_to_string<CodegenIspcVisitor>(
                double_constant);
            REQUIRE(nmodl_constant_result == codegen_output);
        }
    }

    GIVEN("Float constant as string") {
        std::string float_constant = "0.01234567";
        std::string float_constant_with_front_decimal_point = ".01234567";
        std::string double_constant_with_only_decimal_point = "123.";

        std::string codegen_output = "0.01234567f";
        std::string codegen_output_only_decimal_point = "123.f";

        THEN("Codegen ISPC Visitor prints float with same precision") {
            auto nmodl_constant_result = codegen::utils::float_to_string<CodegenIspcVisitor>(
                float_constant);
            REQUIRE(nmodl_constant_result == codegen_output);
            nmodl_constant_result = codegen::utils::float_to_string<CodegenIspcVisitor>(
                float_constant_with_front_decimal_point);
            REQUIRE(nmodl_constant_result == codegen_output);
            nmodl_constant_result = codegen::utils::float_to_string<CodegenIspcVisitor>(
                double_constant_with_only_decimal_point);
            REQUIRE(nmodl_constant_result == codegen_output_only_decimal_point);
        }
    }

    GIVEN("Float constant as string") {
        std::string float_constant = "1";

        std::string codegen_output = "1.0f";

        THEN("Codegen ISPC Visitor prints integer as double number") {
            auto nmodl_constant_result = codegen::utils::float_to_string<CodegenIspcVisitor>(
                float_constant);
            REQUIRE(nmodl_constant_result == codegen_output);
        }
    }

    GIVEN("Double constants in scientific notation as strings") {
        std::string double_constant_with_e = "1e+18";
        std::string double_constant_with_e_minus = "1e-18";
        std::string double_constant_with_decimal_point_e = ".123e18";
        std::string double_constant_with_E = "1E18";


        std::string result_double_constant_with_e = "1d+18";
        std::string result_double_constant_with_e_minus = "1d-18";
        std::string result_double_constant_with_decimal_point_e = ".123d18";
        std::string result_double_constant_with_E = "1d18";

        THEN("Codegen ISPC Visitor prints doubles with scientific notation") {
            auto nmodl_constant_result = codegen::utils::double_to_string<CodegenIspcVisitor>(
                double_constant_with_e);
            REQUIRE(nmodl_constant_result == result_double_constant_with_e);
            nmodl_constant_result = codegen::utils::double_to_string<CodegenIspcVisitor>(
                double_constant_with_e_minus);
            REQUIRE(nmodl_constant_result == result_double_constant_with_e_minus);
            nmodl_constant_result = codegen::utils::double_to_string<CodegenIspcVisitor>(
                double_constant_with_decimal_point_e);
            REQUIRE(nmodl_constant_result == result_double_constant_with_decimal_point_e);
            nmodl_constant_result = codegen::utils::double_to_string<CodegenIspcVisitor>(
                double_constant_with_E);
            REQUIRE(nmodl_constant_result == result_double_constant_with_E);
        }
    }

    GIVEN("Float constants in scientific notation as strings") {
        std::string float_constant_with_e = "1e+18";
        std::string float_constant_with_e_minus = "1e-18";
        std::string float_constant_with_decimal_point_e = ".123e18";
        std::string float_constant_with_E = "1E18";

        THEN("Codegen ISPC Visitor prints doubles with scientific notation") {
            auto nmodl_constant_result = codegen::utils::float_to_string<CodegenIspcVisitor>(
                float_constant_with_e);
            REQUIRE(nmodl_constant_result == float_constant_with_e);
            nmodl_constant_result = codegen::utils::float_to_string<CodegenIspcVisitor>(
                float_constant_with_e_minus);
            REQUIRE(nmodl_constant_result == float_constant_with_e_minus);
            nmodl_constant_result = codegen::utils::float_to_string<CodegenIspcVisitor>(
                float_constant_with_decimal_point_e);
            REQUIRE(nmodl_constant_result == float_constant_with_decimal_point_e);
            nmodl_constant_result = codegen::utils::float_to_string<CodegenIspcVisitor>(
                float_constant_with_E);
            REQUIRE(nmodl_constant_result == float_constant_with_E);
        }
    }
}
