/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "catch/catch.hpp"

#include "parser/nmodl_driver.hpp"
#include "src/config/config.h"
#include "test/utils/nmodl_constructs.hpp"
#include "test/utils/test_utils.hpp"
#include "utils/logger.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/units_visitor.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test_utils;

using nmodl::parser::NmodlDriver;

//=============================================================================
// Unit visitor tests
//=============================================================================

std::string run_units_visitor(const std::string& text) {
    NmodlDriver driver;
    driver.parse_string(text);
    auto ast = driver.get_ast();

    std::stringstream ss;
    std::string units_lib_path(NrnUnitsLib::get_path());
    UnitsVisitor(units_lib_path, ss).visit_program(ast.get());
    return ss.str();
}


SCENARIO("Units Visitor") {
    GIVEN("UNITS block with different cases of units definitions") {
        std::string nmodl_text = R"(
            UNITS {
                (nA)    = (nanoamp)
                (mA)    = (milliamp)
                (mV)    = (millivolt)
                (uS)    = (microsiemens)
                (nS)    = (nanosiemens)
                (pS)    = (picosiemens)
                (umho)  = (micromho)
                (um)    = (micrometers)
                (mM)    = (milli/liter)
                (uM)    = (micro/liter)
                (msM) = (ms mM)
                (fAm) = (femto amp meter)
                (mol) = (1)
                (M) = (1/liter)
                (uM1) = (micro M)
                (mA/cm2) = (nanoamp/cm2)
                (molar) = (1 / liter)
                (S ) = (siemens)
                (mse-1) = (1/millisec)
                (um3) = (liter/1e15)
                (molar1) = (/liter)
                (degK) = (degC)
                FARADAY1 = (faraday) (coulomb)
                FARADAY2 = (faraday) (kilocoulombs)
                FARADAY3 = (faraday) (10000 coulomb)
                PI      = (pi)      (1)
                R1       = (k-mole)  (joule/degC)
                R2 = 8.314 (volt-coul/degC)
                R3 = (mole k) (mV-coulomb/degC)
                R4 = 8.314 (volt-coul/degK)
                R5 = 8.314500000000001 (volt coul/kelvin)
                dummy1  = 123.45    (m 1/sec2)
                dummy2  = 123.45e3  (millimeters/sec2)
                dummy3  = 12345e-2  (m/sec2)
                KTOMV = 0.0853 (mV/degC)
                B = 0.26 (mM-cm2/mA-ms)
                TEMP = 25 (degC)
            }
        )";

        std::string output_nmodl = R"(
        nA 0.00000000: sec-1 coul1
        mA 0.00100000: sec-1 coul1
        mV 0.00100000: m2 kg1 sec-2 coul-1
        uS 0.00000100: m-2 kg-1 sec1 coul2
        nS 0.00000000: m-2 kg-1 sec1 coul2
        pS 0.00000000: m-2 kg-1 sec1 coul2
        umho 0.00000100: m-2 kg-1 sec1 coul2
        um 0.00000100: m1
        mM 1.00000000: m-3
        uM 0.00100000: m-3
        msM 0.00100000: m-3 sec1
        fAm 0.00000000: m1 sec-1 coul1
        mol 1.00000000: constant
        M 1000.00000000: m-3
        uM1 0.00100000: m-3
        mA/cm2 0.00001000: m-2 sec-1 coul1
        molar 1000.00000000: m-3
        S 1.00000000: m-2 kg-1 sec1 coul2
        mse-1 1000.00000000: sec-1
        um3 0.00100000: m3
        molar1 1000.00000000: m-3
        degK 1.00000000: K1
        FARADAY1 96485.30900000: coul1
        FARADAY2 96.48530900: coul1
        FARADAY3 9.64853090: coul1
        PI 3.14159265: constant
        R1 8.31449872: m2 kg1 sec-2 K-1
        R2 8.31400000: m2 kg1 sec-2 K-1
        R3 8314.49871704: m2 kg1 sec-2 K-1
        R4 8.31400000: m2 kg1 sec-2 K-1
        R5 8.31450000: m2 kg1 sec-2 K-1
        dummy1 123.45000000: m1 sec-2
        dummy2 123450.00000000: m1 sec-2
        dummy3 123.45000000: m1 sec-2
        KTOMV 0.08530000: m2 kg1 sec-2 coul-1 K-1
        B 0.26000000: m-1 coul-1
        TEMP 25.00000000: K1
        )";

        THEN("Print the units that were added") {
            std::string input = reindent_text(nmodl_text);
            auto expected_result = reindent_text(output_nmodl);
            auto result = run_units_visitor(input);
            auto reindented_result = reindent_text(result);
            REQUIRE(reindented_result == expected_result);
        }
    }
}
