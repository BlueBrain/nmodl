/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#define CATCH_CONFIG_MAIN

#include <string>
#include <utility>

#include "catch/catch.hpp"
#include "parser/diffeq_driver.hpp"
#include "parser/unit_driver.hpp"

//=============================================================================
// Parser tests
//=============================================================================

// Driver is defined as global to store all the units inserted to it and to be
// able to define complex units based on base units
nmodl::parser::UnitDriver driver;

bool is_valid_construct(const std::string& construct) {
    return driver.parse_string(construct);
}

SCENARIO("Unit parser can read definition of units") {
    GIVEN("A base unit") {
        WHEN("Base unit is *a*") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("m\t\t\t*a*\n"));
            }
        }
        WHEN("Base unit is *b*") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("kg\t\t\t*b*\n"));
            }
        }
        WHEN("Base unit is *d*") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("coul\t\t\t*d*\n"));
            }
        }
        WHEN("Base unit is *i*") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("erlang\t\t\t*i*\n"));
            }
        }
        WHEN("Base unit is *c*") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("sec\t\t\t*c*\n"));
            }
        }
    }
    GIVEN("A double number") {
        WHEN("Double number is writen like 3.14") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("pi\t\t\t3.14159265358979323846\n"));
            }
        }
        WHEN("Double number is writen like 1") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("fuzz\t\t\t1\n"));
            }
        }
    }
    GIVEN("A dimensionless constant") {
        WHEN("Constant expression is double / constant") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("radian\t\t\t.5 / pi\n"));
            }
        }
    }
    GIVEN("A power of another unit") {
        WHEN("Power of 2") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("steradian\t\tradian2\n"));
            }
        }
        WHEN("Power of 3") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("stere\t\t\tm3\n"));
            }
        }
    }
    GIVEN("Divisions and multiplications of units") {
        WHEN("Units are multiplied") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("degree\t\t\t1|180 pi-radian\n"));
            }
        }
        WHEN("There are both divisions and multiplications") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("newton\t\t\tkg-m/sec2\n"));
            }
        }
        WHEN("There is only division") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("dipotre\t\t\t/m\n"));
            }
        }
    }
    GIVEN("A double number and some units") {
        WHEN("Double number is multiplied by a power of 10 with division of multiple units") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("c\t\t\t2.99792458+8 m/sec fuzz\n"));
            }
        }
        WHEN("Double number is writen like .9") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("grade\t\t\t.9 degree\n"));
            }
        }
    }
    GIVEN("A fraction and some units") {
        WHEN("Fraction is writen like 1|2") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("ccs\t\t\t1|36 erlang\n"));
            }
        }
        WHEN("Fraction is writen like 1|8.988e9") {
            THEN("parser accepts without an error") {
                REQUIRE(is_valid_construct("statcoul\t\t1|2.99792458+9 coul\n"));
            }
        }
    }
}
