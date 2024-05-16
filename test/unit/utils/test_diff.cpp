/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <catch2/catch_test_macros.hpp>

#include "test/unit/utils/test_utils.hpp"

using namespace nmodl;
using namespace test_utils;

TEST_CASE("Construction and Identical Strings", "[MyersDiff]") {
    SECTION("Identical strings") {
        MyersDiff diff("Hello\nWorld", "Hello\nWorld");
        for (auto& edit: diff.get_edits()) {
          std::cout << edit << "\n";
        }
        REQUIRE(diff.is_identical() == true);
    }

    SECTION("Completely different strings") {
        MyersDiff diff("Hello\nWorld", "Goodbye\nEarth");
        REQUIRE(diff.is_identical() == false);
    }
}

TEST_CASE("Edit Operations", "[MyersDiff]") {
    SECTION("Insertions and Deletions") {
        MyersDiff diff("Hello", "Hello\nWorld");
        auto edits = diff.get_edits();
        REQUIRE(edits.size() == 1);
        REQUIRE(edits.front().edit == MyersDiff::Edit::etype::ins);
    }

    SECTION("No Edits") {
        MyersDiff diff("", "");
        REQUIRE(diff.is_identical() == true);
        auto edits = diff.get_edits();
        REQUIRE(edits.empty());
    }
}

TEST_CASE("Edge Cases", "[MyersDiff]") {
    SECTION("One input empty") {
        MyersDiff diff("", "Hello\nWorld");
        REQUIRE(diff.is_identical() == false);
        auto edits = diff.get_edits();
        REQUIRE(edits.size() > 0);
    }

    SECTION("Large inputs") {
        // Assuming large_input_a and large_input_b are large multiline strings
        std::string large_input_a = "Large input A content...";
        std::string large_input_b = "Large input B content...";
        MyersDiff diff(large_input_a, large_input_b);
        REQUIRE_NOTHROW(diff.get_edits());
    }
}
