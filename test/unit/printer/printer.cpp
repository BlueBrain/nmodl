/*
 * Copyright 2023 Blue Brain Project, EPFL.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "printer/json_printer.hpp"

using nmodl::printer::JSONPrinter;

TEST_CASE("JSON printer converting object to string form", "[printer][json]") {
    SECTION("Stringstream test 1") {
        std::stringstream ss;
        JSONPrinter p(ss);
        p.compact_json(true);

        p.push_block("A");
        p.add_node("B");
        p.pop_block();
        p.flush();

        auto result = R"({"A":[{"name":"B"}]})";
        REQUIRE(ss.str() == result);
    }

    SECTION("Stringstream test 2") {
        std::stringstream ss;
        JSONPrinter p(ss);
        p.compact_json(true);

        p.push_block("A");
        p.add_node("B");
        p.add_node("C");
        p.push_block("D");
        p.add_node("E");
        p.pop_block();
        p.pop_block();
        p.flush();

        auto result = R"({"A":[{"name":"B"},{"name":"C"},{"D":[{"name":"E"}]}]})";
        REQUIRE(ss.str() == result);
    }

    SECTION("Test with nodes as separate tags") {
        std::stringstream ss;
        JSONPrinter p(ss);
        p.compact_json(true);
        p.expand_keys(true);

        p.push_block("A");
        p.add_node("B");
        p.push_block("D");
        p.add_node("E");
        p.pop_block();
        p.flush();

        auto result =
            R"({"children":[{"name":"B"},{"children":[{"name":"E"}],"name":"D"}],"name":"A"})";
        REQUIRE(ss.str() == result);
    }
}
