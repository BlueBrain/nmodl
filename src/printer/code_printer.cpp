/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fstream>
#include <iostream>

#include "printer/code_printer.hpp"
#include "utils/string_utils.hpp"

#if NMODL_ENABLE_BACKWARD
#include <backward.hpp>
#endif

namespace nmodl {
namespace printer {
CodePrinter::CodePrinter(size_t blame_line)
    : result(std::make_unique<std::ostream>(std::cout.rdbuf()))
    , blame_line(blame_line) {}

CodePrinter::CodePrinter(std::ostream& stream, size_t blame_line)
    : result(std::make_unique<std::ostream>(stream.rdbuf()))
    , blame_line(blame_line) {}

CodePrinter::CodePrinter(const std::string& filename, size_t blame_line)
    : blame_line(blame_line) {
    if (filename.empty()) {
        throw std::runtime_error("Empty filename for CodePrinter");
    }

    ofs.open(filename.c_str());

    if (ofs.fail()) {
        auto msg = "Error while opening file '" + filename + "' for CodePrinter";
        throw std::runtime_error(msg);
    }

    sbuf = ofs.rdbuf();
    result = std::make_unique<std::ostream>(sbuf);
}

CodePrinter::~CodePrinter() {
    ofs.close();
}

void CodePrinter::push_block() {
    add_text('{');
    add_newline();
    indent_level++;
}

void CodePrinter::push_block(const std::string& expression) {
    add_indent();
    add_text(expression, " {");
    add_newline();
    indent_level++;
}

void CodePrinter::chain_block(std::string const& expression) {
    --indent_level;
    add_indent();
    add_text("} ", expression, " {");
    add_newline();
    ++indent_level;
}

void CodePrinter::add_indent() {
    add_text(std::string(indent_level * NUM_SPACES, ' '));
}

void CodePrinter::add_multi_line(const std::string& text) {
    const auto& lines = stringutils::split_string(text, '\n');

    int prefix_length{};
    int start_line{};
    while (start_line < lines.size()) {
        const auto& line = lines[start_line];
        // skip first empty line, if any
        if (line.empty()) {
            ++start_line;
            continue;
        }
        // The common indentation of all blocks if the number of spaces
        // at the beginning of the first non-empty line.
        for (auto line_it = line.begin(); line_it != line.end() && *line_it == ' '; ++line_it) {
            prefix_length += 1;
        }
        break;
    }

    for (const auto& line: lines) {
        if (line.size() < prefix_length) {
            // ignore lines made of ' ' characters
            if (std::find_if_not(line.begin(), line.end(), [](char c) { return c == ' '; }) !=
                line.end()) {
                throw std::invalid_argument("Indentation mismatch");
            }
        } else {
            add_line(line.substr(prefix_length));
        }
    }
}

void CodePrinter::add_newline(std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        add_text('\n');
        ++current_line;
    }
}

void CodePrinter::pop_block() {
    pop_block_nl(1);
}

void CodePrinter::pop_block_nl(std::size_t num_newlines) {
    indent_level--;
    add_indent();
    add_text('}');
    add_newline(num_newlines);
}

void CodePrinter::pop_block(const std::string_view& suffix, std::size_t num_newlines) {
    pop_block_nl(0);
    add_text(suffix);
    add_newline(num_newlines);
}

void CodePrinter::blame() {
#if NMODL_ENABLE_BACKWARD
    if (current_line == blame_line) {
        *result << std::flush;

        std::cout << "\n\n== Blame =======================================================\n";

        backward::StackTrace st;
        st.load_here(32);
        backward::Printer p;
        p.print(st, std::cout);
    }
#endif
}


}  // namespace printer
}  // namespace nmodl
