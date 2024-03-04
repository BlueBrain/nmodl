/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.hpp"
#include "utils/logger.hpp"
#include "utils/string_utils.hpp"

#include "fmt/core.h"

#include <cassert>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace nmodl {
namespace test_utils {

int count_leading_spaces(std::string text) {
    auto const length = text.size();
    text = nmodl::stringutils::ltrim(text);
    auto const num_whitespaces = length - text.size();
    assert(num_whitespaces <= std::numeric_limits<int>::max());
    return static_cast<int>(num_whitespaces);
}

/// check if string has only whitespaces
bool is_empty(const std::string& text) {
    return nmodl::stringutils::trim(text).empty();
}

/** Reindent nmodl text for text-to-text comparison
 *
 * Nmodl constructs defined in test database has extra leading whitespace.
 * This is done for readability reason in nmodl_constructs.cpp. For example,
 * we have following nmodl text with 8 leading whitespaces:


        NEURON {
            RANGE x
        }

 * We convert above paragraph to:

NEURON {
    RANGE x
}

 * i.e. we get first non-empty line and count number of leading whitespaces (X).
 * Then for every sub-sequent line, we remove first X characters (assuming those
 * all are whitespaces). This is done because when ast is transformed back to
 * nmodl, the nmodl output is without "extra" whitespaces in the provided input.
 */

std::string reindent_text(const std::string& text) {
    std::string indented_text;
    int num_whitespaces = 0;
    bool flag = false;
    std::string line;
    std::stringstream stream(text);

    while (std::getline(stream, line)) {
        if (!line.empty()) {
            /// count whitespaces for first non-empty line only
            if (!flag) {
                flag = true;
                num_whitespaces = count_leading_spaces(line);
            }

            /// make sure we don't remove non whitespaces characters
            if (!is_empty(line.substr(0, num_whitespaces))) {
                throw std::runtime_error("Test nmodl input not correctly formatted");
            }

            line.erase(0, num_whitespaces);
            indented_text += line;
        }
        /// discard empty lines at very beginning
        if (!stream.eof() && flag) {
            indented_text += "\n";
        }
    }
    return indented_text;
}

TempFile::TempFile(fs::path path, const std::string& content)
    : path_(std::move(path)) {
    std::ofstream output(path_);
    output << content;
}

TempFile::~TempFile() {
    try {
        fs::remove(path_);
    } catch (...) {
        // TODO: remove .string() once spdlog use fmt 9.1.0
        logger->error("Cannot delete temporary file {}", path_.string());
    }
}
bool MyersDiff::diff() {
    edits = std::deque<Edit>();
    auto x = static_cast<int>(a.size());
    auto y = static_cast<int>(b.size());

    bool identical = true;

    auto trace = shortest_edit();
    auto d = static_cast<int>(trace.size());

    for (auto v_it = trace.rbegin(); v_it != trace.rend(); v_it++, d--) {
        auto v = *v_it;
        int k = x - y;
        int prev_k{};
        if ((k == -d) || ((k != d) && (v[idx(k - 1)] < v[idx(k + 1)]))) {
            prev_k = k + 1;
        } else {
            prev_k = k - 1;
        }
        int prev_x = v[idx(prev_k)];
        int prev_y = prev_x - prev_k;

        while ((x > prev_x) && (y > prev_y)) {
            edits.emplace_front(Edit::etype::eql, &a[x - 1], &b[y - 1]);
            x--;
            y--;
        }
        if (x == prev_x) {
            edits.emplace_front(Edit::etype::ins, nullptr, &b[y - 1]);
            identical = false;
        } else if (y == prev_y) {
            edits.emplace_front(Edit::etype::del, &a[x - 1], nullptr);
            identical = false;
        }
        x = prev_x;
        y = prev_y;
    }
    return identical;
}

std::deque<MyersDiff::Edit> MyersDiff::get_edits() const {
    return edits;
}

std::vector<std::vector<int>> MyersDiff::shortest_edit() {
    auto n = static_cast<int>(a.size());
    auto m = static_cast<int>(b.size());
    int x{};
    int y{};

    auto trace = std::vector<std::vector<int>>();
    auto v = std::vector<int>(2 * max + 1, 0);
    for (int d = 0; d <= max; ++d) {
        trace.push_back(v);
        for (int k = -d; k <= d; k += 2) {
            if ((k == -d) || (k != d && (v[idx(k - 1)] < v[idx(k + 1)]))) {
                x = v[idx(k + 1)];
            } else {
                x = v[idx(k - 1)] + 1;
            }
            y = x - k;
            while ((x < n) && (y < m) && (a[x].second == b[y].second)) {
                x++;
                y++;
            }
            v[idx(k)] = x;

            if ((x >= n) && (y >= m)) {
                trace.push_back(v);
                return trace;
            }
        }
    }
    return trace;
}

string_lines MyersDiff::split_lines(const std::string& txt) const {
    std::size_t pos = 0;
    std::size_t ppos = 0;
    std::size_t lineno = 1;
    string_lines lines;
    while ((pos = txt.find('\n', ppos)) != std::string::npos) {
        lines.emplace_back(lineno++, txt.substr(ppos, pos - ppos));
        ppos = pos + 1;
    }
    if (ppos < txt.length()) {
        lines.emplace_back(lineno, txt.substr(ppos));
    }

    return lines;
}

std::ostringstream& operator<<(std::ostringstream& oss, MyersDiff::Edit& edit) {
    if (edit.edit == MyersDiff::Edit::etype::ins) {
        oss << fmt::format("{} {}", '+', edit.new_line->second);
    } else if (edit.edit == MyersDiff::Edit::etype::del) {
        oss << fmt::format("{} {}", '-', edit.old_line->second);
    } else {
        oss << fmt::format("{} {}", ' ', edit.old_line->second);
    }
    return oss;
}


}  // namespace test_utils
}  // namespace nmodl
