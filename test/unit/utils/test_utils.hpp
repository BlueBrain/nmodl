/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdlib>
#include <deque>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace nmodl {
namespace test_utils {

std::string reindent_text(const std::string& text);

/**
 * \brief Create an empty file which is then removed when the C++ object is destructed
 */
struct TempFile {
    TempFile(std::filesystem::path path, const std::string& content);
    ~TempFile();

  private:
    std::filesystem::path path_;
};

using line = std::pair<std::size_t, std::string>;
using string_lines = std::vector<line>;

class MyersDiff {

  public:

    struct Edit {
        enum etype { ins, del, eql };
        etype edit;
        const line* old_line = nullptr;
        const line* new_line = nullptr;
        Edit(etype e, const line* o, const line* n)
            : edit(e)
            , old_line(o)
            , new_line(n){};

        friend std::ostringstream& operator<<(std::ostringstream& out, Edit&);
    };

    MyersDiff(const std::string& str_a, const std::string& str_b)
        : a(split_lines(str_a))
        , b(split_lines(str_b))
        , max(a.size() + b.size()) {
        identical = diff();
    }

    bool is_identical() const {
        return identical;
    }

    std::deque<Edit> get_edits() const;

  private:

    bool diff();

    struct TraceTuple {
        std::size_t prev_x = 0;
        std::size_t prev_y = 0;
        std::size_t x = 0;
        std::size_t y = 0;
    };

    string_lines a{};
    string_lines b{};
    std::size_t max{};
    bool identical{};
    std::deque<MyersDiff::Edit> edits;

    inline std::size_t idx(int i) const {
        return (i + max) % max;
    }

    string_lines split_lines(const std::string& txt) const;
    std::vector<std::vector<int>> shortest_edit();
};

}  // namespace test_utils
}  // namespace nmodl
