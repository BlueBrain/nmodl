/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

namespace nmodl {
namespace test_utils {

std::string reindent_text(const std::string& text);

/**
 * \brief Create an empty file which is then removed when the C++ object is destructed
 */
struct TempFile {
    explicit TempFile(std::string path);
    TempFile(std::string path, const std::string& content);
    ~TempFile();

  private:
    std::string path_;
};

}  // namespace test_utils
}  // namespace nmodl
