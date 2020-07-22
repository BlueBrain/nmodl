/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "file_library.hpp"

#include <cassert>
#include <cstdlib>
#include <sys/param.h>
#include <unistd.h>

#include "utils/common_utils.hpp"
#include "utils/string_utils.hpp"

namespace nmodl {

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
static constexpr int max_path_len{_MAX_DIR};
#else
static constexpr int max_path_len{MAXPATHLEN};
#endif

FileLibrary FileLibrary::default_instance() {
    FileLibrary library;
    library.append_env_var("NMODL_PATH");
    library.push_cwd();
    return library;
}

void FileLibrary::append_dir(const std::string& path) {
    paths_.insert(paths_.begin(), path);
}

void FileLibrary::append_env_var(const std::string& env_var) {
    const auto value = getenv(env_var.c_str());
    if (value != nullptr) {
        for (const auto& path: stringutils::split_string(value, utils::envpathsep)) {
            if (!path.empty()) {
                append_dir(path);
            }
        }
    }
}

void FileLibrary::push_current_directory(const std::string& path) {
    paths_.push_back(path);
}

void FileLibrary::push_cwd() {
    char cwd[MAXPATHLEN + 1];

    if (nullptr == getcwd(cwd, MAXPATHLEN + 1)) {
        throw std::runtime_error("working directory name too long");
    }
    push_current_directory(std::string(cwd));
}

void FileLibrary::pop_current_directory() {
    assert(!paths_.empty());
    paths_.erase(--paths_.end());
}

std::string FileLibrary::find_file(const std::string& file) {
    if (utils::file_is_abs(file)) {
        if (utils::file_exists(file)) {
            return "";
        }
    }
    for (auto paths_it = paths_.rbegin(); paths_it != paths_.rend(); ++paths_it) {
        auto file_abs = *paths_it + utils::pathsep + file;
        if (utils::file_exists(file_abs)) {
            return *paths_it;
        }
    }
    return "";
}

}  // namespace nmodl
