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

#include "file_library.hpp"

#include <cassert>
#include <filesystem>
#include <sys/param.h>
#include <unistd.h>

#include "utils/common_utils.hpp"
#include "utils/string_utils.hpp"

namespace fs = std::filesystem;

namespace nmodl {


FileLibrary FileLibrary::default_instance() {
    FileLibrary library;
    library.append_env_var("NMODL_PATH");
    library.push_cwd();
    return library;
}

void FileLibrary::append_env_var(const std::string& env_var) {
    const auto value = getenv(env_var.c_str());
    if (value != nullptr) {
        for (const auto& path: stringutils::split_string(value, utils::envpathsep)) {
            if (!path.empty()) {
                paths_.insert(paths_.begin(), path);
            }
        }
    }
}

void FileLibrary::push_current_directory(const fs::path& path) {
    paths_.push_back(path);
}

void FileLibrary::push_cwd() {
    push_current_directory(fs::current_path());
}

void FileLibrary::pop_current_directory() {
    assert(!paths_.empty());
    if (!paths_.empty()) {
        paths_.pop_back();
    }
}

std::string FileLibrary::find_file(const fs::path& file) {
    if (file.is_absolute() && fs::exists(file)) {
        return "";
    }
    for (auto paths_it = paths_.rbegin(); paths_it != paths_.rend(); ++paths_it) {
        auto file_abs = *paths_it / file;
        if (fs::exists(file_abs)) {
            return paths_it->string();
        }
    }
    return "";
}

}  // namespace nmodl
