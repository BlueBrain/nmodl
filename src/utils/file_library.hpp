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

#pragma once

#include <filesystem>
#include <string>
#include <vector>

/**
 *
 * \dir
 * \brief Utility classes and function
 *
 * \file
 * \brief Manage search path.
 */

namespace nmodl {

/**
 * \brief Manage search path.
 *
 * Store search path used for handling paths when processing include NMODL directive
 */
class FileLibrary {
  public:
    /// An empty library
    FileLibrary() = default;
    /**
     *  Initialize the library with the following path:
     *  - current working directory
     *  - paths in the NMODL_PATH environment variable
     */
    static FileLibrary default_instance();

    /**
     * \name Managing inclusion paths.
     * \{
     */
    void append_env_var(const std::string& env_var);
    /** \} */

    /**
     * \name current directory
     * \{
     */
    void push_current_directory(const std::filesystem::path& path);
    void pop_current_directory();
    /** \} */

    /**
     * \brief Search a file.
     * Determine real path of \a file
     * \return Directory containing \a file, or "" if not found.
     */
    std::string find_file(const std::filesystem::path& file);

  private:
    /// push the working directory in the directories stack
    void push_cwd();

    /// inclusion path list
    std::vector<std::filesystem::path> paths_;
};

}  // namespace nmodl
