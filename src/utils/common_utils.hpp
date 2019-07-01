/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <memory>

/**
 *
 * \dir
 * \brief Utility classes and function
 *
 * \file
 * \brief Common utility functions for file/dir manipulation
 */

namespace nmodl {
/// file/string manipulation functions
namespace utils {

/**
 * @defgroup utils Utility Implementation
 * @brief Utility classes and function implementation
 * @{
 */

/// Check if the iterator is pointing to last element in the container
template <typename Iter, typename Cont>
bool is_last(Iter iter, const Cont& cont) {
    return ((iter != cont.end()) && (next(iter) == cont.end()));
}

/// Given full file path, returns only name of the file
template <class T>
T base_name(T const& path, T const& delims = "/\\") {
    return path.substr(path.find_last_of(delims) + 1);
}

/// Given the file name, returns name of the file without extension
template <class T>
T remove_extension(T const& filename) {
    typename T::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
}

/// Given directory path, create sub-directories
bool make_path(const std::string& path);

/// Check if directory with given path exist
bool is_dir_exist(const std::string& path);

std::string gen_random(const int len);

class SingletonRandomString {
  public:
    SingletonRandomString(SingletonRandomString const&) = delete;
    SingletonRandomString& operator=(SingletonRandomString const&) = delete;

    static std::shared_ptr<SingletonRandomString> instance() {
        static std::shared_ptr<SingletonRandomString> s{new SingletonRandomString};
        return s;
    }

    std::string get_random_string_X() {
        return random_string_X;
    }

    std::string reset_random_string_X() {
        random_string_X = gen_random(4);
        return random_string_X;
    }

    std::string get_random_string_J() {
        return random_string_J;
    }

    std::string reset_random_string_J() {
        random_string_J = gen_random(4);
        return random_string_J;
    }

    std::string get_random_string_Jm() {
        return random_string_Jm;
    }

    std::string reset_random_string_Jm() {
        random_string_Jm = gen_random(4);
        return random_string_Jm;
    }

    std::string get_random_string_F() {
        return random_string_F;
    }

    std::string reset_random_string_F() {
        random_string_F = gen_random(4);
        return random_string_F;
    }

  private:
    SingletonRandomString() {}
    std::string random_string_X;
    std::string random_string_J;
    std::string random_string_Jm;
    std::string random_string_F;
};

/** @} */  // end of utils

}  // namespace utils
}  // namespace nmodl