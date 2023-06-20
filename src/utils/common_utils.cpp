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

#include "common_utils.hpp"

#include <array>
#include <cassert>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define IS_WINDOWS
#endif

namespace nmodl {
namespace utils {

std::string generate_random_string(const int len, UseNumbersInString use_numbers) {
    std::string s(len, 0);
    constexpr std::size_t number_of_numbers{10};
    constexpr std::string_view alphanum{
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"};
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(use_numbers ? 0
                                                                              : number_of_numbers,
                                                                  alphanum.size() - 1);
    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[dist(rng)];
    }
    return s;
}

}  // namespace utils
}  // namespace nmodl
