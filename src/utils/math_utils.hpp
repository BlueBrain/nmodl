/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <cmath>
#include <type_traits>

/**
 *
 * \dir
 * Utility classes and function
 *
 * \file
 * Common utility functions for number manipulation
 */

namespace nmodl {
namespace utils {

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> approximaticaly_equal(T a, T b) {
    const auto absa = std::abs(a);
    const auto absb = std::abs(b);
    const auto scale = absa < absb ? absb : absa;
    return std::abs(a - b) < std::numeric_limits<T>::epsilon() * scale;
}

}  // namespace utils
}  // namespace nmodl
