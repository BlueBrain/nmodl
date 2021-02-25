/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#define CATCH_CONFIG_MAIN

#include "codegen/fast_math.hpp"
#include <cmath>

#include <catch/catch.hpp>


template <class T, class = typename std::enable_if<std::is_floating_point<T>::value>::type>
bool check_over_span(T f_ref(T),
                     T f_test(T),
                     const T low_limit,
                     const T high_limit,
                     const size_t npoints) {
    constexpr uint nULP = 4;
    constexpr T eps = std::numeric_limits<T>::epsilon();

    T range = high_limit - low_limit;

    bool ret = true;
    for (size_t i = 0; i < npoints; ++i) {
        T x = low_limit + range * i / npoints;
        T ref = f_ref(x);
        T test = f_test(x);
        T diff = std::abs(ref - test);
        T max = std::max(std::abs(ref), std::abs(test));
        T tol = max * nULP * eps;
        if (diff > tol && diff != 0.0) {
            ret = false;
        }
    }
    return ret;
}

template <class T, class = typename std::enable_if<std::is_floating_point<T>::value>::type>
T exprelr_ref(const T x) {
    return (1.0 + x == 1.0) ? 1.0 : x / (std::exp(x) - 1.0);
};

SCENARIO("Check fast_math") {
    constexpr double low_limit = -700.0;  // limit is 708
    constexpr double high_limit = 700.0;
    constexpr float low_limit_f = -70.0;  // limit is 88
    constexpr float high_limit_f = 70.0;
    constexpr size_t npoints = 2000;

    GIVEN("vexp (double)") {
        auto test = check_over_span(std::exp, vexp, low_limit, high_limit, npoints);

        THEN("error inside threshold") {
            REQUIRE(test);
        }
    }
    GIVEN("vexp (float)") {
        auto test = check_over_span(std::exp, vexp, low_limit_f, high_limit_f, npoints);

        THEN("error inside threshold") {
            REQUIRE(test);
        }
    }
    GIVEN("expm1 (double)") {
        auto test = check_over_span(std::expm1, vexpm1, low_limit, high_limit, npoints);

        THEN("error inside threshold") {
            REQUIRE(test);
        }
    }
    GIVEN("expm1 (float)") {
        auto test = check_over_span(std::expm1, vexpm1, low_limit_f, high_limit_f, npoints);

        THEN("error inside threshold") {
            REQUIRE(test);
        }
    }
    GIVEN("exprelr (double)") {
        auto test = check_over_span(exprelr_ref, exprelr, low_limit, high_limit, npoints);

        THEN("error inside threshold") {
            REQUIRE(test);
        }
    }
    GIVEN("exprelr (float)") {
        auto test = check_over_span(exprelr_ref, exprelr, low_limit_f, high_limit_f, npoints);

        THEN("error inside threshold") {
            REQUIRE(test);
        }
    }
}
