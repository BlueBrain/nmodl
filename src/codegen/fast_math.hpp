/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *
 * Note that the fast ISPC exponentials Based on VDT implementation of
 * D. Piparo et al. See https://github.com/dpiparo/vdt for additional
 * license information.
 * exprelr and expm1 are based on https://arbor.readthedocs.io/en/latest/internals/simd_api.html?highlight=exprelr#operatorname-expm1-x
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief Implementation of different math functions with ISPC
 */

static inline double uint642dp(u_int64_t ll) {
    return *(( double*) (&ll));
}

static inline u_int64_t dp2uint64(double x) {
    return *(( u_int64_t*) (&x));
}

static inline float uint322sp(int32_t x) {
    return *(( float*) (&x));
}

static inline unsigned int sp2uint32(float x) {
    return *(( u_int32_t*) (&x));
}

static inline double dpfloor(const double x) {
    int32_t ret = x;
    ret -= (sp2uint32(x) >> 31);
    return ret;
}

static inline float spfloor(const float x) {
    int32_t ret = x;
    ret -= (sp2uint32(x) >> 31);
    return ret;
}

static inline float f_inf() {
    u_int32_t v = 0x7F800000;
    return *((float*) (&v));
}

static inline double inf() {
    u_int64_t v = 0x7FF0000000000000;
    return *((double*) (&v));
}


static const double EXP_LIMIT = 708.0;

static const double PX1exp = 1.26177193074810590878-4;
static const double PX2exp = 3.02994407707441961300-2;
static const double PX3exp = 9.99999999999999999910-1;
static const double QX1exp = 3.00198505138664455042-6;
static const double QX2exp = 2.52448340349684104192-3;
static const double QX3exp = 2.27265548208155028766-1;
static const double QX4exp = 2.00000000000000000009;

static const double LOG2E = 1.4426950408889634073599; // 1/ln(2)

static const float MAXLOGF = 88.72283905206835f;
static const float MINLOGF = -88.f;

static const float C1F = 0.693359375f;
static const float C2F = -2.12194440e-4f;

static const float PX1expf = 1.9875691500E-4f;
static const float PX2expf = 1.3981999507E-3f;
static const float PX3expf = 8.3334519073E-3f;
static const float PX4expf = 4.1665795894E-2f;
static const float PX5expf = 1.6666665459E-1f;
static const float PX6expf = 5.0000001201E-1f;

static const float LOG2EF = 1.44269504088896341f; // 1/ln(2)

static inline double egm1(double x, double px) {

    x -= px * 6.93145751953125d-1;
    x -= px * 1.42860682030941723212d-6;

    const double xx = x * x;

    px = PX1exp;
    px *= xx;
    px += PX2exp;
    px *= xx;
    px += PX3exp;
    px *= x;

    double qx = QX1exp;
    qx *= xx;
    qx += QX2exp;
    qx *= xx;
    qx += QX3exp;
    qx *= xx;
    qx += QX4exp;

    return 2.0d*px / (qx - px);
}

/// double precision exp function
static inline double vexp(double initial_x) {
    double x = initial_x;
    double px = dpfloor(LOG2E * x + 0.5);
    const int32_t n = px;

    x = 1.0 + egm1(x, px);
    x *= uint642dp((((u_int64_t) n) + 1023) << 52);

    if (initial_x > EXP_LIMIT)
        x = inf();
    if (initial_x < -EXP_LIMIT)
        x = 0.0;

    return x;
}

/// double precision expm1 function
static inline double vexpm1(double initial_x) {
    double x = initial_x;
    double px = dpfloor(LOG2E * x + 0.5);

    const int32_t n = ((px > 0.5) || (px >= -0.5)) ? px : 0;

    x = egm1(x, px);
    const double pow2nm1 = uint642dp((((u_int64_t) n-1) + 1023) << 52);
    x *= pow2nm1;
    x += pow2nm1-1;
    x *= 2;

    if (initial_x > EXP_LIMIT)
        x = inf();
    if (initial_x < -EXP_LIMIT)
        x = -1.0;

    return x;
}

/// double precision exprelr function ()
static inline double exprelr(double initial_x) {
    if (1.0+initial_x == 1.0) {
        return 1.0;
    }

    return initial_x/vexpm1(initial_x);
}

static inline float egm1(float x) {

    float z = x * PX1expf;
    z += PX2expf;
    z *= x;
    z += PX3expf;
    z *= x;
    z += PX4expf;
    z *= x;
    z += PX5expf;
    z *= x;
    z += PX6expf;
    z *= x*x;
    z += x;

    return z;
}

/// single precision exp function
static inline float vexp(float initial_x) {
    float x = initial_x;
    float z = spfloor(LOG2EF * x + 0.5f);

    x -= z * C1F;
    x -= z * C2F;
    const int32_t n = z;

    z = 1.0f + egm1(x);

    z *= uint322sp((n + 0x7f) << 23);

    if (initial_x > MAXLOGF)
        z = f_inf();
    if (initial_x < MINLOGF)
        z = 0.f;

    return z;
}

/// single precision expm1 function
static inline float vexpm1(float initial_x) {
    float x = initial_x;
    float z = spfloor(LOG2EF * x + 0.5f);

    x -= z * C1F;
    x -= z * C2F;
    const int32_t n = z;

    z = egm1(x);
    const double pow2nm1 = uint322sp((n-1 + 0x7f) << 23);
    z *= pow2nm1;
    z += pow2nm1-1;
    z *= 2;

    if (initial_x > MAXLOGF)
        z = f_inf();
    if (initial_x < MINLOGF)
        z = -1.f;

    return z;
}

/// single precision exprelr function
static inline float exprelr(float initial_x) {
    if (1.0+initial_x == 1.0) {
        return 1.0;
    }

    return initial_x/vexpm1(initial_x);
}