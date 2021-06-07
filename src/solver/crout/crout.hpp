/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \dir
 * \brief Solver for a system of linear equations : Crout matrix decomposition
 *
 * \file
 * \brief Implementation of Crout matrix decomposition (LU decomposition) followed by
 * Forward/Backward substitution
 */

#include <Eigen/LU>

namespace nmodl {
namespace crout {

/**
 * \brief Crout matrix decomposition : LU Decomposition of (S)ource matrix stored in (D)estination
 * matrix.
 *
 * LU decomposition function.
 * Implementation details : http://www.sci.utah.edu/~wallstedt/LU.htm (Philip Wallstedt 2007-2008)
 *
 * \param d matrices of size d x d
 * \param S (S)ource matrix (C-style arrays : row-major order)
 * \param D (D)estination matrix (LU decomposition of S-matrix) (C-style arrays : row-major order)
 */
#ifdef _OPENACC
#pragma acc routine seq
#endif
template <typename T>
EIGEN_DEVICE_FUNC inline void Crout(int d, T* S, T* D) {
    for (int k = 0; k < d; ++k) {
        for (int i = k; i < d; ++i) {
            T sum = ((T)(0.));
            for (int p = 0; p < k; ++p)
                sum += D[i * d + p] * D[p * d + k];
            D[i * d + k] = S[i * d + k] - sum;
        }
        for (int j = k + 1; j < d; ++j) {
            T sum = ((T)(0.));
            for (int p = 0; p < k; ++p)
                sum += D[k * d + p] * D[p * d + j];
            D[k * d + j] = (S[k * d + j] - sum) / D[k * d + k];
        }
    }
}

/**
 * \brief Crout matrix decomposition : Forward/Backward substitution.
 *
 * Forward/Backward substitution function.
 * Implementation details : http://www.sci.utah.edu/~wallstedt/LU.htm (Philip Wallstedt 2007-2008)
 *
 * \param d matrices of size d x d
 * \param LU LU-factorized matrix (C-style arrays : row-major order)
 * \param b rhs vector
 * \param x solution of (LU)x=b linear system
 */
#ifdef _OPENACC
#pragma acc routine seq
#endif
template <typename T>
EIGEN_DEVICE_FUNC inline void solveCrout(int d, T* LU, T* b, T* x) {
    T y[d];
    for (int i = 0; i < d; ++i) {
        T sum = ((T)(0.));
        for (int k = 0; k < i; ++k)
            sum += LU[i * d + k] * y[k];
        y[i] = (b[i] - sum) / LU[i * d + i];
    }
    for (int i = d - 1; i >= 0; --i) {
        T sum = ((T)(0.));
        for (int k = i + 1; k < d; ++k)
            sum += LU[i * d + k] * x[k];
        x[i] = (y[i] - sum);
    }
}

}  // namespace crout
}  // namespace nmodl
