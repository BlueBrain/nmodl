/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
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

#include <cmath>
#include <Eigen/Core>

#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
#include "coreneuron/utils/offload.hpp"
#endif

namespace nmodl {
namespace crout {

/**
 * \brief Crout matrix decomposition : in-place LU Decomposition of matrix A.
 *
 * LU decomposition function.
 * Implementation details : (Legacy code) coreneuron/sim/scopmath/crout_thread.cpp
 *
 * \param n The number of rows or columns of the matrix A
 * \param A matrix of size nxn : in-place LU decomposition (C-style arrays : row-major order)
 * \param pivot matrix of size n : The i-th element is the pivot row interchanged with row i
 */
#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
nrn_pragma_acc(routine seq)
nrn_pragma_omp(declare target)
#endif
template <typename T>
EIGEN_DEVICE_FUNC inline void Crout(int n, T* A, int* pivot) {
    int i, j, k;
    T *p_k, *p_row, *p_col;
    T max;

    // For each row and column, k = 0, ..., n-1,
    for (k = 0, p_k = A; k < n; p_k += n, k++) {
        // find the pivot row
        pivot[k] = k;
        max = std::fabs(*(p_k + k));
        for (j = k + 1, p_row = p_k + n; j < n; j++, p_row += n) {
            if (max < std::fabs(*(p_row + k))) {
                max = std::fabs(*(p_row + k));
                pivot[k] = j;
                p_col = p_row;
            }
        }

        // and if the pivot row differs from the current row, then
        // interchange the two rows.
        if (pivot[k] != k)
            for (j = 0; j < n; j++) {
                max = *(p_k + j);
                *(p_k + j) = *(p_col + j);
                *(p_col + j) = max;
            }

        // and if the matrix is singular, return error
        // if ( *(p_k + k) == 0.0 ) return -1;

        // otherwise find the upper triangular matrix elements for row k.
        for (j = k + 1; j < n; j++) {
            *(p_k + j) /= *(p_k + k);
        }

        // update remaining matrix
        for (i = k + 1, p_row = p_k + n; i < n; p_row += n, i++)
            for (j = k + 1; j < n; j++)
                *(p_row + j) -= *(p_row + k) * *(p_k + j);
    }
    // return 0;
}
#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
nrn_pragma_omp(end declare target)
#endif

/**
 * \brief Crout matrix decomposition : Forward/Backward substitution.
 *
 * Forward/Backward substitution function.
 * Implementation details : (Legacy code) coreneuron/sim/scopmath/crout_thread.cpp
 *
 * \param n The number of rows or columns of the matrix LU
 * \param LU LU-factorized matrix (C-style arrays : row-major order)
 * \param B rhs vector
 * \param x solution of (LU)x=B linear system
 * \param pivot matrix of size n : The i-th element is the pivot row interchanged with row i
 */
#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
nrn_pragma_acc(routine seq)
nrn_pragma_omp(declare target)
#endif
template <typename T>
EIGEN_DEVICE_FUNC inline void solveCrout(int n, T* LU, T* B, T* x, int* pivot) {
    int i, k;
    T* p_k;
    T dum;

    // Solve the linear equation Lx = B for x, where L is a lower
    // triangular matrix.
    for (k = 0, p_k = LU; k < n; p_k += n, k++) {
        if (pivot[k] != k) {
            dum = B[k];
            B[k] = B[pivot[k]];
            B[pivot[k]] = dum;
        }
        x[k] = B[k];
        for (i = 0; i < k; i++)
            x[k] -= x[i] * *(p_k + i);
        x[k] /= *(p_k + k);
    }

    // Solve the linear equation Ux = y, where y is the solution
    // obtained above of Lx = B and U is an upper triangular matrix.
    // The diagonal part of the upper triangular part of the matrix is
    // assumed to be 1.0.
    for (k = n - 1, p_k = LU + n * (n - 1); k >= 0; k--, p_k -= n) {
        if (pivot[k] != k) {
            dum = B[k];
            B[k] = B[pivot[k]];
            B[pivot[k]] = dum;
        }
        for (i = k + 1; i < n; i++)
            x[k] -= x[i] * *(p_k + i);
        // if (*(p_k + k) == 0.0) return -1;
    }

    // return 0;
}
#if defined(CORENEURON_ENABLE_GPU) && !defined(DISABLE_OPENACC)
nrn_pragma_omp(end declare target)
#endif

}  // namespace crout
}  // namespace nmodl
