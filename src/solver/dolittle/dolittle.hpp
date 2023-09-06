#include <algorithm>

namespace nmodl::dolittle {

/** @brief Compute the LU decomposition of `a` inplace.
 *
 *  Arguments:
 *    @param n The number of columns/row of `a`.
 *    @param a A square matrix in row major format.
 *    @param p The permutation of the rows.
 */
template <class FloatType, class IndexType>
int factor(IndexType n, FloatType* const a, IndexType* const __restrict__ p) {
    FloatType atol = 1e-12;

    for (IndexType i = 0; i < n; i++) {
        p[i] = i;
    }

    for (IndexType i = 0; i < n; i++) {
        FloatType max_acol = 0.0;
        IndexType imax = i;

        for (IndexType k = i; k < n; k++) {
            FloatType abs_akj = std::abs(a[k * n + i]);
            if (abs_akj > max_acol) {
                max_acol = abs_akj;
                imax = k;
            }
        }

        if (max_acol < atol) {
            return -1;
        }

        if (imax != i) {
            std::swap(p[i], p[imax]);

            FloatType* const __restrict__ ai = a + i * n;
            FloatType* const __restrict__ aimax = a + imax * n;
            for (IndexType k = 0; k < n; ++k) {
                std::swap(ai[k], aimax[k]);
            }
        }

        for (IndexType j = i + 1; j < n; j++) {
            if (std::abs(a[j * n + i]) > atol) {
                FloatType* const __restrict__ aj = a + j * n;
                FloatType const* const __restrict__ ai = a + i * n;

                aj[i] /= ai[i];
                FloatType aji = aj[i];

                for (IndexType k = i + 1; k < n; k++) {
                    aj[k] -= aji * ai[k];
                }
            }
        }
    }

    return 0;
}

/** @brief Solve the system a * x = b.
 *
 *  Given the LU decomposition computed by `factor`, solve the linear system
 *  a * x = b.
 *
 *  Arguments:
 *    @param n The number of columns/row of `a`.
 *    @param a The LU decomposition of `a` (row-major).
 *    @param b The right-hand side of the linear system.
 *    @param x The output vector with the solution of the linear system.
 *    @param p The permutation of the rows, aka pivot matrix.
 */
template <class FloatType, class IndexType>
void solve(IndexType n,
           FloatType const* const lu,
           FloatType const* const __restrict__ b,
           FloatType* const __restrict__ x,
           IndexType const* const __restrict__ p) {
    for (IndexType i = 0; i < n; i++) {
        FloatType xi = b[p[i]];
        FloatType const* const __restrict__ lu_i = lu + i * n;

        for (IndexType k = 0; k < i; k++) {
            xi -= lu_i[k] * x[k];
        }
        x[i] = xi;
    }

    for (IndexType i = n - 1; i >= 0; i--) {
        FloatType xi = x[i];
        FloatType const* const __restrict__ lu_i = lu + i * n;
        for (IndexType k = i + 1; k < n; k++) {
            xi -= lu_i[k] * x[k];
        }

        x[i] = xi / lu_i[i];
    }
}
}  // namespace nmodl::dolittle
