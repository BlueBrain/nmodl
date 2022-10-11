/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#define CATCH_CONFIG_MAIN

#include "nmodl.hpp"

#include <catch2/catch.hpp>

#include <random>

#include "Eigen/Dense"
#include "Eigen/LU"

using namespace nmodl;
using namespace Eigen;
using namespace std;


/// https://stackoverflow.com/questions/15051367/how-to-compare-vectors-approximately-in-eigen
template <typename DerivedA, typename DerivedB>
bool allclose(const Eigen::DenseBase<DerivedA>& a,
              const Eigen::DenseBase<DerivedB>& b,
              const typename DerivedA::RealScalar& rtol =
                  Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
              const typename DerivedA::RealScalar& atol =
                  Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon()) {
    return ((a.derived() - b.derived()).array().abs() <= (atol + rtol * b.derived().array().abs()))
        .all();
}


template <typename T>
bool test_Crout_correctness(T rtol = 1e-8, T atol = 1e-8) {
    std::random_device rd;  // seeding
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> nums(-1e3, 1e3);

    for (int mat_size = 5; mat_size < 15; mat_size++) {
        Matrix<T, Dynamic, Dynamic, Eigen::ColMajor> A_ColMajor(mat_size,
                                                                mat_size);  // default in Eigen!
        Matrix<T, Dynamic, Dynamic, Eigen::RowMajor> A_RowMajor(mat_size, mat_size);
        Matrix<T, Dynamic, 1> b(mat_size);

        for (int repetitions = 0; repetitions < static_cast<int>(1e4); ++repetitions) {
            do {
                // initialization
                for (int r = 0; r < mat_size; r++) {
                    for (int c = 0; c < mat_size; c++) {
                        A_ColMajor(r, c) = nums(mt);
                        A_RowMajor(r, c) = A_ColMajor(r, c);
                        b(r) = nums(mt);
                    }
                }
            } while (!A_ColMajor.fullPivLu().isInvertible());  // Checking Invertibility

            // Eigen (ColMajor)
            Matrix<T, Dynamic, 1> eigen_x_ColMajor(mat_size);
            eigen_x_ColMajor = A_ColMajor.partialPivLu().solve(b);

            // Eigen (RowMajor)
            Matrix<T, Dynamic, 1> eigen_x_RowMajor(mat_size);
            eigen_x_RowMajor = A_RowMajor.partialPivLu().solve(b);

            if (!allclose(eigen_x_ColMajor, eigen_x_RowMajor, rtol, atol)) {
                cerr << "eigen_x_ColMajor vs eigen_x_RowMajor (issue)" << endl;
                return false;
            }

            // Crout with A_ColMajor
            Matrix<T, Dynamic, 1> crout_x_ColMajor(mat_size);
            if (!A_ColMajor.IsRowMajor)
                A_ColMajor.transposeInPlace();
            Matrix<int, Dynamic, 1> pivot(mat_size);
            crout::Crout<T>(mat_size, A_ColMajor.data(), pivot.data());
            crout::solveCrout<T>(
                mat_size, A_ColMajor.data(), b.data(), crout_x_ColMajor.data(), pivot.data());

            if (!allclose(eigen_x_ColMajor, crout_x_ColMajor, rtol, atol)) {
                cerr << "eigen_x_ColMajor vs crout_x_ColMajor (issue)" << endl;
                return false;
            }

            // Crout with A_RowMajor
            Matrix<T, Dynamic, 1> crout_x_RowMajor(mat_size);
            crout::Crout<T>(mat_size, A_RowMajor.data(), pivot.data());
            crout::solveCrout<T>(
                mat_size, A_RowMajor.data(), b.data(), crout_x_RowMajor.data(), pivot.data());

            if (!allclose(eigen_x_RowMajor, crout_x_RowMajor, rtol, atol)) {
                cerr << "eigen_x_RowMajor vs crout_x_RowMajor (issue)" << endl;
                return false;
            }
        }
    }

    return true;
}


SCENARIO("Compare Crout solver with Eigen") {
    GIVEN("crout (double)") {
        constexpr double tol = 1e-8;
        constexpr double rtol = tol;
        constexpr double atol = tol;

        auto test = test_Crout_correctness<double>(rtol, atol);

        THEN("run tests & compare") {
            REQUIRE(test);
        }
    }
}
