/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#define CATCH_CONFIG_MAIN

#include <chrono>
#include <random>

#include <catch/catch.hpp>

#include "Eigen/Dense"
#include "Eigen/LU"

#include "nmodl.hpp"

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
    using MatType = Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>;
    using VecType = Matrix<T, Dynamic, 1>;

    std::random_device rd;  // seeding
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> nums(-1, 1);

    std::chrono::duration<double> eigen_solve_RowMajor(std::chrono::duration<double>::zero());
    std::chrono::duration<double> eigen_solve_ColMajor(std::chrono::duration<double>::zero());
    std::chrono::duration<double> crout_solve_host(std::chrono::duration<double>::zero());

    for (int mat_size = 2; mat_size < 10; mat_size++) {
        MatType A_RowMajor(mat_size, mat_size);
        Matrix<T, Dynamic, Dynamic, Eigen::ColMajor> A_ColMajor(mat_size,
                                                                mat_size);  // default in Eigen!
        VecType b(mat_size);

        for (int repetitions = 0; repetitions < 10000; ++repetitions) {
            do {
                // initialization
                for (int r = 0; r < mat_size; r++) {
                    for (int c = 0; c < mat_size; c++) {
                        A_RowMajor(r, c) = nums(mt);
                        A_ColMajor(r, c) = A_RowMajor(r, c);
                        b(r) = nums(mt);
                    }
                }
            } while (!A_RowMajor.fullPivLu().isInvertible());  // Checking Invertibility

            // Eigen (RowMajor)
            VecType eigen_solution_RowMajor(mat_size);
            auto t1 = std::chrono::high_resolution_clock::now();
            eigen_solution_RowMajor = A_RowMajor.partialPivLu().solve(b);
            auto t2 = std::chrono::high_resolution_clock::now();
            eigen_solve_RowMajor += (t2 - t1);

            // Eigen (ColMajor)
            VecType eigen_solution_ColMajor(mat_size);
            t1 = std::chrono::high_resolution_clock::now();
            eigen_solution_ColMajor = A_ColMajor.partialPivLu().solve(b);
            t2 = std::chrono::high_resolution_clock::now();
            eigen_solve_ColMajor += (t2 - t1);

            if (!allclose(eigen_solution_RowMajor, eigen_solution_ColMajor, rtol, atol)) {
                cerr << "Eigen issue with RowMajor vs ColMajor storage order!" << endl << endl;
                return false;
            }

            // Crout LU-Decomposition CPU (in-place)
            VecType crout_solution_host(mat_size);
            Matrix<int, Dynamic, 1> pivot(mat_size);
            t1 = std::chrono::high_resolution_clock::now();
            crout::Crout<T>(mat_size, A_RowMajor.data(), pivot.data());
            crout::solveCrout<T>(
                mat_size, A_RowMajor.data(), b.data(), crout_solution_host.data(), pivot.data());
            t2 = std::chrono::high_resolution_clock::now();
            crout_solve_host += (t2 - t1);

            if (!allclose(eigen_solution_RowMajor, crout_solution_host, rtol, atol))
                return false;
        }
    }

    return true;
}


SCENARIO("Compare Crout solver with Eigen") {
    GIVEN("crout (double)") {
        constexpr double rtol = 1e-8;
        constexpr double atol = 1e-8;

        auto test = test_Crout_correctness<double>(rtol, atol);

        THEN("run tests & compare") {
            REQUIRE(test);
        }
    }
}
