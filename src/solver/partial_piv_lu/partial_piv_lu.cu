/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "partial_piv_lu/partial_piv_lu.h"

template<int dim>
EIGEN_DEVICE_FUNC 
VecType<dim> partialPivLu(MatType<dim> A, VecType<dim> b)
{
    return A.partialPivLu().solve(b);
}

// Explicit Template Instantiation
template EIGEN_DEVICE_FUNC VecType<1> partialPivLu<1>(MatType<1> A, VecType<1> b);
template EIGEN_DEVICE_FUNC VecType<2> partialPivLu<2>(MatType<2> A, VecType<2> b);
template EIGEN_DEVICE_FUNC VecType<3> partialPivLu<3>(MatType<3> A, VecType<3> b);
template EIGEN_DEVICE_FUNC VecType<4> partialPivLu<4>(MatType<4> A, VecType<4> b);
template EIGEN_DEVICE_FUNC VecType<5> partialPivLu<5>(MatType<5> A, VecType<5> b);
template EIGEN_DEVICE_FUNC VecType<6> partialPivLu<6>(MatType<6> A, VecType<6> b);
template EIGEN_DEVICE_FUNC VecType<7> partialPivLu<7>(MatType<7> A, VecType<7> b);
template EIGEN_DEVICE_FUNC VecType<8> partialPivLu<8>(MatType<8> A, VecType<8> b);
template EIGEN_DEVICE_FUNC VecType<9> partialPivLu<9>(MatType<9> A, VecType<9> b);
template EIGEN_DEVICE_FUNC VecType<10> partialPivLu<10>(MatType<10> A, VecType<10> b);
template EIGEN_DEVICE_FUNC VecType<11> partialPivLu<11>(MatType<11> A, VecType<11> b);
template EIGEN_DEVICE_FUNC VecType<12> partialPivLu<12>(MatType<12> A, VecType<12> b);
template EIGEN_DEVICE_FUNC VecType<13> partialPivLu<13>(MatType<13> A, VecType<13> b);
template EIGEN_DEVICE_FUNC VecType<14> partialPivLu<14>(MatType<14> A, VecType<14> b);
template EIGEN_DEVICE_FUNC VecType<15> partialPivLu<15>(MatType<15> A, VecType<15> b);
template EIGEN_DEVICE_FUNC VecType<16> partialPivLu<16>(MatType<16> A, VecType<16> b);

// Currently there is an issue in Eigen (GPU-branch) for matrices 17x17 and above.
// ToDo: Check in a future release if the issue is resolved!
