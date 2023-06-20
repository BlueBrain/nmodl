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

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::visitor::SteadystateVisitor
 */

#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class SteadystateVisitor
 * \brief %Visitor for STEADYSTATE solve statements
 *
 * For each `STEADYSTATE` solve statement, creates a clone of the corresponding
 * DERIVATIVE block, with `_steadystate` appended to the name of the block. If
 * the original `DERIVATIVE` block was called `X`, the new `DERIVATIVE` block will
 * be called `X_steadystate`. Only difference in new block is that the value of `dt`
 * is changed for the solve, to:
 *   - dt = 1e9 for sparse
 *   - dt = 1e-9 for derivimplicit
 *
 * where these values for `dt` in the steady state solves are taken from NEURON see
 * https://github.com/neuronsimulator/nrn/blob/master/src/scopmath/ssimplic_thread.c
 *
 * Also updates the solve statement to point to the new `DERIVATIVE` block as a
 * `METHOD solve`, not a `STEADYSTATE` one, e.g.
 *
 * \code{.mod}
 *      SOLVE X STEADYSTATE derivimplicit
 * \endcode
 *
 * becomes
 *
 * \code{.mod}
 *      SOLVE X_steadystate METHOD derivimplicit
 * \endcode
 */
class SteadystateVisitor: public AstVisitor {
  private:
    /// create new steady state derivative block for given solve block
    std::shared_ptr<ast::DerivativeBlock> create_steadystate_block(
        const std::shared_ptr<ast::SolveBlock>& solve_block,
        const std::vector<std::shared_ptr<ast::Ast>>& deriv_blocks);

    const double STEADYSTATE_SPARSE_DT = 1e9;

    const double STEADYSTATE_DERIVIMPLICIT_DT = 1e-9;

  public:
    SteadystateVisitor() = default;

    void visit_program(ast::Program& node) override;
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
