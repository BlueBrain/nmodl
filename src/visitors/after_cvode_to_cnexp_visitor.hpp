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
 * \brief \copybrief nmodl::visitor::AfterCVodeToCnexpVisitor
 */

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class AfterCVodeToCnexpVisitor
 * \brief Visitor to change usage of after_cvode solver to cnexp
 *
 *  `CVODE` is not supported in CoreNEURON. If MOD file has `after_cvode` solver then
 *   we can treat that as `cnexp`. In order to re-use existing passes, in this visitor we
 *  replace `after_cvode` with `cnexp`.
 */

class AfterCVodeToCnexpVisitor: public AstVisitor {
  public:
    /// \name Ctor & dtor
    /// \{

    /// Default constructor
    AfterCVodeToCnexpVisitor() = default;

    /// \}
    void visit_solve_block(ast::SolveBlock& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
