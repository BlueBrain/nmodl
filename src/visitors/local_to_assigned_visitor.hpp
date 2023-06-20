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
 * \brief \copybrief nmodl::visitor::LocalToAssignedVisitor
 */

#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class LocalToAssignedVisitor
 * \brief Visitor to convert top level LOCAL variables to ASSIGNED variables
 *
 * Some of the existing mod file include declaration of LOCAL variables in
 * the top level of the mod file. Those variables are normally written in
 * the INITIAL block which is executed potentially by multiple threads.
 * This results into race condition in case of CoreNEURON. To avoid this,
 * such variables are converted to ASSIGNED variables which by default are
 * handled as RANGE.
 *
 * For example:
 *
 * \code{.mod}
 *      NEURON {
 *          SUFFIX test
 *  	    RANGE x
 *      }
 *
 *      LOCAL qt
 *
 *      INITIAL {
 *          qt = 10.0
 *      }
 *
 *      PROCEDURE rates(v (mV)) {
 *          x = qt + 12.2
 *      }
 * \endcode
 *
 * In the above example, `qt` is used as temporary variable to pass value from
 * INITIAL block to PROCEDURE. This works fine in case of serial execution but
 * in parallel execution we end up in race condition. To avoid this, we convert
 * qt to ASSIGNED variable.
 *
 * \todo
 *   - Variables like qt are often constant. As long as INITIAL block is executed
 *     serially or qt is updated in atomic way then we don't have a problem.
 */

class LocalToAssignedVisitor: public AstVisitor {
  public:
    /// \name Ctor & dtor
    /// \{

    /// Default constructor
    LocalToAssignedVisitor() = default;

    /// \}

    /// Visit ast::Program node to transform top level LOCAL variables
    /// to ASSIGNED if they are written in the mod file
    void visit_program(ast::Program& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
