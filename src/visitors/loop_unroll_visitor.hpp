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
 * \brief \copybrief nmodl::visitor::LoopUnrollVisitor
 */

#include <string>

#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class LoopUnrollVisitor
 * \brief Unroll for loop in the AST
 *
 * Derivative and kinetic blocks have for loop with coupled set of ODEs :
 *
 * \code{.mod}
 *     DEFINE NANN 4
 *     KINETIC state {
 *        FROM i=0 TO NANN-2 {
 *    		~ ca[i] <-> ca[i+1] (DFree*frat[i+1]*1(um), DFree*frat[i+1]*1(um))
 *        }
 *     }
 * \endcode
 *
 * To solve these ODEs with SymPy, we need to get all ODEs from such loops.
 * This visitor unroll such loops and insert new expression statement in
 * the AST :
 *
 * \code{.mod}
 *     KINETIC state {
 *       {
 *           ~ ca[0] <-> ca[0+1] (DFree*frat[0+1]*1(um), DFree*frat[0+1]*1(um))
 *           ~ ca[1] <-> ca[1+1] (DFree*frat[1+1]*1(um), DFree*frat[1+1]*1(um))
 *           ~ ca[2] <-> ca[2+1] (DFree*frat[2+1]*1(um), DFree*frat[2+1]*1(um))
 *       }
 *     }
 * \endcode
 *
 * Note that the index `0+1` is not expanded to `1` because we do not run
 * constant folder pass within this loop (but could be easily done).
 */
class LoopUnrollVisitor: public AstVisitor {
  public:
    LoopUnrollVisitor() = default;

    void visit_statement_block(ast::StatementBlock& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
