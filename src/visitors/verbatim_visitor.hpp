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
 * \brief \copybrief nmodl::visitor::VerbatimVisitor
 */

#include <vector>

#include "ast/ast.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

/**
 * @addtogroup visitor_classes
 * @{
 */

/**
 * \class VerbatimVisitor
 * \brief %Visitor for verbatim blocks of AST
 *
 * This is simple example of visitor that uses base AstVisitor
 * interface. We override visitVerbatim method and store all
 * verbatim blocks that we encounter. This could be used for
 * generating report of all verbatim blocks from all mod files
 * in ModelDB.
 */
class VerbatimVisitor: public ConstAstVisitor {
  private:
    /// flag to enable/disable printing blocks as we visit them
    bool verbose = false;

    /// vector containing all verbatim blocks
    std::vector<std::string> blocks;

  public:
    VerbatimVisitor() = default;

    explicit VerbatimVisitor(bool verbose) {
        this->verbose = verbose;
    }

    void visit_verbatim(const ast::Verbatim& node) override;

    const std::vector<std::string>& verbatim_blocks() const noexcept {
        return blocks;
    }
};

/** @} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
