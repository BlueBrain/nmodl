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
 * \brief \copybrief nmodl::visitor::VarUsageVisitor
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
 * \class VarUsageVisitor
 * \brief Check if variable is used in given block
 *
 * \todo Check if macro is considered as variable
 */

class VarUsageVisitor: protected ConstAstVisitor {
  private:
    /// variable to check usage
    std::string var_name;
    bool used = false;

    void visit_name(const ast::Name& node) override;

  public:
    VarUsageVisitor() = default;

    bool variable_used(const ast::Node& node, std::string name);
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
