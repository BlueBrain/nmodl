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
 * \brief \copybrief nmodl::visitor::CodegenTransformVisitor
 */

#include "visitors/ast_visitor.hpp"

namespace nmodl {

/**
 * @addtogroup codegen_details
 * @{
 */

/**
 * \class CodegenTransformVisitor
 * \brief Visitor to make last transformation to AST before codegen
 *
 * Modifications made:
 * - add an argument to the table if it is inside a function. In this case
 *   the argument is the name of the function.
 */

class CodegenTransformVisitor: public visitor::AstVisitor {
  public:
    /// \name Ctor & dtor
    /// \{

    /// Default constructor
    CodegenTransformVisitor() = default;

    /// \}

    void visit_function_block(ast::FunctionBlock& node) override;
};

/** \} */  // end of codegen_details

}  // namespace nmodl
