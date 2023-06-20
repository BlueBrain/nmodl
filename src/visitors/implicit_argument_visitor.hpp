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
 * \brief \copybrief nmodl::visitor::ImplicitArgumentVisitor
 */

#include "visitors/ast_visitor.hpp"


namespace nmodl {
namespace visitor {

/**
 * \addtogroup visitor_classes
 * \{
 */

/**
 * \class ImplicitArgumentVisitor
 * \brief %Visitor for adding implicit arguments to [Core]NEURON functions.
 *
 * This visitor is used to add implicit arguments to functions (nrn_ghk, ...)
 * that have historially been used in MOD files with "fewer" arguments and
 * relied on global state, but which now need "more" arguments to be passed so
 * that they can be pure functions.
 */
struct ImplicitArgumentVisitor: public AstVisitor {
    void visit_function_call(ast::FunctionCall& node) override;
};

/** \} */  // end of visitor_classes

}  // namespace visitor
}  // namespace nmodl
