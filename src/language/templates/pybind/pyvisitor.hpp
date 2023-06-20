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

///
/// THIS FILE IS GENERATED AT BUILD TIME AND SHALL NOT BE EDITED.
///

#pragma once

/**
 * \file
 * \brief Visitors extending base visitors for Python interface
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ast/ast.hpp"
#include "visitors/visitor.hpp"
#include "visitors/ast_visitor.hpp"

using namespace nmodl;
using namespace visitor;

/**
 * \brief Class mirroring nmodl::visitor::Visitor for Python bindings
 *
 * \details \copydetails nmodl::visitor::Visitor
 *
 * This class is used to interface nmodl::visitor::Visitor with the Python
 * world using `pybind11`.
 */
class PyVisitor : public Visitor {
public:
    using Visitor::Visitor;

    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}& node) override;
    {% endfor %}
};


/**
 * \brief Class mirroring nmodl::visitor::AstVisitor for Python bindings
 *
 * \details \copydetails nmodl::visitor::AstVisitor
 *
 * This class is used to interface nmodl::visitor::AstVisitor with the Python
 * world using `pybind11`.
 */
class PyAstVisitor : public AstVisitor {
public:
    using AstVisitor::AstVisitor;

    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}& node) override;
    {% endfor %}
};

/**
 * \brief Class mirroring nmodl::visitor::ConstVisitor for Python bindings
 *
 * \details \copydetails nmodl::visitor::ConstVisitor
 *
 * This class is used to interface nmodl::visitor::ConstVisitor with the Python
 * world using `pybind11`.
 */
class PyConstVisitor : public ConstVisitor {
public:
    using ConstVisitor::ConstVisitor;

    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(const ast::{{ node.class_name }}& node) override;
    {% endfor %}
};


/**
 * \brief Class mirroring nmodl::visitor::ConstAstVisitor for Python bindings
 *
 * \details \copydetails nmodl::visitor::ConstAstVisitor
 *
 * This class is used to interface nmodl::visitor::ConstAstVisitor with the Python
 * world using `pybind11`.
 */
class PyConstAstVisitor : public ConstAstVisitor {
public:
    using ConstAstVisitor::ConstAstVisitor;

    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(const ast::{{ node.class_name }}& node) override;
    {% endfor %}
};

