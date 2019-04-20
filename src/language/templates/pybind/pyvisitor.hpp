/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "ast/ast.hpp"
#include "visitors/visitor.hpp"
#include "visitors/ast_visitor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace nmodl;

/**
 * \brief Class mirroring nmodl::Visitor for Python bindings
 *
 * \details \copydetails nmodl::Visitor
 *
 * This class is used to interface nmodl::Visitor with the Python
 * world using `pybind11`.
 */
class PyVisitor : public Visitor {
public:
    using Visitor::Visitor;

    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}* node) override;
    {% endfor %}
};


/**
 * \brief Class mirroring nmodl::AstVisitor for Python bindings
 *
 * \details \copydetails nmodl::AstVisitor
 *
 * This class is used to interface nmodl::AstVisitor with the Python
 * world using `pybind11`.
 */
class PyAstVisitor : public AstVisitor {
public:
    using AstVisitor::AstVisitor;

    {% for node in nodes %}
    void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}* node) override;
    {% endfor %}
};
