/*
 * Copyright 2024 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ast/all.hpp"
#include "visitors/ast_visitor.hpp"

#include <filesystem>

namespace nmodl {
namespace visitor {

/**
 * \class DefaultSuffixVisitor
 * \brief If no SUFFIX is specified add the default.
 *
 * The default is the stem of the MOD file.
 */
class DefaultSuffixVisitor: public AstVisitor {
    std::filesystem::path mod_path;

  public:
    explicit DefaultSuffixVisitor(std::filesystem::path mod_path)
        : mod_path(std::move(mod_path)) {}
    void visit_neuron_block(ast::NeuronBlock& node) override;
};

}  // namespace visitor
}  // namespace nmodl
