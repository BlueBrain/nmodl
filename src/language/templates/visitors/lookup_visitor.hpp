/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "ast/ast.hpp"
#include "visitors/visitor.hpp"


namespace nmodl {

/**
 * \class AstLookupVisitor
 * \brief Visitor to find ast nodes based on on the ast types
 */

class AstLookupVisitor : public Visitor {
    private:
        /// node types to search in the ast
        std::vector<ast::AstNodeType> types;

        /// matching nodes found in the ast
        std::vector<std::shared_ptr<ast::Ast>> nodes;

    public:

        AstLookupVisitor() = default;

        AstLookupVisitor(ast::AstNodeType type) : types{type} {}

        AstLookupVisitor(const std::vector<ast::AstNodeType>& types) : types(types) {}

        std::vector<std::shared_ptr<ast::Ast>> lookup(ast::Ast *node);

        std::vector<std::shared_ptr<ast::Ast>> lookup(ast::Ast* node, ast::AstNodeType type);

        std::vector<std::shared_ptr<ast::Ast>> lookup(ast::Ast* node, std::vector<ast::AstNodeType>& types);

        const std::vector<std::shared_ptr<ast::Ast>>& get_nodes() const noexcept {
            return nodes;
        }

        void clear() {
            types.clear();
            nodes.clear();
        }

        {% for node in nodes %}
            void visit_{{ node.class_name|snake_case }}(ast::{{ node.class_name }}* node) override;
        {% endfor %}
};

}  // namespace nmodl