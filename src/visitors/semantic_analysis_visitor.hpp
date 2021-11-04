#pragma once

#include "ast/ast.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace visitor {

class SemanticAnalysisVisitor: public ConstAstVisitor {
  private:
    bool one_arg_in_procedure_function = false;
    bool in_procedure_function = false;

  public:
    SemanticAnalysisVisitor() = default;
    void visit_procedure_block(const ast::ProcedureBlock& node) override;
    void visit_function_block(const ast::FunctionBlock& node) override;
    void visit_table_statement(const ast::TableStatement& node) override;
};

}  // namespace visitor
}  // namespace nmodl
