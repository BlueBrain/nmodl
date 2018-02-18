#ifndef NMODL_LOCALIZE_VISITOR_HPP
#define NMODL_LOCALIZE_VISITOR_HPP

#include <map>
#include <stack>

#include "ast/ast.hpp"
#include "printer/json_printer.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/rename_visitor.hpp"
#include "visitors/visitor_utils.hpp"
#include "visitors/local_var_rename_visitor.hpp"
#include "symtab/symbol_table.hpp"

/**
 * \class LocalizeVisitor
 * \brief Visitor to transform global variable usage to local
 *
 * Motivation: As NMODL doesn't support returning multiple values,
 * procedures are often written with use of range variables that
 * can be made local. For example:
 *
 *      NEURON {
 *          RANGE tau, alpha, beta
 *      }
 *
 *      DERIVATIVE states() {
 *          ...
 *          rates()
 *          alpha = tau + beta
 *      }
 *
 *      PROCEDURE rates() {
 *          tau = xx * 0.12 * some_var
 *          beta = yy * 0.11
 *      }
 *
 * In above example we are only interested in variable alpha computed in
 * DERIVATIVE block. If rates() is inlined into DERIVATIVE block then we
 * get:
 *
 *       DERIVATIVE states() {
 *          ...
 *          {
 *              tau = xx * 0.12 * some_var
 *              beta = yy * 0.11
 *          }
 *          alpha = tau + beta
 *      }
 *
 * Now tau and beta could become local variables provided that their values
 * are not used in any other global blocks.
 *
 * Implementation Notes:
 *   - For every global variable in the mod file we have to compute
 *     def-use chains in global blocks (except procedure and functions, which should be
 *     already inlined).
 *   - If every block has "definition" first then that variable is safe to "localize"
 *
 * \todo:
 *   - We are excluding procedures/functions because they will be still using global
 *     variables. We need to have dead-code removal pass to eliminate unused procedures/
 *     functions before localizer pass.
 *   - For conditional block like below we are returning usage as NONE. May be better to
 *     return COND_D so that localizer can declare tau as LOCAL (this is artificial use
 *     case though) :
 *          BREAKPOINT {
 *              IF (1) {
 *                  tau = 11
 *              }
 *          }
 */

class LocalizeVisitor : public AstVisitor {
  private:
    /// ignore verbatim blocks while localizing
    bool ignore_verbatim = false;

    symtab::SymbolTable* program_symtab = nullptr;

    std::vector<std::string> variables_to_optimize();

    bool node_for_def_use_analysis(ast::Node *node);

    bool is_solve_procedure(ast::Node* node);

  public:
    LocalizeVisitor() = default;

    explicit LocalizeVisitor(bool ignore_verbatim) : ignore_verbatim(ignore_verbatim) {
    }

    virtual void visit_program(ast::Program* node) override;
};

#endif
