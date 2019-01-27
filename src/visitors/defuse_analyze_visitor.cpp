#include <algorithm>
#include <utility>

#include "visitors/defuse_analyze_visitor.hpp"

using namespace ast;
using namespace syminfo;

/// DUState to string conversion for pretty-printing
std::string to_string(DUState state) {
    switch (state) {
        case DUState::U:
            return "U";
        case DUState::D:
            return "D";
        case DUState::LU:
            return "LU";
        case DUState::LD:
            return "LD";
        case DUState::CONDITIONAL_BLOCK:
            return "CONDITIONAL_BLOCK";
        case DUState::IF:
            return "IF";
        case DUState::ELSEIF:
            return "ELSEIF";
        case DUState::ELSE:
            return "ELSE";
        case DUState::UNKNOWN:
            return "UNKNOWN";
        case DUState::NONE:
            return "NONE";
        default:
            throw std::runtime_error("Unhandled DUState?");
    }
}

std::ostream& operator<<(std::ostream& os, DUState state) {
    os << to_string(state);
    return os;
}

/// DUInstance to JSON string
void DUInstance::print(JSONPrinter& printer) {
    if (children.empty()) {
        printer.add_node(to_string(state));
    } else {
        printer.push_block(to_string(state));
        for (auto& inst : children) {
            inst.print(printer);
        }
        printer.pop_block();
    }
}

/// DUChain to JSON string
std::string DUChain::to_string(bool compact) {
    std::stringstream stream;
    JSONPrinter printer(stream);
    printer.compact_json(compact);

    printer.push_block(name);
    for (auto& instance : chain) {
        instance.print(printer);
    }
    printer.pop_block();

    printer.flush();
    return stream.str();
}

/** Evaluate sub-blocks like if, elseif and else
 *  As these are innermost blocks, we have to just check first use
 *  of variable in this block and that's the result of this block.
 */
DUState DUInstance::sub_block_eval() {
    DUState result = DUState::NONE;
    for (auto& chain : children) {
        auto child_state = chain.eval();
        if (child_state == DUState::U || child_state == DUState::D) {
            result = child_state;
            break;
        }
    }
    return result;
}

/** Evaluate conditional block that contain sub-blocks like if, elseif and else.
 *  Note that sub-blocks are already evaluated by sub_block_eval() and has only
 *  single value. In order to find effective usage, we are using following rules:
 *  - If variable is "used" in any of the sub-block then it's effectively
 *    "U". This is because any branch can be taken.
 *  - If variable is "defined" in all sub-blocks doesn't mean that it's
 *      effectively "D". This is because if we can just have "if-elseif"
 *      which could be never be taken. Same for empty "if". In order to
 *      decide if it is "D", we make sure there is no empty block and there
 *      must be "else" block with "D". Note that "U" definitions are already
 *      covered in 1) and hence this rule is safe.
 *  - If there is an "if" with "D" or empty "if" followed by "D" in "else"
 *      block, we can't say it's definition. In this case we return "NONE"
 *      which is safe.
 *  - If there is empty "if" followed by "U" in "else" block, we can say
 *      it's "use". This is because for optimizations we don't want to "localize"
 *      this type of variable. This needs to be changed.
 *
 *  \todo: Need to introduce new states like "conditional definition" to make that
 *         the variable is "can be" definition. And then we have to return appropriate
 *         state so that more analysis can be enabled.
 */
DUState DUInstance::conditional_block_eval() {
    DUState result = DUState::NONE;
    bool block_with_none = false;

    for (auto& chain : children) {
        auto child_state = chain.eval();
        if (child_state == DUState::U) {
            result = child_state;
            break;
        }
        if (child_state == DUState::NONE) {
            block_with_none = true;
        }
        if (chain.state == DUState::ELSE && child_state == DUState::D) {
            if (block_with_none) {
                result = DUState::NONE;
            } else {
                result = child_state;
            }
            break;
        }
    }
    return result;
}

/** Find "effective" usage of variable from def-use chain.
 *  Note that we are interested in "global" variable usage
 *  and hence we consider only [U,D] states and not [LU, LD]
 */
DUState DUInstance::eval() {
    auto result = state;
    if (state == DUState::IF || state == DUState::ELSEIF || state == DUState::ELSE) {
        result = sub_block_eval();
    } else if (state == DUState::CONDITIONAL_BLOCK) {
        result = conditional_block_eval();
    }
    return result;
}

/// first usage of a variable in a block decides whether it's definition
/// or usage. Note that if-else blocks already evaluated.
DUState DUChain::eval() {
    auto result = DUState::NONE;
    for (auto& inst : chain) {
        auto re = inst.eval();
        if (re == DUState::U || re == DUState::D) {
            result = re;
            break;
        }
    }
    return result;
}

void DefUseAnalyzeVisitor::visit_unsupported_node(Node* node) {
    unsupported_node = true;
    node->visit_children(this);
    unsupported_node = false;
}

/** Nothing to do if called function is not defined or it's external
 *  but if there is a function call for internal function that means
 *  there is no inlining happened. In this case we mark the call as
 *  unsupported.
 */
void DefUseAnalyzeVisitor::visit_function_call(FunctionCall* node) {
    std::string function_name = node->get_node_name();
    auto symbol = global_symtab->lookup_in_scope(function_name);
    if (symbol == nullptr || symbol->is_external_symbol_only()) {
        node->visit_children(this);
    } else {
        visit_unsupported_node(node);
    }
}

void DefUseAnalyzeVisitor::visit_statement_block(StatementBlock* node) {
    auto symtab = node->get_symbol_table();
    if (symtab != nullptr) {
        current_symtab = symtab;
    }

    symtab_stack.push(current_symtab);
    node->visit_children(this);
    symtab_stack.pop();
    current_symtab = symtab_stack.top();
}

/** Nmodl grammar doesn't allow assignment operator on rhs (e.g. a = b + (b=c)
 *  and hence not necessary to keep track of assignment operator using stack.
 */
void DefUseAnalyzeVisitor::visit_binary_expression(BinaryExpression* node) {
    node->get_rhs()->visit_children(this);
    if (node->get_op().get_value() == BOP_ASSIGN) {
        visiting_lhs = true;
    }
    node->get_lhs()->visit_children(this);
    visiting_lhs = false;
}

void DefUseAnalyzeVisitor::visit_if_statement(IfStatement* node) {
    /// store previous chain
    auto previous_chain = current_chain;

    /// starting new if block
    previous_chain->push_back(DUInstance(DUState::CONDITIONAL_BLOCK));
    current_chain = &(previous_chain->back().children);

    /// visiting if sub-block
    auto last_chain = current_chain;
    start_new_chain(DUState::IF);
    node->get_condition()->accept(this);
    auto block = node->get_block();
    if (block) {
        block->accept(this);
    }
    current_chain = last_chain;

    /// visiting else if sub-blocks
    for (const auto& item : node->get_elseifs()) {
        visit_with_new_chain(item.get(), DUState::ELSEIF);
    }

    /// visiting else sub-block
    if (node->get_elses()) {
        visit_with_new_chain(node->get_elses().get(), DUState::ELSE);
    }

    /// restore to previous chain
    current_chain = previous_chain;
}

/** We are not analyzing verbatim blocks yet and hence if there is
 *  a verbatim block we assume there is variable usage.
 *
 * \todo: one simple way would be to look for p_name in the string
 *        of verbatim block to find the variable usage.
 */
void DefUseAnalyzeVisitor::visit_verbatim(Verbatim* node) {
    if (!ignore_verbatim) {
        current_chain->push_back(DUInstance(DUState::U));
    }
}

/** Update def-use chain if we encounter a variable that we are looking for.
 * If we encounter non-supported construct then we mark that variable as "use"
 * because we haven't completely analyzed the usage. Marking that variable "U"
 * make sures that won't get optimized. Then we distinguish between local and
 * non-local variables. All variables that appear on lhs are maked as "definitions"
 * whereas the one on rhs are marked as "usages".
 */
void DefUseAnalyzeVisitor::update_defuse_chain(const std::string& name) {
    if (name == variable_name) {
        auto symbol = current_symtab->lookup_in_scope(name);
        // variable properties that make it local
        auto properties = NmodlType::local_var | NmodlType::argument;
        auto is_local = symbol->has_properties(properties);

        if (unsupported_node) {
            current_chain->push_back(DUInstance(DUState::U));
        } else if (visiting_lhs) {
            if (is_local) {
                current_chain->push_back(DUInstance(DUState::LD));
            } else {
                current_chain->push_back(DUInstance(DUState::D));
            }
        } else {
            if (is_local) {
                current_chain->push_back(DUInstance(DUState::LU));
            } else {
                current_chain->push_back(DUInstance(DUState::U));
            }
        }
    }
}

void DefUseAnalyzeVisitor::visit_with_new_chain(Node* node, DUState state) {
    auto last_chain = current_chain;
    start_new_chain(state);
    node->visit_children(this);
    current_chain = last_chain;
}

void DefUseAnalyzeVisitor::start_new_chain(DUState state) {
    current_chain->push_back(DUInstance(state));
    current_chain = &current_chain->back().children;
}

DUChain DefUseAnalyzeVisitor::analyze(ast::Node* node, const std::string& name) {
    /// re-initialize state
    variable_name = name;
    visiting_lhs = false;
    current_symtab = global_symtab;
    unsupported_node = false;

    /// new chain
    DUChain usage(node->get_node_type_name());
    current_chain = &usage.chain;

    /// analyze given node
    symtab_stack.push(current_symtab);
    node->visit_children(this);
    symtab_stack.pop();

    return usage;
}
