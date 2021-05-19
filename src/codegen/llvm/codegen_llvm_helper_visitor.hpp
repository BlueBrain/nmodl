/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

/**
 * \file
 * \brief \copybrief nmodl::codegen::CodegenLLVMHelperVisitor
 */

#include <string>

#include "ast/instance_struct.hpp"
#include "codegen/codegen_info.hpp"
#include "symtab/symbol_table.hpp"
#include "visitors/ast_visitor.hpp"

namespace nmodl {
namespace codegen {

using namespace fmt::literals;
typedef std::vector<std::shared_ptr<ast::CodegenFunction>> CodegenFunctionVector;

/**
 * @addtogroup llvm_codegen_details
 * @{
 */

/**
 * \class InstanceVarHelper
 * \brief Helper to query instance variables information
 *
 * For LLVM IR generation we need to know the variable, it's type and
 * location in the instance structure. This helper provides convenient
 * functions to query this information.
 */
struct InstanceVarHelper {
    /// pointer to instance node in the AST
    std::shared_ptr<ast::InstanceStruct> instance;

    /// find variable with given name and return the iterator
    ast::CodegenVarWithTypeVector::const_iterator find_variable(
        const ast::CodegenVarWithTypeVector& vars,
        const std::string& name) {
        return find_if(vars.begin(),
                       vars.end(),
                       [&](const std::shared_ptr<ast::CodegenVarWithType>& v) {
                           return v->get_node_name() == name;
                       });
    }

    /// check if given variable is instance variable
    bool is_an_instance_variable(const std::string& name) {
        const auto& vars = instance->get_codegen_vars();
        return find_variable(vars, name) != vars.end();
    }

    /// return codegen variable with a given name
    const std::shared_ptr<ast::CodegenVarWithType>& get_variable(const std::string& name) {
        const auto& vars = instance->get_codegen_vars();
        auto it = find_variable(vars, name);
        if (it == vars.end()) {
            throw std::runtime_error("Can not find variable with name {}"_format(name));
        }
        return *it;
    }

    /// return position of the variable in the instance structure
    int get_variable_index(const std::string& name) {
        const auto& vars = instance->get_codegen_vars();
        auto it = find_variable(vars, name);
        if (it == vars.end()) {
            throw std::runtime_error("Can not find codegen variable with name {}"_format(name));
        }
        return (it - vars.begin());
    }
};


/**
 * \class CodegenLLVMHelperVisitor
 * \brief Helper visitor for AST information to help code generation backends
 *
 * Code generation backends convert NMODL AST to C++ code. But during this
 * C++ code generation, various transformations happens and final code generated
 * is quite different / large than actual kernel represented in MOD file ro
 * NMODL AST.
 *
 * Currently, these transformations are embedded into code generation backends
 * like ast::CodegenCVisitor. If we have to generate code for new simulator, there
 * will be duplication of these transformations. Also, for completely new
 * backends like NEURON simulator or SIMD library, we will have code duplication.
 *
 * In order to avoid this, we perform maximum transformations in this visitor.
 * Currently we focus on transformations that will help LLVM backend but later
 * these will be common across all backends.
 */
class CodegenLLVMHelperVisitor: public visitor::AstVisitor {
    // explicit vectorisation width
    int vector_width;

    /// newly generated code generation specific functions
    CodegenFunctionVector codegen_functions;

    /// ast information for code generation
    codegen::CodegenInfo info;

    /// mechanism data helper
    InstanceVarHelper instance_var_helper;

    /// create new function for FUNCTION or PROCEDURE block
    void create_function_for_node(ast::Block& node);

    /// create new InstanceStruct
    std::shared_ptr<ast::InstanceStruct> create_instance_struct();

  public:
    /// default integer and float node type
    static const ast::AstNodeType INTEGER_TYPE;
    static const ast::AstNodeType FLOAT_TYPE;

    // node count, voltage and node index variables
    static const std::string NODECOUNT_VAR;
    static const std::string VOLTAGE_VAR;
    static const std::string NODE_INDEX_VAR;

    CodegenLLVMHelperVisitor(int vector_width)
        : vector_width(vector_width){};

    const InstanceVarHelper& get_instance_var_helper() {
        return instance_var_helper;
    }

    std::string get_kernel_id() {
        return naming::INDUCTION_VAR;
    }

    /// run visitor and return code generation functions
    CodegenFunctionVector get_codegen_functions(const ast::Program& node);

    void ion_read_statements(BlockType type,
                             std::vector<std::string>& int_variables,
                             std::vector<std::string>& double_variables,
                             ast::StatementVector& index_statements,
                             ast::StatementVector& body_statements);

    void ion_write_statements(BlockType type,
                              std::vector<std::string>& int_variables,
                              std::vector<std::string>& double_variables,
                              ast::StatementVector& index_statements,
                              ast::StatementVector& body_statements);

    void convert_to_instance_variable(ast::Node& node, std::string& index_var);

    void convert_local_statement(ast::StatementBlock& node);
    void rename_local_variables(ast::StatementBlock& node);

    void visit_procedure_block(ast::ProcedureBlock& node) override;
    void visit_function_block(ast::FunctionBlock& node) override;
    void visit_nrn_state_block(ast::NrnStateBlock& node) override;
    void visit_program(ast::Program& node) override;
};

/** @} */  // end of llvm_codegen_details

}  // namespace codegen
}  // namespace nmodl
