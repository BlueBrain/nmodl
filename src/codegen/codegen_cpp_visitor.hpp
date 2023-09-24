/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * \dir
 * \brief Code generation backend implementations for CoreNEURON
 *
 * \file
 * \brief \copybrief nmodl::codegen::CodegenCppVisitor
 */

#include <algorithm>
#include <cmath>
#include <ctime>
#include <numeric>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "codegen/codegen_info.hpp"
#include "codegen/codegen_naming.hpp"
#include "printer/code_printer.hpp"
#include "symtab/symbol_table.hpp"
#include "utils/logger.hpp"
#include "visitors/ast_visitor.hpp"

/// encapsulates code generation backend implementations
namespace nmodl {

namespace codegen {

/**
 * \defgroup codegen Code Generation Implementation
 * \brief Implementations of code generation backends
 *
 * \defgroup codegen_details Codegen Helpers
 * \ingroup codegen
 * \brief Helper routines/types for code generation
 * \{
 */

/**
 * \enum BlockType
 * \brief Helper to represent various block types
 *
 * Note: do not assign integers to these enums
 *
 */
enum class BlockType {
    /// initial block
    Initial,

    /// constructor block
    Constructor,

    /// destructor block
    Destructor,

    /// breakpoint block
    Equation,

    /// derivative block
    State,

    /// watch block
    Watch,

    /// net_receive block
    NetReceive,

    /// before / after block
    BeforeAfter,

    /// fake ending block type for loops on the enums. Keep it at the end
    BlockTypeEnd
};


/**
 * \enum MemberType
 * \brief Helper to represent various variables types
 *
 */
enum class MemberType {
    /// index / int variables
    index,

    /// range / double variables
    range,

    /// global variables
    global,

    /// thread variables
    thread
};


/**
 * \class IndexVariableInfo
 * \brief Helper to represent information about index/int variables
 *
 */
struct IndexVariableInfo {
    /// symbol for the variable
    const std::shared_ptr<symtab::Symbol> symbol;

    /// if variable resides in vdata field of NrnThread
    /// typically true for bbcore pointer
    bool is_vdata = false;

    /// if this is pure index (e.g. style_ion) variables is directly
    /// index and shouldn't be printed with data/vdata
    bool is_index = false;

    /// if this is an integer (e.g. tqitem, point_process) variable which
    /// is printed as array accesses
    bool is_integer = false;

    /// if the variable is qualified as constant (this is property of IndexVariable)
    bool is_constant = false;

    explicit IndexVariableInfo(std::shared_ptr<symtab::Symbol> symbol,
                               bool is_vdata = false,
                               bool is_index = false,
                               bool is_integer = false)
        : symbol(std::move(symbol))
        , is_vdata(is_vdata)
        , is_index(is_index)
        , is_integer(is_integer) {}
};


/**
 * \class ShadowUseStatement
 * \brief Represents ion write statement during code generation
 *
 * Ion update statement needs use of shadow vectors for certain backends
 * as atomics operations are not supported on cpu backend.
 *
 * \todo If shadow_lhs is empty then we assume shadow statement not required
 */
struct ShadowUseStatement {
    std::string lhs;
    std::string op;
    std::string rhs;
};

/** \} */  // end of codegen_details


using printer::CodePrinter;


/**
 * \defgroup codegen_backends Codegen Backends
 * \ingroup codegen
 * \brief Code generation backends for CoreNEURON
 * \{
 */

/**
 * \class CodegenCppVisitor
 * \brief %Visitor for printing C++ code compatible with legacy api of CoreNEURON
 *
 * \todo
 *  - Handle define statement (i.e. macros)
 *  - If there is a return statement in the verbatim block
 *    of inlined function then it will be error. Need better
 *    error checking. For example, see netstim.mod where we
 *    have removed return from verbatim block.
 */
class CodegenCppVisitor: public visitor::ConstAstVisitor {
protected:
    /**
     * Code printer object for target (C++)
     */
    std::unique_ptr<CodePrinter> printer;

    /**
     * Name of mod file (without .mod suffix)
     */
    std::string mod_filename;

    /**
     * Data type of floating point variables
     */
    std::string float_type = codegen::naming::DEFAULT_FLOAT_TYPE;

    /**
     * Flag to indicate if visitor should avoid ion variable copies
     */
    bool optimize_ionvar_copies = true;

    /**
     * All ast information for code generation
     */
    codegen::CodegenInfo info;

    /**
     * Name of the simulator the code was generated for
     */
    virtual std::string simulator_name() = 0;

    /**
     * Check if function or procedure node has parameter with given name
     *
     * \tparam T Node type (either procedure or function)
     * \param node AST node (either procedure or function)
     * \param name Name of parameter
     * \return True if argument with name exist
     */
    template <typename T>
    bool has_parameter_of_name(const T& node, const std::string& name);

    /**
     * Rename function/procedure arguments that conflict with default arguments
     */
    void rename_function_arguments();

    /**
     * Arguments for "_threadargs_" macro in neuron implementation
     */
    virtual std::string nrn_thread_arguments() const = 0;

    /// This constructor is private, see the public section below to find how to create an instance
    /// of this class.
    CodegenCppVisitor(std::string mod_filename,
                      const std::string& output_dir,
                      std::string float_type,
                      const bool optimize_ionvar_copies)
        : printer(std::make_unique<CodePrinter>(output_dir + "/" + mod_filename + ".cpp"))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}

    /// This constructor is private, see the public section below to find how to create an instance
    /// of this class.
    CodegenCppVisitor(std::string mod_filename,
                      std::ostream& stream,
                      std::string float_type,
                      const bool optimize_ionvar_copies)
        : printer(std::make_unique<CodePrinter>(stream))
        , mod_filename(std::move(mod_filename))
        , float_type(std::move(float_type))
        , optimize_ionvar_copies(optimize_ionvar_copies) {}
};

/** \} */  // end of codegen_backends

}  // namespace codegen
}  // namespace nmodl