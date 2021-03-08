/*************************************************************************
 * Copyright (C) 2018-2021 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "ast/ast.hpp"

/// \file
/// \brief Generate test data for testing and benchmarking compute kernels

namespace nmodl {
namespace codegen {

/**
 * \class CodegenInstanceData
 * \brief Wrapper class to pack data allocate for instance
 */
struct CodegenInstanceData {
    /// base pointer which can be type casted
    /// to instance struct at run time
    void* base_ptr;

    /// length of each member variable
    size_t num_elements;

    /// offset relative to base_ptr to locate
    /// each member variable in instance struct
    std::vector<size_t> offsets;

    /// pointer to array allocated for each member variable
    /// i.e. *(base_ptr + offsets[0]) will be members[0]
    std::vector<void*> members;
};


/**
 * \class CodegenDataHelper
 * \brief Helper to allocate and initialize data for benchmarking
 *
 * The `ast::InstanceStruct` is has different number of member
 * variables for different MOD files and hence we can't instantiate
 * it at compile time. This class helps to inspect the variables
 * information gathered from AST and allocate memory block that
 * can be type cast to the `ast::InstanceStruct` corresponding
 * to the MOD file.
 */
class CodegenDataHelper {
    std::shared_ptr<ast::Program> program;
    std::shared_ptr<ast::InstanceStruct> instance;

  public:
    CodegenDataHelper() = delete;
    CodegenDataHelper(std::shared_ptr<ast::Program>& program,
                      std::shared_ptr<ast::InstanceStruct>& instance)
        : program(program)
        , instance(instance) {}

    CodegenInstanceData create_data(size_t num_elements, size_t seed);
};

}  // namespace codegen
}  // namespace nmodl