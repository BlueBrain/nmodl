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

/// common scalar variables
extern const double default_nthread_dt_value;
extern const double default_nthread_t_value;
extern const double default_celsius_value;
extern const int default_second_order_value;

/**
 * \class CodegenInstanceData
 * \brief Wrapper class to pack data allocate for instance
 */
struct CodegenInstanceData {
    /// base pointer which can be type casted
    /// to instance struct at run time
    void* base_ptr = nullptr;

    /// length of each member of pointer type
    size_t num_elements = 0;

    /// number of pointer members
    size_t num_ptr_members = 0;

    /// offset relative to base_ptr to locate
    /// each member variable in instance struct
    std::vector<size_t> offsets;

    /// pointer to array allocated for each member variable
    /// i.e. *(base_ptr + offsets[0]) will be members[0]
    std::vector<void*> members;

    /// size in bytes
    size_t num_bytes = 0;

    // cleanup all memory allocated for type and member variables
    ~CodegenInstanceData();
};


/**
 * Generate vector of dummy data according to the template type specified
 *
 * For double or float type: generate vector starting from `initial_value`
 *                  with an increment of 1e-5. The increment can be any other
 *                  value but 1e-5 is chosen because when we benchmark with
 *                  a million elements then the values are in the range of
 *                  <initial_value, initial_value + 10>.
 * For int type:    generate vector starting from initial_value with an
 *                  increments of 1
 *
 * \param inital_value Base value for initializing the data
 * \param num_elements Number of element of the generated vector
 * \return std::vector<T> of dummy data for testing purposes
 */
template <typename T>
std::vector<T> generate_dummy_data(size_t initial_value, size_t num_elements) {
    std::vector<T> data(num_elements);
    T increment;
    if (std::is_same<T, int>::value) {
        increment = 1;
    } else {
        increment = 1e-5;
    }
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = initial_value + increment * i;
    }
    return data;
}

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
    CodegenDataHelper(const std::shared_ptr<ast::Program>& program,
                      const std::shared_ptr<ast::InstanceStruct>& instance)
        : program(program)
        , instance(instance) {}

    CodegenInstanceData create_data(size_t num_elements, size_t seed);
};

}  // namespace codegen
}  // namespace nmodl
