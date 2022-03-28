/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/target_platform.hpp"

#include <stdexcept>

namespace nmodl {
namespace codegen {

const std::string Platform::DEFAULT_PLATFORM_NAME = "default";
const std::string Platform::DEFAULT_MATH_LIBRARY = "none";

bool Platform::is_default_platform() {
    // Default platform is a CPU.
    return platform_id == PlatformID::CPU && name == Platform::DEFAULT_PLATFORM_NAME;
}

bool Platform::is_cpu() {
    return platform_id == PlatformID::CPU;
}

bool Platform::is_cpu_with_simd() {
    return platform_id == PlatformID::CPU && instruction_width > 1;
}

bool Platform::is_gpu() {
    return platform_id == PlatformID::GPU;
}

bool Platform::is_CUDA_gpu() {
  return platform_id == PlatformID::GPU && (name == "nvptx" || name == "nvptx64");
}

bool Platform::is_single_precision() {
  return use_single_precision;
}

std::string Platform::get_name() const {
    return name;
}

std::string Platform::get_subtarget_name() const {
    if (platform_id != PlatformID::GPU)
        throw std::runtime_error("Error: platform must be a GPU to query the subtarget!\n");
    return subtarget_name;
}

std::string Platform::get_math_library() const {
    return math_library;
}

int Platform::get_instruction_width() const {
    return instruction_width;
}

int Platform::get_precision() const {
    return use_single_precision? 32 : 64;
}

}  // namespace codegen
}  // namespace nmodl
