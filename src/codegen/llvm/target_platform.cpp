/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "codegen/llvm/target_platform.hpp"

namespace nmodl {
namespace codegen {

const std::string Target::DEFAULT_TARGET_NAME = "default";
const std::string Target::DEFAULT_MATH_LIBRARY = "none";

// ========================================================================== //
//                                 Builders.                                  //
// ========================================================================== //

Target* Target::build_target(std::string& cpu_name, std::string& gpu_name) {
    if (cpu_name == Target::DEFAULT_TARGET_NAME && gpu_name != Target::DEFAULT_TARGET_NAME)
        return Target::build_gpu(gpu_name);
    // By default, we use CPU as a target.
    return Target::build_cpu(cpu_name);
}

Target* Target::build_default_target() {
    return Target::build_cpu();
}

CPUTarget* Target::build_cpu() {
    return new CPUTarget(/*instruction_width=*/1);
}

CPUTarget* Target::build_cpu(std::string& cpu_name) {
    return new CPUTarget(cpu_name, /*instruction_width=*/1);
}

GPUTarget* Target::build_gpu(std::string& gpu_name) {
    return new GPUTarget(gpu_name);
}

// ========================================================================== //
//                                  Checks.                                   //
// ========================================================================== //

bool Target::is_default_platform() {
    // Default platform is a CPU.
    return platform_id == TargetPlatformID::CPU && name == Target::DEFAULT_TARGET_NAME;
}

bool Target::is_cpu() {
    return platform_id == TargetPlatformID::CPU;
}

bool Target::is_cpu_with_simd() {
    return is_cpu() && instruction_width > 1;
}

bool Target::is_cpu_with_vla_simd() {
    return is_cpu_with_simd() && dynamic_cast<CPUTarget*>(this)->has_VLA_support();
}

bool Target::is_gpu() {
    return platform_id == TargetPlatformID::GPU;
}

bool CPUTarget::has_VLA_support() {
    return is_VLA;
}

// ========================================================================== //
//                                  Setters.                                  //
// ========================================================================== //

Target* Target::with_math_library(std::string& math_library) {
    this->math_library = math_library;
    return this;
}

Target* Target::with_instruction_width(int instruction_width) {
    if (is_gpu() && instruction_width > 1)
        throw std::runtime_error("GPU target has instruction width of 1!");
    this->instruction_width = instruction_width;
    return this;
}

Target* Target::with_precision(int precision) {
    this->precision = precision;
    return this;
}

Target* CPUTarget::with_VLA(bool is_VLA) {
    this->is_VLA = is_VLA;
    return this;
}

Target* GPUTarget::with_VLA(bool is_VLA) {
    // Do nothing.
    return this;
}
}  // namespace codegen
}  // namespace nmodl