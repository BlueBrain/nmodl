/*************************************************************************
 * Copyright (C) 2018-2020 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

namespace nmodl {
namespace codegen {

enum PlatformID {
    CPU,
    GPU
};

/**
 * \class Platform
 * \brief A class that represents the target platform. It is needed to
 * reduce the amount of code passed to LLVM visitor and its helpers.
 */
class Platform {
  public:
    /// Default name of the target and math library.
    static const std::string DEFAULT_PLATFORM_NAME;
    static const std::string DEFAULT_MATH_LIBRARY;

  private:
    /// Name of the platform.
    const std::string name = Platform::DEFAULT_PLATFORM_NAME;

    /// Target chip for GPUs.
    /// TODO: this should only be available to GPUs! If we refactor target
    /// classes so that GPUPlatform <: Platform, it will be nicer!
    const std::string subtarget_name = "sm_70";

    /// Target-specific id to compare platforms easily.
    PlatformID platform_id;

    /// User-provided width that is used to construct LLVM instructions
    ///  and types.
    int instruction_width = 1;

    /// Use single-precision floating-point types.
    bool use_single_precision = false;

    /// A name of user-provided math library.
    std::string math_library = Platform::DEFAULT_MATH_LIBRARY;

  public:
    Platform(PlatformID platform_id,
             const std::string& name,
             const std::string& subtarget_name,
             std::string& math_library,
             bool use_single_precision = false,
             int instruction_width = 1)
              : platform_id(platform_id)
              , name(name)
              , subtarget_name(subtarget_name)
              , math_library(math_library)
              , use_single_precision(use_single_precision)
              , instruction_width(instruction_width) {}

    Platform(PlatformID platform_id,
             const std::string& name,
             std::string& math_library,
             bool use_single_precision = false,
             int instruction_width = 1)
              : platform_id(platform_id)
              , name(name)
              , math_library(math_library)
              , use_single_precision(use_single_precision)
              , instruction_width(instruction_width) {}

    Platform(bool use_single_precision,
             int instruction_width)
            : platform_id(PlatformID::CPU)
            , use_single_precision(use_single_precision)
            , instruction_width(instruction_width) {}

    Platform() : platform_id(PlatformID::CPU) {}

    /// Checks if this platform is a default platform.
    bool is_default_platform() const;

    /// Checks if this platform is a CPU.
    bool is_cpu() const;

    /// Checks if this platform is a CPU with SIMD support.
    bool is_cpu_with_simd() const;

    /// Checks if this platform is a GPU.
    bool is_gpu() const;

    /// Checks if this platform is CUDA platform.
    bool is_CUDA_gpu() const;

    bool is_single_precision();

    std::string get_name() const;

    std::string get_subtarget_name() const;

    std::string get_math_library() const;

    int get_instruction_width() const;

    int get_precision() const;
};

}  // namespace codegen
}  // namespace nmodl
