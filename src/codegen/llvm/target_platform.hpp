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

enum TargetPlatformID {
    CPU,
    GPU
};

class CPUTarget;
class GPUTarget;

/**
 * \class Target
 * \brief An abstract class that represents a target platform. It is needed to
 * reduce the amount of code passed to LLVM visitor and its helpers.
 */
class Target {
  public:
    /// Default name of the target & math library.
    static const std::string DEFAULT_TARGET_NAME;
    static const std::string DEFAULT_MATH_LIBRARY;

  protected:
    /// Name of the target.
    const std::string name;

    /// Target-specific id to compare targets easily.
    TargetPlatformID platform_id;

    /// User-provided width that is used to construct LLVM vectors. If 1, then an
    /// assumption is made that no vector ISA is supported.
    int instruction_width = 1;

    /// A name of user-provided math (SIMD) library.
    std::string math_library = Target::DEFAULT_MATH_LIBRARY;

    /// Floating-point precision used for this target (64-bit by default).
    int precision = 64;

  private:
    /// CPU/GPU builder methods.
    static CPUTarget* build_cpu();
    static CPUTarget* build_cpu(std::string& cpu_name);
    static GPUTarget* build_gpu(std::string& gpu_name);

  protected:
    Target(TargetPlatformID platform_id,
           const std::string& name,
           std::string& math_library,
           int instruction_width)
            : platform_id(platform_id)
            , name(name)
            , math_library(math_library)
            , instruction_width(instruction_width) {}

    Target(TargetPlatformID platform_id,
           const std::string& name,
           int instruction_width)
            : platform_id(platform_id)
            , name(name)
            , instruction_width(instruction_width) {}

  public:
    virtual ~Target() = default;

    /// Checks if this target is a CPU or a GPU.
    bool is_default_platform();
    bool is_cpu();
    bool is_cpu_with_simd();
    bool is_cpu_with_vla_simd();
    bool is_gpu();

    /// Generic target builder.
    static Target* build_target(std::string& cpu_name, std::string& gpu_name);
    static Target* build_default_target();

    /// Setters.
    Target* with_math_library(std::string& math_library);
    Target* with_instruction_width(int instruction_width);
    Target* with_precision(int precision);
    virtual Target* with_VLA(bool is_VLA) = 0;

    /// Some getters.
    std::string get_name() const {
        return name;
    }

    int get_precision() const {
        return precision;
    }

    int get_instruction_width() const {
        return instruction_width;
    }
};

/**
 * \class CPUTarget
 * \brief A class that represents a CPU target platform.
 */
class CPUTarget: public Target {
  private:

    /// Flag to specify if the target has VLA ISA.
    bool is_VLA;

  public:
    CPUTarget(const std::string& name,
              std::string& math_library,
              int instruction_width = 1,
              bool is_VLA = false)
              : Target(TargetPlatformID::CPU, name, math_library, instruction_width)
              , is_VLA(is_VLA) {}

    CPUTarget(const std::string& name,
              int instruction_width = 1,
              bool is_VLA = false)
            : Target(TargetPlatformID::CPU, name, instruction_width)
            , is_VLA(is_VLA) {}

    CPUTarget(int instruction_width = 1,
              bool is_VLA = false)
            : Target(TargetPlatformID::CPU, Target::DEFAULT_TARGET_NAME, instruction_width)
            , is_VLA(is_VLA) {}

    Target* with_VLA(bool is_VLA) override;

    /// Returns true if this CPU has VLA support that allows us to target scalable vectors.
    bool has_VLA_support();
};

/**
 * \class GPUTarget
 * \brief A class that represents a GPU target platform.
 */
class GPUTarget: public Target {
  public:
    GPUTarget(std::string& name,
              std::string& math_library)
            : Target(TargetPlatformID::GPU, name, math_library, /*instruction_width=*/1) {}
    explicit GPUTarget(std::string& name)
            : Target(TargetPlatformID::GPU, name, /*instruction_width=*/1) {}

    Target* with_VLA(bool is_VLA) override;
};

}  // namespace codegen
}  // namespace nmodl