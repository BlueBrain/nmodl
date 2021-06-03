# =============================================================================
# LLVM/Clang needs to be linked with either libc++ or libstdc++
# =============================================================================

find_package(LLVM REQUIRED CONFIG)

# include LLVM libraries
set(NMODL_LLVM_COMPONENTS
    aggressiveinstcombine
    analysis
    codegen
    core
    executionengine
    instcombine
    ipo
    mc
    native
    orcjit
    target
    transformutils
    scalaropts
    support)

if(NMODL_ENABLE_JIT_EVENT_LISTENERS)
  list(APPEND NMODL_LLVM_COMPONENTS inteljitevents perfjitevents)
endif()

llvm_map_components_to_libnames(LLVM_LIBS_TO_LINK ${NMODL_LLVM_COMPONENTS})

set(CMAKE_REQUIRED_INCLUDES ${LLVM_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${LLVM_LIBS_TO_LINK})

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NMODL_ENABLE_LLVM)
  include(CheckCXXSourceCompiles)

  # simple code to test LLVM library linking
  set(CODE_TO_TEST
      "
    #include <llvm/IR/IRBuilder.h>
    using namespace llvm;
    int main(int argc, char* argv[]) {
        std::unique_ptr<IRBuilder<>> Builder;
    }")

  # first compile without any flags
  check_cxx_source_compiles("${CODE_TO_TEST}" LLVM_LIB_LINK_TEST)

  # if standard compilation fails
  if(NOT LLVM_LIB_LINK_TEST)
    # try libstdc++ first
    set(CMAKE_REQUIRED_FLAGS "-stdlib=libstdc++")
    check_cxx_source_compiles("${CODE_TO_TEST}" LLVM_LIBSTDCPP_TEST)
    # on failure, try libc++
    if(NOT LLVM_LIBSTDCPP_TEST)
      set(CMAKE_REQUIRED_FLAGS "-stdlib=libc++")
      check_cxx_source_compiles("${CODE_TO_TEST}" LLVM_LIBCPP_TEST)
    endif()
    # if either library works then add it to CXX flags
    if(LLVM_LIBSTDCPP_TEST OR LLVM_LIBCPP_TEST)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_REQUIRED_FLAGS}")
      message(
        STATUS
          "Adding ${CMAKE_REQUIRED_FLAGS} to CMAKE_CXX_FLAGS, required to link with LLVM libraries")
    else()
      message(
        STATUS
          "WARNING : -stdlib=libstdcx++ or -stdlib=libc++ didn't work to link with LLVM library")
    endif()
  endif()
endif()
