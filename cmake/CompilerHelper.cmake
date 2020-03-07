# minimal check for c++11 compliant gnu compiler
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
  if(NOT (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9))
    message(FATAL_ERROR "${PROJECT_NAME} requires g++ >= 4.9 (for c++11 support)")
  endif()
endif()

# PGI adds standard complaint flag "-A" which breaks compilation of of spdlog and fmt
if(CMAKE_CXX_COMPILER_ID MATCHES "PGI")
  set(CMAKE_CXX11_STANDARD_COMPILE_OPTION  --c++11)
  set(CMAKE_CXX14_STANDARD_COMPILE_OPTION  --c++14)
 endif()
