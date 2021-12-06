# Often older version of flex is available in /usr. Even we set PATH for newer flex, CMake will set
# FLEX_INCLUDE_DIRS to /usr/include. This will result in compilation errors. Hence we check for flex
# include directory for the corresponding FLEX_EXECUTABLE. If found, we add that first and then we
# include include path from CMake.

get_filename_component(FLEX_BIN_DIR ${FLEX_EXECUTABLE} DIRECTORY)

if(NOT FLEX_BIN_DIR MATCHES "/usr/bin")
  get_filename_component(FLEX_INCLUDE_PATH ${FLEX_BIN_DIR} PATH)
  set(FLEX_INCLUDE_PATH ${FLEX_INCLUDE_PATH}/include/)
  if(EXISTS "${FLEX_INCLUDE_PATH}/FlexLexer.h")
    message(STATUS " Adding Flex include path as : ${FLEX_INCLUDE_PATH}")
    include_directories(SYSTEM ${FLEX_INCLUDE_PATH})
  endif()
endif()
