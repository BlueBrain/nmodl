# =============================================================================
# Solver sources
# =============================================================================
set(SOLVER_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/newton/newton.hpp)

# =============================================================================
# Solver target for dependencies (only headers for now)
# =============================================================================

# =============================================================================
# Install solver headers and Eigen dependency headers
# =============================================================================
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/newton
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h*")

install(DIRECTORY ${NMODL_PROJECT_SOURCE_DIR}/ext/eigen/Eigen
        DESTINATION include)
