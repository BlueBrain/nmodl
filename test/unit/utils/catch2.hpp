#if NMODL_CATCH2_VERSION_MAJOR == 3
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Equals;
using Catch::Matchers::StartsWith;

#elif NMODL_CATCH2_VERSION_MAJOR == 2

#include <catch2/catch.hpp>

using Catch::WithinRel;
using Catch::StartsWith;

inline auto ContainsSubstring(const std::string &x) { return Catch::Contains(x); }

#else
#error "Invalid 'NMODL_CATCH2_VERSION_MAJOR'."
#endif

