#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "parser/nmodl_driver.hpp"
#include "visitors/rename_function_arguments.hpp"
#include "visitors/visitor_utils.hpp"

using namespace nmodl;
using namespace visitor;

using Catch::Matchers::ContainsSubstring;
using nmodl::parser::NmodlDriver;


TEST_CASE("Rename function arguments") {
    std::string mod_code = R"(
NEURON {
    RANGE tau
}

FUNCTION foo(a, v, cai) {
    exp(tau + a + v*cai)
}

PROCEDURE bar(tau, a, b, d) {
    foo(tau, a*b, d)
}
)";

    NmodlDriver driver;
    const auto& ast = driver.parse_string(mod_code);

    auto visitor = nmodl::visitor::RenameFunctionArgumentsVisitor();
    visitor.visit_program(*ast);

    auto transformed_mod = nmodl::to_nmodl(*ast);

    SECTION("FUNCTION") {
        std::string expected = R"(
FUNCTION foo(_arg_a, _arg_v, _arg_cai) {
    exp(tau+_arg_a+_arg_v*_arg_cai)
}
)";

        REQUIRE_THAT(transformed_mod, ContainsSubstring(expected));
    }

    SECTION("PROCEDURE") {
        std::string expected = R"(
PROCEDURE bar(_arg_tau, _arg_a, _arg_b, _arg_d) {
    foo(_arg_tau, _arg_a*_arg_b, _arg_d)
}
)";

        REQUIRE_THAT(transformed_mod, ContainsSubstring(expected));
    }
}
