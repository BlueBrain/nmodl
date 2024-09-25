#include <catch2/catch_test_macros.hpp>

#include "ast/program.hpp"
#include "parser/nmodl_driver.hpp"
#include "test/unit/utils/test_utils.hpp"
#include "visitors/checkparent_visitor.hpp"
#include "visitors/derivative_original_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/visitor_utils.hpp"

using namespace nmodl;
using namespace visitor;
using namespace test;
using namespace test_utils;

using nmodl::parser::NmodlDriver;


auto run_derivative_original_visitor(const std::string& text) {
    NmodlDriver driver;
    const auto& ast = driver.parse_string(text);
    SymtabVisitor().visit_program(*ast);
    DerivativeOriginalVisitor().visit_program(*ast);

    return ast;
}


TEST_CASE("Make sure DERIVATIVE block is copied properly", "[visitor][derivative_original]") {
    GIVEN("DERIVATIVE block") {
        std::string nmodl_text = R"(
            NEURON	{
                SUFFIX example
            }

            STATE {x z}

            DERIVATIVE equation {
                x' = -x + z * z
                z' = z * x
            }
)";
        auto ast = run_derivative_original_visitor(nmodl_text);
        THEN("DERIVATIVE_ORIGINAL_FUNCTION block is added") {
            auto block = collect_nodes(*ast,
                                       {ast::AstNodeType::DERIVATIVE_ORIGINAL_FUNCTION_BLOCK});
            REQUIRE(!block.empty());
            THEN("No primed variables exist in the DERIVATIVE_ORIGINAL_FUNCTION block") {
                auto primed_vars = collect_nodes(*block[0], {ast::AstNodeType::PRIME_NAME});
                REQUIRE(primed_vars.empty());
            }
        }
    }
}
