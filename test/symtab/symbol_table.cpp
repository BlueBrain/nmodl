#define CATCH_CONFIG_MAIN

#include <string>

#include "catch/catch.hpp"
#include "ast/ast.hpp"
#include "symtab/symbol.hpp"
#include "symtab/symbol_table.hpp"

using namespace symtab;
using namespace symtab::details;

extern bool has_property(const SymbolInfo& obj, NmodlInfo property);

//=============================================================================
// Symbol properties test
//=============================================================================

SCENARIO("Symbol properties can be added and converted to string") {
    SymbolInfo prop1{flags::empty};
    SymbolInfo prop2 = NmodlInfo::local_var;
    SymbolInfo prop3 = NmodlInfo::global_var;

    GIVEN("A empty property") {
        WHEN("converted to string") {
            THEN("returns empty string") {
                REQUIRE(to_string(prop1).empty());
            }
        }
        WHEN("checked for property") {
            THEN("doesn't have any property") {
                REQUIRE_FALSE(has_property(prop1, NmodlInfo::local_var));
            }
        }
        WHEN("adding another empty property") {
            SymbolInfo result = prop1 | prop1;
            THEN("to_string still returns empty string") {
                REQUIRE(to_string(result).empty());
            }
        }
        WHEN("added some other property") {
            SymbolInfo result = prop1 | prop2;
            THEN("to_string returns added property") {
                REQUIRE(to_string(result) == "local");
            }
            WHEN("checked for property") {
                THEN("has required property") {
                    REQUIRE(has_property(result, NmodlInfo::local_var) == true);
                }
            }
        }
        WHEN("added multiple properties") {
            SymbolInfo result = prop1 | prop2 | prop3;
            result |= NmodlInfo::write_ion_var;
            THEN("to_string returns all added properties") {
                REQUIRE_THAT(to_string(result), Catch::Contains("local"));
                REQUIRE_THAT(to_string(result), Catch::Contains("global"));
                REQUIRE_THAT(to_string(result), Catch::Contains("write_ion"));
            }
            WHEN("checked for property") {
                THEN("has all added properties") {
                    REQUIRE(has_property(result, NmodlInfo::local_var) == true);
                    REQUIRE(has_property(result, NmodlInfo::global_var) == true);
                    REQUIRE(has_property(result, NmodlInfo::write_ion_var) == true);
                    REQUIRE_FALSE(has_property(result, NmodlInfo::read_ion_var));
                }
            }
        }
    }
}

//=============================================================================
// Symbol test
//=============================================================================

SCENARIO("Symbol operations") {
    SymbolInfo property1 = NmodlInfo::argument;
    SymbolInfo property2 = NmodlInfo::range_var;
    SymbolInfo property3 = NmodlInfo::param_assign;
    GIVEN("A symbol") {
        ModToken token(true);
        Symbol symbol("alpha", token);
        WHEN("added external property") {
            symbol.add_property(NmodlInfo::extern_neuron_variable);
            THEN("symbol becomes external") {
                REQUIRE(symbol.is_external_symbol_only() == true);
            }
        }
        WHEN("added multiple properties to symbol") {
            symbol.add_property(property1);
            symbol.add_property(property2);
            THEN("symbol has multiple properties") {
                REQUIRE(symbol.has_properties(property1) == true);

                REQUIRE(symbol.has_properties(property3) == false);

                symbol.add_property(property3);
                REQUIRE(symbol.has_properties(property3) == true);

                auto property = property1 | property2;
                REQUIRE(symbol.has_all_properties(property) == true);

                property |= property3;
                REQUIRE(symbol.has_all_properties(property) == true);

                property = property2 | property3;
                REQUIRE(symbol.has_all_properties(property) == true);

                property |= NmodlInfo::to_solve;
                REQUIRE(symbol.has_all_properties(property) == false);
            }
        }
        WHEN("combined properties") {
            SymbolInfo property = NmodlInfo::factor_def | NmodlInfo::global_var;
            THEN("symbol has union of all properties") {
                REQUIRE(symbol.has_properties(property) == false);
                symbol.combine_properties(property);
                REQUIRE(symbol.has_properties(property) == true);
                property |= symbol.get_properties();
                REQUIRE(symbol.get_properties() == property);
            }
        }
    }
}

//=============================================================================
// Symbol table test
//=============================================================================

SCENARIO("Symbol table operations") {
    GIVEN("A global SymbolTable") {
        auto program = std::make_shared<ast::Program>();
        auto table = std::make_shared<SymbolTable>("Na", program.get(), true);
        auto symbol = std::make_shared<Symbol>("alpha", ModToken());

        WHEN("checked methods and member variables") {
            THEN("all members are initialized") {
                REQUIRE(table->under_global_scope());
                REQUIRE_THAT(table->name(), Catch::Contains("Na"));
                REQUIRE_THAT(table->type(), Catch::Contains("Program"));
                REQUIRE_THAT(table->get_parent_table_name(), Catch::Contains("None"));
                REQUIRE_THAT(table->position(), Catch::Contains("UNKNOWN"));
            }
        }
        WHEN("insert symbol") {
            table->insert(symbol);
            THEN("table size increases") {
                REQUIRE(table->symbol_count() == 1);
            }
            THEN("lookup returns a inserted symbol") {
                REQUIRE(table->lookup("alpha") != nullptr);
                REQUIRE(table->lookup("beta") == nullptr);
            }
            WHEN("re-inserting the same symbol") {
                THEN("throws an exception") {
                    REQUIRE_THROWS_WITH(table->insert(symbol), Catch::Contains("re-insert"));
                }
            }
            WHEN("inserting another symbol") {
                auto next_symbol = std::make_shared<Symbol>("beta", ModToken());
                table->insert(next_symbol);
                THEN("symbol gets added and table size increases") {
                    REQUIRE(table->symbol_count() == 2);
                    REQUIRE(table->lookup("beta") != nullptr);
                }
            }
        }
        WHEN("checked for global variables") {
            table->insert(symbol);
            auto variables = table->get_variables_with_properties(NmodlInfo::range_var);
            THEN("table doesn't have any global variables") {
                REQUIRE(variables.empty());
                WHEN("added global symbol") {
                    auto next_symbol = std::make_shared<Symbol>("gamma", ModToken());
                    next_symbol->add_property(NmodlInfo::dependent_def);
                    table->insert(next_symbol);
                    auto variables = table->get_variables_with_properties(NmodlInfo::dependent_def);
                    THEN("table has global variable") {
                        REQUIRE(variables.size() == 1);
                    }
                }
            }
        }
        WHEN("added another symbol table as children") {
            table->insert(symbol);
            auto next_program = std::make_shared<ast::Program>();
            auto next_table = std::make_shared<SymbolTable>("Ca", next_program.get(), true);
            next_table->set_parent_table(table.get());
            THEN("children symbol table can lookup into parent table scope") {
                REQUIRE(next_table->lookup("alpha") == nullptr);
                REQUIRE(next_table->lookup_in_scope("alpha") != nullptr);
            }
        }
        WHEN("query for symbol with and without properties") {
            auto symbol1 = std::make_shared<Symbol>("alpha", ModToken());
            auto symbol2 = std::make_shared<Symbol>("beta", ModToken());
            auto symbol3 = std::make_shared<Symbol>("gamma", ModToken());
            auto symbol4 = std::make_shared<Symbol>("delta", ModToken());

            symbol1->add_property(NmodlInfo::range_var | NmodlInfo::param_assign);
            symbol2->add_property(NmodlInfo::range_var | NmodlInfo::param_assign |
                                  NmodlInfo::state_var);
            symbol3->add_property(NmodlInfo::range_var | NmodlInfo::dependent_def |
                                  NmodlInfo::pointer_var);
            symbol4->add_property(NmodlInfo::range_var);

            table->insert(symbol1);
            table->insert(symbol2);
            table->insert(symbol3);
            table->insert(symbol4);

            auto result = table->get_variables_with_properties(NmodlInfo::range_var);
            REQUIRE(result.size() == 4);

            result =
                table->get_variables_with_properties(NmodlInfo::range_var | NmodlInfo::pointer_var);
            REQUIRE(result.size() == 4);

            auto with = NmodlInfo::range_var | NmodlInfo::param_assign;
            auto without = NmodlInfo::state_var | NmodlInfo::pointer_var;
            result = table->get_variables(with, without);
            REQUIRE(result.size() == 1);
            REQUIRE(result[0]->get_name() == "alpha");


            with = NmodlInfo::range_var;
            without = NmodlInfo::param_assign | NmodlInfo::dependent_def;
            result = table->get_variables(with, without);
            REQUIRE(result.size() == 1);
            REQUIRE(result[0]->get_name() == "delta");

            with = NmodlInfo::range_var;
            without = NmodlInfo::range_var;
            result = table->get_variables(with, without);
            REQUIRE(result.empty());
        }
    }
}

//=============================================================================
// Model symbol table test
//=============================================================================

SCENARIO("Model symbol table operations") {
    GIVEN("A Model symbolTable") {
        ModelSymbolTable mod_symtab;

        auto program = std::make_shared<ast::Program>();
        auto symbol1 = std::make_shared<Symbol>("alpha", ModToken());
        auto symbol2 = std::make_shared<Symbol>("alpha", ModToken());
        auto symbol3 = std::make_shared<Symbol>("alpha", ModToken());

        symbol1->add_property(NmodlInfo::param_assign);
        symbol2->add_property(NmodlInfo::range_var);
        symbol3->add_property(NmodlInfo::range_var);

        SymbolTable* old_symtab = nullptr;

        WHEN("trying to exit scope without entering") {
            THEN("throws an exception") {
                REQUIRE_THROWS_WITH(mod_symtab.leave_scope(), Catch::Contains("without entering"));
            }
        }
        WHEN("trying to enter scope without valid node") {
            THEN("throws an exception") {
                REQUIRE_THROWS_WITH(mod_symtab.enter_scope("scope", nullptr, true, old_symtab),
                                    Catch::Contains("empty node"));
            }
        }
        WHEN("trying to insert without entering scope") {
            THEN("throws an exception") {
                auto symbol = std::make_shared<Symbol>("alpha", ModToken());
                REQUIRE_THROWS_WITH(mod_symtab.insert(symbol), Catch::Contains("Can not insert"));
            }
        }
        WHEN("enter scope multipel times") {
            auto program1 = std::make_shared<ast::Program>();
            auto program2 = std::make_shared<ast::Program>();
            mod_symtab.enter_scope("scope1", program1.get(), false, old_symtab);
            mod_symtab.enter_scope("scope2", program2.get(), false, old_symtab);
            THEN("can leave scope multiple times") {
                mod_symtab.leave_scope();
                mod_symtab.leave_scope();
            }
        }
        WHEN("added same symbol with different properties in global scope") {
            mod_symtab.enter_scope("scope", program.get(), true, old_symtab);
            mod_symtab.insert(symbol1);
            mod_symtab.insert(symbol2);
            THEN("only one symbol gets added with combined properties") {
                auto symbol = mod_symtab.lookup("alpha");
                auto properties = NmodlInfo::param_assign | NmodlInfo::range_var;
                REQUIRE(symbol->get_properties() == properties);
            }
        }
        WHEN("added same symbol with exisiting property") {
            mod_symtab.enter_scope("scope", program.get(), true, old_symtab);
            mod_symtab.insert(symbol1);
            mod_symtab.insert(symbol2);
            THEN("throws an exception") {
                REQUIRE_THROWS_WITH(mod_symtab.insert(symbol3), Catch::Contains("Re-declaration"));
            }
        }
        WHEN("added same symbol in children scope") {
            mod_symtab.enter_scope("scope1", program.get(), true, old_symtab);
            mod_symtab.insert(symbol2);
            THEN("it's ok, just get overshadow warning") {
                mod_symtab.enter_scope("scope2", program.get(), false, old_symtab);
                mod_symtab.insert(symbol3);
                ///\todo : not sure how to capture std::cout
            }
        }
    }
}
