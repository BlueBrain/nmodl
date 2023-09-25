/*
 * Copyright 2023 Blue Brain Project, EPFL.
 * See the top-level LICENSE file for details.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>


namespace nmodl {
namespace parser {

/// flex generated scanner class (extends base lexer class of flex)
class CLexer;

/// parser class generated by bison
class CParser;

class location;

/**
 * \addtogroup parser
 * \{
 */

/**
 * \class CDriver
 * \brief Class that binds all pieces together for parsing C verbatim blocks
 */
class CDriver {
  private:
    /// all typedefs
    std::map<std::string, std::string> typedefs;

    /// constants defined in enum
    std::vector<std::string> enum_constants;

    /// all tokens encountered
    std::vector<std::string> tokens;

    /// enable debug output in the flex scanner
    bool trace_scanner = false;

    /// enable debug output in the bison parser
    bool trace_parser = false;

    /// pointer to the lexer instance being used
    std::unique_ptr<CLexer> lexer;

    /// pointer to the parser instance being used
    std::unique_ptr<CParser> parser;

    /// print messages from lexer/parser
    bool verbose = false;

  public:
    /// file or input stream name (used by scanner for position), see todo
    std::string streamname;

    CDriver();
    CDriver(bool strace, bool ptrace);
    ~CDriver();

    static void error(const std::string& m);

    bool parse_stream(std::istream& in);
    bool parse_string(const std::string& input);
    bool parse_file(const std::string& filename);
    void scan_string(const std::string& text);
    void add_token(const std::string&);

    static void error(const std::string& m, const location& l);

    void set_verbose(bool b) noexcept {
        verbose = b;
    }

    bool is_verbose() const noexcept {
        return verbose;
    }

    bool is_typedef(const std::string& type) const noexcept {
        return typedefs.find(type) != typedefs.end();
    }

    bool is_enum_constant(const std::string& constant) const noexcept {
        return std::find(enum_constants.begin(), enum_constants.end(), constant) !=
               enum_constants.end();
    }

    const std::vector<std::string>& all_tokens() const noexcept {
        return tokens;
    }

    bool has_token(const std::string& token) const noexcept {
        if (std::find(tokens.begin(), tokens.end(), token) != tokens.end()) {
            return true;
        }
        return false;
    }
};

/** \} */  // end of parser

}  // namespace parser
}  // namespace nmodl