/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include "units/units.hpp"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nmodl {
namespace parser {

/// flex generated scanner class (extends base lexer class of flex)
class UnitLexer;

/// parser class generated by bison
class UnitParser;

/// represent location of the token
class location;

/**
 * @addtogroup parser
 * @addtogroup units
 * @{
 */

/**
 * \class UnitDriver
 * \brief Class that binds all pieces together for parsing C units
 */
class UnitDriver {
  private:
    /// pointer to the lexer instance being used
    UnitLexer* lexer = nullptr;

    /// pointer to the parser instance being used
    UnitParser* parser = nullptr;

    /// print messages from lexer/parser
    bool verbose = false;

  public:
    /// shared pointer to the UnitTable that stores all the unit definitions
    std::shared_ptr<nmodl::units::UnitTable> table = std::make_shared<nmodl::units::UnitTable>();

    /// file or input stream name (used by scanner for position), see todo
    std::string stream_name;

    /// \name Ctor & dtor
    /// \{

    UnitDriver() = default;
    UnitDriver(bool strace, bool ptrace);

    /// \}

    bool parse_stream(std::istream& in);
    bool parse_string(const std::string& input);
    bool parse_file(const std::string& filename);

    void set_verbose(bool b) {
        verbose = b;
    }

    bool is_verbose() const {
        return verbose;
    }
};

}  // namespace parser
}  // namespace nmodl
