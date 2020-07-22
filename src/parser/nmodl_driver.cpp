/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <fstream>
#include <sstream>

#include "lexer/nmodl_lexer.hpp"
#include "parser/nmodl_driver.hpp"
#include "utils/logger.hpp"

namespace nmodl {
namespace parser {

NmodlDriver::NmodlDriver(bool strace, bool ptrace)
    : trace_scanner(strace)
    , trace_parser(ptrace) {}

/// parse nmodl file provided as istream
std::shared_ptr<ast::Program> NmodlDriver::parse_stream(std::istream& in) {
    NmodlLexer scanner(*this, &in);
    NmodlParser parser(scanner, *this);

    scanner.set_debug(trace_scanner);
    parser.set_debug_level(trace_parser);
    parser.parse();
    return astRoot;
}

std::shared_ptr<ast::Program> NmodlDriver::parse_file(const std::string& filename,
                                                      const location* loc) {
    std::ifstream in(filename.c_str());
    stream_name = filename;

    if (!in.good()) {
        std::ostringstream oss;
        if (loc == nullptr) {
            oss << "NMODL Parser Error : ";
        }
        oss << "can not open file : " << filename;
        if (loc != nullptr) {
            parse_error(*loc, oss.str());
        } else {
            throw std::runtime_error(oss.str());
        }
    }
    parse_stream(in);
    return astRoot;
}

std::shared_ptr<ast::Program> NmodlDriver::parse_string(const std::string& input) {
    std::istringstream iss(input);
    parse_stream(iss);
    return astRoot;
}

std::shared_ptr<ast::Include> NmodlDriver::parse_include(const std::string& name,
                                                         const location& loc) {
    // Try to find directory containing the file to import
    const auto directory_path = library.find_file(name);

    // Complete path of file (directory + filename).
    std::string absolute_path = name;

    if (!directory_path.empty()) {
        absolute_path = directory_path + std::string(1, utils::pathsep) + name;
    }

    // Detect recursive inclusion.
    if (open_files.find(absolute_path) != open_files.end()) {
        std::ostringstream oss;
        oss << name << ": recursive inclusion.\n"
            << open_files[absolute_path] << ": initial inclusion was here.";
        parse_error(loc, oss.str());
    }
    library.push_current_directory(directory_path);
    open_files.emplace(absolute_path, loc);

    std::shared_ptr<ast::Program> program;
    program.swap(astRoot);

    parse_file(absolute_path, &loc);

    program.swap(astRoot);
    open_files.erase(absolute_path);
    library.pop_current_directory();
    auto filename_node = std::shared_ptr<ast::String>(
        new ast::String(std::string(1, '"') + name + std::string(1, '"')));
    return std::shared_ptr<ast::Include>(new ast::Include(filename_node, program));
}

void NmodlDriver::add_defined_var(const std::string& name, int value) {
    defined_var[name] = value;
}

bool NmodlDriver::is_defined_var(const std::string& name) const {
    return !(defined_var.find(name) == defined_var.end());
}

int NmodlDriver::get_defined_var_value(const std::string& name) const {
    const auto var_it = defined_var.find(name);
    if (var_it != defined_var.end()) {
        return var_it->second;
    }
    throw std::runtime_error("Trying to get undefined macro / define :" + name);
}

void NmodlDriver::parse_error(const location& location, const std::string& message) {
    std::ostringstream oss;
    oss << "NMODL Parser Error : " << message << " [Location : " << location << ']';
    throw std::runtime_error(oss.str());
}

std::string NmodlDriver::check_include_argument(const location& location,
                                                const std::string& filename) {
    if (filename.empty()) {
        parse_error(location, "empty filename in INCLUDE directive");
    } else if (filename.front() != '"' && filename.back() != '"') {
        parse_error(location, "filename may start and end with \" character");
    } else if (filename.size() == 3) {
        parse_error(location, "filename is empty");
    }
    return filename.substr(1, filename.size() - 2);
}

}  // namespace parser
}  // namespace nmodl
