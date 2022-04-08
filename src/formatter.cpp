
/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "ast/program.hpp"
#include "config/config.h"
#include "parser/nmodl_driver.hpp"
#include "pybind/pyembed.hpp"
#include "utils/common_utils.hpp"
#include "utils/logger.hpp"
#include "visitors/nmodl_visitor.hpp"

/**
 * \dir
 * \brief Main NMODL code generation program
 */

using namespace fmt::literals;
using namespace nmodl;
using namespace visitor;
using nmodl::parser::NmodlDriver;

int main(int argc, const char* argv[]) {
    CLI::App app{
        "NMODL : Source-to-Source Code Generation Framework [{}]"_format(Version::to_string())};

    /// list of mod files to process
    std::vector<std::string> mod_files;

    /// true if debug logger statements should be shown
    std::string verbose("info");

    app.get_formatter()->column_width(40);

    app.add_option("--verbose", verbose, "Verbosity of logger output")
        ->capture_default_str()
        ->ignore_case()
        ->check(CLI::IsMember({"trace", "debug", "info", "warning", "error", "critical", "off"}));

    app.add_option("file", mod_files, "One or more MOD files to process")
        ->ignore_case()
        ->required()
        ->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    logger->set_level(spdlog::level::from_str(verbose));

    for (const auto& file: mod_files) {
        logger->info("Processing {}", file);

        const auto modfile = utils::remove_extension(utils::base_name(file));

        /// driver object creates lexer and parser, just call parser method
        NmodlDriver driver;

        /// parse mod file and construct ast
        const auto& ast = driver.parse_file(file);

        NmodlPrintVisitor("{}.clean"_format(modfile)).visit_program(*ast);
    }
}
