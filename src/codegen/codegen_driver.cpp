/*************************************************************************
 * Copyright (C) 2018-2022 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include <string>

#include "codegen/codegen_driver.hpp"
#include "codegen_compatibility_visitor.hpp"
#include "utils/logger.hpp"
#include "visitors/after_cvode_to_cnexp_visitor.hpp"
#include "visitors/ast_visitor.hpp"
#include "visitors/constant_folder_visitor.hpp"
#include "visitors/global_var_visitor.hpp"
#include "visitors/inline_visitor.hpp"
#include "visitors/ispc_rename_visitor.hpp"
#include "visitors/kinetic_block_visitor.hpp"
#include "visitors/local_to_assigned_visitor.hpp"
#include "visitors/local_var_rename_visitor.hpp"
#include "visitors/localize_visitor.hpp"
#include "visitors/loop_unroll_visitor.hpp"
#include "visitors/neuron_solve_visitor.hpp"
#include "visitors/nmodl_visitor.hpp"
#include "visitors/perf_visitor.hpp"
#include "visitors/semantic_analysis_visitor.hpp"
#include "visitors/solve_block_visitor.hpp"
#include "visitors/steadystate_visitor.hpp"
#include "visitors/sympy_conductance_visitor.hpp"
#include "visitors/sympy_solver_visitor.hpp"
#include "visitors/symtab_visitor.hpp"
#include "visitors/units_visitor.hpp"
#include "visitors/verbatim_var_rename_visitor.hpp"

using namespace nmodl;
using namespace codegen;
using namespace visitor;

bool CodegenDriver::prepare_mod(std::shared_ptr<ast::Program> node) {
    /// whether to update existing symbol table or create new
    /// one whenever we run symtab visitor.
    bool update_symtab = false;

    std::string modfile;
    std::string scratch_dir = "tmp";
    auto filepath = [scratch_dir, modfile](const std::string& suffix, const std::string& ext) {
        static int count = 0;
        return fmt::format(
            "{}/{}.{}.{}.{}", scratch_dir, modfile, std::to_string(count++), suffix, ext);
    };

    /// just visit the ast
    AstVisitor().visit_program(*node);

    /// Check some rules that ast should follow
    {
        logger->info("Running semantic analysis visitor");
        if (SemanticAnalysisVisitor().check(*node)) {
            return false;
        }
    }

    /// construct symbol table
    {
        logger->info("Running symtab visitor");
        SymtabVisitor(update_symtab).visit_program(*node);
    }

    /// use cnexp instead of after_cvode solve method
    {
        logger->info("Running CVode to cnexp visitor");
        AfterCVodeToCnexpVisitor().visit_program(*node);
        ast_to_nmodl(*node, filepath("after_cvode_to_cnexp", "mod"));
    }

    /// Rename variables that match ISPC compiler double constants
    if (cfg.ispc_backend) {
        logger->info("Running ISPC variables rename visitor");
        IspcRenameVisitor(node).visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("ispc_double_rename", "mod"));
    }

    /// GLOBAL to RANGE rename visitor
    if (cfg.nmodl_global_to_range) {
        // make sure to run perf visitor because code generator
        // looks for read/write counts const/non-const declaration
        PerfVisitor().visit_program(*node);
        // make sure to run the GlobalToRange visitor after all the
        // reinitializations of Symtab
        logger->info("Running GlobalToRange visitor");
        GlobalToRangeVisitor(*node).visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("ispc_double_rename", "mod"));
    }

    /// LOCAL to ASSIGNED visitor
    if (cfg.nmodl_local_to_range) {
        logger->info("Running LOCAL to ASSIGNED visitor");
        PerfVisitor().visit_program(*node);
        LocalToAssignedVisitor().visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("global_to_range", "mod"));
    }

    {
        // Compatibility Checking
        logger->info("Running code compatibility checker");
        // run perfvisitor to update read/write counts
        PerfVisitor().visit_program(*node);

        auto ast_has_unhandled_nodes = CodegenCompatibilityVisitor().find_unhandled_ast_nodes(
            *node);
        // If we want to just check compatibility we return the result
        if (cfg.only_check_compatibility) {
            return !ast_has_unhandled_nodes;  // negate since this function returns false on failure
        }

        // If there is an incompatible construct and code generation is not forced exit NMODL
        if (ast_has_unhandled_nodes && !cfg.force_codegen) {
            return false;
        }
    }

    ast_to_nmodl(*node, filepath("ast", "mod"));
    ast_to_json(*node, filepath("ast", "json"));

    if (cfg.verbatim_rename) {
        logger->info("Running verbatim rename visitor");
        VerbatimVarRenameVisitor().visit_program(*node);
        ast_to_nmodl(*node, filepath("verbatim_rename", "mod"));
    }

    if (cfg.nmodl_const_folding) {
        logger->info("Running nmodl constant folding visitor");
        ConstantFolderVisitor().visit_program(*node);
        ast_to_nmodl(*node, filepath("constfold", "mod"));
    }

    if (cfg.nmodl_unroll) {
        logger->info("Running nmodl loop unroll visitor");
        LoopUnrollVisitor().visit_program(*node);
        ConstantFolderVisitor().visit_program(*node);
        ast_to_nmodl(*node, filepath("unroll", "mod"));
        SymtabVisitor(update_symtab).visit_program(*node);
    }

    /// note that we can not symtab visitor in update mode as we
    /// replace kinetic block with derivative block of same name
    /// in global scope
    {
        logger->info("Running KINETIC block visitor");
        auto kineticBlockVisitor = KineticBlockVisitor();
        kineticBlockVisitor.visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        const auto filename = filepath("kinetic", "mod");
        ast_to_nmodl(*node, filename);
        if (cfg.nmodl_ast && kineticBlockVisitor.get_conserve_statement_count()) {
            logger->warn(
                fmt::format("{} presents non-standard CONSERVE statements in DERIVATIVE blocks. Use it only for debugging/developing",
                    filename));
        }
    }

    {
        logger->info("Running STEADYSTATE visitor");
        SteadystateVisitor().visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("steadystate", "mod"));
    }

    /// Parsing units fron "nrnunits.lib" and mod files
    {
        logger->info("Parsing Units");
        UnitsVisitor(cfg.units_dir).visit_program(*node);
    }

    /// once we start modifying (especially removing) older constructs
    /// from ast then we should run symtab visitor in update mode so
    /// that old symbols (e.g. prime variables) are not lost
    update_symtab = true;

    if (cfg.nmodl_inline) {
        logger->info("Running nmodl inline visitor");
        InlineVisitor().visit_program(*node);
        ast_to_nmodl(*node, filepath("inline", "mod"));
    }

    if (cfg.local_rename) {
        logger->info("Running local variable rename visitor");
        LocalVarRenameVisitor().visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("local_rename", "mod"));
    }

    if (cfg.nmodl_localize) {
        // localize pass must follow rename pass to avoid conflict
        logger->info("Running localize visitor");
        LocalizeVisitor(cfg.localize_verbatim).visit_program(*node);
        LocalVarRenameVisitor().visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("localize", "mod"));
    }

    if (cfg.sympy_conductance) {
        logger->info("Running sympy conductance visitor");
        SympyConductanceVisitor().visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("sympy_conductance", "mod"));
    }

    if (cfg.sympy_analytic || sparse_solver_exists(*node)) {
        if (!cfg.sympy_analytic) {
            logger->info(
                "Automatically enable sympy_analytic because it exists solver of type sparse");
        }
        logger->info("Running sympy solve visitor");
        SympySolverVisitor(cfg.sympy_pade, cfg.sympy_cse).visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("sympy_solve", "mod"));
    }

    {
        logger->info("Running cnexp visitor");
        NeuronSolveVisitor().visit_program(*node);
        ast_to_nmodl(*node, filepath("cnexp", "mod"));
    }

    {
        SolveBlockVisitor().visit_program(*node);
        SymtabVisitor(update_symtab).visit_program(*node);
        ast_to_nmodl(*node, filepath("solveblock", "mod"));
    }

    if (cfg.json_perfstat) {
        auto file = scratch_dir + "/" + modfile + ".perf.json";
        logger->info("Writing performance statistics to {}", file);
        PerfVisitor(file).visit_program(*node);
    }

    {
        // make sure to run perf visitor because code generator
        // looks for read/write counts const/non-const declaration
        PerfVisitor().visit_program(*node);
    }
    return true;
}

void CodegenDriver::ast_to_nmodl(Program& ast, const std::string& filepath) const {
    if (cfg.nmodl_ast) {
        NmodlPrintVisitor(filepath).visit_program(ast);
        logger->info("AST to NMODL transformation written to {}", filepath);
    }
};

void CodegenDriver::ast_to_json(ast::Program& ast, const std::string& filepath) const {
    if (cfg.json_ast) {
        JSONVisitor(filepath).write(ast);
        logger->info("AST to JSON transformation written to {}", filepath);
    }
};
