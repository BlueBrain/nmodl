/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <string>

#include "fmt/format.h"
#include "printer/code_printer.hpp"


using namespace fmt::literals;

namespace instrument {

/// code instrumentor types
enum class InstrumentorType { None, Caliper, Likwid };


/**
 * \class InstrumentorBase
 * \brief Base class for code instrumentation
 *
 * We need code instrumentation mechanism for performance
 * analysis and optimization. Different profiling tools provide
 * different markers for code instrumentation. This base class
 * defines common interface for different tools.
 */
class InstrumentorBase {
  protected:
    /// printer object
    CodePrinter* printer;

  public:
    InstrumentorBase(CodePrinter* printer)
        : printer(printer) {}

    virtual ~InstrumentorBase() = default;

    /// header include
    virtual void header_include(){};

    /// initialize measurement (at the beginning)
    virtual void initialize() {}

    /// finalize measurement (at the end)
    virtual void finalize() {}

    /// start of region
    virtual void region_begin(const std::string& name){};

    /// end of region
    virtual void region_end(const std::string& name){};

    /// start of loop
    virtual void loop_begin(const std::string& id, const std::string& name){};

    /// end of loop
    virtual void loop_end(const std::string& id, const std::string& name){};
};


/**
 * \class Instrumentor
 * \brief Default instrumentor that does not do any instrumentation
 */
template <InstrumentorType T>
struct Instrumentor: public InstrumentorBase {
    Instrumentor(CodePrinter* printer)
        : InstrumentorBase(printer) {}
};


/**
 * \class Instrumentor<InstrumentorType::Caliper>:
 * \brief Define Caliper based instrumentation
 *
 * Caliper is a program instrumentation and performance measurement
 * framework designed to bake performance analysis capabilities directly
 * into applications and activate them at runtime. This class takes care
 * of implementing necessary instrumentation in code generator.
 */
template <>
struct Instrumentor<InstrumentorType::Caliper>: public InstrumentorBase {
    Instrumentor(CodePrinter* printer)
        : InstrumentorBase(printer) {}

    void header_include() override {
        printer->add_line("#include <caliper/cali.h>");
    }

    void region_begin(const std::string& name) override {
        printer->add_line("CALI_MARK_BEGIN(\"{}\");"_format(name));
    }

    void region_end(const std::string& name) override {
        printer->add_line("CALI_MARK_END(\"{}\");"_format(name));
    }

    void loop_begin(const std::string& id, const std::string& name) override {
        printer->add_line("CALI_CXX_MARK_LOOP_BEGIN({}, \"{}\");"_format(id, name));
    }

    void loop_end(const std::string& id, const std::string& name) override {
        printer->add_line("CALI_CXX_MARK_LOOP_END({});"_format(id));
    }
};


/**
 * \class Instrumentor<InstrumentorType::Likwid>
 * \brief Define LIKWID based instrumentation
 *
 *  Likwid is a performance measurement framework for Intel and AMD processors.
 *  This class takes care of instrumentation with LIKWID markers.
 */
template <>
struct Instrumentor<InstrumentorType::Likwid>: public InstrumentorBase {
    Instrumentor() = default;

    Instrumentor(CodePrinter* printer)
        : InstrumentorBase(printer) {}

    void header_include() override {
        printer->add_line("#include <likwid.h>");
    }

    void initialize() override {
        printer->add_line("LIKWID_MARKER_INIT;");
    }

    void finalize() override {
        printer->add_line("LIKWID_MARKER_CLOSE;");
    }

    void region_begin(const std::string& name) override {
        printer->add_line("LIKWID_MARKER_START(\"{}\");"_format(name));
    }

    void region_end(const std::string& name) override {
        printer->add_line("LIKWID_MARKER_STOP(\"{}\");"_format(name));
    }
};

}  // namespace instrument
