/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <utility>
#include <fstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iomanip>


namespace nmodl {
namespace units {

    static const int max_dims = 10;

    class UnitTable;

class unit {
private:
    double m_factor = 1.0;
    std::array<int, max_dims> m_dim = {0};
    std::string m_name;
    std::vector<std::string> nominator;
    std::vector<std::string> denominator;

public:
    unit() = default;

    unit(const std::string t_name)
            : m_name(std::move(t_name)) {}

    // table(std::string&& t_name) noexcept
    //        : m_name(std::move(t_name)) {}

    unit(const double t_factor, std::array<int, max_dims> t_dim, std::string t_name)
            : m_factor(t_factor), m_dim(t_dim), m_name(std::move(t_name)) {}

    void addUnit(std::string t_name);

    void addBaseUnit(std::string t_name);

    void addNominatorDouble(std::string t_double);

    void addNominatorDims(std::array<int, max_dims> t_dim);

    void addDenominatorDims(std::array<int, max_dims> t_dim);

    void addNominatorUnit(std::string t_nom);

    void addNominatorUnit(const std::vector<std::string> *t_nom);

    void addDenominatorUnit(std::string t_denom);

    void addDenominatorUnit(const std::vector<std::string> *t_denom);

    void mulFactor(double prefixFactor);

    void addFraction(const std::string& t_fraction);

    double doubleParsing(const std::string& t_double);

    std::vector<std::string> getNominatorUnit() const {
        return nominator;
    }

    std::vector<std::string> getDenominatorUnit() const {
        return denominator;
    }

    std::string get_name() const {
        return m_name;
    }

    double get_factor() const {
        return m_factor;
    }

    std::array<int, max_dims> get_dims() const {
        return m_dim;
    }

};

class prefix {
private:
    double m_factor = 1;
    std::string m_name;
    std::string m_factorname;

public:

    prefix() = default;

    prefix(std::string t_name, std::string t_factor) {
        if (t_name.back() == '-') {
            t_name.pop_back();
        }
        m_name = t_name;
        if ((t_factor.front() >= '0' && t_factor.front() <= '9') || t_factor.front() == '.') {
            m_factor = std::stod(t_factor);
        } else {
            m_factorname = t_factor;
        }
    }

    std::string get_name() const {
        return m_name;
    }

    std::string get_factorname() const {
        return m_factorname;
    }

    double get_factor() const {
        return m_factor;
    }

};

class UnitTable {
private:
    std::unordered_map<std::string, unit *> Table;
    std::unordered_map<std::string, double> Prefixes;
    std::array<std::string, max_dims> BaseUnitsNames;

public:

    UnitTable() = default;

    void calcNominatorDims(unit *unit, std::string nominator_name);

    void calcDenominatorDims(unit *unit, std::string denominator_name);

    void insert(unit *unit);

    void insertPrefix(prefix *prfx);

    void printUnits() const;

    void printBaseUnits() const;
};

}  // namespace unit
}  // namespace nmodl