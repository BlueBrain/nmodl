#include <utility>

/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#pragma once

#include <utility>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iomanip>
#include <array>


namespace nmodl {
namespace units {

/// Maximum number of dimensions of units (maximum number of base units)
static const int MAX_DIMS = 10;

/**
 * \class unit
 * \brief Class that stores all the data of a unit
 */
class unit {
private:
    /// Double factor of the unit
    double m_factor = 1.0;
    
    /// Array of MAX_DIMS size that keeps the unit's dimensions
    std::array<int, MAX_DIMS> m_dim = {0};

    /// Name of the unit
    std::string m_name;

    /// Vector of nominators of the unit
    std::vector<std::string> nominator;

    /// Vector of denominators of the unit
    std::vector<std::string> denominator;

public:
    unit() = default;

    explicit unit(std::string t_name)
            : m_name(std::move(t_name)) {}

    unit(const double t_factor, std::array<int, MAX_DIMS> t_dim, std::string t_name)
            : m_factor(t_factor), m_dim(t_dim), m_name(std::move(t_name)) {}

    void add_unit(std::string t_name);

    void add_base_unit(std::string t_name);

    void add_nominator_double(std::string t_double);

    void add_nominator_dims(std::array<int, MAX_DIMS> t_dim);

    void add_denominator_dims(std::array<int, MAX_DIMS> t_dim);

    void add_nominator_unit(std::string t_nom);

    void add_nominator_unit(const std::vector<std::string> *t_nom);

    void add_denominator_unit(std::string t_denom);

    void add_denominator_unit(const std::vector<std::string> *t_denom);

    void mul_factor(double prefixFactor);

    void add_fraction(const std::string &t_fraction);

    double double_parsing(const std::string &t_double);

    std::vector<std::string> get_nominator_unit() const {
        return nominator;
    }

    std::vector<std::string> get_denominator_unit() const {
        return denominator;
    }

    std::string get_name() const {
        return m_name;
    }

    double get_factor() const {
        return m_factor;
    }

    std::array<int, MAX_DIMS> get_dims() const {
        return m_dim;
    }

};

/**
 * \class prefix
 * \brief Class that stores all the data of a prefix
 */
class prefix {
private:
    /// Prefix's double factor
    double m_factor = 1;

    /// Prefix's name
    std::string m_name;

    /// Prefix's name that this prefix is based on
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

/**
 * \class UnitTable
 * \brief Class that stores all the units, prefixes and names of base units used
 */
class UnitTable {
private:
    /// Hash map that stores all the units
    std::unordered_map<std::string, unit *> Table;

    /// Hash map that stores all the prefixes
    std::unordered_map<std::string, double> Prefixes;

    /// Hash map that stores all the base units' names
    std::array<std::string, MAX_DIMS> BaseUnitsNames;

public:

    UnitTable() = default;

    void calc_nominator_dims(unit *unit, std::string nominator_name);

    void calc_denominator_dims(unit *unit, std::string denominator_name);

    void insert(unit *unit);

    void insert_prefix(prefix *prfx);

    void print_units() const;

    void print_base_units() const;
};

}  // namespace unit
}  // namespace nmodl