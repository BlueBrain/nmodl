/*************************************************************************
 * Copyright (C) 2018-2019 Blue Brain Project
 *
 * This file is part of NMODL distributed under the terms of the GNU
 * Lesser General Public License. See top-level LICENSE file for details.
 *************************************************************************/

#include "units.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace nmodl {
namespace units {

void nmodl::units::unit::add_unit(std::string t_name) {
    m_name = t_name;
}

void nmodl::units::unit::add_base_unit(std::string t_name) {
    // t_name = "*[a-j]*" which is a base unit
    const auto dim_name = t_name[1];
    const int dim_no = dim_name - 'a';
    m_dim[dim_no] = 1;
    add_nominator_unit(t_name);
}

void nmodl::units::unit::add_nominator_double(std::string t_double) {
    m_factor = double_parsing(t_double);
}

void nmodl::units::unit::add_nominator_dims(std::array<int, MAX_DIMS> t_dim) {
    std::transform(m_dim.begin(), m_dim.end(), t_dim.begin(), m_dim.begin(), std::plus<int>());
}

void nmodl::units::unit::add_denominator_dims(std::array<int, MAX_DIMS> t_dim) {
    std::transform(m_dim.begin(), m_dim.end(), t_dim.begin(), m_dim.begin(), std::minus<int>());
}

void nmodl::units::unit::add_nominator_unit(std::string t_nom) {
    nominator.push_back(t_nom);
}

void nmodl::units::unit::add_nominator_unit(const std::vector<std::string>* t_nom) {
    nominator.insert(nominator.end(), t_nom->begin(), t_nom->end());
}

void nmodl::units::unit::add_denominator_unit(std::string t_denom) {
    denominator.push_back(t_denom);
}

void nmodl::units::unit::add_denominator_unit(const std::vector<std::string>* t_denom) {
    denominator.insert(denominator.end(), t_denom->begin(), t_denom->end());
}

void nmodl::units::unit::mul_factor(const double prefixFactor) {
    m_factor *= prefixFactor;
}

void nmodl::units::unit::add_fraction(const std::string& t_fraction) {
    double nom, denom;
    std::string nominator;
    std::string denominator;
    std::string::const_iterator it;
    for (it = t_fraction.begin(); it != t_fraction.end() && *it != '|'; ++it) {
        nominator.push_back(*it);
    }
    // pass "|" char
    ++it;
    for (auto itm = it; itm != t_fraction.end(); ++itm) {
        denominator.push_back(*itm);
    }
    nom = double_parsing(nominator);
    denom = double_parsing(denominator);
    m_factor = nom / denom;
}

double nmodl::units::unit::double_parsing(std::string t_double) {
    double d_number, d_magnitude;
    std::string s_number;
    std::string s_magnitude;
    std::string::const_iterator it;
    int sign = 1;
    if (t_double.front() == '-') {
        sign = -1;
        t_double.erase(t_double.begin());
    }
    // if *it reached an exponent related char, then the whole double number is read
    for (it = t_double.begin(); it != t_double.end() && *it != 'e' && *it != '+' && *it != '-';
         ++it) {
        s_number.push_back(*it);
    }
    // then read the magnitude of the double number
    for (auto itm = it; itm != t_double.end(); ++itm) {
        if (*itm != 'e') {
            s_magnitude.push_back(*itm);
        }
    }
    d_number = std::stod(s_number);
    if (s_magnitude.empty()) {
        d_magnitude = 0.0;
    } else {
        d_magnitude = std::stod(s_magnitude);
    }
    return d_number * std::pow(10.0, d_magnitude) * sign;
}

void nmodl::units::UnitTable::calc_nominator_dims(unit* unit, std::string nominator_name) {
    double nominator_prefix_factor = 1.0;
    int nominator_power = 1;
    std::string nom_name = nominator_name;
    auto nominator = Table.find(nominator_name);

    // if the nominator_name is not in the table, check if there are any prefixes or power
    if (nominator == Table.end()) {
        int changed_nominator_name = 1;

        while (changed_nominator_name) {
            changed_nominator_name = 0;
            for (const auto& it: Prefixes) {
                auto res = std::mismatch(it.first.begin(), it.first.end(), nominator_name.begin());
                if (res.first == it.first.end()) {
                    changed_nominator_name = 1;
                    nominator_prefix_factor *= it.second;
                    nominator_name.erase(nominator_name.begin(),
                                         nominator_name.begin() + it.first.size());
                }
            }
        }
        // if the nominator is only a prefix, just multiply the prefix factor with the unit factor
        if (nominator_name.empty()) {
            for (const auto& it: Prefixes) {
                auto res = std::mismatch(it.first.begin(), it.first.end(), nom_name.begin());
                if (res.first == it.first.end()) {
                    unit->mul_factor(it.second);
                    return;
                }
            }
        }

        char nominator_back = nominator_name.back();
        if (nominator_back >= '2' && nominator_back <= '9') {
            nominator_power = nominator_back - '0';
            nominator_name.pop_back();
        }

        nominator = Table.find(nominator_name);

        if (nominator == Table.end()) {
            if (nominator_name.back() == 's') {
                nominator_name.pop_back();
            }
            nominator = Table.find(nominator_name);
        }
    }

    if (nominator == Table.end()) {
        std::stringstream ss;
        ss << "Unit " << nominator_name << " not defined!" << std::endl;
        throw std::runtime_error(ss.str());
    } else {
        for (int i = 0; i < nominator_power; i++) {
            unit->mul_factor(nominator_prefix_factor * nominator->second->get_factor());
            unit->add_nominator_dims(nominator->second->get_dims());
        }
    }
}

void nmodl::units::UnitTable::calc_denominator_dims(unit* unit, std::string denominator_name) {
    double denominator_prefix_factor = 1.0;
    int denominator_power = 1;
    std::string denom_name = denominator_name;
    auto denominator = Table.find(denominator_name);

    // if the denominator_name is not in the table, check if there are any prefixes or power
    if (denominator == Table.end()) {
        int changed_denominator_name = 1;

        while (changed_denominator_name) {
            changed_denominator_name = 0;
            for (const auto& it: Prefixes) {
                auto res = std::mismatch(it.first.begin(), it.first.end(),
                                         denominator_name.begin());
                if (res.first == it.first.end()) {
                    changed_denominator_name = 1;
                    denominator_prefix_factor *= it.second;
                    denominator_name.erase(denominator_name.begin(),
                                           denominator_name.begin() + it.first.size());
                }
            }
        }
        // if the denominator is only a prefix, just multiply the prefix factor with the unit factor
        if (denominator_name.empty()) {
            for (const auto& it: Prefixes) {
                auto res = std::mismatch(it.first.begin(), it.first.end(), denom_name.begin());
                if (res.first == it.first.end()) {
                    unit->mul_factor(it.second);
                    return;
                }
            }
        }

        char denominator_back = denominator_name.back();
        if (denominator_back >= '2' && denominator_back <= '9') {
            denominator_power = denominator_back - '0';
            denominator_name.pop_back();
        }

        denominator = Table.find(denominator_name);

        if (denominator == Table.end()) {
            if (denominator_name.back() == 's') {
                denominator_name.pop_back();
            }
            denominator = Table.find(denominator_name);
        }
    }

    if (denominator == Table.end()) {
        std::stringstream ss;
        ss << "Unit " << denominator_name << " not defined!" << std::endl;
        throw std::runtime_error(ss.str());
    } else {
        for (int i = 0; i < denominator_power; i++) {
            unit->mul_factor(1.0 / (denominator_prefix_factor * denominator->second->get_factor()));
            unit->add_denominator_dims(denominator->second->get_dims());
        }
    }
}

void nmodl::units::UnitTable::insert(unit* unit) {
    // check if the unit is a base unit and
    // then add it to the base units vector
    auto unit_nominator = unit->get_nominator_unit();
    auto only_base_unit_nominator =
        unit_nominator.size() == 1 && unit_nominator.front().size() == 3 &&
        (unit_nominator.front().front() == '*' && unit_nominator.front().back() == '*');
    if (only_base_unit_nominator) {
        // BaseUnitsNames[i] = "*i-th base unit*" (ex. BaseUnitsNames[0] = "*a*")
        BaseUnitsNames[unit_nominator.front()[1] - 'a'] = unit->get_name();
        // if  unit is found in table replace it
        auto find_unit_name = Table.find(unit->get_name());
        if (find_unit_name == Table.end()) {
            Table.insert({unit->get_name(), unit});
        } else {
            Table.erase(unit->get_name());
            Table.insert({unit->get_name(), unit});
        }
        return;
    }
    for (const auto& it: unit->get_nominator_unit()) {
        calc_nominator_dims(unit, it);
    }
    for (const auto& it: unit->get_denominator_unit()) {
        calc_denominator_dims(unit, it);
    }
    // if  unit is found in table replace it
    auto find_unit_name = Table.find(unit->get_name());
    if (find_unit_name == Table.end()) {
        Table.insert({unit->get_name(), unit});
    } else {
        Table.erase(unit->get_name());
        Table.insert({unit->get_name(), unit});
    }
}

void nmodl::units::UnitTable::insert_prefix(prefix* prfx) {
    // if the factorname is not empty, then this prefix is based on another one (rename)
    // else the factor of the prefix is calculated and should be added to the Prefixes
    auto rename = !prfx->get_factorname().empty();
    if (rename) {
        auto find_prfx = Prefixes.find(prfx->get_factorname());
        if (find_prfx == Prefixes.end()) {
            std::stringstream ss;
            ss << "Prefix " << prfx->get_factorname() << " not defined!" << std::endl;
            throw std::runtime_error(ss.str());
        } else {
            Prefixes.insert({prfx->get_name(), find_prfx->second});
        }
    } else {
        Prefixes.insert({prfx->get_name(), prfx->get_factor()});
    }
}

void nmodl::units::UnitTable::print_units() const {
    for (const auto& it: Table) {
        std::cout << std::fixed << std::setprecision(8) << it.first << " "
                  << it.second->get_factor() << ": ";
        for (const auto& dims: it.second->get_dims()) {
            std::cout << dims << " ";
        }
        std::cout << "\n";
    }
}

void nmodl::units::UnitTable::print_base_units() const {
    for (const auto& it: BaseUnitsNames) {
        std::cout << it << " ";
    }
    std::cout << "\n";
}

}  // namespace units
}  // namespace nmodl