//
// Created by magkanar on 4/1/19.
//

#ifndef NMODL_UNITS_H
#define NMODL_UNITS_H

#include <fstream>
#include <iostream>
#include <regex>
#include <unordered_map>
#include <vector>
namespace unit {

static const int max_dims = 10;

class String;
class unit {
  public:
    unit() = default;

    unit(double t_factor, std::vector<int> t_dim, int t_isnum)
        : m_factor(t_factor)
        , m_dim(t_dim)
        , m_isnum(t_isnum) {}

  private:
    double m_factor;
    std::vector<int> m_dim;
    int m_isnum;
};

class table {
  public:
    table() = default;

    table(const std::string t_name)
        : m_name(std::move(t_name)) {}

    // table(std::string&& t_name) noexcept
    //        : m_name(std::move(t_name)) {}

    table(const double t_factor, const std::array<int,max_dims> t_dim, const std::string& t_name)
        : m_factor(t_factor)
        , m_dim(t_dim)
        , m_name(t_name) {}

    void addUnit(const std::string t_name) {
        m_name = t_name;
    }

    void addBaseUnit(const std::string t_name) {
        /// t_name = "*[a-j]*"
        const auto dim_name = t_name[1];
        const int dim_no = dim_name - 'a';
        m_dim[dim_no] = 1;
    }

    void addNominatorDouble(const std::string t_double) {
        m_factor *= std::stod(t_double);
    }

    std::string get_name() {
        return m_name;
    }

    double get_factor() {
        return m_factor;
    }

  private:
    double m_factor = 1;
    std::array<int,max_dims> m_dim;
    std::string m_name;
};

class UnitTable {
  public:
    std::unordered_map<std::string, table*> Table;
    UnitTable() = default;
    UnitTable(table* unit) {
        Table.insert({unit->get_name(), unit});
    }
    void insert(table* unit) {
        std::cout << "Inserting: " << unit->get_name() << " " << unit->get_factor() << std::endl;
        Table.insert({unit->get_name(), unit});
    }
    void printUnits() {
        for (auto it = Table.begin(); it != Table.end(); it++) {
            std::cout << (*it).first << " " << (*it).second->get_factor() << std::endl;
        }
    }
};
}  // namespace unit
#endif  // NMODL_UNITS_H
