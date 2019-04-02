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
#include <cmath>
#include <iomanip>


namespace unit {

static const int max_dims = 10;

class UnitTable;

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

    double doubleParsing(const std::string t_double){
        double nmbr, mgntd;
        std::string number;
        std::string magnitude;
        std::string::const_iterator it;
        for(it = t_double.begin(); it != t_double.end() && *it != 'e' && *it != '+' && *it != '-'; ++it ){
            number.push_back(*it);
        }
        for(auto itm = it; itm != t_double.end(); ++itm){
            if(*itm != 'e'){
                magnitude.push_back(*itm);
            }
        }
        nmbr = std::stod(number);
        if(magnitude.empty()){
            mgntd = 0.0;
        }
        else {
            mgntd = std::stod(magnitude);
        }
        return nmbr*std::pow(10.0, mgntd);
    }

    void addNominatorDouble(const std::string t_double) {
        m_factor = doubleParsing(t_double);
    }

    void addFraction(const std::string t_double) {
        double nom, denom;
        std::string nominator;
        std::string denominator;
        std::string::const_iterator it;
        for(it = t_double.begin(); it != t_double.end() && *it != '|'; ++it ){
            nominator.push_back(*it);
        }
        for(auto itm = it; itm != t_double.end(); ++itm){
            if(*itm != '|') {
                denominator.push_back(*itm);
            }
        }
        nom = doubleParsing(nominator);
        denom = doubleParsing(denominator);
        m_factor = nom/denom;
    }

    void addNominatorDims(std::array<int,max_dims> t_dim){
        std::transform (m_dim.begin(), m_dim.end(), t_dim.begin(), m_dim.begin(), std::plus<int>());
    }

    void addDenominatorDims(std::array<int,max_dims> t_dim){
        std::transform (m_dim.begin(), m_dim.end(), t_dim.begin(), m_dim.begin(), std::minus<int>());
    }

    void mulFactor(double prefixFactor){
        m_factor *= prefixFactor;
    }

    void addNominatorUnit(std::string t_nom){
        nominator.push_back(t_nom);
    }

    void addNominatorUnit(std::vector<std::string>* t_nom){
        nominator.insert(nominator.end(), t_nom->begin(), t_nom->end());
    }

    std::vector<std::string> getNominatorUnit(){
        return nominator;
    }

    void addDenominatorUnit(std::string t_denom){
        denominator.push_back(t_denom);
    }

    void addDenominatorUnit(std::vector<std::string>* t_denom){
        denominator.insert(denominator.end(), t_denom->begin(), t_denom->end());
    }

    std::vector<std::string> getDenominatorUnit(){
        return denominator;
    }

    std::string get_name() {
        return m_name;
    }

    double get_factor() {
        return m_factor;
    }

    std::array<int,max_dims> get_dims(){
        return m_dim;
    }

  private:
    double m_factor = 1;
    std::array<int,max_dims> m_dim = { 0 };
    std::string m_name;
    std::vector<std::string> nominator;
    std::vector<std::string> denominator;
};

class prefix {
public:

    prefix(std::string t_name, std::string t_factor){
        if(t_name.back() == '-'){
            t_name.pop_back();
        }
        m_name = t_name;
        if((t_factor.front() >= '0' && t_factor.front() <= '9') || t_factor.front() == '.' ){
            m_factor = std::stod(t_factor);
        }
        else{
            m_factorname = t_factor;
        }
    }

    std::string get_name() {
        return m_name;
    }

    std::string get_factorname() {
        return m_factorname;
    }

    double get_factor() {
        return m_factor;
    }

private:
    double m_factor = 1;
    std::string m_name;
    std::string m_factorname;
};

class UnitTable {
  public:
    std::unordered_map<std::string, table*> Table;
    std::unordered_map<std::string, double> Prefixes;

    UnitTable() = default;
    UnitTable(table* unit) {
        Table.insert({unit->get_name(), unit});
    }

    void calcNominatorDims(table* unit, std::string nominator_name){

        double nominator_prefix_factor = 1.0;
        int nominator_power = 1;

        auto nominator = Table.find(nominator_name);

        if(nominator == Table.end()) {

            int changed_nominator_name = 1;

            while (changed_nominator_name) {
                changed_nominator_name = 0;
                for (auto it : Prefixes) {
                    auto res = std::mismatch(it.first.begin(), it.first.end(), nominator_name.begin());
                    if (res.first == it.first.end()) {
                        changed_nominator_name = 1;
                        nominator_prefix_factor = it.second;
                        nominator_name.erase(nominator_name.begin(), nominator_name.begin() + it.first.size());
                    }
                }
            }

            char nominator_back = nominator_name.back();
            if (nominator_back >= '2' && nominator_back <= '9') {
                nominator_power = nominator_back - '0';
                nominator_name.pop_back();
            }

            nominator = Table.find(nominator_name);
        }

        if(nominator == Table.end()){
            std::stringstream ss;
            ss << "Unit " << nominator_name << " not defined!" << std::endl;
            throw std::runtime_error(ss.str());
        }
        else{
            for(int i = 0; i < nominator_power; i++) {
                unit->mulFactor(nominator_prefix_factor*nominator->second->get_factor());
                unit->addNominatorDims(nominator->second->get_dims());
            }
        }

    }

    void calcDenominatorDims(table* unit, std::string denominator_name){

        double denominator_prefix_factor = 1.0;
        int denominator_power = 1;

        auto denominator = Table.find(denominator_name);

        if(denominator == Table.end()) {

            int changed_denominator_name = 1;

            while (changed_denominator_name) {
                changed_denominator_name = 0;
                for (auto it : Prefixes) {
                    auto res = std::mismatch(it.first.begin(), it.first.end(), denominator_name.begin());
                    if (res.first == it.first.end()) {
                        changed_denominator_name = 1;
                        denominator_prefix_factor = it.second;
                        denominator_name.erase(denominator_name.begin(), denominator_name.begin() + it.first.size());
                    }
                }
            }

            char denominator_back = denominator_name.back();
            if (denominator_back >= '2' && denominator_back <= '9') {
                denominator_power = denominator_back - '0';
                denominator_name.pop_back();
            }

            denominator = Table.find(denominator_name);

        }

        if(denominator == Table.end()){
            std::stringstream ss;
            ss << "Unit " << denominator_name << " not defined!" << std::endl;
            throw std::runtime_error(ss.str());
        }
        else{
            for(int i = 0; i < denominator_power; i++) {
                unit->mulFactor(1.0/(denominator_prefix_factor*denominator->second->get_factor()));
                unit->addDenominatorDims(denominator->second->get_dims());
            }
        }

    }

    void insert(table* unit) {
        for(auto it : unit->getNominatorUnit()){
            calcNominatorDims(unit, it);
        }
        for(auto it : unit->getDenominatorUnit()){
            calcDenominatorDims(unit, it);
        }
        Table.insert({unit->get_name(), unit});
    }
    void insertPrefix(prefix* prfx){
        auto rename = !prfx->get_factorname().empty();
        if(rename) {
            auto find_prfx = Prefixes.find(prfx->get_factorname());
            if (find_prfx == Prefixes.end()) {
                std::stringstream ss;
                ss << "Prefix " << prfx->get_factorname() << " not defined!" << std::endl;
                throw std::runtime_error(ss.str());
            } else {
                Prefixes.insert({prfx->get_name(), find_prfx->second});
            }
        }
        else{
            Prefixes.insert({prfx->get_name(), prfx->get_factor()});
        }
    }
    void printUnits() {
        for (auto it = Table.begin(); it != Table.end(); it++) {
            std::cout << std::fixed << std::setprecision(8) << (*it).first << " " << (*it).second->get_factor() << ": ";
            for(auto dims : (*it).second->get_dims()){
                std::cout << dims << " ";
            }
            std::cout << std::endl;
        }
    }
};
}  // namespace unit
#endif  // NMODL_UNITS_H
