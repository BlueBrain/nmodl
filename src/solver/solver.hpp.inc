#include <string>
#include <vector>

// This file is generated from `crout/crout.hpp` and `newton/newton.hpp`.
//
// To avoid a dependency of the `nmodl` binary on a specific header include directory and
// to avoid mod file compilation using wrong headers, we include them like this.
//
// However, because we want to be able to test the headers separately we can't
// move them here.

namespace nmodl::solver {
const std::vector<std::string> get_names() {
    return {"crout", "newton"};
}

const std::string get_hpp(const std::string& solver) {
    if (solver == "crout") {
        return R"jiowi(@NMODL_CROUT_HPP@)jiowi";
    } else if (solver == "newton") {
        return R"jiowi(@NMODL_NEWTON_HPP@)jiowi";
    } else {
        throw std::runtime_error("unknown solver '" + solver + "'");
    }
};

}
