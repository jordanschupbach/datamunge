/// @file string_metrics.cpp
/// @brief Dynamic-programming edit metrics, fixed-byte comparison and lexical similarities.
#include <datamunge/algorithms/string_metrics.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::algorithms {

std::size_t hamming_distance(const std::string& a, const std::string& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("hamming_distance: strings must have equal length");
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++d;
    return d;
}


} // namespace datamunge::algorithms
