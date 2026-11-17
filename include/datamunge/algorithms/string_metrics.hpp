#pragma once
#include <datamunge/algorithms/levenshtein.hpp>
#include <datamunge/algorithms/jaro_winkler.hpp>
/// @file string_metrics.hpp
/// @brief Byte-string edit distances, positional distance and lexical similarity measures.

#include <datamunge/algorithms/damerau_levenshtein.hpp>
#include <datamunge/algorithms/dice_coefficient.hpp>
#include <cstddef>
#include <string>

namespace datamunge::algorithms {

/// @brief Counts unequal byte positions in two fixed-width strings (Hamming distance).
/// The strings must have equal byte length. Embedded NULs and every byte value are
/// compared literally; there is no UTF-8 decoding, Unicode normalization, padding or
/// realignment. Both empty strings give zero. Inputs are borrowed and unchanged,
/// with no shared mutable state. A single scan costs O(n) time and O(1) storage.
/// @param a First byte string.
/// @param b Second string, of the same byte length.
/// @return Number of unequal positions, between zero and a.size().
/// @throws std::invalid_argument Lengths differ.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hamming_distance("karolin","kathrin")); // 3
/// std::print("{}\n",dm::algorithms::hamming_distance("","")); // 0
/// @endcode
/// @par References
/// - R. W. Hamming (1950), Error Detecting and Error Correcting Codes,
///   Bell System Technical Journal 29(2), 147-160.
///   https://doi.org/10.1002/j.1538-7305.1950.tb00463.x
/// @see levenshtein_distance()
std::size_t hamming_distance(const std::string& a, const std::string& b);


} // namespace datamunge::algorithms
