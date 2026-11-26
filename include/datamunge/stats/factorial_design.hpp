#pragma once

/// @file factorial_design.hpp
/// @brief Two-level full and regular fractional factorial designs with defining relations and alias sets.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief A two-level design in standard order with its aliasing structure.
/// @par Example
/// @code
/// const auto d = dm::stats::full_factorial_design(3);
/// std::println("{} runs, resolution {}", d.runs.rows(), d.resolution); // 8 runs, resolution 0
/// @endcode
struct TwoLevelDesign {
    /// Runs by factors with levels \f$\pm1\f$ in standard (Yates) order: the first factor
    /// alternates fastest, and generated columns are products of base columns.
    linalg::DenseMatrix<double> runs;
    std::vector<std::string> factors;            ///< Factor names A, B, ... (I is skipped).
    std::vector<std::string> generators;         ///< Generated factors, e.g. "E=ABC".

    /// Words of the defining relation \f$I=\dots\f$ (every product of generator words),
    /// sorted by length and then alphabetically; empty for a full factorial.
    std::vector<std::string> defining_relation;

    /// Number of words of length 3, 4, ..., k in the defining relation (the word length
    /// pattern used to rank designs by aberration).
    std::vector<std::size_t> word_length_pattern;
    std::size_t resolution{};                    ///< Shortest word length; 0 for a full factorial.
};

/// @brief Full \f$2^k\f$ factorial design in standard order.
/// @param factors Number of factors k, from 1 to 20.
/// @return Design with \f$2^k\f$ runs and no defining relation.
/// @throws std::invalid_argument For k outside 1..20.
/// @par Example
/// @code
/// const auto d = dm::stats::full_factorial_design(2);
/// std::println("{} {} {} {}", d.runs(0, 0), d.runs(1, 0), d.runs(2, 0), d.runs(2, 1)); // -1 1 -1 1
/// @endcode
/// @par References
/// - G. E. P. Box, J. S. Hunter and W. G. Hunter (2005). Statistics for Experimenters, 2nd ed.
///   Wiley, ch. 5. ISBN 978-0-471-71813-0
/// @see fractional_factorial_design()
[[nodiscard]] TwoLevelDesign full_factorial_design(std::size_t factors);

/// @brief Regular \f$2^{k-q}\f$ fractional factorial design from generator words.
///
/// The first \f$k-q\f$ factors form a full factorial; each generator word, written with
/// those base factor letters (e.g. "ABC"), defines the next factor as the product of
/// its columns, as in `FrF2(generators = ...)` in R. The defining relation contains all
/// \f$2^q-1\f$ products of the generator words multiplied by their new factors; the
/// resolution is its shortest word length.
/// @param factors Total number of factors k, from 2 to 20.
/// @param generators q generator words over the base letters, each of length at least two, distinct.
/// @return Design with \f$2^{k-q}\f$ runs and its defining relation, word length pattern and resolution.
/// @throws std::invalid_argument For k outside 2..20, no generators or q not below k, a letter outside
/// the base factors, a repeated letter, a word shorter than two letters, or duplicate words.
/// @par Example
/// @code
/// const auto d = dm::stats::fractional_factorial_design(6, {"ABC", "BCD"});
/// std::println("{} {} {}", d.runs.rows(), d.defining_relation[0], d.resolution); // 16 ABCE 4
/// @endcode
/// @par References
/// - G. E. P. Box and J. S. Hunter (1961). "The 2^(k-p) fractional factorial designs, part I".
///   Technometrics 3(3), 311-351. doi:10.1080/00401706.1961.10489951
/// - A. Fries and W. G. Hunter (1980). "Minimum aberration 2^(k-p) designs". Technometrics 22(4),
///   601-608. doi:10.1080/00401706.1980.10486210
/// - https://cran.r-project.org/package=FrF2
/// @see alias_sets(), full_factorial_design()
[[nodiscard]] TwoLevelDesign fractional_factorial_design(std::size_t factors,
                                                         const std::vector<std::string>& generators);

/// @brief Sets of effects of order at most `max_order` that share a column of the design.
///
/// Two effects are aliased when their product is a word of the defining relation. Each
/// returned set lists such effects in order of length, then alphabetically, and only sets
/// with two or more members are returned. With `max_order = 2` this reproduces the
/// main-effect and two-factor-interaction alias lists of FrF2. Effects aliased with the
/// mean (words of the defining relation) are not reported here.
/// @param design Design from full_factorial_design() or fractional_factorial_design().
/// @param max_order Largest interaction order to list, from 1 to the number of factors.
/// @return Alias sets, ordered by their first member.
/// @throws std::invalid_argument For max_order outside 1..k.
/// @par Example
/// @code
/// const auto d = dm::stats::fractional_factorial_design(4, {"ABC"});
/// const auto sets = dm::stats::alias_sets(d);
/// std::println("{}={}", sets[0][0], sets[0][1]); // AB=CD
/// @endcode
/// @see fractional_factorial_design()
[[nodiscard]] std::vector<std::vector<std::string>> alias_sets(const TwoLevelDesign& design,
                                                               std::size_t max_order = 2);

} // namespace datamunge::stats
