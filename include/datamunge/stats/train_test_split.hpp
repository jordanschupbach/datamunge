#pragma once

/// @file train_test_split.hpp
/// @brief Random (optionally stratified) split of a data frame into training and test rows.

#include <datamunge/dstruct/dataframe.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace datamunge::stats {

/// @brief Splits @p data into a training frame and a test frame by rows, like scikit-learn's
///        `train_test_split()` or rsample's `initial_split()`.
///
/// Rows are shuffled with a seeded generator, so the split is reproducible. With @p stratify,
/// each level of that string column is split separately, so both parts keep (to within one row
/// per level) the class proportions of the whole frame. Both parts keep the original row order.
///
/// @param data The frame to split.
/// @param test_fraction Share of rows (per level, when stratified) that go to the test frame,
///        rounded to the nearest row; must be in (0, 1).
/// @param seed Seed for the shuffle.
/// @param stratify Name of a string column to stratify by; empty for a plain random split.
/// @return `{train, test}`.
/// @throws std::invalid_argument if @p test_fraction is not in (0, 1), or @p stratify names a
///         column that does not exist or is not a string column.
///
/// @par Example
/// @code
/// const auto [train, test] = dm::stats::train_test_split(dm::datasets::breast_cancer(), 0.3, 1, "diagnosis");
/// std::print("{} training rows, {} test rows\n", train.nrows(), test.nrows());   // 398, 171
/// @endcode
std::pair<dstruct::DataFrame, dstruct::DataFrame> train_test_split(const dstruct::DataFrame& data,
                                                                   double test_fraction = 0.25,
                                                                   std::uint64_t seed = 42,
                                                                   const std::string& stratify = "");

} // namespace datamunge::stats
