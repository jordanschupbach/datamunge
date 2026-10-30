/// @file train_test_split.cpp
/// @brief Seeded, optionally stratified row split of a data frame.

#include <datamunge/random/random.hpp>
#include <datamunge/stats/train_test_split.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace datamunge::stats {

std::pair<dstruct::DataFrame, dstruct::DataFrame> train_test_split(const dstruct::DataFrame& data,
                                                                   double test_fraction, std::uint64_t seed,
                                                                   const std::string& stratify) {
  if (!(test_fraction > 0.0 && test_fraction < 1.0))
    throw std::invalid_argument("train_test_split: test_fraction must be in (0, 1)");
  using size_type = dstruct::DataFrame::size_type;

  // Group row indices by stratum (one group when not stratified), in first-appearance order.
  std::vector<std::vector<size_type>> groups;
  if (stratify.empty()) {
    groups.emplace_back(data.nrows());
    for (size_type r = 0; r < data.nrows(); ++r) groups[0][r] = r;
  } else {
    if (!data.has_column(stratify))
      throw std::invalid_argument("train_test_split: no column named '" + stratify + "'");
    if (data.column_type(stratify) != dstruct::DataFrame::ColumnType::String)
      throw std::invalid_argument("train_test_split: stratify column '" + stratify + "' must be a string column");
    const auto& column = data.get_column<std::string>(stratify);
    std::map<std::string, std::size_t> index;
    for (size_type r = 0; r < data.nrows(); ++r) {
      const std::string key = column.is_null(r) ? std::string("\x01NA") : column.values()[r];
      const auto [it, added] = index.try_emplace(key, groups.size());
      if (added) groups.emplace_back();
      groups[it->second].push_back(r);
    }
  }

  random::SplitMix64 rng(seed);
  std::vector<size_type> train, test;
  for (auto& rows : groups) {
    for (std::size_t i = rows.size(); i > 1; --i) std::swap(rows[i - 1], rows[rng.uniform_u64(0, i - 1)]);
    const auto n_test = static_cast<std::size_t>(std::lround(test_fraction * static_cast<double>(rows.size())));
    test.insert(test.end(), rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(n_test));
    train.insert(train.end(), rows.begin() + static_cast<std::ptrdiff_t>(n_test), rows.end());
  }
  std::sort(train.begin(), train.end());
  std::sort(test.begin(), test.end());
  return {data.take_rows(train), data.take_rows(test)};
}

} // namespace datamunge::stats
