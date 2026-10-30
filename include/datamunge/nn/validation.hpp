// datamunge::nn port of tflow_examples src/common/ml/validation.h (host-only, unchanged).
// Deterministic k-fold/stratified splits and sample-weighted cross-validation.
// Every row is held out exactly once. Stratification requires at least k rows
// per class and distributes each class round-robin after seeded shuffling.
// A callback receives training and validation row indices; it must fit both
// preprocessing and model exclusively on training rows, then return a mean
// validation loss. Refit the selected setting on all development rows before
// evaluating a separate test set. Cross-validation scores select a model;
// they are not an unbiased final performance estimate after that selection.
#pragma once

/// @file validation.hpp
/// @brief Balanced folds and sample-weighted cross-validation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::nn::ml {
/// @brief One complementary training/validation partition with zero-based row positions.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto folds=m::KFold(7,3,17,false);
/// const auto held_out=folds[0].validation; // {0,1,2}
/// @endcode
/// @see CompleteFolds()
struct Fold {
  std::vector<size_t> training;   ///< Complementary fitting rows, zero-based.
  std::vector<size_t> validation; ///< Held-out rows for this fold, zero-based.
};
/// @brief Validate exact hold-out coverage and construct training complements.
/// @param rows Number of source rows.
/// @param validation Nonempty validation lists; each source index must occur once overall.
/// @return Owned folds in supplied order, training positions sorted ascending.
/// @throws std::invalid_argument On duplicate/out-of-range positions, empty partitions or
/// incomplete coverage.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto folds=m::CompleteFolds(4,{{0,1},{2,3}});
/// @endcode
/// @see KFold()
inline std::vector<Fold> CompleteFolds(size_t rows, std::vector<std::vector<size_t>> validation) {
  std::vector<Fold> result;
  std::vector<int>  held_out(rows);
  for (auto& ids : validation) {
    if (ids.empty())
      throw std::invalid_argument("validation fold must be nonempty");
    std::vector<bool> used(rows);
    for (auto id : ids) {
      if (id >= rows || used[id])
        throw std::invalid_argument("invalid validation index");
      used[id] = true;
      ++held_out[id];
    }
    Fold fold;
    fold.validation = std::move(ids);
    for (size_t row = 0; row < rows; ++row)
      if (!used[row])
        fold.training.push_back(row);
    if (fold.training.empty())
      throw std::invalid_argument("training fold must be nonempty");
    result.push_back(std::move(fold));
  }
  for (int count : held_out)
    if (count != 1)
      throw std::invalid_argument("each row must be held out exactly once");
  return result;
}

/// @brief Create balanced disjoint validation folds, optionally shuffled.
///
/// Reproducibility of std::shuffle is within a fixed standard-library implementation.
/// Random splitting does not account for temporal or grouped dependence.
/// @param rows Number of source rows.
/// @param k Fold count in [2,rows].
/// @param seed mt19937_64 shuffle seed.
/// @param shuffle Whether to shuffle first; true by default.
/// @return Owned folds, first rows%k having one extra validation observation.
/// @throws std::invalid_argument On invalid k.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto folds=m::KFold(7,3,17,false);
/// @endcode
///
/// @par References
/// - Hastie, Tibshirani and Friedman, The Elements of Statistical Learning, chapter 7,
/// https://hastie.su.domains/ElemStatLearn/
/// @see StratifiedKFold()
inline std::vector<Fold> KFold(size_t rows, int k, uint64_t seed, bool shuffle = true) {
  if (k < 2 || size_t(k) > rows)
    throw std::invalid_argument("k must be in [2, rows]");
  std::vector<size_t> indices(rows);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937_64 rng(seed);
  if (shuffle)
    std::shuffle(indices.begin(), indices.end(), rng);
  std::vector<std::vector<size_t>> validation(k);
  size_t                           cursor = 0;
  for (int fold = 0; fold < k; ++fold) {
    const size_t count = rows / k + (size_t(fold) < rows % k);
    validation[fold].assign(indices.begin() + cursor, indices.begin() + cursor + count);
    cursor += count;
  }
  return CompleteFolds(rows, std::move(validation));
}

/// @brief Shuffle each class and distribute its rows round-robin across folds.
/// @param labels Class IDs in source order; integer values need not be contiguous.
/// @param k At least two folds, with at least k observations per class.
/// @param seed mt19937_64 seed, consumed across sorted class IDs.
/// @return Balanced folds with per-class counts differing by at most one.
/// @throws std::invalid_argument On invalid class counts or k.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto folds=m::StratifiedKFold({0,0,0,1,1,1},3,17);
/// @endcode
/// @see KFold()
inline std::vector<Fold> StratifiedKFold(const std::vector<int>& labels, int k, uint64_t seed) {
  if (k < 2 || labels.empty())
    throw std::invalid_argument("stratification needs labels and k >= 2");
  std::map<int, std::vector<size_t>> by_class;
  for (size_t i = 0; i < labels.size(); ++i)
    by_class[labels[i]].push_back(i);
  std::mt19937_64                  rng(seed);
  std::vector<std::vector<size_t>> validation(k);
  size_t                           offset = 0;
  for (auto& [label, rows] : by_class) {
    (void)label;
    if (rows.size() < size_t(k))
      throw std::invalid_argument("each class needs at least k rows");
    std::shuffle(rows.begin(), rows.end(), rng);
    for (size_t i = 0; i < rows.size(); ++i)
      validation[(offset + i) % k].push_back(rows[i]);
    offset = (offset + rows.size()) % k;
  }
  return CompleteFolds(labels.size(), std::move(validation));
}

/// @brief Sample-weighted validation mean and individual fold means.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto score=m::CrossValidate(m::KFold(6,3,17),[](const auto&,const auto&){return 2.;}); // mean is 2
/// @endcode
/// @see CrossValidate()
struct CrossValidationScore {
  double              mean = 0;    ///< Mean loss weighted by validation observation counts.
  std::vector<double> fold_losses; ///< Callback mean per fold, in fold order.
};
/// @brief Evaluate folds and weight each mean loss by validation sample count.
///
/// The callback must fit model and preprocessing exclusively on its training
/// indices. This aggregation helper does not validate index coverage or prevent leakage.
/// Use KFold/StratifiedKFold or CompleteFolds to establish partition invariants.
/// @tparam Evaluate Callable type returning a per-observation mean loss.
/// @param folds Nonempty folds with nonempty training and validation partitions.
/// @param evaluate Callable(training_indices,validation_indices) returning finite mean loss.
/// @return Weighted overall mean and fold means in input order.
/// @throws std::invalid_argument On empty folds/partitions; std::runtime_error on nonfinite
/// callback loss.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// const auto score=m::CrossValidate(m::KFold(6,3,17),[](const auto&,const auto&){return 2.;});
/// @endcode
/// @see KFold()
template<class Evaluate>
CrossValidationScore CrossValidate(const std::vector<Fold>& folds, Evaluate evaluate) {
  if (folds.empty())
    throw std::invalid_argument("cross-validation needs folds");
  CrossValidationScore result;
  size_t               samples = 0;
  for (const auto& fold : folds) {
    if (fold.training.empty() || fold.validation.empty())
      throw std::invalid_argument("fold partitions must be nonempty");
    const double loss = evaluate(fold.training, fold.validation);
    if (!std::isfinite(loss))
      throw std::runtime_error("nonfinite validation loss");
    result.fold_losses.push_back(loss);
    result.mean += loss * fold.validation.size();
    samples += fold.validation.size();
  }
  result.mean /= samples;
  return result;
}
} // namespace datamunge::nn::ml
