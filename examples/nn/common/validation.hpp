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

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {
struct Fold { std::vector<size_t> training, validation; };
inline std::vector<Fold> CompleteFolds(size_t rows, std::vector<std::vector<size_t>> validation) {
  std::vector<Fold> result;
  std::vector<int> held_out(rows);
  for (auto& ids : validation) {
    if (ids.empty()) throw std::invalid_argument("validation fold must be nonempty");
    std::vector<bool> used(rows);
    for (auto id : ids) {
      if (id >= rows || used[id]) throw std::invalid_argument("invalid validation index");
      used[id] = true; ++held_out[id];
    }
    Fold fold; fold.validation = std::move(ids);
    for (size_t row = 0; row < rows; ++row) if (!used[row]) fold.training.push_back(row);
    if (fold.training.empty()) throw std::invalid_argument("training fold must be nonempty");
    result.push_back(std::move(fold));
  }
  for (int count : held_out) if (count != 1) throw std::invalid_argument("each row must be held out exactly once");
  return result;
}

inline std::vector<Fold> KFold(size_t rows, int k, uint64_t seed, bool shuffle = true) {
  if (k < 2 || size_t(k) > rows) throw std::invalid_argument("k must be in [2, rows]");
  std::vector<size_t> indices(rows);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937_64 rng(seed);
  if (shuffle) std::shuffle(indices.begin(), indices.end(), rng);
  std::vector<std::vector<size_t>> validation(k);
  size_t cursor = 0;
  for (int fold = 0; fold < k; ++fold) {
    const size_t count = rows/k + (size_t(fold) < rows%k);
    validation[fold].assign(indices.begin()+cursor, indices.begin()+cursor+count);
    cursor += count;
  }
  return CompleteFolds(rows, std::move(validation));
}

inline std::vector<Fold> StratifiedKFold(const std::vector<int>& labels, int k, uint64_t seed) {
  if (k < 2 || labels.empty()) throw std::invalid_argument("stratification needs labels and k >= 2");
  std::map<int, std::vector<size_t>> by_class;
  for (size_t i = 0; i < labels.size(); ++i) by_class[labels[i]].push_back(i);
  std::mt19937_64 rng(seed);
  std::vector<std::vector<size_t>> validation(k);
  size_t offset = 0;
  for (auto& [label, rows] : by_class) {
    (void)label;
    if (rows.size() < size_t(k)) throw std::invalid_argument("each class needs at least k rows");
    std::shuffle(rows.begin(), rows.end(), rng);
    for (size_t i = 0; i < rows.size(); ++i) validation[(offset+i)%k].push_back(rows[i]);
    offset = (offset+rows.size())%k;
  }
  return CompleteFolds(labels.size(), std::move(validation));
}

struct CrossValidationScore { double mean = 0; std::vector<double> fold_losses; };
template<class Evaluate> CrossValidationScore CrossValidate(const std::vector<Fold>& folds, Evaluate evaluate) {
  if (folds.empty()) throw std::invalid_argument("cross-validation needs folds");
  CrossValidationScore result;
  size_t samples = 0;
  for (const auto& fold : folds) {
    if (fold.training.empty() || fold.validation.empty()) throw std::invalid_argument("fold partitions must be nonempty");
    const double loss = evaluate(fold.training, fold.validation);
    if (!std::isfinite(loss)) throw std::runtime_error("nonfinite validation loss");
    result.fold_losses.push_back(loss);
    result.mean += loss*fold.validation.size(); samples += fold.validation.size();
  }
  result.mean /= samples;
  return result;
}
}  // namespace ml
