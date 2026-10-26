// Small real-word recognition data and metrics (port of tflow's common/ml/ocr.h):
// COCO-Text grayscale word crops [N, 32, 100, 1], labels [N, 12] over
// "0-9a-z" padded with -1, CTC blank = 36, greedy decoding and edit distance.
#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/sequence.hpp>

#include "common/checks.hpp"

namespace ocr {

namespace nn = datamunge::nn;
using nn::Tensor;

inline const std::string kAlphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
constexpr int            kBlank = 36, kSteps = 26, kClasses = 37;

inline std::string CtcCollapse(const std::vector<std::int64_t>& path, int blank = kBlank) {
  std::string text;
  for (auto c : path) check::Require(c >= 0 && c <= blank, "CTC path class");
  for (auto c : nn::ctc_collapse(path, blank)) text += kAlphabet[static_cast<std::size_t>(c)];
  return text;
}

inline int OcrDistance(const std::string& a, const std::string& b) {
  std::vector<int> row(b.size() + 1);
  std::iota(row.begin(), row.end(), 0);
  for (std::size_t i = 0; i < a.size(); ++i) {
    auto old = row;
    row[0]   = static_cast<int>(i + 1);
    for (std::size_t j = 0; j < b.size(); ++j) row[j + 1] = std::min({old[j + 1] + 1, row[j] + 1, old[j] + (a[i] != b[j])});
  }
  return row.back();
}

struct OcrData {
  Tensor pixels, labels, lengths, image_ids, annotation_ids;
};

inline OcrData OcrRows(const OcrData& d, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(d.pixels, rows), nn::take_rows(d.labels, rows), nn::take_rows(d.lengths, rows),
          nn::take_rows(d.image_ids, rows), nn::take_rows(d.annotation_ids, rows)};
}

inline std::string OcrText(const OcrData& d, int n) {
  const auto  length = d.lengths.to_vector<int>()[n];
  const auto  labels = d.labels.to_vector<int>();
  std::string result;
  for (int i = 0; i < length; ++i) result += kAlphabet[labels[n * 12 + i]];
  return result;
}

inline std::vector<OcrData> LoadOcr(const std::filesystem::path& root) {
  const auto arrays = nn::load_npz((root / "ocr.npz").string());
  auto       get    = [&](const std::string& key) {
    for (const auto& [name, t] : arrays)
      if (name == key) return t;
    throw std::runtime_error("ocr.npz: missing " + key);
  };
  std::vector<OcrData> out;
  std::set<int>        previous, annotations;
  for (const std::string split : {"fit", "validation", "test"}) {
    OcrData d{get(split + "_pixels"), get(split + "_labels"), get(split + "_lengths"), get(split + "_image_ids"),
              get(split + "_annotation_ids")};
    const auto n = d.pixels.shape()[0];
    check::Require(n > 0 && d.pixels.dtype() == nn::DType::Float32 && d.pixels.shape() == nn::Shape{n, 32, 100, 1},
                   "OCR image shape/dtype");
    for (const Tensor* t : {&d.labels, &d.lengths, &d.image_ids, &d.annotation_ids})
      check::Require(t->dtype() == nn::DType::Int32 && t->shape()[0] == n, "OCR aligned int metadata");
    check::Require(d.labels.shape() == nn::Shape{n, 12} && d.lengths.dim() == 1, "OCR target dimensions");
    const auto    lengths = d.lengths.to_vector<int>(), labels = d.labels.to_vector<int>();
    const auto    images = d.image_ids.to_vector<int>(), notes = d.annotation_ids.to_vector<int>();
    std::set<int> current;
    for (int i = 0; i < n; ++i) {
      const int length = lengths[i];
      check::Require(length >= 1 && length <= 12, "variable word label length");
      int minimum = length;
      for (int j = 0; j < 12; ++j) {
        const int c = labels[i * 12 + j];
        check::Require(j < length ? (c >= 0 && c < 36) : c == -1, "CTC character/padding range");
        if (j > 0 && j < length && c == labels[i * 12 + j - 1]) ++minimum;
      }
      check::Require(minimum <= kSteps, "CTC repeat-aware feasible alignment");
      check::Require(!previous.count(images[i]) && annotations.insert(notes[i]).second,
                     "source images disjoint and annotations unique");
      current.insert(images[i]);
    }
    for (float v : d.pixels.to_vector<float>())
      check::Require(std::isfinite(v) && v >= 0 && v <= 1, "observed grayscale pixels");
    previous.insert(current.begin(), current.end());
    std::cout << " OCR " << split << ":" << n << " words/" << current.size() << " source images\n";
    out.push_back(d);
  }
  return out;
}

/// Greedy CTC decoding of time-major logits [26, N, 37].
inline std::vector<std::string> OcrDecode(const Tensor& logits) {
  check::Require(logits.dim() == 3 && logits.shape()[0] == kSteps && logits.shape()[2] == kClasses, "OCR logits TBC");
  for (float v : logits.to_vector<float>()) check::Require(std::isfinite(v), "finite OCR logit");
  std::vector<std::string> result;
  for (const auto& seq : nn::ctc_greedy_decode(logits).sequences) {
    std::string text;
    for (auto c : seq) text += kAlphabet[static_cast<std::size_t>(c)];
    result.push_back(text);
  }
  return result;
}

inline void OcrMetrics(const std::string& name, const OcrData& d, const std::vector<std::string>& predicted) {
  int errors = 0, characters = 0, exact = 0;
  for (std::size_t n = 0; n < predicted.size(); ++n) {
    const auto target = OcrText(d, static_cast<int>(n));
    errors += OcrDistance(target, predicted[n]);
    characters += static_cast<int>(target.size());
    exact += target == predicted[n];
  }
  std::cout << " " << name << ": word exact=" << exact << "/" << predicted.size()
            << ",character edit rate=" << double(errors) / characters << " (" << errors << " edits/" << characters
            << " reference characters)\n";
}

}  // namespace ocr
