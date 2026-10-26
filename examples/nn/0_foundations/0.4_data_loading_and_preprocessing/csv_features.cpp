// CMD: just run-nn csv_features
// CSV parsing and training-only numeric/categorical feature columns.
//
// Port of tflow's csv_features. Read a table containing quoted commas, escaped
// quotes, embedded newlines, CRLF, missing values and held-out categories.
// Fit numeric means/scales and sorted categorical vocabularies on four
// training rows, then apply that fixed transformation to two validation rows.
// Numeric missingness and categorical missing/unknown values have separate
// indicator columns. Column lookup uses header names, so a reordered input
// table produces the same model features.
//
// Check exact numeric moments and one-hot positions, verify validation values
// cannot change fitted state, and feed the dense float32 matrix to an nn
// linear prediction. This example demonstrates preprocessing, not model
// generalization. The shared loader also accepts a caller-selected delimiter.
//
// Differences from TF: the parser and feature columns live in
// common/csv.hpp (tflow's common/data/csv.h, returning nn tensors); the
// prediction is an eager nn::matmul instead of a session graph. The fixture
// is written to output/nn/fixtures/tabular/features.csv.

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/csv.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void ParserChecks() {
  const auto quoted = data::ParseCsv("a,b\r\n\"one, two\",\"say \"\"hi\"\"\"\r\n\"line1\nline2\",\"\"\r\n");
  check::Require(quoted.rows.size() == 2 && quoted.At(0, 0) == "one, two" && quoted.At(0, 1) == "say \"hi\"" &&
                 quoted.At(1, 0) == "line1\nline2" && quoted.At(1, 1).empty(), "quoted CSV fields");
  check::Require(data::ParseCsv("\xef\xbb\xbf" "a;b\n1;2", ';').At(0, 1) == "2", "BOM and alternate delimiter");
  check::Require(data::ParseCsv("a,b\n,\n").rows == std::vector<std::vector<std::string>>({{"", ""}}), "empty fields and terminal newline");
  check::Require(data::ParseCsv("a\n  x  ").At(0, 0) == "  x  ", "preserve whitespace");
  size_t rejected = 0;
  for (const std::string malformed : {"", "a,a\n1,2", ",b\n1,2", "a,b\n1", "a\n1,2",
                                     "a\n\"open", "a\nraw\"quote", "a\n\"closed\"tail", "a\r1"}) {
    bool bad = false;
    try { data::ParseCsv(malformed); } catch (const std::invalid_argument&) { bad = true; }
    check::Require(bad, "malformed CSV must fail"); ++rejected;
  }
  for (const std::string invalid : {"1x", "NaN", "inf", "1e999", "+-1"}) {
    bool bad = false;
    try { data::Numeric(invalid); } catch (const std::invalid_argument&) { bad = true; }
    check::Require(bad, "invalid/nonfinite numeric must fail"); ++rejected;
  }
  check::Near(*data::Numeric(" +1.25e2 "), 125, 0, "numeric whitespace/sign/exponent");
  check::Require(!data::Numeric(" \t"), "empty numeric cell is missing");
  std::cout << "  quoting, multiline fields, CRLF/LF, BOM, delimiters and empty cells passed\n"
            << "  " << rejected << " malformed CSV/numeric cases rejected\n";
}

void Features() {
  const std::string text =
      "age,income,city,target,note\r\n"
      "20,100,Denver,0,first\r\n"
      "30,200,Boulder,1,\"said \"\"hello\"\"\"\r\n"
      ",300,Denver,0,\"two\nlines\"\r\n"
      "50,400,Austin,1,last training row\r\n"
      "1000,500,\"Rome, IT\",1,validation only\r\n"
      ",,,0,validation missing\r\n";
  const auto path = paths::OutputDir("fixtures/tabular") / "features.csv";
  data::WriteBinary(path, data::Bytes(text.begin(), text.end()));
  const auto table = data::LoadCsv(path);
  const std::vector<data::FeatureSpec> specs{{"age", data::FeatureKind::kNumeric},
      {"income", data::FeatureKind::kNumeric}, {"city", data::FeatureKind::kCategorical}};
  const auto features = data::FeatureColumns::Fit(table, {0, 1, 2, 3}, specs);
  check::Near(features.features()[0].mean, 100. / 3, 1e-12, "training age mean excludes validation");
  check::Near(features.features()[0].scale, std::sqrt(1400. / 9), 1e-12, "observed population age deviation");
  check::Near(features.features()[1].mean, 250, 0, "training income mean");
  check::Near(features.features()[1].scale, std::sqrt(12500.), 1e-12, "population income deviation");
  check::Require(features.features()[2].vocabulary == std::vector<std::string>({"Austin", "Boulder", "Denver"}), "training-only sorted vocabulary");
  const auto training = features.Transform(table, {0, 1, 2, 3}), validation = features.Transform(table, {4, 5});
  check::Require(training.shape() == nn::Shape({4, 9}) && validation.shape() == nn::Shape({2, 9}), "feature dimensions");
  const auto tr = training.to_vector<float>(), va = validation.to_vector<float>();
  check::Near(tr[2 * 9 + 0], 0, 0, "impute standardized mean");
  check::Near(tr[2 * 9 + 1], 1, 0, "numeric missing indicator");
  check::Near(va[0], (1000 - 100. / 3) / std::sqrt(1400. / 9), 1e-5, "held-out numeric transform");
  check::Near(va[5], 1, 0, "unknown category bucket");
  check::Near(va[9 + 4], 1, 0, "missing category bucket");
  auto reordered = table;
  std::swap(reordered.header[0], reordered.header[2]);
  for (auto& row : reordered.rows) std::swap(row[0], row[2]);
  const auto transformed = features.Transform(reordered, {4, 5});
  check::Require(transformed.to_vector<float>() == va, "column reorder must preserve features");
  const auto sparse = data::ParseCsv("constant,missing\n4,\n4,\n");
  const auto fixed = data::FeatureColumns::Fit(sparse, {0, 1}, {{"constant", data::FeatureKind::kNumeric}, {"missing", data::FeatureKind::kNumeric}});
  const auto zeros = fixed.Transform(sparse, {0, 1}).to_vector<float>();
  for (int row = 0; row < 2; ++row) {
    check::Near(zeros[row * 4 + 0], 0, 0, "constant column");
    check::Near(zeros[row * 4 + 2], 0, 0, "all-missing imputation");
    check::Near(zeros[row * 4 + 3], 1, 0, "all-missing indicator");
  }
  const Tensor weights    = nn::full({9, 1}, .1);
  const auto   out        = nn::matmul(validation, weights).to_vector<float>();
  for (int row = 0; row < 2; ++row) {
    double expected = 0;
    for (int col = 0; col < 9; ++col) expected += va[row * 9 + col] * .1;
    check::Near(out[row], expected, 1e-6, "nn consumes feature matrix");
  }
  std::cout << "  training rows=4, validation rows=2; features=" << features.names().size() << '\n';
  for (const auto& name : features.names()) std::cout << "    " << name << '\n';
  std::cout << "  training age mean=" << features.features()[0].mean << ", scale=" << features.features()[0].scale
            << "; held-out age 1000 -> " << va[0] << '\n'
            << "  unknown/missing buckets, fixed statistics and reordered columns passed\n"
            << "  nn validation predictions=(" << out[0] << ", " << out[1] << ")\n";
}

}  // namespace

int main() {
  try {
    nn::set_num_threads(2);
    std::cout << "== CSV parser checks\n";
    ParserChecks();
    std::cout << "\n== Fitted feature columns\n";
    Features();
    std::cout << "All CSV and feature-column checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
