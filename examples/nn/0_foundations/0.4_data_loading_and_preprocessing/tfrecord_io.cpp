// CMD: just run-nn tfrecord_io
// TFRecord writing, streaming reads, compression and tf.Example parsing.
//
// Port of tflow's tfrecord_io. Serialize float, integer and bytes features in
// tf.Example records. Round-trip identical payloads through plain, GZIP and
// ZLIB TFRecord files, then parse required and defaulted features with a
// ParseSingleExample-style schema. Check empty records versus EOF, embedded
// NUL bytes and an empty file. Corrupted length headers, corrupted payloads
// and truncated records must fail checksum validation. Missing required
// features must fail parsing.
//
// Differences from TF: TFRecord and tf.Example are out of the nn library's
// scope (no TensorFlow or protobuf runtime), so common/tfrecord.hpp
// implements both from the published formats: CRC-32C framing, gzip/zlib
// wrapping and a minimal Example protobuf codec. Parsing is a host function
// (no ParseSingleExample graph op); float and int64 features become nn
// tensors, bytes features std::strings (nn has no string dtype). The shared
// codec compresses GZIP/ZLIB and reads stored/fixed/dynamic DEFLATE streams,
// including ZLIB streams without a declared output size. Files go to
// output/nn/fixtures/tfrecord/.
//
// Option verify=DIR reads plain/GZIP/ZLIB files written by a TF run
// (tflow's output/fixtures/tfrecord), checking CRCs and every parsed feature.

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/tfrecord.hpp"

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using data::Feature;

namespace {

std::vector<std::string> Examples() {
  std::vector<std::string> records;
  for (int i = 0; i < 3; ++i) {
    data::Example      example;
    std::vector<float> x;
    for (int j = 0; j < 3; ++j) x.push_back(i + j * .25f);
    example["x"]     = Feature::Floats(x);
    example["label"] = Feature::Int64s({i % 2});
    example["id"]    = Feature::Bytes({"sample-" + std::to_string(i)});
    if (i != 1) example["weight"] = Feature::Floats({2});
    records.push_back(data::SerializeExample(example));
  }
  return records;
}

const std::vector<data::DenseSpec>& Schema() {
  static const std::vector<data::DenseSpec> specs{{"x", Feature::Kind::kFloat, 3, std::nullopt},
                                                  {"label", Feature::Kind::kInt64, 1, std::nullopt},
                                                  {"id", Feature::Kind::kBytes, 1, std::nullopt},
                                                  {"weight", Feature::Kind::kFloat, 1, Feature::Floats({1})}};
  return specs;
}

/// Parses `records` with the schema and checks every value.
void CheckParsed(const std::vector<std::string>& records) {
  check::Require(records.size() == 3, "three Example records");
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto      parsed = data::ParseSingleExample(records[i], Schema());
    const nn::Tensor x     = nn::from_vector(parsed[0].floats, {3});
    const nn::Tensor label = nn::from_vector(parsed[1].ints, {1}).reshape({});
    const auto       xv    = x.to_vector<float>();
    for (int j = 0; j < 3; ++j) check::Near(xv[j], i + j * .25, 0, "float feature");
    check::Require(label.item<std::int64_t>() == std::int64_t(i % 2), "integer feature");
    check::Require(parsed[2].bytes[0] == "sample-" + std::to_string(i), "bytes feature");
    check::Near(parsed[3].floats[0], i == 1 ? 1 : 2, 0, "defaulted feature");
  }
}

void ParseSchema(const std::vector<std::string>& records) {
  CheckParsed(records);
  // A valid empty Example proto lacks every required feature.
  check::Throws([] { data::ParseSingleExample("", Schema()); }, "missing required feature must fail");
  std::cout << "  parsed float[3], int64 label, bytes id and defaulted weight\n"
               "  weights=(2,1,2); missing required features rejected\n";
}

void RoundTrip() {
  const fs::path root    = paths::OutputDir("fixtures/tfrecord");
  const auto     records = Examples();
  for (const std::string compression : {"", "GZIP", "ZLIB"}) {
    const auto path = root / ((compression.empty() ? "plain" : compression) + std::string(".tfrecord"));
    data::WriteTFRecords(path, records, compression);
    const auto restored = data::ReadTFRecords(path, compression);
    check::Require(restored == records, "exact TFRecord payload round trip");
    std::cout << "  " << (compression.empty() ? "plain" : compression) << ": " << restored.size()
              << " exact records, file bytes=" << fs::file_size(path) << '\n';
  }
  const std::vector<std::string> bytes{"", std::string("a\0b", 3), "last"};
  data::WriteTFRecords(root / "bytes.tfrecord", bytes);
  check::Require(data::ReadTFRecords(root / "bytes.tfrecord") == bytes, "empty and binary records distinct from EOF");
  data::WriteTFRecords(root / "empty.tfrecord", {});
  check::Require(data::ReadTFRecords(root / "empty.tfrecord").empty(), "empty TFRecord file");
  const auto  original = data::ReadBinary(root / "plain.tfrecord");
  std::size_t failures = 0;
  for (int mode = 0; mode < 3; ++mode) {
    auto damaged = original;
    if (mode == 0) damaged[0] ^= 1;
    else if (mode == 1) damaged[12] ^= 1;
    else damaged.pop_back();
    const auto path = root / ("damaged-" + std::to_string(mode) + ".tfrecord");
    data::WriteBinary(path, damaged);
    bool rejected = false;
    try {
      data::ReadTFRecords(path);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    check::Require(rejected, "corrupt/truncated TFRecord must fail");
    ++failures;
  }
  bool rejected = false;
  try {
    data::ReadTFRecords(root / "plain.tfrecord", "unknown");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected, "unsupported compression cannot silently fall back");
  std::cout << "  empty/binary payloads and EOF passed; " << failures << " corrupt/truncated streams rejected\n";
  ParseSchema(records);
}

void VerifyTF(const fs::path& dir) {
  for (const std::string compression : {"", "GZIP", "ZLIB"}) {
    const auto path = dir / ((compression.empty() ? "plain" : compression) + std::string(".tfrecord"));
    CheckParsed(data::ReadTFRecords(path, compression));
    std::cout << "  TF-written " << path.filename().string() << " (" << fs::file_size(path)
              << " bytes): CRCs valid, every feature identical\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    fs::path verify;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("verify=", 0) == 0 && arg.size() > 7) verify = arg.substr(7);
      else throw std::invalid_argument("expected verify=DIR");
    }
    std::cout << "== TFRecord round trips and schema parsing\n";
    RoundTrip();
    if (!verify.empty()) VerifyTF(verify);
    std::cout << "All TFRecord checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
