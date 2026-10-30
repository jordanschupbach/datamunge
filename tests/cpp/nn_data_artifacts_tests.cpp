#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <limits>

namespace n = datamunge::nn;
namespace d = n::data;
namespace t = d::text;

TEST(NnDataFormats, CsvQuotingAndMalformedInput) {
  const auto x = d::ParseCsv(
      "\xef\xbb\xbf"
      "a,b\r\n\"x,y\",\"say \"\"hi\"\"\"\r\n\"two\nlines\",\r\n");
  EXPECT_EQ(x.At(0, 0), "x,y");
  EXPECT_EQ(x.At(0, 1), "say \"hi\"");
  EXPECT_EQ(x.At(1, 0), "two\nlines");
  EXPECT_EQ(x.At(1, 1), "");
  for (const auto bad : {"", "a,a\n1,2", "a,b\n1", "a\n\"open", "a\n\"x\"tail", "a\r1"})
    EXPECT_THROW(d::ParseCsv(bad), std::invalid_argument);
  EXPECT_EQ(d::ParseCsv("a;b\n1;2", ';').At(0, 1), "2");
}
TEST(NnDataFormats, NumericStrictness) {
  EXPECT_DOUBLE_EQ(*d::Numeric(" +1.25e2 "), 125);
  EXPECT_FALSE(d::Numeric(" \t"));
  for (const auto bad : {"NaN", "inf", "1x", "1e999", "+-1"})
    EXPECT_THROW(d::Numeric(bad), std::invalid_argument);
}
TEST(NnDataFormats, TrainingOnlyMomentsVocabularyAndHeaderOrder) {
  auto       x = d::ParseCsv("x,city\n1,B\n3,A\n,\n999,C\n,\n");
  const auto f = d::FeatureColumns::Fit(
      x, {0, 1, 2}, {{"x", d::FeatureKind::kNumeric}, {"city", d::FeatureKind::kCategorical}});
  EXPECT_DOUBLE_EQ(f.features()[0].mean, 2);
  EXPECT_DOUBLE_EQ(f.features()[0].scale, 1);
  EXPECT_EQ(f.features()[1].vocabulary, (std::vector<std::string>{"A", "B"}));
  const auto v = f.Transform(x, {3, 4});
  EXPECT_EQ(v.shape(), (n::Shape{2, 6}));
  EXPECT_EQ(v.at({0, 0}), 997);
  EXPECT_EQ(v.at({0, 3}), 1);
  EXPECT_EQ(v.at({1, 1}), 1);
  EXPECT_EQ(v.at({1, 2}), 1);
  std::swap(x.header[0], x.header[1]);
  for (auto& row : x.rows)
    std::swap(row[0], row[1]);
  EXPECT_EQ(f.Transform(x, {3, 4}).to_vector<float>(), v.to_vector<float>());
  EXPECT_EQ(f.Transform(x, {}).shape(), (n::Shape{0, 6}));
}
TEST(NnDataFormats, CifarCoordinateOracleAndLabels) {
  for (const auto kind : {d::Cifar::k10, d::Cifar::k100}) {
    d::Bytes bytes;
    if (kind == d::Cifar::k100)
      bytes.push_back(4);
    bytes.push_back(7);
    for (int c = 0; c < 3; ++c)
      for (int p = 0; p < 1024; ++p)
        bytes.push_back((61 * c + p) % 256);
    const auto x = d::ParseCifar({bytes}, kind);
    EXPECT_EQ(x.labels.item<int>(), 7);
    EXPECT_EQ(x.images.dtype(), n::DType::Int32);
    if (kind == d::Cifar::k100)
      EXPECT_EQ(x.coarse_labels.item<int>(), 4);
    for (int c = 0; c < 3; ++c)
      for (int p = 0; p < 1024; ++p)
        EXPECT_EQ(x.images.at({0, p / 32, p % 32, c}), (61 * c + p) % 256);
    auto bad = bytes;
    bad.pop_back();
    EXPECT_THROW(d::ParseCifar({bad}, kind), std::invalid_argument);
    bad    = bytes;
    bad[0] = 255;
    EXPECT_THROW(d::ParseCifar({bad}, kind), std::invalid_argument);
  }
}
TEST(NnDataFormats, UnicodeScalarsAndInvalidEncodings) {
  EXPECT_EQ(t::Characters("café😀").size(), 5);
  EXPECT_EQ(t::Whitespace("a\tB!\n"), (std::vector<std::string>{"a", "B!"}));
  for (const std::string bad :
       {std::string("\xc0\x80", 2),
        std::string("\xed\xa0\x80", 3),
        std::string("\xf4\x90\x80\x80", 4),
        std::string("\xe2\x82", 2)})
    EXPECT_THROW(t::Characters(bad), std::invalid_argument);
}
TEST(NnDataFormats, BpeFrequencyEndMarkerTieAndReload) {
  const auto unique = t::Bpe::Train({{"aaa", 3}, {"ab", 1}}, 1);
  EXPECT_EQ(unique.merges()[0], (t::Bpe::Pair{"a", "a"}));
  const auto tie = t::Bpe::Train({{"aa", 3}, {"ab", 1}}, 1);
  EXPECT_EQ(tie.merges()[0], (t::Bpe::Pair{"a", "</w>"}));
  const auto copy = t::Bpe::Load(unique.Save());
  for (const auto w : {"aa", "aaa", "ab", "z"})
    EXPECT_EQ(unique.EncodeWord(w), copy.EncodeWord(w));
  EXPECT_THROW(t::Bpe::Train({{"a", 0}}, 1), std::invalid_argument);
  EXPECT_THROW(t::Bpe::Load(unique.Save() + "junk"), std::invalid_argument);
}
TEST(NnDataFormats, WordPieceWholeWordUnknownAndMask) {
  const t::WordPiece wp({"[PAD]", "[UNK]", "play", "##ing", "ball"});
  EXPECT_EQ(wp.Encode("playing playx ball"), (std::vector<int32_t>{2, 3, 1, 4}));
  const auto b = t::PadTokens({{2, 3, 4}, {1}, {}}, 2);
  EXPECT_EQ(b.ids.to_vector<int>(), (std::vector<int>{2, 3, 1, 0, 0, 0}));
  EXPECT_EQ(b.mask.to_vector<int>(), (std::vector<int>{1, 1, 1, 0, 0, 0}));
  EXPECT_THROW(t::PadTokens({{-1}}, 2), std::invalid_argument);
}
TEST(NnDataFormats, Crc32cKnownVectorAndProtoSchema) {
  const std::string s = "123456789";
  EXPECT_EQ(d::Crc32c(reinterpret_cast<const uint8_t*>(s.data()), s.size()), 0xe3069283u);
  const d::Example e{
      {"x", d::Feature::Floats({1, .5f})},
      {"y", d::Feature::Int64s({-7, INT64_MAX})},
      {"id", d::Feature::Bytes({std::string("a\0b", 3)})}};
  const auto decoded = d::ParseExample(d::SerializeExample(e));
  EXPECT_EQ(decoded.at("y").ints, (std::vector<int64_t>{-7, INT64_MAX}));
  EXPECT_EQ(decoded.at("id").bytes[0], std::string("a\0b", 3));
  const std::vector<d::DenseSpec> schema{
      {"x", d::Feature::Kind::kFloat, 2, std::nullopt},
      {"weight", d::Feature::Kind::kFloat, 1, d::Feature::Floats({1})}};
  EXPECT_EQ(d::ParseSingleExample(d::SerializeExample(e), schema)[1].floats[0], 1);
  EXPECT_THROW(d::ParseSingleExample("", schema), std::invalid_argument);
  EXPECT_THROW(
      d::ParseSingleExample(
          d::SerializeExample(e), {{"x", d::Feature::Kind::kFloat, 1, std::nullopt}}),
      std::invalid_argument);
}
TEST(NnDataFormats, FramingCompressionBinaryPayloadAndCorruption) {
  const auto root = std::filesystem::temp_directory_path() / "datamunge-data-formats-gtest";
  std::filesystem::create_directories(root);
  const std::vector<std::string> records{"", std::string("a\0b", 3), "last"};
  for (const auto mode : {"", "GZIP", "ZLIB"}) {
    const auto p = root / (std::string("records") + mode);
    d::WriteTFRecords(p, records, mode);
    EXPECT_EQ(d::ReadTFRecords(p, mode), records);
  }
  auto raw = d::ReadBinary(root / "records");
  EXPECT_EQ(raw.size(), 55);
  raw[28] ^= 1;
  d::WriteBinary(root / "broken", raw);
  EXPECT_THROW(d::ReadTFRecords(root / "broken"), std::runtime_error);
  EXPECT_THROW(d::ReadBinary(root / "records", 4), std::runtime_error);
  std::filesystem::remove_all(root);
}
TEST(NnArtifacts, RestorationValidatesBeforeAssigningAndStagesAliases) {
  auto a = n::tensor({1, 2}), b = n::tensor({3, 4});
  EXPECT_THROW(
      n::restore_tensors({{"a", a}, {"b", b}}, {{"a", n::tensor({9, 9})}, {"b", n::tensor({1})}}),
      std::invalid_argument);
  EXPECT_EQ(a.at({0}), 1);
  n::restore_tensors({{"a", a}, {"b", b}}, {{"a", b}, {"b", a}});
  EXPECT_EQ(a.at({0}), 3);
  EXPECT_EQ(b.at({0}), 1);
  EXPECT_THROW(n::restore_tensors({{"a", a}}, {{"a", a}, {"extra", b}}), std::invalid_argument);
  EXPECT_THROW(n::restore_tensors({{"a", a}}, {{"a", a}, {"a", a}}), std::invalid_argument);
}
TEST(NnArtifacts, StringsPreserveArbitraryBytesAndRejectOffsets) {
  const std::vector<std::string> words{"café", std::string("a\0b", 3), "", std::string("\xff", 1)};
  const auto                     encoded = n::encode_strings(words);
  EXPECT_EQ(n::decode_strings(encoded), words);
  EXPECT_TRUE(n::decode_strings(n::encode_strings({})).empty());
  auto bad      = encoded;
  bad[1].second = n::from_vector(std::vector<int64_t>{0, 5, 3, 8, 9}, {5});
  EXPECT_THROW(n::decode_strings(bad), std::invalid_argument);
  bad           = encoded;
  bad[0].second = n::full({9}, 256, n::DType::Int32);
  EXPECT_THROW(n::decode_strings(bad), std::invalid_argument);
}
TEST(NnArtifacts, ConnectivityIncludesZeroDerivativesAndSnapshotIsIndependent) {
  n::ParameterStore s;
  auto              w = s.get("w", {}, n::init::ones());
  s.get("unused", {}, n::init::ones());
  const auto frozen = n::inference_parameters(s, {w * 0});
  ASSERT_EQ(frozen.size(), 1);
  EXPECT_EQ(frozen[0].first, "w");
  EXPECT_FALSE(frozen[0].second.requires_grad());
  {
    n::NoGradGuard guard;
    w.copy_(n::scalar(5));
  }
  EXPECT_EQ(frozen[0].second.item(), 1);
  EXPECT_THROW(n::inference_parameters(s, {}), std::invalid_argument);
}
TEST(NnArtifacts, AffineContractSupportsDynamicBatchAndRejectsDtypeShape) {
  const auto w = n::tensor({.75, -1.5}, {2, 1}), b = n::tensor({.25});
  EXPECT_EQ(n::affine_inference(n::tensor({{2, -3}}), w, b).item(), 6.25);
  EXPECT_EQ(n::affine_inference(n::zeros({3, 2}), w, b).shape(), (n::Shape{3, 1}));
  EXPECT_THROW(n::affine_inference(n::zeros({1, 3}), w, b), std::invalid_argument);
  EXPECT_THROW(
      n::affine_inference(n::zeros({1, 2}, n::DType::Float64), w, b), std::invalid_argument);
  EXPECT_THROW(n::affine_inference({}, w, b), std::invalid_argument);
}
