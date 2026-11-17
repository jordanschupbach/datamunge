#include <gtest/gtest.h>

#include <dm/dm.hpp>
#include <fstream>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::filesystem::path
SafeFile(const std::string& name, const std::string& header, const std::vector<float>& values) {
  auto          p = std::filesystem::temp_directory_path() / name;
  std::ofstream out(p, std::ios::binary);
  std::uint64_t size = header.size();
  for (int i = 0; i < 8; ++i)
    out.put(char((size >> (8 * i)) & 255));
  out << header;
  out.write(reinterpret_cast<const char*>(values.data()), values.size() * 4);
  return p;
}
} // namespace
TEST(NnCompletion, SafeTensorsReadsOwningFloatValues) {
  auto p = SafeFile(
      "dm-safe-valid.safetensors",
      R"({"values":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
      {1.25f, -2.f});
  auto a = n::load_safetensors(p.string());
  std::filesystem::remove(p);
  ASSERT_EQ(a.size(), 1);
  EXPECT_EQ(a[0].first, "values");
  EXPECT_EQ(a[0].second.to_vector<float>(), (std::vector<float>{1.25f, -2}));
}
TEST(NnCompletion, SafeTensorsRejectsTruncationAndOverlap) {
  auto p = SafeFile(
      "dm-safe-bad.safetensors",
      R"({"values":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}})",
      {1, 2});
  EXPECT_THROW(n::load_safetensors(p.string()), std::invalid_argument);
  std::filesystem::remove(p);
  auto q = SafeFile(
      "dm-safe-overlap.safetensors",
      R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
      {1, 2});
  EXPECT_THROW(n::load_safetensors(q.string()), std::invalid_argument);
  std::filesystem::remove(q);
}
TEST(NnCompletion, ForecastLiveInputPathAndCensus) {
  n::ParameterStore p(44803);
  auto              x = n::ones({2, 12, 1}).requires_grad_();
  auto              y = m::forecast::LstmForecast(p, x);
  EXPECT_EQ(y.shape(), (n::Shape{2, 3}));
  EXPECT_EQ(p.count(), 347);
  auto g = n::grad(n::sum(y), {x})[0];
  EXPECT_GT(n::sum(n::abs(g)).item(), 0);
}
TEST(NnCompletion, MidiStepMatchesUnroll) {
  n::ParameterStore p(44903);
  auto              x = n::one_hot(n::zeros({1, 16}, n::DType::Int32), 13);
  auto              y = m::midi::MidiLstm(p, x);
  n::LstmState      state{n::zeros({1, 8}), n::zeros({1, 8})};
  for (int t = 0; t < 16; ++t) {
    auto step = m::midi::MidiLstmStep(p, n::select(x, 1, t), state);
    EXPECT_LT(n::max(n::abs(step.logits - n::select(y.logits, 1, t))).item(), 1e-6);
    state = step.state;
  }
  EXPECT_EQ(p.count(), 756);
}
TEST(NnCompletion, CtcPaddingCannotReachValidStates) {
  n::ParameterStore p(44703);
  auto x = n::ones({1, 8, 49}).requires_grad_(), length = n::full({1}, 3, n::DType::Int32);
  auto y = m::ctc_handwriting::CtcHandwriting(p, x, length);
  auto g = n::grad(n::sum(n::slice_axis(y, 0, 0, 3)), {x})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(g, 1, 3, 8))).item(), 0);
  EXPECT_EQ(p.count(), 1043);
}
TEST(NnCompletion, PlainSequenceFreezesPaddedEncoder) {
  n::ParameterStore p(44103);
  auto              x = n::zeros({1, 6, 14}).requires_grad_(), z = n::zeros({1, 8}),
       length = n::full({1}, 2, n::DType::Int32);
  auto y      = m::seq2seq::Plain(p, x, x, length, length, z, z, 14);
  auto g      = n::grad(n::sum(y.bridge_h), {x})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(g, 1, 2, 6))).item(), 0);
}
TEST(NnCompletion, BertPaddingZeroAndLiveRightContext) {
  n::ParameterStore p(46403);
  auto              x = n::one_hot(n::zeros({1, 32}, n::DType::Int32), 70).requires_grad_(),
       segments = n::zeros({1, 32}, n::DType::Int32), length = n::full({1}, 5, n::DType::Int32);
  auto y = m::bert::BertMlm(p, x, segments, length);
  EXPECT_EQ(p.count(), 1606);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(y.hidden, 1, 5, 32))).item(), 0);
  auto g = n::grad(n::sum(n::slice(y.logits, {0, 0, 0}, {1, 1, 1})), {x})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(g, 1, 5, 32))).item(), 0);
  EXPECT_GT(n::sum(n::abs(n::slice_axis(g, 1, 1, 5))).item(), 0);
}
TEST(NnCompletion, VisionPatchesRetainExactRgbOrder) {
  n::ParameterStore p(48301);
  auto              image = n::reshape(n::arange(0, 3072, 1, n::DType::Float32), {1, 32, 32, 3});
  auto              y     = m::vit::Forward(p, image);
  EXPECT_EQ(p.count(), 2994);
  auto patches = y.patches.to_vector<float>();
  EXPECT_EQ(patches[0], 0);
  EXPECT_EQ(patches[191], (7 * 32 + 7) * 3 + 2);
  EXPECT_EQ(patches[192], 8 * 3);
}
TEST(NnCompletion, SwitchCapacityUsesOnlyEligibleRows) {
  n::ParameterStore p(47103);
  auto y = m::switch_lm::Switch(p, n::ones({1, 4, 8}), n::zeros({1, 4}, n::DType::Bool), 0);
  EXPECT_EQ(n::sum(n::abs(y.fields.at("output"))).item(), 0);
  for (auto& t : y.indices)
    EXPECT_EQ(t.numel(), 0);
}
TEST(NnCompletion, RelativeMemoryIsExplicitlyDetached) {
  n::ParameterStore p(46903);
  auto              previous = n::ones({1, 3, 8}).requires_grad_();
  m::xl::Memory     memory{previous, previous};
  auto              x = n::zeros({1, 2, 65}).requires_grad_();
  auto              y = m::xl::Forward(p, x, memory);
  n::GradOptions    options;
  options.allow_unused = true;
  auto g               = n::grad(std::vector<n::Tensor>{n::sum(y.logits)}, {previous}, {}, options);
  EXPECT_TRUE(!g[0].defined() || n::sum(n::abs(g[0])).item() == 0);
}
TEST(NnCompletion, SafeTensorsRejectsDuplicateKeysAndMalformedDtype) {
  auto p = SafeFile(
      "dm-safe-duplicate.safetensors",
      R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
      {1});
  EXPECT_THROW(n::load_safetensors(p.string()), std::invalid_argument);
  std::filesystem::remove(p);
  auto q = SafeFile(
      "dm-safe-dtype.safetensors", R"({"x":{"dtype":123,"shape":[1],"data_offsets":[0,4]}})", {1});
  EXPECT_THROW(n::load_safetensors(q.string()), std::invalid_argument);
  std::filesystem::remove(q);
}
