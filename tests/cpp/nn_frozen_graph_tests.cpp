#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <datamunge/nn/frozen_graph.hpp>
#include <datamunge/nn/ops.hpp>
#include <filesystem>
#include <fstream>
#include <future>

namespace n = datamunge::nn;
namespace {
std::string V(std::uint64_t x) {
  std::string s;
  while (x > 127) {
    s.push_back(char((x & 127) | 128));
    x >>= 7;
  }
  s.push_back(char(x));
  return s;
}
std::string Int(unsigned field, std::uint64_t x) {
  return V(field * 8) + V(x);
}
std::string Blob(unsigned field, const std::string& bytes) {
  return V(field * 8 + 2) + V(bytes.size()) + bytes;
}
std::string Fixed(unsigned field, std::uint64_t bits, unsigned width) {
  auto out = V(field * 8 + (width == 4 ? 5 : 1));
  for (unsigned i = 0; i < width; ++i)
    out.push_back(char(bits >> (8 * i)));
  return out;
}
std::string Shape(const n::Shape& shape) {
  std::string out;
  for (auto size : shape)
    out += Blob(2, Int(1, std::uint64_t(size)));
  return out;
}
std::string Float(float x) {
  return Fixed(5, std::bit_cast<std::uint32_t>(x), 4);
}
std::string Double(double x) {
  return Fixed(6, std::bit_cast<std::uint64_t>(x), 8);
}
std::string Tensor(int type, const n::Shape& shape, const std::string& values) {
  return Int(1, type) + Blob(2, Shape(shape)) + values;
}
using Attrs = std::vector<std::pair<std::string, std::string>>;
std::string Node(
    const std::string&              name,
    const std::string&              op,
    const std::vector<std::string>& inputs = {},
    const Attrs&                    attrs  = {}) {
  auto out = Blob(1, name) + Blob(2, op);
  for (const auto& x : inputs)
    out += Blob(3, x);
  for (const auto& [key, value] : attrs)
    out += Blob(5, Blob(1, key) + Blob(2, value));
  return out;
}
std::string Constant(const std::string& name, const std::string& tensor) {
  return Node(name, "Const", {}, {{"value", Blob(8, tensor)}});
}
std::string F(const std::string& name, float x, n::Shape shape = {}) {
  return Constant(name, Tensor(1, shape, Float(x)));
}
std::string I(const std::string& name, std::int32_t x, n::Shape shape = {}) {
  return Constant(name, Tensor(3, shape, Int(7, std::uint64_t(std::int64_t(x)))));
}
std::string List(n::Shape x) {
  std::string out;
  for (auto v : x)
    out += V(std::uint64_t(v));
  return Blob(1, Blob(3, out));
}
std::vector<unsigned char> Bytes(const std::vector<std::string>& nodes) {
  std::string s;
  for (const auto& node : nodes)
    s += Blob(1, node);
  return {s.begin(), s.end()};
}
n::FrozenGraph Graph(const std::vector<std::string>& nodes) {
  return n::FrozenGraph(Bytes(nodes));
}
n::Tensor
Get(const n::FrozenGraph& graph, const std::string& fetch, const n::NamedTensors& feeds = {}) {
  return graph.evaluate(feeds, {fetch})[0].second;
}
std::string Placeholder(const std::string& name, int dtype = 1) {
  return Node(name, "Placeholder", {}, {{"dtype", Int(6, dtype)}});
}
Attrs Spatial(
    n::Shape    window  = {1, 2, 2, 1},
    n::Shape    strides = {1, 1, 1, 1},
    std::string padding = "SAME") {
  return {
      {"ksize", List(window)},
      {"strides", List(strides)},
      {"padding", Blob(2, padding)},
      {"T", Int(6, 1)}};
}
} // namespace

TEST(NnFrozenGraph, NumericTensorProtoPreservesScalarRankRawBitsAndWideIntegers) {
  std::string raw;
  auto        wide = std::uint64_t(9007199254740993LL);
  for (unsigned i = 0; i < 8; ++i)
    raw.push_back(char(wide >> (8 * i)));
  auto g = Graph(
      {F("scalar", -0.f),
       Constant("double", Tensor(2, {2}, Double(1.25) + Double(-3.5))),
       Constant("wide", Tensor(9, {}, Blob(4, raw))),
       Constant("bool", Tensor(10, {2}, Int(11, 1) + Int(11, 0)))});
  auto x = g.constant("scalar");
  EXPECT_EQ(x.dim(), 0);
  EXPECT_TRUE(std::signbit(x.item()));
  EXPECT_EQ(g.constant("double").to_vector<double>(), (std::vector<double>{1.25, -3.5}));
  EXPECT_EQ(g.constant("wide").item<std::int64_t>(), 9007199254740993LL);
  EXPECT_EQ(g.constant("bool").dtype(), n::DType::Bool);
  EXPECT_EQ(g.constant("bool").to_vector<bool>(), (std::vector<bool>{true, false}));
}
TEST(NnFrozenGraph, RepeatedFieldsSupportPackedUnpackedLastValueFillAndEmptyAxes) {
  auto g = Graph(
      {F("fill", 2, {4}),
       Constant("packed", Tensor(3, {3}, Blob(7, V(1) + V(2) + V(3)))),
       Constant("short", Tensor(2, {4}, Double(1) + Double(2))),
       Constant("empty", Tensor(1, {0, 3}, "")),
       Constant("zero", Tensor(1, {3}, ""))});
  EXPECT_EQ(g.constant("fill").to_vector<float>(), (std::vector<float>{2, 2, 2, 2}));
  EXPECT_EQ(g.constant("packed").to_vector<std::int32_t>(), (std::vector<std::int32_t>{1, 2, 3}));
  EXPECT_EQ(g.constant("short").to_vector<double>(), (std::vector<double>{1, 2, 2, 2}));
  EXPECT_EQ(g.constant("empty").shape(), (n::Shape{0, 3}));
  EXPECT_EQ(g.constant("zero").to_vector<float>(), (std::vector<float>{0, 0, 0}));
}
TEST(NnFrozenGraph, ConstructorBoundsRejectMalformedWireNamesShapesAndPayloads) {
  EXPECT_THROW(n::FrozenGraph(std::vector<unsigned char>{}), std::invalid_argument);
  EXPECT_THROW(n::FrozenGraph(std::vector<unsigned char>{10, 255}), std::invalid_argument);
  EXPECT_THROW(n::FrozenGraph(std::vector<unsigned char>{0}), std::invalid_argument);
  EXPECT_THROW(
      n::FrozenGraph(std::vector<unsigned char>{8, 128, 128, 128, 128, 128, 128, 128, 128, 128, 2}),
      std::invalid_argument);
  EXPECT_THROW(Graph({F("x", 1), F("x", 2)}), std::invalid_argument);
  EXPECT_THROW(Graph({F("bad name", 1)}), std::invalid_argument);
  EXPECT_THROW(Graph({F("_bad", 1)}), std::invalid_argument);
  EXPECT_THROW(Graph({Node("a", "Identity", {"absent"})}), std::invalid_argument);
  EXPECT_THROW(Graph({Constant("x", Tensor(1, {-1}, Float(1)))}), std::invalid_argument);
  EXPECT_THROW(Graph({Constant("x", Tensor(1, {1}, Float(1) + Float(2)))}), std::invalid_argument);
  EXPECT_THROW(Graph({Constant("x", Tensor(1, {2}, Blob(4, "xx")))}), std::invalid_argument);
  EXPECT_THROW(Graph({Constant("x", Tensor(10, {1}, Int(11, 2)))}), std::invalid_argument);
  auto b = Bytes({F("x", 1, {1000})});
  EXPECT_THROW(n::FrozenGraph(b, 100), std::invalid_argument);
  EXPECT_THROW(n::FrozenGraph(b, 0), std::invalid_argument);
  EXPECT_THROW(Graph({Constant("x", Tensor(1, {INT64_MAX, 2}, Float(1)))}), std::invalid_argument);
}
TEST(NnFrozenGraph, FeedCutsUnsupportedAncestryAndResultsAlwaysOwnDetachedCopies) {
  auto g = Graph(
      {Node("opaque", "DecodeJpeg"),
       Node("fed", "Identity", {"opaque"}, {{"T", Int(6, 1)}}),
       Node("out", "Relu", {"fed"}, {{"T", Int(6, 1)}})});
  auto x   = n::tensor({-1., 2.}, {2}).requires_grad_();
  auto out = g.evaluate({{"fed:0", x}}, {"out:0", "fed", "out"});
  EXPECT_EQ(out.size(), 3);
  EXPECT_EQ(out[0].first, "out:0");
  EXPECT_FALSE(out[0].second.requires_grad());
  EXPECT_FALSE(out[1].second.requires_grad());
  out[0].second.data_ptr<float>()[1] = 7;
  EXPECT_EQ(out[2].second.at({1}), 2);
  EXPECT_EQ(x.at({1}), 2);
  EXPECT_THROW(Get(g, "out"), std::invalid_argument);
}
TEST(NnFrozenGraph, ConstantAndReturnedFeedMutationsCannotAlterTheModel) {
  auto g                      = Graph({F("c", 2)});
  auto copy                   = g;
  auto c                      = g.constant("c");
  c.data_ptr<float>()[0]      = 100;
  auto output                 = Get(g, "c");
  output.data_ptr<float>()[0] = 200;
  EXPECT_EQ(Get(copy, "c").item(), 2);
  auto feed                   = n::full({}, 3).requires_grad_();
  auto result                 = Get(g, "c", {{"c", feed}});
  result.data_ptr<float>()[0] = 5;
  EXPECT_EQ(feed.item(), 3);
  EXPECT_EQ(Get(g, "c").item(), 2);
}
TEST(NnFrozenGraph, ControlDependencyRunsCheckNumericsAndCyclesRejectBeforeExecution) {
  auto g = Graph(
      {Placeholder("x"),
       Node("check", "CheckNumerics", {"x"}, {{"T", Int(6, 1)}, {"message", Blob(2, "bad input")}}),
       F("safe", 2),
       Node("out", "Identity", {"safe", "^check"})});
  EXPECT_THROW(Get(g, "out", {{"x", n::full({}, NAN)}}), std::invalid_argument);
  EXPECT_EQ(Get(g, "out", {{"x", n::full({}, 3)}}).item(), 2);
  auto cyclic = Graph({Node("a", "Identity", {"b"}), Node("b", "Identity", {"a"})});
  EXPECT_THROW(Get(cyclic, "a"), std::invalid_argument);
  EXPECT_EQ(Get(cyclic, "a", {{"a", n::full({}, 7)}}).item(), 7);
}
TEST(NnFrozenGraph, InvalidFeedsPortsMetadataDtypesAndMissingPlaceholdersReject) {
  auto g = Graph(
      {Placeholder("x"),
       I("shape", 2, {1}),
       Node("out", "Reshape", {"x", "shape"}, {{"T", Int(6, 1)}})});
  EXPECT_THROW(Get(g, "out"), std::invalid_argument);
  EXPECT_THROW(Get(g, "out", {{"x", n::Tensor()}}), std::invalid_argument);
  EXPECT_THROW(Get(g, "out", {{"x", n::zeros({2}, n::DType::Float64)}}), std::invalid_argument);
  EXPECT_THROW(
      Get(g, "out", {{"x", n::zeros({2})}, {"x:0", n::zeros({2})}}), std::invalid_argument);
  EXPECT_THROW(g.operation("x:1"), std::invalid_argument);
  EXPECT_THROW(g.operation("^x"), std::invalid_argument);
  EXPECT_THROW(g.constant("x"), std::invalid_argument);
  EXPECT_TRUE(g.evaluate({}, {}).empty());
  auto conflict =
      Node("c", "Const", {}, {{"value", Blob(8, Tensor(1, {}, Float(2)))}, {"dtype", Int(6, 2)}});
  EXPECT_THROW(Graph({conflict}), std::invalid_argument);
}
TEST(NnFrozenGraph, ReshapeExpandSqueezeAndBothConcatOrdersKeepExactAxes) {
  auto g = Graph(
      {Placeholder("x"),
       I("shape", 2, {1}),
       I("axis", 0),
       Node("flat", "Reshape", {"x", "shape"}),
       Node("expand", "ExpandDims", {"flat", "axis"}),
       Node("sq", "Squeeze", {"expand"}, {{"squeeze_dims", List({0})}}),
       Node("old", "Concat", {"axis", "flat", "sq"}, {{"N", Int(3, 2)}}),
       Node("new", "ConcatV2", {"flat", "sq", "axis"}, {{"N", Int(3, 2)}})});
  auto x = n::tensor({1., 2.}, {1, 2});
  EXPECT_EQ(Get(g, "old", {{"x", x}}).to_vector<float>(), (std::vector<float>{1, 2, 1, 2}));
  EXPECT_TRUE(n::equal_bits(Get(g, "old", {{"x", x}}), Get(g, "new", {{"x", x}})));
}
TEST(NnFrozenGraph, NativeArithmeticMatMulTransposeBiasAndSoftmax) {
  auto g = Graph(
      {Placeholder("x"),
       F("scale", 2),
       F("shift", 1),
       F("w", 1, {2, 2}),
       F("bias", .5, {2}),
       Node("mul", "Mul", {"x", "scale"}),
       Node("sub", "Sub", {"mul", "shift"}),
       Node("div", "RealDiv", {"sub", "scale"}),
       Node("mm", "MatMul", {"div", "w"}, {{"transpose_a", Int(5, 1)}}),
       Node("bias_add", "BiasAdd", {"mm", "bias"}),
       Node("out", "Softmax", {"bias_add"})});
  auto x      = n::tensor({1., 2., 3., 4.}, {2, 2});
  auto logits = Get(g, "bias_add", {{"x", x}});
  EXPECT_EQ(logits.to_vector<float>(), (std::vector<float>{3.5, 3.5, 5.5, 5.5}));
  EXPECT_TRUE(n::equal_bits(Get(g, "out", {{"x", x}}), n::full({2, 2}, .5)));
}
TEST(NnFrozenGraph, LegacyBatchNormHonorsFloatEpsilonAndOptionalGamma) {
  auto attrs        = Attrs{{"variance_epsilon", Fixed(4, std::bit_cast<std::uint32_t>(1.f), 4)}};
  auto attrs_scaled = attrs;
  attrs_scaled.emplace_back("scale_after_normalization", Int(5, 1));
  auto g = Graph(
      {Placeholder("x"),
       F("mean", 1, {1}),
       F("var", 3, {1}),
       F("beta", 5, {1}),
       F("gamma", 3, {1}),
       Node(
           "plain",
           "BatchNormWithGlobalNormalization",
           {"x", "mean", "var", "beta", "gamma"},
           attrs),
       Node(
           "scaled",
           "BatchNormWithGlobalNormalization",
           {"x", "mean", "var", "beta", "gamma"},
           attrs_scaled)});
  auto input = n::full({1, 2, 3, 1}, 2);
  EXPECT_EQ(Get(g, "plain", {{"x", input}}).at({0, 0, 0, 0}), 5.5);
  EXPECT_EQ(Get(g, "scaled", {{"x", input}}).at({0, 0, 0, 0}), 6.5);
}
TEST(NnFrozenGraph, NHWCConvolutionPoolingAndLegacyResizeUseNativeGeometry) {
  auto attrs     = Spatial({1, 2, 2, 1}, {1, 1, 1, 1}, "SAME");
  auto convattrs = Spatial({1, 1, 1, 1}, {1, 1, 1, 1}, "SAME");
  auto g         = Graph(
      {Placeholder("x"),
               F("w", 2, {1, 1, 1, 1}),
               Node("conv", "Conv2D", {"x", "w"}, convattrs),
               Node("avg", "AvgPool", {"conv"}, attrs),
               Node("max", "MaxPool", {"conv"}, attrs),
               I("size", 2, {2}),
               Node("resize", "ResizeBilinear", {"x", "size"})});
  auto x = n::tensor({1., 2., 3., 4.}, {1, 2, 2, 1});
  EXPECT_EQ(Get(g, "avg", {{"x", x}}).to_vector<float>(), (std::vector<float>{5, 6, 7, 8}));
  EXPECT_EQ(Get(g, "max", {{"x", x}}).to_vector<float>(), (std::vector<float>{8, 8, 8, 8}));
  EXPECT_TRUE(n::equal_bits(Get(g, "resize", {{"x", x}}), x));
  auto bad = attrs;
  bad.emplace_back("data_format", Blob(2, "NCHW"));
  auto invalid = Graph({Placeholder("x"), Node("pool", "AvgPool", {"x"}, bad)});
  EXPECT_THROW(Get(invalid, "pool", {{"x", x}}), std::invalid_argument);
}
TEST(NnFrozenGraph, ReadOnlyEvaluationsMayRunConcurrentlyAndFileLoadIsBounded) {
  auto bytes = Bytes({F("c", 2)});
  auto g     = n::FrozenGraph(bytes);
  auto a     = std::async(std::launch::async, [&] { return Get(g, "c").item(); });
  auto b     = std::async(std::launch::async, [&] { return Get(g, "c").item(); });
  EXPECT_EQ(a.get(), 2);
  EXPECT_EQ(b.get(), 2);
  auto file = std::filesystem::temp_directory_path() /
              ("dm-frozen-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".pb");
  {
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  auto loaded = n::FrozenGraph::load(file.string());
  EXPECT_EQ(loaded.node_names(), (std::vector<std::string>{"c"}));
  EXPECT_EQ(loaded.operation("c"), "Const");
  EXPECT_EQ(Get(loaded, "c").item(), 2);
  EXPECT_THROW(n::FrozenGraph::load(file.string(), 2), std::invalid_argument);
  std::filesystem::remove(file);
  EXPECT_THROW(n::FrozenGraph::load(file.string()), std::runtime_error);
}
