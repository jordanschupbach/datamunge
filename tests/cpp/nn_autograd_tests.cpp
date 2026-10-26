// Autograd tests for datamunge::nn: a finite-difference gradient check for
// every differentiable op (float64), second-order checks, and engine behaviour.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <string>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

using Inputs = std::vector<Tensor>;

std::vector<double> vals(const Tensor& t) { return t.to_vector<double>(); }

/// float64 leaf with values in [low, high).
Tensor input(const nn::Shape& shape, double low, double high, std::uint64_t seed) {
  nn::Generator gen(seed);
  return nn::uniform(shape, low, high, gen, DType::Float64).requires_grad_();
}

/// float64 leaf with |x| in [0.2, 1.2): keeps away from kinks at 0.
Tensor away_from_zero(const nn::Shape& shape, std::uint64_t seed) {
  nn::Generator gen(seed);
  Tensor        u = nn::uniform(shape, 0.2, 1.2, gen, DType::Float64);
  Tensor        s = nn::where(nn::less(nn::uniform(shape, 0.0, 1.0, gen, DType::Float64), 0.5), -1.0, nn::ones(shape, DType::Float64));
  return (u * s).detach().requires_grad_();
}

/// float64 leaf holding distinct values (a shuffled grid), for max/min/top_k.
Tensor distinct(const nn::Shape& shape, std::uint64_t seed) {
  nn::Generator      gen(seed);
  const std::int64_t n = nn::shape_numel(shape);
  Tensor             grid = nn::arange(0, static_cast<double>(n), 1, DType::Float64) * 0.37 - 1.0;
  return nn::gather(grid, nn::permutation(n, gen)).reshape(shape).detach().requires_grad_();
}

struct Case {
  std::string        name;
  nn::TensorFunction f;
  Inputs             inputs;
};

void expect_gradcheck(const Case& c, nn::GradcheckOptions opts = {}) {
  const auto r = nn::gradcheck(c.f, c.inputs, opts);
  EXPECT_TRUE(r.ok) << c.name << ": " << r.message << " (max error " << r.max_abs_error << ")";
}

void expect_gradgradcheck(const Case& c, nn::GradcheckOptions opts = {}) {
  opts.atol = std::max(opts.atol, 1e-4);
  const auto r = nn::gradgradcheck(c.f, c.inputs, opts);
  EXPECT_TRUE(r.ok) << c.name << " (second order): " << r.message << " (max error " << r.max_abs_error << ")";
}

using Unary = std::function<Tensor(const Tensor&)>;

Case unary(const std::string& name, Unary op, Tensor x) {
  return {name, [op](const Inputs& in) { return op(in[0]); }, {x}};
}

using Binary = std::function<Tensor(const Tensor&, const Tensor&)>;

Case binary(const std::string& name, Binary op, Tensor a, Tensor b) {
  return {name, [op](const Inputs& in) { return op(in[0], in[1]); }, {a, b}};
}

std::vector<Case> unary_cases() {
  const nn::Shape s{2, 3};
  return {
      unary("neg", [](const Tensor& x) { return nn::neg(x); }, input(s, -2, 2, 1)),
      unary("square", [](const Tensor& x) { return nn::square(x); }, input(s, -2, 2, 2)),
      unary("sqrt", [](const Tensor& x) { return nn::sqrt(x); }, input(s, 0.5, 3, 3)),
      unary("rsqrt", [](const Tensor& x) { return nn::rsqrt(x); }, input(s, 0.5, 3, 4)),
      unary("reciprocal", [](const Tensor& x) { return nn::reciprocal(x); }, input(s, 0.5, 3, 5)),
      unary("exp", [](const Tensor& x) { return nn::exp(x); }, input(s, -2, 2, 6)),
      unary("log", [](const Tensor& x) { return nn::log(x); }, input(s, 0.5, 3, 7)),
      unary("log1p", [](const Tensor& x) { return nn::log1p(x); }, input(s, -0.5, 3, 8)),
      unary("expm1", [](const Tensor& x) { return nn::expm1(x); }, input(s, -2, 2, 9)),
      unary("abs", [](const Tensor& x) { return nn::abs(x); }, away_from_zero(s, 10)),
      unary("sin", [](const Tensor& x) { return nn::sin(x); }, input(s, -3, 3, 11)),
      unary("cos", [](const Tensor& x) { return nn::cos(x); }, input(s, -3, 3, 12)),
      unary("tan", [](const Tensor& x) { return nn::tan(x); }, input(s, -1, 1, 13)),
      unary("asin", [](const Tensor& x) { return nn::asin(x); }, input(s, -0.8, 0.8, 14)),
      unary("acos", [](const Tensor& x) { return nn::acos(x); }, input(s, -0.8, 0.8, 15)),
      unary("atan", [](const Tensor& x) { return nn::atan(x); }, input(s, -3, 3, 16)),
      unary("sinh", [](const Tensor& x) { return nn::sinh(x); }, input(s, -2, 2, 17)),
      unary("cosh", [](const Tensor& x) { return nn::cosh(x); }, input(s, -2, 2, 18)),
      unary("erf", [](const Tensor& x) { return nn::erf(x); }, input(s, -2, 2, 19)),
      unary("erfc", [](const Tensor& x) { return nn::erfc(x); }, input(s, -2, 2, 20)),
      unary("lgamma", [](const Tensor& x) { return nn::lgamma(x); }, input(s, 0.5, 4, 21)),
      unary("digamma", [](const Tensor& x) { return nn::digamma(x); }, input(s, 0.5, 4, 22)),
      unary("clip", [](const Tensor& x) { return nn::clip(x, -0.5, 0.5); }, away_from_zero(s, 23) * 0.4 + 0.0),
      unary("relu", [](const Tensor& x) { return nn::relu(x); }, away_from_zero(s, 24)),
      unary("relu6", [](const Tensor& x) { return nn::relu6(x * 4.0); }, away_from_zero(s, 25)),
      unary("leaky_relu", [](const Tensor& x) { return nn::leaky_relu(x, 0.1); }, away_from_zero(s, 26)),
      unary("elu", [](const Tensor& x) { return nn::elu(x, 1.3); }, away_from_zero(s, 27)),
      unary("selu", [](const Tensor& x) { return nn::selu(x); }, away_from_zero(s, 28)),
      unary("sigmoid", [](const Tensor& x) { return nn::sigmoid(x); }, input(s, -4, 4, 29)),
      unary("tanh", [](const Tensor& x) { return nn::tanh(x); }, input(s, -3, 3, 30)),
      unary("softplus", [](const Tensor& x) { return nn::softplus(x); }, input(s, -4, 4, 31)),
      unary("softsign", [](const Tensor& x) { return nn::softsign(x); }, away_from_zero(s, 32)),
      unary("silu", [](const Tensor& x) { return nn::silu(x); }, input(s, -3, 3, 33)),
      unary("gelu", [](const Tensor& x) { return nn::gelu(x); }, input(s, -3, 3, 34)),
      unary("gelu_tanh", [](const Tensor& x) { return nn::gelu(x, true); }, input(s, -3, 3, 35)),
      unary("mish", [](const Tensor& x) { return nn::mish(x); }, input(s, -3, 3, 36)),
      unary("log_sigmoid", [](const Tensor& x) { return nn::log_sigmoid(x); }, input(s, -5, 5, 37)),
      unary("pow2", [](const Tensor& x) { return nn::pow(x, 2.0); }, input(s, -2, 2, 38)),
      unary("pow3", [](const Tensor& x) { return nn::pow(x, 3.0); }, input(s, -2, 2, 39)),
      unary("pow_half", [](const Tensor& x) { return nn::pow(x, 0.5); }, input(s, 0.5, 3, 40)),
      unary("pow_neg", [](const Tensor& x) { return nn::pow(x, -1.5); }, input(s, 0.5, 3, 41)),
      unary("scalar_ops", [](const Tensor& x) { return (2.0 - x) * 3.0 / (x + 4.0) - 1.0 / (x + 5.0); },
            input(s, -2, 2, 42)),
  };
}

std::vector<Case> binary_cases() {
  return {
      binary("add_bcast", [](const Tensor& a, const Tensor& b) { return a + b; }, input({2, 3}, -1, 1, 50),
             input({3}, -1, 1, 51)),
      binary("sub_bcast", [](const Tensor& a, const Tensor& b) { return a - b; }, input({2, 1}, -1, 1, 52),
             input({1, 3}, -1, 1, 53)),
      binary("mul_bcast", [](const Tensor& a, const Tensor& b) { return a * b; }, input({2, 3}, -1, 1, 54),
             input({2, 1}, -1, 1, 55)),
      binary("mul_scalar_tensor", [](const Tensor& a, const Tensor& b) { return a * b; }, input({2, 3}, -1, 1, 56),
             input({}, -1, 1, 57)),
      binary("div_bcast", [](const Tensor& a, const Tensor& b) { return a / b; }, input({2, 3}, -1, 1, 58),
             input({3}, 0.5, 2, 59)),
      binary("pow_tensor", [](const Tensor& a, const Tensor& b) { return nn::pow(a, b); }, input({2, 3}, 0.5, 2, 60),
             input({2, 3}, -1.5, 1.5, 61)),
      binary("maximum", [](const Tensor& a, const Tensor& b) { return nn::maximum(a, b); }, distinct({2, 3}, 62),
             distinct({3}, 63) * 1.01 + 0.003),
      binary("minimum", [](const Tensor& a, const Tensor& b) { return nn::minimum(a, b); }, distinct({2, 3}, 64),
             distinct({3}, 65) * 1.01 + 0.003),
      binary("squared_difference", [](const Tensor& a, const Tensor& b) { return nn::squared_difference(a, b); },
             input({2, 3}, -1, 1, 66), input({3}, -1, 1, 67)),
      binary("atan2", [](const Tensor& a, const Tensor& b) { return nn::atan2(a, b); }, input({2, 3}, -1, 1, 68),
             away_from_zero({2, 3}, 69)),
      binary("where", [](const Tensor& a, const Tensor& b) { return nn::where(nn::greater(a, 0.0), a * 2.0, b); },
             away_from_zero({2, 3}, 70), input({3}, -1, 1, 71)),
      binary("matmul", [](const Tensor& a, const Tensor& b) { return nn::matmul(a, b); }, input({3, 4}, -1, 1, 72),
             input({4, 2}, -1, 1, 73)),
      binary("matmul_ta", [](const Tensor& a, const Tensor& b) { return nn::matmul(a, b, true, false); },
             input({4, 3}, -1, 1, 74), input({4, 2}, -1, 1, 75)),
      binary("matmul_tb", [](const Tensor& a, const Tensor& b) { return nn::matmul(a, b, false, true); },
             input({3, 4}, -1, 1, 76), input({2, 4}, -1, 1, 77)),
      binary("matmul_tab", [](const Tensor& a, const Tensor& b) { return nn::matmul(a, b, true, true); },
             input({4, 3}, -1, 1, 78), input({2, 4}, -1, 1, 79)),
      binary("matmul_batched_bcast", [](const Tensor& a, const Tensor& b) { return nn::matmul(a, b); },
             input({2, 1, 3, 4}, -1, 1, 80), input({3, 4, 2}, -1, 1, 81)),
      binary("bias_add", [](const Tensor& a, const Tensor& b) { return nn::bias_add(a, b); },
             input({2, 2, 3}, -1, 1, 82), input({3}, -1, 1, 83)),
      binary("dot", [](const Tensor& a, const Tensor& b) { return nn::dot(a, b); }, input({4}, -1, 1, 84),
             input({4}, -1, 1, 85)),
      binary("outer", [](const Tensor& a, const Tensor& b) { return nn::outer(a, b); }, input({3}, -1, 1, 86),
             input({2}, -1, 1, 87)),
      binary("concat", [](const Tensor& a, const Tensor& b) { return nn::concat({a, b, a}, 1); },
             input({2, 3}, -1, 1, 88), input({2, 2}, -1, 1, 89)),
      binary("stack", [](const Tensor& a, const Tensor& b) { return nn::stack({a, b}, 1); }, input({2, 3}, -1, 1, 90),
             input({2, 3}, -1, 1, 91)),
      binary("index_add", [](const Tensor& a, const Tensor& b) {
               return nn::index_add(a, 1, nn::from_vector(std::vector<std::int64_t>{2, 0, 2}), b);
             },
             input({2, 4}, -1, 1, 92), input({2, 3}, -1, 1, 93)),
      binary("scatter_nd_add", [](const Tensor& a, const Tensor& b) {
               return nn::scatter_nd_add(a, nn::from_vector(std::vector<std::int64_t>{0, 1, 2, 0, 0, 1}, {3, 2}), b);
             },
             input({3, 2}, -1, 1, 94), input({3}, -1, 1, 95)),
  };
}

std::vector<Case> reduction_cases() {
  const nn::Shape s{2, 3, 4};
  return {
      unary("sum_all", [](const Tensor& x) { return nn::sum(x); }, input(s, -1, 1, 100)),
      unary("sum_axes", [](const Tensor& x) { return nn::sum(x, {0, 2}); }, input(s, -1, 1, 101)),
      unary("sum_keep_neg", [](const Tensor& x) { return nn::sum(x, {-1}, true); }, input(s, -1, 1, 102)),
      unary("mean", [](const Tensor& x) { return nn::mean(x, {1}); }, input(s, -1, 1, 103)),
      unary("max_axis", [](const Tensor& x) { return nn::max(x, {1}); }, distinct(s, 104)),
      unary("max_all", [](const Tensor& x) { return nn::max(x); }, distinct(s, 105)),
      unary("min_axes", [](const Tensor& x) { return nn::min(x, {0, 2}, true); }, distinct(s, 106)),
      unary("prod", [](const Tensor& x) { return nn::prod(x, {2}); }, input(s, 0.5, 1.5, 107)),
      unary("prod_with_zero",
            [](const Tensor& x) {
              // One exact zero and one double zero group: d/dx of the product of others.
              const Tensor mask = nn::tensor({1, 0, 1, 1, 0, 0, 1, 1}, {2, 4}, DType::Float64);
              return nn::prod(x * mask, {1});
            },
            input({2, 4}, 0.5, 1.5, 108)),
      unary("variance", [](const Tensor& x) { return nn::variance(x, {2}); }, input(s, -1, 1, 109)),
      unary("std_dev", [](const Tensor& x) { return nn::std_dev(x, {0}); }, input(s, -1, 1, 110)),
      unary("logsumexp", [](const Tensor& x) { return nn::logsumexp(x, {1}); }, input(s, -3, 3, 111)),
      unary("softmax", [](const Tensor& x) { return nn::softmax(x, 1); }, input(s, -3, 3, 112)),
      unary("log_softmax", [](const Tensor& x) { return nn::log_softmax(x, -1); }, input(s, -3, 3, 113)),
      unary("cumsum", [](const Tensor& x) { return nn::cumsum(x, 1); }, input(s, -1, 1, 114)),
      unary("cumsum_excl_rev", [](const Tensor& x) { return nn::cumsum(x, 2, true, true); }, input(s, -1, 1, 115)),
      unary("norm", [](const Tensor& x) { return nn::norm(x, {2}); }, input(s, -1, 1, 116)),
  };
}

std::vector<Case> shape_cases() {
  const nn::Shape s{2, 3, 4};
  using P = std::vector<std::pair<std::int64_t, std::int64_t>>;
  return {
      unary("reshape", [](const Tensor& x) { return nn::reshape(x, {4, -1}) * nn::arange(0, 6, 1, DType::Float64); },
            input(s, -1, 1, 120)),
      unary("flatten", [](const Tensor& x) { return nn::flatten(nn::transpose(x)); }, input(s, -1, 1, 121)),
      unary("expand_squeeze", [](const Tensor& x) { return nn::squeeze(nn::expand_dims(x, 1) * 2.0, {1}); },
            input(s, -1, 1, 122)),
      unary("transpose", [](const Tensor& x) { return nn::transpose(x, {2, 0, 1}) * nn::arange(0, 3, 1, DType::Float64); },
            input(s, -1, 1, 123)),
      unary("swapaxes", [](const Tensor& x) { return nn::swapaxes(x, 0, 2); }, input(s, -1, 1, 124)),
      unary("slice", [](const Tensor& x) { return nn::slice(x, {0, 1, 1}, {-1, 2, 2}); }, input(s, -1, 1, 125)),
      unary("slice_axis_neg", [](const Tensor& x) { return nn::slice_axis(x, 2, -1, -5, -2); }, input(s, -1, 1, 126)),
      unary("strided_slice", [](const Tensor& x) { return nn::strided_slice(x, {1, 0}, {2, 3}, {1, 2}); },
            input(s, -1, 1, 127)),
      unary("select", [](const Tensor& x) { return nn::select(x, 1, -1); }, input(s, -1, 1, 128)),
      unary("split",
            [](const Tensor& x) {
              const auto parts = nn::split(x, {1, -1}, 2);
              return parts[1] * 2.0;  // parts[0] unused: its gradient is zero
            },
            input(s, -1, 1, 129)),
      unary("unstack",
            [](const Tensor& x) {
              const auto parts = nn::unstack(x, 1);
              return parts[0] * parts[2];
            },
            input(s, -1, 1, 130)),
      unary("tile", [](const Tensor& x) { return nn::tile(x, {2, 1, 3}); }, input(s, -1, 1, 131)),
      unary("repeat", [](const Tensor& x) { return nn::repeat(x, 3, 1); }, input(s, -1, 1, 132)),
      unary("broadcast_to", [](const Tensor& x) { return nn::broadcast_to(x, {2, 2, 3, 4}) * 1.5; },
            input(s, -1, 1, 133)),
      unary("sum_to", [](const Tensor& x) { return nn::sum_to(x, {3, 1}); }, input(s, -1, 1, 134)),
      unary("pad_constant", [](const Tensor& x) { return nn::pad(x, P{{1, 0}, {0, 2}, {1, 1}}, nn::PadMode::Constant, 3.0); },
            input(s, -1, 1, 135)),
      unary("pad_reflect", [](const Tensor& x) { return nn::pad(x, P{{0, 0}, {2, 1}, {1, 3}}, nn::PadMode::Reflect); },
            input(s, -1, 1, 136)),
      unary("pad_symmetric",
            [](const Tensor& x) { return nn::pad(x, P{{1, 1}, {0, 0}, {2, 2}}, nn::PadMode::Symmetric); },
            input(s, -1, 1, 137)),
      unary("reverse", [](const Tensor& x) { return nn::reverse(x, {0, 2}) * nn::arange(0, 4, 1, DType::Float64); },
            input(s, -1, 1, 138)),
      unary("contiguous", [](const Tensor& x) { return nn::transpose(x).contiguous(); }, input(s, -1, 1, 139)),
      unary("clone", [](const Tensor& x) { return x.clone() * x; }, input(s, -1, 1, 140)),
      unary("gather_axis1",
            [](const Tensor& x) {
              return nn::gather(x, nn::from_vector(std::vector<std::int64_t>{2, 0, 2, -1}, {2, 2}), 1);
            },
            input(s, -1, 1, 141)),
      unary("gather_nd",
            [](const Tensor& x) {
              return nn::gather_nd(x, nn::from_vector(std::vector<std::int64_t>{1, 2, 0, 0, 1, 2}, {3, 2}));
            },
            input(s, -1, 1, 142)),
      unary("segment_sum",
            [](const Tensor& x) {
              return nn::unsorted_segment_sum(x, nn::from_vector(std::vector<std::int32_t>{2, 0}), 4);
            },
            input(s, -1, 1, 143)),
      unary("segment_mean",
            [](const Tensor& x) {
              return nn::unsorted_segment_mean(x, nn::from_vector(std::vector<std::int64_t>{1, 1, 5}, {3}), 3);
            },
            input({3, 4}, -1, 1, 144)),
      unary("scatter_nd",
            [](const Tensor& x) {
              return nn::scatter_nd(nn::from_vector(std::vector<std::int64_t>{1, 0, 1}, {3, 1}), x, {2, 4});
            },
            input({3, 4}, -1, 1, 145)),
      unary("top_k_values", [](const Tensor& x) { return nn::top_k(x, 2).first; }, distinct(s, 146)),
      unary("cast_roundtrip", [](const Tensor& x) { return nn::cast(nn::cast(x, DType::Float64), DType::Float64) * x; },
            input(s, -1, 1, 147)),
  };
}

}  // namespace

// ---- first order: every differentiable op ------------------------------------------

TEST(NnGradcheck, UnaryOps) {
  for (const auto& c : unary_cases()) expect_gradcheck(c);
}

TEST(NnGradcheck, BinaryOps) {
  for (const auto& c : binary_cases()) expect_gradcheck(c);
}

TEST(NnGradcheck, Reductions) {
  for (const auto& c : reduction_cases()) expect_gradcheck(c);
}

TEST(NnGradcheck, ShapeAndIndexOps) {
  for (const auto& c : shape_cases()) expect_gradcheck(c);
}

TEST(NnGradcheck, DetectsWrongGradient) {
  // A custom function with a deliberately wrong backward must fail the check.
  nn::Function bad("BadSquare", [](const Inputs& in) { return Inputs{nn::square(in[0])}; },
                   [](const Inputs& in, const Inputs&, const Inputs& g) { return Inputs{g[0] * in[0]}; });
  const auto r = nn::gradcheck([&](const Inputs& in) { return bad(in[0]); }, {input({3}, -1, 1, 160)});
  EXPECT_FALSE(r.ok);
  EXPECT_FALSE(r.message.empty());
}

// ---- second order --------------------------------------------------------------------

TEST(NnGradcheck, SecondOrder) {
  const nn::Shape s{2, 3};
  std::vector<Case> cases = {
      unary("exp", [](const Tensor& x) { return nn::exp(x); }, input(s, -1, 1, 200)),
      unary("log", [](const Tensor& x) { return nn::log(x); }, input(s, 0.5, 2, 201)),
      unary("sqrt", [](const Tensor& x) { return nn::sqrt(x); }, input(s, 0.5, 2, 202)),
      unary("tanh", [](const Tensor& x) { return nn::tanh(x); }, input(s, -2, 2, 203)),
      unary("sigmoid", [](const Tensor& x) { return nn::sigmoid(x); }, input(s, -2, 2, 204)),
      unary("softplus", [](const Tensor& x) { return nn::softplus(x); }, input(s, -2, 2, 205)),
      unary("sin_cos", [](const Tensor& x) { return nn::sin(x) * nn::cos(x * 2.0); }, input(s, -2, 2, 206)),
      unary("erf", [](const Tensor& x) { return nn::erf(x); }, input(s, -2, 2, 207)),
      unary("elu", [](const Tensor& x) { return nn::elu(x); }, away_from_zero(s, 208)),
      unary("pow_scalar", [](const Tensor& x) { return nn::pow(x, 3.0); }, input(s, -2, 2, 209)),
      unary("softmax", [](const Tensor& x) { return nn::softmax(x, -1); }, input(s, -2, 2, 210)),
      unary("log_softmax", [](const Tensor& x) { return nn::log_softmax(x, 0); }, input(s, -2, 2, 211)),
      unary("logsumexp", [](const Tensor& x) { return nn::logsumexp(x, {1}); }, input(s, -2, 2, 212)),
      unary("variance", [](const Tensor& x) { return nn::variance(x, {1}); }, input(s, -2, 2, 213)),
      unary("prod", [](const Tensor& x) { return nn::prod(x, {1}); }, input(s, 0.5, 1.5, 214)),
      unary("cumsum_sq", [](const Tensor& x) { return nn::square(nn::cumsum(x, 1)); }, input(s, -1, 1, 215)),
      unary("gather_sq",
            [](const Tensor& x) {
              return nn::square(nn::gather(x, nn::from_vector(std::vector<std::int64_t>{2, 2, 0}), 1));
            },
            input(s, -1, 1, 216)),
      unary("slice_concat",
            [](const Tensor& x) { return nn::square(nn::concat({nn::slice_axis(x, 1, 1, 3), x}, 1)); },
            input(s, -1, 1, 217)),
      binary("mul", [](const Tensor& a, const Tensor& b) { return a * b * a; }, input(s, -1, 1, 218),
             input({3}, -1, 1, 219)),
      binary("div", [](const Tensor& a, const Tensor& b) { return a / b; }, input(s, -1, 1, 220),
             input(s, 0.5, 2, 221)),
      binary("pow_tensor", [](const Tensor& a, const Tensor& b) { return nn::pow(a, b); }, input(s, 0.5, 2, 222),
             input(s, -1, 1, 223)),
      binary("matmul_tanh", [](const Tensor& a, const Tensor& b) { return nn::tanh(nn::matmul(a, b)); },
             input({2, 3}, -1, 1, 224), input({3, 2}, -1, 1, 225)),
      binary("where", [](const Tensor& a, const Tensor& b) { return nn::where(nn::greater(a, 0.0), a * a, b * b * b); },
             away_from_zero(s, 226), input(s, -1, 1, 227)),
  };
  for (const auto& c : cases) expect_gradgradcheck(c);
}

TEST(NnAutograd, HessianVectorProduct) {
  // f(x) = x^T A x + sum(x^3): H = A + A^T + diag(6x).
  const Tensor A = nn::tensor({{2, 1, 0}, {0, 3, -1}, {1, 0, 1}}, DType::Float64);
  Tensor       x = nn::tensor({0.5, -1.0, 2.0}, {3}, DType::Float64).requires_grad_();
  const Tensor v = nn::tensor({1.0, 2.0, -1.0}, {3}, DType::Float64);
  auto         f = [&](const Tensor& in) {
    const Tensor col = nn::reshape(in, {3, 1});
    return nn::sum(nn::matmul(nn::matmul(col, A, true, false), col)) + nn::sum(nn::pow(in, 3.0));
  };
  nn::GradOptions create;
  create.create_graph = true;
  const Tensor g      = nn::grad(f(x), {x}, create)[0];
  const Tensor hv     = nn::grad({nn::dot(g, v)}, {x})[0];
  // Expected: (A + A^T) v + 6 x * v
  const auto xs = vals(x), vs = vals(v), as = vals(A);
  for (int i = 0; i < 3; ++i) {
    double expected = 6 * xs[i] * vs[i];
    for (int j = 0; j < 3; ++j) expected += (as[i * 3 + j] + as[j * 3 + i]) * vs[j];
    EXPECT_NEAR(vals(hv)[i], expected, 1e-12);
  }
}

// ---- engine ------------------------------------------------------------------------

TEST(NnAutograd, ScalarDerivatives) {
  // The tflow `autodiff` example: y = x^3 + 2x, dy/dx = 3x^2 + 2, d2y/dx2 = 6x.
  for (double xv : {0.5, 1.0, 2.0}) {
    Tensor          x = nn::scalar(xv, DType::Float32).requires_grad_();
    Tensor          y = nn::pow(x, 3.0) + 2.0 * x;
    nn::GradOptions create;
    create.create_graph = true;
    Tensor dy           = nn::grad(y, {x}, create)[0];
    Tensor d2y          = nn::grad(dy, {x})[0];
    EXPECT_NEAR(dy.item(), 3 * xv * xv + 2, 1e-5);
    EXPECT_NEAR(d2y.item(), 6 * xv, 1e-5);
    EXPECT_EQ(dy.dtype(), DType::Float32);
  }
}

TEST(NnAutograd, JacobianRowsViaSeeds) {
  Tensor       x  = nn::tensor({2.0, 3.0}).requires_grad_();
  const Tensor x0 = nn::select(x, 0, 0), x1 = nn::select(x, 0, 1);
  const Tensor y  = nn::stack({x0 * x1, x0 + x1, nn::sin(x0)});
  const std::vector<std::vector<double>> expected = {{3, 2}, {1, 1}, {std::cos(2.0), 0}};
  for (int row = 0; row < 3; ++row) {
    nn::GradOptions retain;
    retain.retain_graph = true;
    const Tensor seed   = nn::one_hot(nn::scalar(row, DType::Int64), 3);
    const auto   g      = nn::grad({y}, {x}, {seed}, retain);
    EXPECT_NEAR(vals(g[0])[0], expected[row][0], 1e-6);
    EXPECT_NEAR(vals(g[0])[1], expected[row][1], 1e-6);
  }
}

TEST(NnAutograd, StopGradient) {
  Tensor x = nn::scalar(3.0).requires_grad_();
  EXPECT_NEAR(nn::grad(x * nn::stop_gradient(x), {x})[0].item(), 3.0, 1e-6);
  EXPECT_NEAR(nn::grad(x * x, {x})[0].item(), 6.0, 1e-6);
  // The straight-through trick: forward ceil(x), gradient as if identity.
  Tensor       v = nn::tensor({0.2, 1.7, -0.4}).requires_grad_();
  const Tensor y = v + nn::stop_gradient(nn::ceil(v) - v);
  const auto   g = nn::grad(nn::sum(nn::square(y)), {v})[0];
  EXPECT_EQ(vals(y), (std::vector<double>{1, 2, -0.0}));
  EXPECT_EQ(vals(g), (std::vector<double>{2, 4, 0}));
}

TEST(NnAutograd, BackwardAccumulatesIntoLeaves) {
  Tensor w = nn::tensor({1.0, 2.0}).requires_grad_();
  nn::sum(w * w).backward();
  EXPECT_EQ(vals(w.grad()), (std::vector<double>{2, 4}));
  nn::sum(w * 3.0).backward();
  EXPECT_EQ(vals(w.grad()), (std::vector<double>{5, 7}));
  w.zero_grad();
  EXPECT_FALSE(w.grad().defined());
  // grad() leaves .grad() alone.
  (void)nn::grad(nn::sum(w), {w});
  EXPECT_FALSE(w.grad().defined());
}

TEST(NnAutograd, DiamondAndReuse) {
  // x feeds two paths that meet again: gradients add.
  Tensor       x = nn::scalar(1.5, DType::Float64).requires_grad_();
  const Tensor a = nn::exp(x), b = nn::sin(x);
  const Tensor y = a * b + a;
  const double xv = 1.5;
  EXPECT_NEAR(nn::grad(y, {x})[0].item(), std::exp(xv) * std::sin(xv) + std::exp(xv) * std::cos(xv) + std::exp(xv),
              1e-12);
}

TEST(NnAutograd, NonLeafInputsAndMultipleOutputs) {
  Tensor       x = nn::tensor({1.0, 2.0, 3.0}, {3}, DType::Float64).requires_grad_();
  const Tensor h = x * 2.0;
  const Tensor y = nn::sum(nn::square(h));
  const Tensor z = nn::sum(h);
  const auto   g = nn::grad({y, z}, {h, x}, {}, {});
  EXPECT_EQ(vals(g[0]), (std::vector<double>{2 * 2 + 1, 4 * 2 + 1, 6 * 2 + 1}));
  EXPECT_EQ(vals(g[1]), (std::vector<double>{10, 18, 26}));
}

TEST(NnAutograd, Errors) {
  Tensor x = nn::tensor({1.0, 2.0}).requires_grad_();
  Tensor u = nn::tensor({1.0}).requires_grad_();
  // Non-scalar output without a seed.
  EXPECT_THROW(nn::grad(x * 2.0, {x}), std::invalid_argument);
  // Unused input.
  EXPECT_THROW(nn::grad(nn::sum(x), {u}), std::invalid_argument);
  nn::GradOptions allow;
  allow.allow_unused = true;
  const auto g       = nn::grad({nn::sum(x)}, {x, u}, {}, allow);
  EXPECT_TRUE(g[0].defined());
  EXPECT_FALSE(g[1].defined());
  // Output without history.
  EXPECT_THROW(nn::grad(nn::sum(nn::ones({2})), {x}), std::invalid_argument);
  // Only float tensors can require grad.
  Tensor i = nn::zeros({2}, DType::Int64);
  EXPECT_THROW(i.requires_grad_(), std::invalid_argument);
  // Zero-gradient ops (ceil) leave no history: like TF, asking for the gradient is an error.
  EXPECT_THROW(nn::grad(nn::sum(nn::ceil(x)), {x}), std::invalid_argument);
}

TEST(NnAutograd, GraphIsFreedUnlessRetained) {
  Tensor       x = nn::scalar(2.0).requires_grad_();
  const Tensor y = nn::exp(x) * x;
  (void)nn::grad(y, {x});
  EXPECT_THROW(nn::grad(y, {x}), std::logic_error);

  const Tensor    y2 = nn::exp(x) * x;
  nn::GradOptions retain;
  retain.retain_graph = true;
  const double g1     = nn::grad(y2, {x}, retain)[0].item();
  const double g2     = nn::grad(y2, {x})[0].item();
  EXPECT_DOUBLE_EQ(g1, g2);
}

TEST(NnAutograd, InPlaceAfterSaveIsDetected) {
  Tensor       w = nn::tensor({1.0, 2.0}).requires_grad_();
  Tensor       c = nn::tensor({3.0, 4.0});
  const Tensor y = nn::sum(w * c);
  c.add_(nn::ones({2}));  // c was saved by Mul for w's gradient
  EXPECT_THROW(nn::grad(y, {w}), std::logic_error);

  // In-place on a tensor that requires grad needs NoGradGuard.
  EXPECT_THROW(w.add_(nn::ones({2})), std::logic_error);
  {
    nn::NoGradGuard guard;
    w.add_(nn::ones({2}), -0.5);
  }
  EXPECT_EQ(vals(w), (std::vector<double>{0.5, 1.5}));
  EXPECT_TRUE(w.is_leaf());
}

TEST(NnAutograd, SelfMultiplySharedEdgeAndHigherDerivatives) {
  const auto fn = [](const Inputs& in) { return nn::mul(in[0], in[0]); };
  auto       x  = input({3, 5}, -2, 2, 713);
  EXPECT_TRUE(nn::gradcheck(fn, {x}).ok);
  EXPECT_TRUE(nn::gradgradcheck(fn, {x}).ok);
  auto y = nn::mul(x, x);
  {
    nn::NoGradGuard guard;
    x.add_(nn::ones(x.shape(), x.dtype()));
  }
  EXPECT_THROW(nn::grad(nn::sum(y), {x}), std::logic_error);

  // Multiplying the gradient first avoids overflowing 2*x unnecessarily.
  auto large      = nn::full({1}, 2e38, DType::Float32).requires_grad_();
  auto seed       = nn::full({1}, 1e-38, DType::Float32);
  auto derivative = nn::grad({nn::mul(large, large)}, {large}, {seed})[0];
  EXPECT_NEAR(derivative.item(), 4.0, 1e-5);
}

TEST(NnAutograd, SquareVjpRetainsScaleOrderAndDifferentiableSeeds) {
  auto x = nn::tensor({1.0, -2.0, 3.0}).requires_grad_();
  auto            seed = nn::tensor({4.0, 5.0, -6.0}).requires_grad_();
  nn::GradOptions options;
  options.create_graph = true;
  auto derivative      = nn::grad({nn::square(x)}, {x}, {seed}, options)[0];
  EXPECT_EQ(vals(derivative), (std::vector<double>{8, -20, -36}));
  auto second = nn::grad(nn::sum(derivative), {x, seed});
  EXPECT_EQ(vals(second[0]), (std::vector<double>{8, 10, -12}));
  EXPECT_EQ(vals(second[1]), (std::vector<double>{2, -4, 6}));
  auto large      = nn::full({1}, 2e38, DType::Float32).requires_grad_();
  auto small_seed = nn::full({1}, 1e-38, DType::Float32);
  EXPECT_TRUE(std::isinf(nn::grad({nn::square(large)}, {large}, {small_seed})[0].item()));
  for (auto dtype : {DType::Float32, DType::Float64}) {
    auto strided = nn::transpose(nn::cast(nn::tensor({{1, -2, 3}, {4, 5, -6}}), dtype)).requires_grad_();
    auto varied_seed = nn::cast(nn::tensor({{2, -1}, {3, -2}, {4, -3}}), dtype);
    auto ordinary    = nn::grad({nn::square(strided)}, {strided}, {varied_seed})[0];
    EXPECT_EQ(vals(ordinary), (std::vector<double>{4, -8, -12, -20, 24, 36}));
    ordinary.add_(nn::ones(ordinary.shape(), dtype));
    EXPECT_EQ(vals(strided), (std::vector<double>{1, 4, -2, 5, 3, -6}));
  }
}

TEST(NnAutograd, NoGradGuard) {
  Tensor x = nn::tensor({1.0}).requires_grad_();
  {
    nn::NoGradGuard guard;
    EXPECT_FALSE(nn::is_grad_enabled());
    const Tensor y = x * 2.0;
    EXPECT_FALSE(y.requires_grad());
    {
      nn::EnableGradGuard enable;
      EXPECT_TRUE((x * 2.0).requires_grad());
    }
  }
  EXPECT_TRUE(nn::is_grad_enabled());
  EXPECT_EQ((x * 2.0).grad_fn_name(), "Mul");
}

TEST(NnAutograd, MixedDtypeGradientsKeepInputDtype) {
  Tensor       a = nn::tensor({1.0, 2.0}, {2}, DType::Float32).requires_grad_();
  Tensor       b = nn::tensor({3.0, 4.0}, {2}, DType::Float64).requires_grad_();
  const Tensor y = nn::sum(a * b);
  EXPECT_EQ(y.dtype(), DType::Float64);
  const auto g = nn::grad(y, {a, b});
  EXPECT_EQ(g[0].dtype(), DType::Float32);
  EXPECT_EQ(g[1].dtype(), DType::Float64);
  EXPECT_EQ(vals(g[0]), (std::vector<double>{3, 4}));
}

TEST(NnAutograd, CustomFunctions) {
  // Erfc with a registered gradient, as in the tflow `custom_gradient` example.
  nn::Function erfc_fn(
      "MyErfc", [](const Inputs& in) { return Inputs{nn::erfc(in[0])}; },
      [](const Inputs& in, const Inputs&, const Inputs& g) {
        return Inputs{g[0] * nn::exp(-nn::square(in[0])) * (-2.0 / std::sqrt(3.14159265358979323846))};
      });
  Tensor x = input({5}, -1.5, 1.5, 300);
  EXPECT_TRUE(nn::gradcheck([&](const Inputs& in) { return erfc_fn(in[0]); }, {x}).ok);
  EXPECT_TRUE(nn::gradgradcheck([&](const Inputs& in) { return erfc_fn(in[0]); }, {x}, {1e-6, 1e-4, 1e-3}).ok);

  // Straight-through estimator for round: forward quantises, backward is identity.
  nn::Function ste("RoundSTE", [](const Inputs& in) { return Inputs{nn::round(in[0])}; },
                   [](const Inputs&, const Inputs&, const Inputs& g) { return Inputs{g[0]}; });
  Tensor       q = nn::scalar(0.1).requires_grad_();
  // Quantise to a 0.25 grid and pull toward 2.5 (the tflow STE demo).
  for (int step = 0; step < 30; ++step) {
    const Tensor quantised = ste(q * 4.0) / 4.0;
    const Tensor loss      = nn::square(quantised - 2.5);
    const Tensor g         = nn::grad(loss, {q})[0];
    nn::NoGradGuard guard;
    q.sub_(g, 0.1);
  }
  EXPECT_NEAR(q.item(), 2.5, 0.13);

  // Gradient reversal: identity forward, negated gradient.
  nn::Function reverse_grad("GradReverse", [](const Inputs& in) { return Inputs{in[0]}; },
                            [](const Inputs&, const Inputs&, const Inputs& g) { return Inputs{-g[0]}; });
  Tensor       r = nn::tensor({1.0, 2.0}).requires_grad_();
  const auto   gr = nn::grad(nn::sum(reverse_grad(r) * 3.0), {r})[0];
  EXPECT_EQ(vals(gr), (std::vector<double>{-3, -3}));
  EXPECT_EQ(vals(reverse_grad(r)), vals(r));

  // A backward with the wrong number of gradients is reported.
  nn::Function broken("Broken", [](const Inputs& in) { return Inputs{in[0] * 2.0}; },
                      [](const Inputs&, const Inputs&, const Inputs&) { return Inputs{}; });
  EXPECT_THROW(nn::grad(nn::sum(broken(r)), {r}), std::logic_error);
}

TEST(NnAutograd, DeepGraphsDoNotOverflowTheStack) {
  Tensor x = nn::scalar(1.0).requires_grad_();
  Tensor y = x;
  for (int i = 0; i < 100000; ++i) y = y + 1e-6;
  EXPECT_NEAR(nn::grad(y, {x})[0].item(), 1.0, 1e-6);
  Tensor z = x;
  for (int i = 0; i < 100000; ++i) z = z * 1.0;
  z = Tensor();  // dropping the graph must not recurse
  SUCCEED();
}

TEST(NnAutograd, GradcheckMatchesFiniteDifferenceOnLoss) {
  // The tflow autodiff example's matrix check: loss = mean((xW + b - t)^2).
  const Tensor x = nn::tensor({{1, 2, 3}, {0, -1, 1}, {2, 2, 2}, {-1, 0, 1}}, DType::Float64);
  const Tensor t = nn::tensor({{1, 0}, {0, 1}, {1, 1}, {0, 0}}, DType::Float64);
  Tensor       W = nn::tensor({{0.1, -0.2}, {0.3, 0.4}, {-0.5, 0.6}}, DType::Float64).requires_grad_();
  Tensor       b = nn::tensor({0.05, -0.05}, {2}, DType::Float64).requires_grad_();
  const auto   r = nn::gradcheck(
      [&](const Inputs& in) { return nn::mean(nn::squared_difference(nn::matmul(x, in[0]) + in[1], t)); }, {W, b});
  EXPECT_TRUE(r.ok) << r.message;
  EXPECT_LT(r.max_abs_error, 1e-8);
}

TEST(NnAutograd, DenseTraversalHandlesGrowthAncestorRootsDuplicateCapturesAndHigherOrder) {
  auto x = nn::scalar(2.0, DType::Float64).requires_grad_();
  auto unused = nn::scalar(7.0, DType::Float64).requires_grad_();
  auto y = x;
  for (int i = 0; i < 1024; ++i) y = nn::add(y, 0.001);
  auto z = nn::square(y);
  nn::GradOptions options;
  options.retain_graph = true;
  options.create_graph = true;
  options.allow_unused = true;
  const auto gradients = nn::grad({y, z}, {x, x, unused},
      {nn::scalar(3.0, DType::Float64), nn::scalar(5.0, DType::Float64)}, options);
  ASSERT_EQ(gradients.size(), 3u);
  EXPECT_NEAR(gradients[0].item(), 3 + 10 * y.item(), 1e-11);
  EXPECT_EQ(gradients[0].item(), gradients[1].item());
  EXPECT_FALSE(gradients[2].defined());
  EXPECT_NEAR(nn::grad(gradients[0], {x}, nn::GradOptions{true, false, false})[0].item(), 10, 1e-12);
  z.backward();
  EXPECT_NEAR(x.grad().item(), 2 * y.item(), 1e-12);
}

TEST(NnAutograd, WideCapturePermutationAndDuplicatesKeepGradientPositions) {
  std::vector<Tensor> leaves;
  auto loss = nn::scalar(0, DType::Float64);
  for (int i = 0; i < 257; ++i) {
    leaves.push_back(nn::scalar(i, DType::Float64).requires_grad_());
    loss = nn::add(loss, nn::mul(leaves.back(), i + 1));
  }
  std::rotate(leaves.begin(), leaves.begin() + 93, leaves.end());
  leaves.push_back(leaves.front());
  const auto gradients = nn::grad(loss, leaves);
  ASSERT_EQ(gradients.size(), leaves.size());
  for (std::size_t i = 0; i < leaves.size(); ++i)
    EXPECT_DOUBLE_EQ(gradients[i].item(), leaves[i].item() + 1);
}

TEST(NnAutograd, FusedSoftmaxVjpMatchesStridedFallbackAndKeepsHigherDerivatives) {
  for (auto dtype : {DType::Float32, DType::Float64})
    for (std::int64_t width : {1, 7, 32, 129, 257}) {
      auto x =
          nn::cast(
              nn::reshape(nn::arange(0, 3 * width, 1, DType::Float64), {3, width}) / 128.0, dtype)
              .requires_grad_();
      auto seed = nn::cast(
          nn::reshape(nn::arange(-3 * width, 0, 1, DType::Float64), {3, width}) / 64.0, dtype);
      auto            strided = nn::transpose(nn::transpose(seed).clone());
      auto            output  = nn::softmax(x);
      nn::GradOptions retain;
      retain.retain_graph = true;
      const auto packed   = nn::grad({output}, {x}, {seed}, retain)[0].to_vector<double>();
      const auto fallback = nn::grad({output}, {x}, {strided}, retain)[0].to_vector<double>();
      ASSERT_EQ(packed.size(), fallback.size());
      for (std::size_t i = 0; i < packed.size(); ++i)
        EXPECT_NEAR(packed[i], fallback[i], dtype == DType::Float32 ? 3e-7 : 2e-14);
    }
  auto x = input({2, 7}, -1, 1, 992);
  expect_gradgradcheck(
      unary("softmax_fused_fallback", [](const Tensor& v) { return nn::softmax(v); }, x));
}
