// Known-value tests for the nn ops (forward values only; gradients are tested elsewhere).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <vector>

using namespace datamunge;
using nn::DType;
using nn::Shape;
using nn::Tensor;

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<double> vals(const Tensor& t) { return t.to_vector<double>(); }

void expect_vals(const Tensor& t, const std::vector<double>& expected, double tol = 1e-5) {
  const auto v = vals(t);
  ASSERT_EQ(v.size(), expected.size()) << t.str();
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (std::isnan(expected[i]))
      EXPECT_TRUE(std::isnan(v[i])) << "element " << i << " of " << t.str();
    else if (std::isinf(expected[i]))
      EXPECT_EQ(v[i], expected[i]) << "element " << i << " of " << t.str();
    else
      EXPECT_NEAR(v[i], expected[i], tol) << "element " << i << " of " << t.str();
  }
}

Tensor f64(std::initializer_list<double> v) { return nn::tensor(v, {}, DType::Float64); }
Tensor i64(std::initializer_list<double> v) { return nn::tensor(v, {}, DType::Int64); }
Tensor i32(std::initializer_list<double> v) { return nn::tensor(v, {}, DType::Int32); }

// The [2, 3] matrix used throughout tensor_basics.cc.
Tensor x23() { return nn::tensor({{1, 2, 3}, {4, 5, 6}}); }

}  // namespace

// ---- broadcasting arithmetic -----------------------------------------------------------------

TEST(NnOps, BroadcastAddRow) {
  // TF: x + row  [2,3]+[3]
  Tensor r = x23() + nn::tensor({10, 20, 30});
  EXPECT_EQ(r.shape(), (Shape{2, 3}));
  expect_vals(r, {11, 22, 33, 14, 25, 36});
}

TEST(NnOps, BroadcastMulColumn) {
  // TF: x * col  [2,3]*[2,1]
  Tensor r = x23() * nn::tensor({1, -1}, {2, 1});
  expect_vals(r, {1, 2, 3, -4, -5, -6});
}

TEST(NnOps, BroadcastRowPlusColumn) {
  // TF: row + col [3]+[2,1] -> [2,3]
  Tensor r = nn::tensor({10, 20, 30}) + nn::tensor({1, -1}, {2, 1});
  EXPECT_EQ(r.shape(), (Shape{2, 3}));
  expect_vals(r, {11, 21, 31, 9, 19, 29});
}

TEST(NnOps, ScalarOperands) {
  expect_vals(x23() * 2.0 + 1.0, {3, 5, 7, 9, 11, 13});
  expect_vals(10.0 - nn::tensor({1, 2}), {9, 8});
  expect_vals(1.0 / nn::tensor({2, 4}), {0.5, 0.25});
  expect_vals(nn::tensor({2, 4}) / 2.0, {1, 2});
  expect_vals(3.0 * nn::tensor({1, 2}), {3, 6});
  expect_vals(-nn::tensor({1, -2}), {-1, 2});
  expect_vals(nn::tensor({1, 2}) + nn::scalar(1), {2, 3});
  // A double scalar keeps the tensor's float dtype.
  EXPECT_EQ((nn::tensor({1, 2}) * 2.0).dtype(), DType::Float32);
  EXPECT_EQ((f64({1, 2}) * 2.0).dtype(), DType::Float64);
}

TEST(NnOps, IncompatibleShapesThrow) {
  // TF: AddV2([2,3],[2]) fails shape inference.
  EXPECT_THROW(x23() + nn::tensor({1, 2}), std::invalid_argument);
  EXPECT_THROW(nn::broadcast_shapes({2, 3}, {4, 3}), std::invalid_argument);
  EXPECT_EQ(nn::broadcast_shapes({5, 1, 3}, {4, 1}), (Shape{5, 4, 3}));
  EXPECT_EQ(nn::broadcast_shapes({}, {2}), (Shape{2}));
}

TEST(NnOps, DtypePromotion) {
  EXPECT_EQ((i32({1, 2}) + nn::tensor({1, 2})).dtype(), DType::Float32);
  EXPECT_EQ((i32({1, 2}) + i64({1, 2})).dtype(), DType::Int64);
  EXPECT_EQ((nn::tensor({1, 2}) + f64({1, 2})).dtype(), DType::Float64);
  // bool arithmetic promotes to int64
  Tensor b = nn::tensor({1, 1, 0}, {}, DType::Bool);
  Tensor s = nn::add(b, b);
  EXPECT_EQ(s.dtype(), DType::Int64);
  expect_vals(s, {2, 2, 0});
  // int scalar keeps int; non-integral scalar goes to float32
  EXPECT_EQ((i32({1, 2}) * 2.0).dtype(), DType::Int32);
  Tensor h = i32({1, 2}) * 2.5;
  EXPECT_EQ(h.dtype(), DType::Float32);
  expect_vals(h, {2.5, 5});
  // division of ints is true division
  Tensor d = nn::div(i32({1, 3}), i32({2, 2}));
  EXPECT_EQ(d.dtype(), DType::Float32);
  expect_vals(d, {0.5, 1.5});
  // integer arithmetic stays exact
  Tensor m = i64({3, -7}) * i64({4, 2});
  EXPECT_EQ(m.dtype(), DType::Int64);
  expect_vals(m, {12, -14});
  EXPECT_EQ(nn::scalar_like(2.0, i32({1})).dtype(), DType::Int32);
  EXPECT_EQ(nn::scalar_like(2.0, nn::tensor({1}, {}, DType::Bool)).dtype(), DType::Int64);
  EXPECT_EQ(nn::scalar_like(0.5, i32({1})).dtype(), DType::Float32);
}

TEST(NnOps, BinaryNamedOps) {
  expect_vals(nn::sub(nn::tensor({5, 3}), nn::tensor({1, 4})), {4, -1});
  expect_vals(nn::pow(nn::tensor({2, 3, 4}), nn::tensor({3, 2, 0.5})), {8, 9, 2});
  expect_vals(nn::pow(nn::tensor({2, -3}), 2.0), {4, 9});
  expect_vals(nn::maximum(nn::tensor({1, 5, 3}), nn::tensor({4, 2, 3})), {4, 5, 3});
  expect_vals(nn::minimum(nn::tensor({1, 5, 3}), nn::tensor({4, 2, 3})), {1, 2, 3});
  expect_vals(nn::maximum(x23(), nn::scalar(3)), {3, 3, 3, 4, 5, 6});
  expect_vals(nn::squared_difference(nn::tensor({1, 5}), nn::tensor({3, 2})), {4, 9});
  expect_vals(nn::atan2(f64({1, 1, -1}), f64({1, -1, 0})), {kPi / 4, 3 * kPi / 4, -kPi / 2}, 1e-12);
  expect_vals(nn::add(nn::tensor({1}), 2.0), {3});
  expect_vals(nn::sub(1.0, nn::tensor({3})), {-2});
  expect_vals(nn::mul(nn::tensor({3}), 2.0), {6});
  expect_vals(nn::div(nn::tensor({3}), 2.0), {1.5});
  expect_vals(nn::div(3.0, nn::tensor({2})), {1.5});
}

// ---- unary math ---------------------------------------------------------------------------------

TEST(NnOps, UnaryMath) {
  Tensor x = f64({0.25, 1, 4});
  expect_vals(nn::square(x), {0.0625, 1, 16}, 1e-12);
  expect_vals(nn::sqrt(x), {0.5, 1, 2}, 1e-12);
  expect_vals(nn::rsqrt(x), {2, 1, 0.5}, 1e-12);
  expect_vals(nn::reciprocal(x), {4, 1, 0.25}, 1e-12);
  expect_vals(nn::exp(f64({0, 1})), {1, std::exp(1.0)}, 1e-12);
  expect_vals(nn::log(f64({1, std::exp(2.0)})), {0, 2}, 1e-12);
  expect_vals(nn::log1p(f64({0, 1e-10})), {0, 1e-10}, 1e-20);
  expect_vals(nn::expm1(f64({0, 1e-10})), {0, 1e-10}, 1e-20);
  expect_vals(nn::abs(nn::tensor({-2, 0, 3})), {2, 0, 3});
  expect_vals(nn::abs(i32({-2, 3})), {2, 3});
  EXPECT_EQ(nn::abs(i32({-2})).dtype(), DType::Int32);
  expect_vals(nn::sign(nn::tensor({-2, 0, 3})), {-1, 0, 1});
  expect_vals(nn::floor(nn::tensor({-1.5, 1.5})), {-2, 1});
  expect_vals(nn::ceil(nn::tensor({-1.5, 1.5})), {-1, 2});
  expect_vals(nn::neg(i32({1, -2})), {-1, 2});
  EXPECT_EQ(nn::neg(i32({1})).dtype(), DType::Int32);
  expect_vals(nn::square(i32({3})), {9});
}

TEST(NnOps, RoundHalfToEven) {
  expect_vals(nn::round(nn::tensor({0.5, 1.5, 2.5, -0.5, -1.5, 2.6, -2.4})), {0, 2, 2, 0, -2, 3, -2});
  expect_vals(nn::round(i32({3})), {3});
}

TEST(NnOps, Trig) {
  Tensor x = f64({0.5});
  expect_vals(nn::sin(x), {std::sin(0.5)}, 1e-12);
  expect_vals(nn::cos(x), {std::cos(0.5)}, 1e-12);
  expect_vals(nn::tan(x), {std::tan(0.5)}, 1e-12);
  expect_vals(nn::asin(x), {std::asin(0.5)}, 1e-12);
  expect_vals(nn::acos(x), {std::acos(0.5)}, 1e-12);
  expect_vals(nn::atan(x), {std::atan(0.5)}, 1e-12);
  expect_vals(nn::sinh(x), {std::sinh(0.5)}, 1e-12);
  expect_vals(nn::cosh(x), {std::cosh(0.5)}, 1e-12);
}

TEST(NnOps, IntegerInputToFloatOps) {
  Tensor e = nn::exp(i32({0, 1}));
  EXPECT_EQ(e.dtype(), DType::Float32);
  expect_vals(e, {1, std::exp(1.0)});
  EXPECT_EQ(nn::sqrt(i64({4})).dtype(), DType::Float32);
}

TEST(NnOps, SpecialFunctions) {
  constexpr double euler = 0.57721566490153286;
  expect_vals(nn::erf(f64({0, 1, -1})), {0, 0.8427007929497149, -0.8427007929497149}, 1e-12);
  expect_vals(nn::erfc(f64({0, 1})), {1, 0.15729920705028513}, 1e-12);
  expect_vals(nn::lgamma(f64({1, 0.5, 5})), {0, 0.5 * std::log(kPi), std::log(24.0)}, 1e-12);
  // digamma(1) = -gamma, digamma(0.5) = -gamma - 2 ln 2, digamma(10) = H_9 - gamma
  double h9 = 0;
  for (int k = 1; k <= 9; ++k) h9 += 1.0 / k;
  expect_vals(nn::digamma(f64({1, 0.5, 10, -0.5})),
              {-euler, -euler - 2 * std::log(2.0), h9 - euler, 0.03648997397857652}, 1e-10);
  // trigamma(1) = pi^2/6, trigamma(0.5) = pi^2/2, trigamma(-0.5) = pi^2/2 + 4
  expect_vals(nn::trigamma(f64({1, 0.5, -0.5})), {kPi * kPi / 6, kPi * kPi / 2, kPi * kPi / 2 + 4}, 1e-9);
  expect_vals(nn::digamma(f64({0})), {std::numeric_limits<double>::quiet_NaN()});
}

TEST(NnOps, Clip) {
  expect_vals(nn::clip(nn::tensor({-2, 0.5, 3}), 0, 1), {0, 0.5, 1});
  expect_vals(nn::clip(nn::tensor({-2, 0.5, 3}), nn::tensor({-1, 1, 0}), nn::scalar(2)), {-1, 1, 2});
  EXPECT_THROW(nn::clip(nn::tensor({1}), 2, 1), std::invalid_argument);
}

// ---- activations -----------------------------------------------------------------------------------

TEST(NnOps, Activations) {
  Tensor x = f64({-1, 0, 1, 7});
  expect_vals(nn::relu(x), {0, 0, 1, 7});
  expect_vals(nn::relu6(x), {0, 0, 1, 6});
  expect_vals(nn::leaky_relu(x), {-0.2, 0, 1, 7});
  expect_vals(nn::leaky_relu(x, 0.1), {-0.1, 0, 1, 7});
  expect_vals(nn::elu(x), {std::exp(-1.0) - 1, 0, 1, 7}, 1e-12);
  const double scale = 1.0507009873554805, alpha = 1.6732632423543772;
  expect_vals(nn::selu(x), {scale * alpha * (std::exp(-1.0) - 1), 0, scale, 7 * scale}, 1e-12);
  const double s1 = 1 / (1 + std::exp(-1.0));
  expect_vals(nn::sigmoid(x), {1 - s1, 0.5, s1, 1 / (1 + std::exp(-7.0))}, 1e-12);
  expect_vals(nn::tanh(x), {std::tanh(-1.0), 0, std::tanh(1.0), std::tanh(7.0)}, 1e-12);
  expect_vals(nn::softplus(x), {std::log1p(std::exp(-1.0)), std::log(2.0), std::log1p(std::exp(1.0)),
                                std::log1p(std::exp(7.0))},
              1e-12);
  expect_vals(nn::softsign(x), {-0.5, 0, 0.5, 7.0 / 8}, 1e-12);
  expect_vals(nn::silu(f64({1})), {s1}, 1e-12);
  expect_vals(nn::gelu(f64({1, -1})), {0.8413447460685429, -0.15865525393145707}, 1e-12);
  expect_vals(nn::gelu(f64({1}), true), {0.8411919906082768}, 1e-12);
  expect_vals(nn::mish(f64({1})), {std::tanh(std::log1p(std::exp(1.0)))}, 1e-12);
  expect_vals(nn::log_sigmoid(f64({0})), {-std::log(2.0)}, 1e-12);
}

TEST(NnOps, ActivationsAreStableAtExtremes) {
  Tensor big = nn::tensor({-1000, 1000});
  expect_vals(nn::sigmoid(big), {0, 1});
  expect_vals(nn::softplus(big), {0, 1000});
  expect_vals(nn::log_sigmoid(big), {-1000, 0});
  expect_vals(nn::tanh(big), {-1, 1});
}

// ---- comparisons / logical / where --------------------------------------------------------------------

TEST(NnOps, Comparisons) {
  Tensor a = nn::tensor({1, 2, 3}), b = nn::tensor({3, 2, 1});
  EXPECT_EQ(nn::equal(a, b).dtype(), DType::Bool);
  expect_vals(nn::equal(a, b), {0, 1, 0});
  expect_vals(nn::not_equal(a, b), {1, 0, 1});
  expect_vals(nn::less(a, b), {1, 0, 0});
  expect_vals(nn::less_equal(a, b), {1, 1, 0});
  expect_vals(nn::greater(a, b), {0, 0, 1});
  expect_vals(nn::greater_equal(a, b), {0, 1, 1});
  expect_vals(nn::greater(a, 1.5), {0, 1, 1});
  expect_vals(nn::equal(a, 2.0), {0, 1, 0});
  expect_vals(nn::not_equal(a, 2.0), {1, 0, 1});
  expect_vals(nn::less(a, 2.0), {1, 0, 0});
  expect_vals(nn::less_equal(a, 2.0), {1, 1, 0});
  expect_vals(nn::greater_equal(a, 2.0), {0, 1, 1});
  // comparing ints with a fractional scalar does not truncate it
  expect_vals(nn::less(i32({2, 3}), 2.5), {1, 0});
  // broadcasting
  EXPECT_EQ(nn::less(x23(), nn::tensor({2, 5, 1})).shape(), (Shape{2, 3}));
}

TEST(NnOps, Logical) {
  Tensor t = nn::tensor({1, 1, 0, 0}, {}, DType::Bool), f = nn::tensor({1, 0, 1, 0}, {}, DType::Bool);
  expect_vals(nn::logical_and(t, f), {1, 0, 0, 0});
  expect_vals(nn::logical_or(t, f), {1, 1, 1, 0});
  expect_vals(nn::logical_xor(t, f), {0, 1, 1, 0});
  expect_vals(nn::logical_not(t), {0, 0, 1, 1});
  // non-bool inputs are treated as nonzero
  expect_vals(nn::logical_and(nn::tensor({2, 0}), nn::tensor({-1, 1})), {1, 0});
}

TEST(NnOps, FiniteNan) {
  const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
  Tensor       x   = f64({1, inf, -inf, nan});
  expect_vals(nn::is_finite(x), {1, 0, 0, 0});
  expect_vals(nn::is_nan(x), {0, 0, 0, 1});
  expect_vals(nn::is_finite(i32({1})), {1});
  expect_vals(nn::is_nan(i32({1})), {0});
}

TEST(NnOps, WhereBroadcasts) {
  Tensor cond = nn::tensor({1, 0}, {2, 1}, DType::Bool);
  Tensor r    = nn::where(cond, nn::tensor({1, 2, 3}), nn::scalar(-1));
  EXPECT_EQ(r.shape(), (Shape{2, 3}));
  expect_vals(r, {1, 2, 3, -1, -1, -1});
  expect_vals(nn::where(nn::greater(nn::tensor({-1, 2}), 0.0), nn::tensor({10, 20}), 0.0), {0, 20});
  expect_vals(nn::where(nn::tensor({1, 0}, {}, DType::Bool), 5.0, nn::tensor({7, 8})), {5, 8});
  // promotes the branches
  EXPECT_EQ(nn::where(nn::tensor({1}, {}, DType::Bool), i32({1}), f64({2})).dtype(), DType::Float64);
  // float conditions are cast to bool
  expect_vals(nn::where(nn::tensor({0.5, 0}), nn::tensor({1, 1}), nn::tensor({2, 2})), {1, 2});
}

// ---- reductions --------------------------------------------------------------------------------------

TEST(NnOps, SumAxes) {
  Tensor x = x23();
  expect_vals(nn::sum(x), {21});
  EXPECT_EQ(nn::sum(x).dim(), 0);
  expect_vals(nn::sum(x, {0}), {5, 7, 9});
  expect_vals(nn::sum(x, {1}), {6, 15});  // tensor_basics: a.sum(axis=1)
  expect_vals(nn::sum(x, {-1}), {6, 15});
  expect_vals(nn::sum(x, {0, 1}), {21});
  EXPECT_EQ(nn::sum(x, {1}, true).shape(), (Shape{2, 1}));
  EXPECT_EQ(nn::sum(x, {}, true).shape(), (Shape{1, 1}));
  EXPECT_EQ(x.sum({0}).shape(), (Shape{3}));
  EXPECT_THROW(nn::sum(x, {2}), std::invalid_argument);
  EXPECT_THROW(nn::sum(x, {0, 0}), std::invalid_argument);
  // works on views
  expect_vals(nn::sum(x.transpose(), {1}), {5, 7, 9});
  // dtypes
  EXPECT_EQ(nn::sum(i32({1, 2})).dtype(), DType::Int32);
  Tensor bs = nn::sum(nn::tensor({1, 1, 0}, {}, DType::Bool));
  EXPECT_EQ(bs.dtype(), DType::Int64);
  EXPECT_EQ(bs.item(), 2);
  // empty sums are zero
  expect_vals(nn::sum(nn::zeros({0, 3}), {0}), {0, 0, 0});
}

TEST(NnOps, SumMiddleAxisOf3D) {
  Tensor x = nn::arange(0, 24).reshape({2, 3, 4});
  Tensor s = nn::sum(x, {1});
  EXPECT_EQ(s.shape(), (Shape{2, 4}));
  expect_vals(s, {12, 15, 18, 21, 48, 51, 54, 57});
  expect_vals(nn::sum(x, {0, 2}), {60, 92, 124});
}

TEST(NnOps, Mean) {
  expect_vals(nn::mean(x23()), {3.5});
  expect_vals(nn::mean(x23(), {0}), {2.5, 3.5, 4.5});
  Tensor mi = nn::mean(i32({1, 2}));
  EXPECT_EQ(mi.dtype(), DType::Float32);
  expect_vals(mi, {1.5});
  EXPECT_EQ(x23().mean({1}, true).shape(), (Shape{2, 1}));
}

TEST(NnOps, MaxMinProd) {
  Tensor x = nn::tensor({{1, 7, 3}, {4, -5, 6}});
  expect_vals(nn::max(x), {7});
  expect_vals(nn::max(x, {0}), {4, 7, 6});
  expect_vals(nn::min(x, {1}), {1, -5});
  expect_vals(nn::min(x, {1}, true), {1, -5});
  EXPECT_EQ(nn::max(i32({3, 9})).dtype(), DType::Int32);
  expect_vals(nn::max(i32({3, 9})), {9});
  expect_vals(nn::prod(x23(), {1}), {6, 120});
  expect_vals(nn::prod(x23()), {720});
  expect_vals(nn::prod(i64({2, 3})), {6});
  EXPECT_THROW(nn::max(nn::zeros({0})), std::invalid_argument);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  expect_vals(nn::max(f64({1, nan, 3})), {nan});
}

TEST(NnOps, VarianceStd) {
  Tensor x = f64({1, 2, 3, 4});
  expect_vals(nn::variance(x), {1.25}, 1e-12);
  expect_vals(nn::std_dev(x), {std::sqrt(1.25)}, 1e-12);
  expect_vals(nn::variance(nn::tensor({{1, 3}, {2, 2}}), {1}), {1, 0});
}

TEST(NnOps, LogSumExpStable) {
  expect_vals(nn::logsumexp(f64({1000, 1000})), {1000 + std::log(2.0)}, 1e-9);
  expect_vals(nn::logsumexp(f64({-1000, -1000})), {-1000 + std::log(2.0)}, 1e-9);
  expect_vals(nn::logsumexp(f64({0, std::log(3.0)})), {std::log(4.0)}, 1e-12);
  Tensor l = nn::logsumexp(nn::tensor({{0, 0}, {1000, 1000}}), {1});
  EXPECT_EQ(l.shape(), (Shape{2}));
  expect_vals(l, {std::log(2.0), 1000 + std::log(2.0)}, 1e-3);
  EXPECT_EQ(nn::logsumexp(x23(), {1}, true).shape(), (Shape{2, 1}));
  const double ninf = -std::numeric_limits<double>::infinity();
  expect_vals(nn::logsumexp(f64({ninf, ninf})), {ninf});
}

TEST(NnOps, ArgMaxArgMin) {
  Tensor x = nn::tensor({{1, 3, 3}, {9, 2, 9}});
  Tensor a = nn::argmax(x, 1);
  EXPECT_EQ(a.dtype(), DType::Int64);
  expect_vals(a, {1, 0});  // first occurrence
  expect_vals(nn::argmax(x, 0), {1, 0, 1});
  expect_vals(nn::argmin(x, -1), {0, 1});
  EXPECT_EQ(nn::argmax(x, 1, true).shape(), (Shape{2, 1}));
  expect_vals(nn::argmax(nn::tensor({5, 1, 7})), {2});
}

TEST(NnOps, AnyAllCountNonzero) {
  Tensor b = nn::tensor({{1, 0}, {0, 0}}, DType::Bool);
  expect_vals(nn::any(b), {1});
  expect_vals(nn::all(b), {0});
  expect_vals(nn::any(b, {1}), {1, 0});
  expect_vals(nn::all(nn::tensor({1, 2})), {1});
  EXPECT_EQ(nn::any(b).dtype(), DType::Bool);
  Tensor c = nn::count_nonzero(nn::tensor({{0, 1, 2}, {0, 0, 3}}), {1});
  EXPECT_EQ(c.dtype(), DType::Int64);
  expect_vals(c, {2, 1});
}

TEST(NnOps, SoftmaxRowsSumToOne) {
  Tensor x = nn::tensor({{1, 2, 3}, {1000, 1000, 1000}});
  Tensor s = nn::softmax(x);
  expect_vals(nn::sum(s, {1}), {1, 1});
  const double e = std::exp(1.0), z = 1 + e + e * e;
  expect_vals(s, {1 / z, e / z, e * e / z, 1.0 / 3, 1.0 / 3, 1.0 / 3});
  Tensor ls = nn::log_softmax(x);
  expect_vals(ls, {-std::log(z), 1 - std::log(z), 2 - std::log(z), -std::log(3.0), -std::log(3.0), -std::log(3.0)});
  expect_vals(nn::sum(nn::exp(ls), {1}), {1, 1});
  // other axis
  expect_vals(nn::sum(nn::softmax(x23(), 0), {0}), {1, 1, 1});
  EXPECT_EQ(nn::softmax(i32({1, 2})).dtype(), DType::Float32);
}

TEST(NnOps, Cumsum) {
  Tensor x = nn::tensor({1, 2, 3});
  expect_vals(nn::cumsum(x, 0), {1, 3, 6});
  expect_vals(nn::cumsum(x, 0, true), {0, 1, 3});
  expect_vals(nn::cumsum(x, 0, false, true), {6, 5, 3});
  expect_vals(nn::cumsum(x, 0, true, true), {5, 3, 0});
  expect_vals(nn::cumsum(x23(), 0), {1, 2, 3, 5, 7, 9});
  expect_vals(nn::cumsum(x23(), -1), {1, 3, 6, 4, 9, 15});
  EXPECT_EQ(nn::cumsum(i32({1, 2}), 0).dtype(), DType::Int32);
}

TEST(NnOps, Norm) {
  expect_vals(nn::norm(nn::tensor({3, 4})), {5});
  // tensor_basics: Frobenius norm of [[1,2,3],[4,5,6]]
  expect_vals(nn::norm(x23()), {std::sqrt(91.0)});
  expect_vals(nn::norm(nn::tensor({{3, 4}, {6, 8}}), {1}), {5, 10});
}

// ---- shape ops ------------------------------------------------------------------------------------------

TEST(NnOps, Reshape) {
  Tensor x = x23();
  EXPECT_EQ(nn::reshape(x, {3, 2}).shape(), (Shape{3, 2}));  // TF Reshape(x,[3,2])
  expect_vals(nn::reshape(x, {3, 2}), {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(nn::reshape(x, {-1}).shape(), (Shape{6}));  // TF Reshape(x,[-1])
  EXPECT_EQ(nn::reshape(x, {2, -1, 1}).shape(), (Shape{2, 3, 1}));
  EXPECT_EQ(nn::flatten(x).shape(), (Shape{6}));
  EXPECT_THROW(nn::reshape(x, {4, -1}), std::invalid_argument);
  EXPECT_THROW(nn::reshape(x, {-1, -1}), std::invalid_argument);
  EXPECT_THROW(nn::reshape(x, {5}), std::invalid_argument);
  // reshape of a non-contiguous view copies in logical order
  expect_vals(nn::reshape(x.transpose(), {6}), {1, 4, 2, 5, 3, 6});
  EXPECT_EQ(nn::reshape(nn::scalar(1), {1, 1}).shape(), (Shape{1, 1}));
}

TEST(NnOps, ExpandDimsSqueeze) {
  Tensor x = x23();
  EXPECT_EQ(nn::expand_dims(x, 0).shape(), (Shape{1, 2, 3}));  // TF ExpandDims(x,0)
  EXPECT_EQ(nn::expand_dims(x, -1).shape(), (Shape{2, 3, 1}));
  EXPECT_EQ(nn::expand_dims(x, 1).shape(), (Shape{2, 1, 3}));
  EXPECT_EQ(nn::squeeze(nn::expand_dims(x, 0)).shape(), (Shape{2, 3}));  // TF Squeeze
  EXPECT_EQ(nn::squeeze(nn::zeros({1, 2, 1}), {0}).shape(), (Shape{2, 1}));
  EXPECT_EQ(nn::squeeze(nn::zeros({1, 2, 1}), {-1}).shape(), (Shape{1, 2}));
  EXPECT_THROW(nn::squeeze(x, {0}), std::invalid_argument);
}

TEST(NnOps, Transpose) {
  Tensor x = x23();
  Tensor t = nn::transpose(x, {1, 0});  // TF Transpose(x,[1,0])
  EXPECT_EQ(t.shape(), (Shape{3, 2}));
  expect_vals(t, {1, 4, 2, 5, 3, 6});
  Tensor c = nn::arange(0, 24).reshape({2, 3, 4});
  Tensor p = nn::transpose(c, {2, 0, 1});
  EXPECT_EQ(p.shape(), (Shape{4, 2, 3}));
  EXPECT_EQ(p.at({3, 1, 2}), c.at({1, 2, 3}));
  EXPECT_EQ(nn::transpose(c).shape(), (Shape{4, 3, 2}));
  EXPECT_EQ(nn::swapaxes(c, 0, -1).shape(), (Shape{4, 3, 2}));
  EXPECT_EQ(nn::swapaxes(c, 0, -1).at({3, 2, 1}), c.at({1, 2, 3}));
  EXPECT_THROW(nn::transpose(c, {0, 0, 1}), std::invalid_argument);
  EXPECT_THROW(nn::transpose(c, {0, 1}), std::invalid_argument);
}

TEST(NnOps, SliceTfSemantics) {
  Tensor x = x23();
  Tensor s = nn::slice(x, {0, 1}, {2, 2});  // TF Slice(x,[0,1],[2,2])
  EXPECT_EQ(s.shape(), (Shape{2, 2}));
  expect_vals(s, {2, 3, 5, 6});
  expect_vals(nn::slice(x, {1, 0}, {-1, -1}), {4, 5, 6});
  EXPECT_THROW(nn::slice(x, {0, 2}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(nn::slice(x, {0}, {1}), std::invalid_argument);
}

TEST(NnOps, SliceAxisPythonSemantics) {
  Tensor x = nn::arange(0, 6);
  expect_vals(nn::slice_axis(x, 0, 1, 4), {1, 2, 3});
  expect_vals(nn::slice_axis(x, 0, -2, 100), {4, 5});
  expect_vals(nn::slice_axis(x, 0, 4, 0, -2), {4, 2});
  expect_vals(nn::slice_axis(x, 0, -1, -7, -1), {5, 4, 3, 2, 1, 0});
  expect_vals(nn::slice_axis(x, 0, 0, 6, 4), {0, 4});
  EXPECT_EQ(nn::slice_axis(x, 0, 4, 2).numel(), 0);
  EXPECT_THROW(nn::slice_axis(x, 0, 0, 1, 0), std::invalid_argument);
  expect_vals(nn::slice_axis(x23(), 1, 1, 3), {2, 3, 5, 6});
}

TEST(NnOps, StridedSlice) {
  Tensor x = x23();
  // TF StridedSlice(x,[0,0],[2,3],[1,2])
  Tensor s = nn::strided_slice(x, {0, 0}, {2, 3}, {1, 2});
  EXPECT_EQ(s.shape(), (Shape{2, 2}));
  expect_vals(s, {1, 3, 4, 6});
  // only the leading axes
  expect_vals(nn::strided_slice(x, {1}, {2}), {4, 5, 6});
  // reverse rows
  expect_vals(nn::strided_slice(x, {-1}, {-3}, {-1}), {4, 5, 6, 1, 2, 3});
}

TEST(NnOps, Select) {
  Tensor x = x23();
  Tensor r = nn::select(x, 0, 1);  // TF x[1] via StridedSlice shrink
  EXPECT_EQ(r.shape(), (Shape{3}));
  expect_vals(r, {4, 5, 6});
  expect_vals(nn::select(x, 1, -1), {3, 6});
  EXPECT_THROW(nn::select(x, 0, 2), std::out_of_range);
}

TEST(NnOps, ConcatSplitStack) {
  Tensor x = x23();
  Tensor c0 = nn::concat({x, x}, 0);  // TF Concat([x,x],axis=0)
  EXPECT_EQ(c0.shape(), (Shape{4, 3}));
  expect_vals(c0, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
  Tensor c1 = nn::concat({x, nn::tensor({{7}, {8}})}, 1);
  expect_vals(c1, {1, 2, 3, 7, 4, 5, 6, 8});
  EXPECT_THROW(nn::concat({x, nn::zeros({3, 3})}, 1), std::invalid_argument);
  EXPECT_EQ(nn::concat({i32({1}), nn::tensor({2})}, 0).dtype(), DType::Float32);

  auto parts = nn::split(x, Shape{1, -1}, 1);
  ASSERT_EQ(parts.size(), 2u);
  expect_vals(parts[0], {1, 4});
  expect_vals(parts[1], {2, 3, 5, 6});
  auto thirds = nn::split(x, 3, 1);
  ASSERT_EQ(thirds.size(), 3u);
  expect_vals(thirds[2], {3, 6});
  EXPECT_THROW(nn::split(x, 2, 1), std::invalid_argument);
  EXPECT_THROW(nn::split(x, Shape{1, 1}, 1), std::invalid_argument);

  Tensor st = nn::stack({x, x});  // TF Stack([x,x])
  EXPECT_EQ(st.shape(), (Shape{2, 2, 3}));
  Tensor st1 = nn::stack({nn::tensor({1, 2}), nn::tensor({3, 4})}, 1);
  EXPECT_EQ(st1.shape(), (Shape{2, 2}));
  expect_vals(st1, {1, 3, 2, 4});
  EXPECT_THROW(nn::stack({nn::tensor({1}), nn::tensor({1, 2})}), std::invalid_argument);

  auto rows = nn::unstack(x);
  ASSERT_EQ(rows.size(), 2u);
  EXPECT_EQ(rows[1].shape(), (Shape{3}));
  expect_vals(rows[1], {4, 5, 6});
  auto cols = nn::unstack(x, 1);
  ASSERT_EQ(cols.size(), 3u);
  expect_vals(cols[0], {1, 4});
}

TEST(NnOps, TileRepeat) {
  Tensor t = nn::tile(nn::tensor({{1, 2}}), {2, 2});
  EXPECT_EQ(t.shape(), (Shape{2, 4}));
  expect_vals(t, {1, 2, 1, 2, 1, 2, 1, 2});
  expect_vals(nn::tile(nn::tensor({1, 2}), {3}), {1, 2, 1, 2, 1, 2});
  EXPECT_THROW(nn::tile(nn::tensor({1, 2}), {2, 2}), std::invalid_argument);
  expect_vals(nn::repeat(nn::tensor({1, 2}), 2, 0), {1, 1, 2, 2});
  Tensor r = nn::repeat(nn::tensor({{1, 2}, {3, 4}}), 2, 1);
  EXPECT_EQ(r.shape(), (Shape{2, 4}));
  expect_vals(r, {1, 1, 2, 2, 3, 3, 4, 4});
}

TEST(NnOps, BroadcastToSumTo) {
  Tensor row = nn::tensor({10, 20, 30});
  Tensor b   = nn::broadcast_to(row, {2, 3});  // TF BroadcastTo(row,[2,3])
  EXPECT_EQ(b.shape(), (Shape{2, 3}));
  EXPECT_EQ(b.impl()->storage, row.impl()->storage);
  expect_vals(b, {10, 20, 30, 10, 20, 30});
  EXPECT_THROW(nn::broadcast_to(row, {2, 2}), std::invalid_argument);
  EXPECT_TRUE(nn::broadcast_to(row, {3}).is_same(row));
  expect_vals(nn::sum_to(x23(), {3}), {5, 7, 9});
  expect_vals(nn::sum_to(x23(), {2, 1}), {6, 15});
  expect_vals(nn::sum_to(x23(), {}), {21});
  EXPECT_EQ(nn::sum_to(x23(), {1, 3}).shape(), (Shape{1, 3}));
  EXPECT_THROW(nn::sum_to(x23(), {2}), std::invalid_argument);
}

TEST(NnOps, PadConstant) {
  Tensor p = nn::pad(nn::tensor({{1, 2}, {3, 4}}), {{1, 0}, {0, 2}});
  EXPECT_EQ(p.shape(), (Shape{3, 4}));
  expect_vals(p, {0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0});
  expect_vals(nn::pad(nn::tensor({1}), {{1, 1}}, nn::PadMode::Constant, 9), {9, 1, 9});
  EXPECT_THROW(nn::pad(nn::tensor({1}), {{-1, 0}}), std::invalid_argument);
  EXPECT_THROW(nn::pad(nn::tensor({1}), {{1, 0}, {0, 0}}), std::invalid_argument);
}

TEST(NnOps, PadMirror) {
  // numpy.pad([1,2,3], 2, 'reflect') = [3,2,1,2,3,2,1]; 'symmetric' = [2,1,1,2,3,3,2]
  expect_vals(nn::pad(nn::tensor({1, 2, 3}), {{2, 2}}, nn::PadMode::Reflect), {3, 2, 1, 2, 3, 2, 1});
  expect_vals(nn::pad(nn::tensor({1, 2, 3}), {{2, 2}}, nn::PadMode::Symmetric), {2, 1, 1, 2, 3, 3, 2});
  // 2-D: numpy.pad([[1,2,3],[4,5,6]], ((1,1),(1,1)), 'reflect')
  Tensor r = nn::pad(x23(), {{1, 1}, {1, 1}}, nn::PadMode::Reflect);
  EXPECT_EQ(r.shape(), (Shape{4, 5}));
  expect_vals(r, {5, 4, 5, 6, 5, 2, 1, 2, 3, 2, 5, 4, 5, 6, 5, 2, 1, 2, 3, 2});
  EXPECT_THROW(nn::pad(nn::tensor({1, 2, 3}), {{3, 0}}, nn::PadMode::Reflect), std::invalid_argument);
  EXPECT_NO_THROW(nn::pad(nn::tensor({1, 2, 3}), {{3, 0}}, nn::PadMode::Symmetric));
}

TEST(NnOps, Reverse) {
  expect_vals(nn::reverse(x23(), {1}), {3, 2, 1, 6, 5, 4});
  expect_vals(nn::reverse(x23(), {0}), {4, 5, 6, 1, 2, 3});
  expect_vals(nn::reverse(x23(), {0, -1}), {6, 5, 4, 3, 2, 1});
}

// ---- index ops -----------------------------------------------------------------------------------------

TEST(NnOps, Gather) {
  Tensor x = nn::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  expect_vals(nn::gather(x, i64({2, 0})), {7, 8, 9, 1, 2, 3});
  Tensor g1 = nn::gather(x, i32({2, 0}), 1);
  EXPECT_EQ(g1.shape(), (Shape{3, 2}));
  expect_vals(g1, {3, 1, 6, 4, 9, 7});
  // 2-D indices: shape x[:axis] + indices.shape + x[axis+1:]
  Tensor idx = nn::tensor({0, 1, 1, 2}, {2, 2}, DType::Int64);
  Tensor g2  = nn::gather(x, idx, 0);
  EXPECT_EQ(g2.shape(), (Shape{2, 2, 3}));
  expect_vals(nn::gather(x, idx, 1), {1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9});
  // scalar index drops the axis
  Tensor g3 = nn::gather(x, nn::scalar(1, DType::Int64), 0);
  EXPECT_EQ(g3.shape(), (Shape{3}));
  expect_vals(g3, {4, 5, 6});
  // negative indices count from the end
  expect_vals(nn::gather(nn::tensor({1, 2, 3}), i64({-1})), {3});
  EXPECT_THROW(nn::gather(x, i64({3})), std::out_of_range);
  EXPECT_THROW(nn::gather(x, nn::tensor({0})), std::invalid_argument);
}

TEST(NnOps, GatherNd) {
  Tensor x = nn::tensor({{1, 2}, {3, 4}});
  expect_vals(nn::gather_nd(x, nn::tensor({0, 1, 1, 0}, {2, 2}, DType::Int64)), {2, 3});
  Tensor rows = nn::gather_nd(x, nn::tensor({1}, {1, 1}, DType::Int64));
  EXPECT_EQ(rows.shape(), (Shape{1, 2}));
  expect_vals(rows, {3, 4});
  Tensor single = nn::gather_nd(x, i64({1, 1}));
  EXPECT_EQ(single.dim(), 0);
  expect_vals(single, {4});
  EXPECT_THROW(nn::gather_nd(x, nn::tensor({2, 0}, {1, 2}, DType::Int64)), std::out_of_range);
}

TEST(NnOps, OneHot) {
  Tensor o = nn::one_hot(i64({0, 2, -1, 5}), 3);
  EXPECT_EQ(o.shape(), (Shape{4, 3}));
  EXPECT_EQ(o.dtype(), DType::Float32);
  expect_vals(o, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  Tensor o2 = nn::one_hot(nn::tensor({1, 0}, {2, 1}, DType::Int32), 2, 5, -1, DType::Int64);
  EXPECT_EQ(o2.shape(), (Shape{2, 1, 2}));
  EXPECT_EQ(o2.dtype(), DType::Int64);
  expect_vals(o2, {-1, 5, 5, -1});
}

TEST(NnOps, IndexAdd) {
  Tensor r = nn::index_add(nn::zeros({3, 2}), 0, i64({0, 0, 2}), nn::tensor({{1, 1}, {2, 2}, {3, 3}}));
  expect_vals(r, {3, 3, 0, 0, 3, 3});
  Tensor c = nn::index_add(nn::ones({2, 3}), 1, i64({2}), nn::tensor({{5}, {6}}));
  expect_vals(c, {1, 1, 6, 1, 1, 7});
}

TEST(NnOps, UnsortedSegmentSumMean) {
  // ids outside [0, n) are dropped
  expect_vals(nn::unsorted_segment_sum(nn::tensor({1, 2, 3, 4}), i64({0, 2, 0, 5}), 3), {4, 0, 2});
  expect_vals(nn::unsorted_segment_sum(nn::tensor({1, 2, 3}), i64({-1, 1, 1}), 2), {0, 5});
  Tensor rows = nn::unsorted_segment_sum(nn::tensor({{1, 2}, {3, 4}, {5, 6}}), i32({1, 0, 1}), 2);
  EXPECT_EQ(rows.shape(), (Shape{2, 2}));
  expect_vals(rows, {3, 4, 6, 8});
  expect_vals(nn::unsorted_segment_mean(nn::tensor({1, 3, 5}), i64({0, 0, 1}), 3), {2, 5, 0});
  Tensor m = nn::unsorted_segment_mean(nn::tensor({{1, 2}, {3, 4}}), i64({1, 1}), 2);
  expect_vals(m, {0, 0, 2, 3});
  EXPECT_THROW(nn::unsorted_segment_sum(nn::tensor({1, 2}), i64({0, 0, 0}), 2), std::invalid_argument);
}

TEST(NnOps, ScatterNd) {
  Tensor idx = nn::tensor({1, 3, 1}, {3, 1}, DType::Int64);
  expect_vals(nn::scatter_nd(idx, nn::tensor({10, 20, 5}), {4}), {0, 15, 0, 20});
  expect_vals(nn::scatter_nd_add(nn::ones({4}), idx, nn::tensor({10, 20, 5})), {1, 16, 1, 21});
  // row updates
  Tensor rows = nn::scatter_nd(nn::tensor({1}, {1, 1}, DType::Int64), nn::tensor({{7, 8}}), {2, 2});
  expect_vals(rows, {0, 0, 7, 8});
  // full coordinates
  Tensor pts = nn::scatter_nd(nn::tensor({0, 1, 1, 0}, {2, 2}, DType::Int64), nn::tensor({3, 4}), {2, 2});
  expect_vals(pts, {0, 3, 4, 0});
  EXPECT_THROW(nn::scatter_nd(idx, nn::tensor({1, 2}), {4}), std::invalid_argument);
}

TEST(NnOps, TopK) {
  auto [v, i] = nn::top_k(nn::tensor({1, 3, 3, 2}), 2);
  expect_vals(v, {3, 3});
  expect_vals(i, {1, 2});  // ties keep the lower index first
  EXPECT_EQ(i.dtype(), DType::Int64);
  auto [v2, i2] = nn::top_k(nn::tensor({{5, 1, 4}, {0, 9, 2}}), 2);
  EXPECT_EQ(v2.shape(), (Shape{2, 2}));
  expect_vals(v2, {5, 4, 9, 2});
  expect_vals(i2, {0, 2, 1, 2});
  auto [v3, i3] = nn::top_k(nn::tensor({2, 1}), 2);
  expect_vals(v3, {2, 1});
  EXPECT_THROW(nn::top_k(nn::tensor({1}), 2), std::invalid_argument);
}

TEST(NnOps, ArgsortStable) {
  expect_vals(nn::argsort(nn::tensor({3, 1, 2, 1})), {1, 3, 2, 0});
  expect_vals(nn::argsort(nn::tensor({3, 1, 2, 1}), true), {0, 2, 1, 3});
  expect_vals(nn::argsort(nn::tensor({{2, 1}, {0, 5}})), {1, 0, 0, 1});
}

// ---- matrix ops ---------------------------------------------------------------------------------------

TEST(NnOps, Matmul2D) {
  // tensor_basics: a.contract(b) with b = [[1,0],[0,1],[1,1]]
  Tensor a = x23();
  Tensor b = nn::tensor({{1, 0}, {0, 1}, {1, 1}});
  Tensor c = nn::matmul(a, b);
  EXPECT_EQ(c.shape(), (Shape{2, 2}));
  expect_vals(c, {4, 5, 10, 11});
  // a * a^T (tensor_basics gram matrix)
  expect_vals(nn::matmul(a, a, false, true), {14, 32, 32, 77});
  expect_vals(nn::matmul(a, a, true, false), {17, 22, 27, 22, 29, 36, 27, 36, 45});
  expect_vals(nn::matmul(b, a, true, true), {4, 10, 5, 11});
  // non-contiguous inputs
  expect_vals(nn::matmul(a.transpose(), nn::tensor({{1}, {1}})), {5, 7, 9});
}

TEST(NnOps, MatmulBatched) {
  Tensor a = nn::arange(0, 12).reshape({2, 2, 3});
  Tensor b = nn::tensor({{1, 0}, {0, 1}, {1, 1}});
  Tensor c = nn::matmul(a, b);  // batch [2] broadcast against []
  EXPECT_EQ(c.shape(), (Shape{2, 2, 2}));
  expect_vals(c, {2, 3, 8, 9, 14, 15, 20, 21});
  // [2,1,2,3] @ [3,3,2] -> [2,3,2,2]
  Tensor a4 = nn::ones({2, 1, 2, 3});
  Tensor b3 = nn::arange(0, 18).reshape({3, 3, 2});
  Tensor c4 = nn::matmul(a4, b3);
  EXPECT_EQ(c4.shape(), (Shape{2, 3, 2, 2}));
  expect_vals(nn::select(nn::select(c4, 0, 1), 0, 2), {42, 45, 42, 45});
}

TEST(NnOps, MatmulDtypesAndErrors) {
  Tensor ai = nn::tensor({{1, 2}, {3, 4}}, DType::Int64);
  Tensor ci = nn::matmul(ai, ai);
  EXPECT_EQ(ci.dtype(), DType::Int64);
  expect_vals(ci, {7, 10, 15, 22});
  EXPECT_EQ(nn::matmul(nn::cast(ai, DType::Int32), nn::eye(2, DType::Float64)).dtype(), DType::Float64);
  EXPECT_THROW(nn::matmul(nn::ones({2, 3}), nn::ones({2, 3})), std::invalid_argument);
  EXPECT_THROW(nn::matmul(nn::ones({3}), nn::ones({3, 1})), std::invalid_argument);
  EXPECT_THROW(nn::matmul(nn::ones({2, 2, 3}), nn::ones({3, 3, 1})), std::invalid_argument);
}

TEST(NnOps, BiasAddDotOuter) {
  expect_vals(nn::bias_add(x23(), nn::tensor({10, 20, 30})), {11, 22, 33, 14, 25, 36});
  EXPECT_THROW(nn::bias_add(x23(), nn::tensor({1, 2})), std::invalid_argument);
  expect_vals(nn::dot(nn::tensor({1, 2, 3}), nn::tensor({4, 5, 6})), {32});
  EXPECT_THROW(nn::dot(nn::tensor({1, 2}), nn::tensor({1})), std::invalid_argument);
  Tensor o = nn::outer(nn::tensor({1, 2}), nn::tensor({3, 4, 5}));
  EXPECT_EQ(o.shape(), (Shape{2, 3}));
  expect_vals(o, {3, 4, 5, 6, 8, 10});
}
