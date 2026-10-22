// Value tests for nn::Tensor: storage, factories, dtypes, views, in-place ops, printing.

#include <gtest/gtest.h>

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace datamunge;
using nn::DType;
using nn::Shape;
using nn::Tensor;

namespace {

std::vector<double> vals(const Tensor& t) { return t.to_vector<double>(); }

void expect_vals(const Tensor& t, const std::vector<double>& expected, double tol = 1e-6) {
  const auto v = vals(t);
  ASSERT_EQ(v.size(), expected.size()) << t.str();
  for (std::size_t i = 0; i < v.size(); ++i) EXPECT_NEAR(v[i], expected[i], tol) << "element " << i << " of " << t.str();
}

}  // namespace

// ---- factories -------------------------------------------------------------------------

TEST(NnTensor, EmptyHasShapeAndDtype) {
  Tensor t = nn::empty({2, 3}, DType::Float64);
  EXPECT_EQ(t.shape(), (Shape{2, 3}));
  EXPECT_EQ(t.dtype(), DType::Float64);
  EXPECT_EQ(t.numel(), 6);
  EXPECT_EQ(t.dim(), 2);
  EXPECT_EQ(t.size(-1), 3);
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_EQ(t.strides(), (Shape{3, 1}));
}

TEST(NnTensor, ZerosOnesFull) {
  expect_vals(nn::zeros({2, 2}), {0, 0, 0, 0});
  expect_vals(nn::ones({3}, DType::Int32), {1, 1, 1});
  expect_vals(nn::full({2, 2}, 7), {7, 7, 7, 7});  // TF Fill([2,2], 7)
  EXPECT_EQ(nn::full({2}, 7, DType::Int64).dtype(), DType::Int64);
  Tensor z = nn::zeros_like(nn::ones({2, 3}, DType::Float64));
  EXPECT_EQ(z.shape(), (Shape{2, 3}));
  EXPECT_EQ(z.dtype(), DType::Float64);
  expect_vals(nn::full_like(z, 2.5), {2.5, 2.5, 2.5, 2.5, 2.5, 2.5});
  expect_vals(nn::ones_like(z), {1, 1, 1, 1, 1, 1});
}

TEST(NnTensor, ScalarIsRankZero) {
  Tensor s = nn::scalar(3.14);
  EXPECT_EQ(s.dim(), 0);
  EXPECT_EQ(s.numel(), 1);
  EXPECT_TRUE(s.shape().empty());
  EXPECT_NEAR(s.item(), 3.14, 1e-6);
  EXPECT_EQ(s.dtype(), DType::Float32);
}

TEST(NnTensor, Arange) {
  // TF Range(0, 10, 3) -> [0, 3, 6, 9]
  expect_vals(nn::arange(0, 10, 3), {0, 3, 6, 9});
  expect_vals(nn::arange(5, 0, -2), {5, 3, 1});
  EXPECT_EQ(nn::arange(3, 3).numel(), 0);
  EXPECT_EQ(nn::arange(0, 4, 1, DType::Int32).dtype(), DType::Int32);
  EXPECT_THROW(nn::arange(0, 1, 0), std::invalid_argument);
}

TEST(NnTensor, Linspace) {
  expect_vals(nn::linspace(0, 1, 5), {0, 0.25, 0.5, 0.75, 1});
  expect_vals(nn::linspace(2, 9, 1), {2});
  EXPECT_EQ(nn::linspace(0, 1, 0).numel(), 0);
}

TEST(NnTensor, Eye) { expect_vals(nn::eye(3), {1, 0, 0, 0, 1, 0, 0, 0, 1}); }

TEST(NnTensor, FromVectorInfersDtype) {
  EXPECT_EQ(nn::from_vector(std::vector<float>{1, 2}).dtype(), DType::Float32);
  EXPECT_EQ(nn::from_vector(std::vector<double>{1, 2}).dtype(), DType::Float64);
  EXPECT_EQ(nn::from_vector(std::vector<std::int32_t>{1, 2}).dtype(), DType::Int32);
  EXPECT_EQ(nn::from_vector(std::vector<std::int64_t>{1, 2}).dtype(), DType::Int64);
  EXPECT_EQ(nn::from_vector(std::vector<bool>{true, false}).dtype(), DType::Bool);
  Tensor m = nn::from_vector(std::vector<double>{1, 2, 3, 4, 5, 6}, {2, 3});
  EXPECT_EQ(m.shape(), (Shape{2, 3}));
  EXPECT_EQ(m.at({1, 0}), 4);
  EXPECT_THROW(nn::from_vector(std::vector<double>{1, 2, 3}, {2, 2}), std::invalid_argument);
}

TEST(NnTensor, Literals) {
  Tensor v = nn::tensor({1, 2, 3});
  EXPECT_EQ(v.shape(), (Shape{3}));
  EXPECT_EQ(v.dtype(), DType::Float32);
  Tensor m = nn::tensor({1, 2, 3, 4}, {2, 2}, DType::Float64);
  EXPECT_EQ(m.shape(), (Shape{2, 2}));
  EXPECT_EQ(m.dtype(), DType::Float64);
  Tensor r = nn::tensor({{1, 2, 3}, {4, 5, 6}});
  EXPECT_EQ(r.shape(), (Shape{2, 3}));
  expect_vals(r, {1, 2, 3, 4, 5, 6});
  EXPECT_THROW(nn::tensor({{1, 2}, {3}}), std::invalid_argument);
}

TEST(NnTensor, UndefinedTensor) {
  Tensor t;
  EXPECT_FALSE(t.defined());
  EXPECT_THROW((void)t.shape(), std::logic_error);
  EXPECT_EQ(t.str(), "Tensor(undefined)");
}

// ---- dtypes -------------------------------------------------------------------------------

TEST(NnTensor, DtypeHelpers) {
  EXPECT_EQ(nn::dtype_name(DType::Float32), "float32");
  EXPECT_EQ(nn::dtype_name(DType::Int64), "int64");
  EXPECT_EQ(nn::dtype_name(DType::Bool), "bool");
  EXPECT_EQ(nn::dtype_size(DType::Float64), 8u);
  EXPECT_EQ(nn::dtype_size(DType::Int32), 4u);
  EXPECT_EQ(nn::dtype_size(DType::Bool), 1u);
  EXPECT_TRUE(nn::is_floating(DType::Float32));
  EXPECT_FALSE(nn::is_floating(DType::Int64));
  EXPECT_TRUE(nn::is_integral(DType::Int32));
  EXPECT_FALSE(nn::is_integral(DType::Bool));
  EXPECT_EQ(nn::promote_types(DType::Int32, DType::Float32), DType::Float32);
  EXPECT_EQ(nn::promote_types(DType::Float32, DType::Float64), DType::Float64);
  EXPECT_EQ(nn::promote_types(DType::Bool, DType::Int32), DType::Int32);
  EXPECT_EQ(nn::promote_types(DType::Int64, DType::Int32), DType::Int64);
  EXPECT_EQ(nn::to_floating(DType::Int64), DType::Float32);
  EXPECT_EQ(nn::to_floating(DType::Float64), DType::Float64);
}

TEST(NnTensor, DataPtrChecksDtype) {
  Tensor t = nn::zeros({2});
  EXPECT_NO_THROW(t.data_ptr<float>());
  EXPECT_THROW(t.data_ptr<double>(), std::invalid_argument);
}

// ---- cast ---------------------------------------------------------------------------------

TEST(NnTensor, CastTruncatesTowardZero) {
  // TF: Cast(x * 1.7, int32) truncates.
  Tensor x = nn::tensor({{1, 2, 3}, {4, 5, 6}}) * 1.7;
  Tensor i = nn::cast(x, DType::Int32);
  EXPECT_EQ(i.dtype(), DType::Int32);
  expect_vals(i, {1, 3, 5, 6, 8, 10});
  expect_vals(nn::cast(nn::tensor({-1.7, -0.2, 2.9}), DType::Int64), {-1, 0, 2});
}

TEST(NnTensor, CastToBoolIsNonzero) {
  Tensor b = nn::cast(nn::tensor({0, 0.5, -2, 0}), DType::Bool);
  EXPECT_EQ(b.dtype(), DType::Bool);
  expect_vals(b, {0, 1, 1, 0});
  expect_vals(nn::cast(b, DType::Float32), {0, 1, 1, 0});
}

TEST(NnTensor, CastSameDtypeIsIdentity) {
  Tensor x = nn::tensor({1, 2});
  EXPECT_TRUE(nn::cast(x, DType::Float32).is_same(x));
  EXPECT_EQ(x.to(DType::Float64).dtype(), DType::Float64);
}

// ---- views ----------------------------------------------------------------------------------

TEST(NnTensor, ReshapeSharesStorage) {
  Tensor x = nn::arange(0, 12).reshape({4, 3});
  Tensor y = x.reshape({3, 4});
  EXPECT_EQ(y.impl()->storage, x.impl()->storage);
  {
    nn::NoGradGuard g;
    nn::select(x, 0, 0).fill_(-1);
  }
  EXPECT_EQ(y.at({0, 0}), -1);
  EXPECT_EQ(y.at({0, 2}), -1);
}

TEST(NnTensor, TransposeIsAStridedView) {
  Tensor x = nn::tensor({{1, 2, 3}, {4, 5, 6}});
  Tensor t = x.transpose();
  EXPECT_EQ(t.shape(), (Shape{3, 2}));
  EXPECT_EQ(t.impl()->storage, x.impl()->storage);
  EXPECT_FALSE(t.is_contiguous());
  EXPECT_EQ(t.strides(), (Shape{1, 3}));
  expect_vals(t, {1, 4, 2, 5, 3, 6});
  Tensor c = t.contiguous();
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_NE(c.impl()->storage, x.impl()->storage);
  expect_vals(c, {1, 4, 2, 5, 3, 6});
}

TEST(NnTensor, SliceWritesAreVisibleInBase) {
  // tensor_basics: write -1 into rows [1, 3) of a [4, 3] matrix.
  Tensor mat  = nn::arange(0, 12).reshape({4, 3});
  Tensor rows = nn::slice(mat, {1, 0}, {2, -1});
  EXPECT_EQ(rows.shape(), (Shape{2, 3}));
  EXPECT_EQ(rows.impl()->storage, mat.impl()->storage);
  EXPECT_EQ(rows.offset(), 3);
  rows.copy_(nn::scalar(-1));
  EXPECT_EQ(mat.at({1, 0}), -1);
  EXPECT_EQ(mat.at({2, 2}), -1);
  EXPECT_EQ(mat.at({0, 0}), 0);
  EXPECT_EQ(mat.at({3, 0}), 9);
}

TEST(NnTensor, ContiguousReturnsSelfWhenPacked) {
  Tensor x = nn::ones({2, 2});
  EXPECT_TRUE(x.contiguous().is_same(x));
}

TEST(NnTensor, CloneIsIndependent) {
  Tensor x = nn::tensor({1, 2, 3});
  Tensor c = x.clone();
  EXPECT_NE(c.impl()->storage, x.impl()->storage);
  c.fill_(100);
  expect_vals(x, {1, 2, 3});
  expect_vals(c, {100, 100, 100});
}

TEST(NnTensor, DetachSharesDataWithoutHistory) {
  Tensor x = nn::tensor({1, 2}).requires_grad_();
  Tensor y = x * 2.0;
  Tensor d = y.detach();
  EXPECT_FALSE(d.requires_grad());
  EXPECT_TRUE(d.is_leaf());
  EXPECT_EQ(d.impl()->storage, y.impl()->storage);
  expect_vals(d, {2, 4});
}

// ---- element access ---------------------------------------------------------------------------

TEST(NnTensor, AtItemToVector) {
  Tensor cube = nn::arange(0, 8, 1, DType::Float64).reshape({2, 2, 2}) * 0.5;
  EXPECT_DOUBLE_EQ(cube.at({1, 0, 1}), 2.5);  // tensor_basics: cube(1,0,1)
  EXPECT_DOUBLE_EQ(cube.at({-1, -1, -1}), 3.5);
  EXPECT_THROW((void)cube.at({2, 0, 0}), std::out_of_range);
  EXPECT_THROW((void)cube.at({0, 0}), std::invalid_argument);
  EXPECT_EQ(nn::scalar(4).item<int>(), 4);
  EXPECT_THROW(cube.item(), std::invalid_argument);
  const auto iv = nn::tensor({1.9, 2.1}).to_vector<int>();
  EXPECT_EQ(iv, (std::vector<int>{1, 2}));
  // to_vector reads views in logical order.
  const auto tv = nn::tensor({{1, 2}, {3, 4}}).transpose().to_vector<double>();
  EXPECT_EQ(tv, (std::vector<double>{1, 3, 2, 4}));
}

// ---- in-place -------------------------------------------------------------------------------------

TEST(NnTensor, CopyBroadcastsAndConverts) {
  Tensor x = nn::zeros({2, 3});
  x.copy_(nn::tensor({1, 2, 3}, {}, DType::Float64));
  expect_vals(x, {1, 2, 3, 1, 2, 3});
  EXPECT_EQ(x.dtype(), DType::Float32);
  EXPECT_THROW(x.copy_(nn::tensor({1, 2})), std::invalid_argument);
}

TEST(NnTensor, CopyIntoTransposedView) {
  Tensor x = nn::zeros({2, 3});
  Tensor t = x.transpose();
  t.copy_(nn::tensor({{1, 2}, {3, 4}, {5, 6}}));
  expect_vals(x, {1, 3, 5, 2, 4, 6});
}

TEST(NnTensor, CopyFromOverlappingSource) {
  Tensor x = nn::tensor({1, 2, 3, 4});
  x.copy_(nn::reverse(x, {0}));
  expect_vals(x, {4, 3, 2, 1});
}

TEST(NnTensor, FillAddMul) {
  Tensor x = nn::zeros({3});
  x.fill_(2);
  expect_vals(x, {2, 2, 2});
  x.add_(nn::tensor({1, 2, 3}));
  expect_vals(x, {3, 4, 5});
  x.add_(nn::tensor({1, 1, 1}), 0.5);
  expect_vals(x, {3.5, 4.5, 5.5});
  x.sub_(nn::tensor({0.5, 0.5, 0.5}), 1.0);
  expect_vals(x, {3, 4, 5});
  x.mul_(2.0);
  expect_vals(x, {6, 8, 10});
  x.mul_(nn::tensor({1, 0, -1}));
  expect_vals(x, {6, 0, -10});
  x.zero_();
  expect_vals(x, {0, 0, 0});
}

TEST(NnTensor, InPlaceBumpsVersion) {
  Tensor        x  = nn::zeros({2});
  std::uint64_t v0 = x.impl()->storage->version;
  x.fill_(1);
  EXPECT_GT(x.impl()->storage->version, v0);
  std::uint64_t v1 = x.impl()->storage->version;
  x.add_(nn::ones({2}));
  EXPECT_GT(x.impl()->storage->version, v1);
}

TEST(NnTensor, InPlaceRefusedOnRequiresGradUnderGradMode) {
  Tensor w = nn::zeros({2}).requires_grad_();
  EXPECT_THROW(w.fill_(1), std::logic_error);
  EXPECT_THROW(w.add_(nn::ones({2})), std::logic_error);
  EXPECT_THROW(w.copy_(nn::ones({2})), std::logic_error);
  {
    nn::NoGradGuard guard;
    EXPECT_NO_THROW(w.add_(nn::ones({2}), 3.0));
  }
  expect_vals(w, {3, 3});
  EXPECT_TRUE(w.requires_grad());
}

TEST(NnTensor, RequiresGradOnlyForFloats) {
  Tensor i = nn::zeros({2}, DType::Int32);
  EXPECT_THROW(i.requires_grad_(), std::invalid_argument);
  Tensor f = nn::zeros({2});
  f.requires_grad_();
  EXPECT_TRUE(f.requires_grad());
  EXPECT_TRUE(f.is_leaf());
}

// ---- printing ------------------------------------------------------------------------------------

TEST(NnTensor, Printing) {
  Tensor m = nn::tensor({{1, 2, 3}, {4, 5, 6}});
  EXPECT_EQ(m.str(), "float32[2, 3] [[1, 2, 3], [4, 5, 6]]");
  EXPECT_EQ(m.values_str(), "[[1, 2, 3], [4, 5, 6]]");
  EXPECT_EQ(nn::scalar(3.5).str(), "float32[] 3.5");
  EXPECT_EQ(nn::tensor({1, 0}, {}, DType::Bool).values_str(), "[true, false]");
  EXPECT_EQ(nn::arange(0, 3, 1, DType::Int64).str(), "int64[3] [0, 1, 2]");
  std::ostringstream os;
  os << m;
  EXPECT_EQ(os.str(), m.str());
}

TEST(NnTensor, PrintingElides) {
  Tensor big = nn::arange(0, 100);
  EXPECT_EQ(big.values_str(10), "[0, 1, 2, ..., 97, 98, 99]");
  EXPECT_EQ(nn::arange(0, 5).values_str(2), "[0, 1, 2, 3, 4]");  // short axes are not elided
}

TEST(NnTensor, PrintingShowsGradState) {
  Tensor x = nn::tensor({1}).requires_grad_();
  EXPECT_NE(x.str().find("requires_grad"), std::string::npos);
  Tensor y = x * 2.0;
  EXPECT_NE(y.str().find("grad_fn="), std::string::npos);
}

TEST(NnTensor, ShapeHelpers) {
  EXPECT_EQ(nn::shape_numel({}), 1);
  EXPECT_EQ(nn::shape_numel({2, 0, 3}), 0);
  EXPECT_EQ(nn::shape_str({2, 3}), "[2, 3]");
  EXPECT_EQ(nn::shape_str({}), "[]");
  EXPECT_EQ(nn::contiguous_strides({2, 3, 4}), (Shape{12, 4, 1}));
  EXPECT_THROW(nn::shape_numel({-1}), std::invalid_argument);
}

// ---- linalg interop --------------------------------------------------------------------------------

TEST(NnTensor, LinalgRoundTrip) {
  Tensor         x = nn::tensor({{1.5, 2, 3}, {4, 5, -6}}, DType::Float64);
  linalg::Tensor l = nn::to_linalg(x);
  EXPECT_EQ(l.shape(), (std::vector<std::size_t>{2, 3}));
  EXPECT_DOUBLE_EQ(l.at({1, 2}), -6);
  Tensor back = nn::from_linalg(l);
  EXPECT_EQ(back.dtype(), DType::Float64);
  EXPECT_EQ(back.shape(), (Shape{2, 3}));
  expect_vals(back, {1.5, 2, 3, 4, 5, -6});

  // float32 and int become Float64; bool stays Bool.
  EXPECT_EQ(nn::to_linalg(nn::tensor({1, 2})).dtype(), linalg::TensorDType::Float64);
  Tensor         b  = nn::tensor({1, 0, 1}, {}, DType::Bool);
  linalg::Tensor lb = nn::to_linalg(b);
  EXPECT_EQ(lb.dtype(), linalg::TensorDType::Bool);
  Tensor bb = nn::from_linalg(lb);
  EXPECT_EQ(bb.dtype(), DType::Bool);
  expect_vals(bb, {1, 0, 1});

  linalg::Tensor s = linalg::Tensor::from_string_values({1}, {"a"});
  EXPECT_THROW(nn::from_linalg(s), std::invalid_argument);
}
