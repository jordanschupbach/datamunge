#include <gtest/gtest.h>

#include <datamunge/linalg/tensor.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::linalg::Tensor;
using datamunge::linalg::TensorDType;

TEST(Tensor, ZerosOnesFullHaveExpectedShapeAndValues) {
    auto z = Tensor::zeros({2, 3});
    EXPECT_EQ(z.ndim(), 2u);
    EXPECT_EQ(z.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(z.size(), 6u);
    EXPECT_EQ(z.dtype_name(), "float64");
    for (std::size_t i = 0; i < z.size(); ++i) EXPECT_DOUBLE_EQ(z.at_flat(i), 0.0);

    auto o = Tensor::ones({4});
    for (std::size_t i = 0; i < o.size(); ++i) EXPECT_DOUBLE_EQ(o.at_flat(i), 1.0);

    auto f = Tensor::full({2, 2}, 7.5);
    for (std::size_t i = 0; i < f.size(); ++i) EXPECT_DOUBLE_EQ(f.at_flat(i), 7.5);
}

TEST(Tensor, FromValuesRoundTripsThroughMultiIndex) {
    auto t = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_DOUBLE_EQ(t.at({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(t.at({0, 2}), 3.0);
    EXPECT_DOUBLE_EQ(t.at({1, 0}), 4.0);
    EXPECT_DOUBLE_EQ(t.at({1, 2}), 6.0);
    t.set({1, 1}, 42.0);
    EXPECT_DOUBLE_EQ(t.at_flat(4), 42.0);
}

TEST(Tensor, FromValuesRejectsSizeMismatch) {
    EXPECT_THROW(Tensor::from_values({2, 3}, {1, 2, 3}), std::invalid_argument);
}

TEST(Tensor, BoolDtypeStoresZeroOneAndParticipatesInArithmetic) {
    auto b = Tensor::from_bool_values({4}, {1, 0, 1, 1});
    EXPECT_EQ(b.dtype_name(), "bool");
    EXPECT_DOUBLE_EQ(b.at_flat(0), 1.0);
    EXPECT_DOUBLE_EQ(b.at_flat(1), 0.0);
    EXPECT_DOUBLE_EQ(b.sum(), 3.0);

    auto doubled = b.multiply_scalar(2.0);
    EXPECT_EQ(doubled.dtype_name(), "float64");
    EXPECT_DOUBLE_EQ(doubled.at_flat(0), 2.0);
}

TEST(Tensor, StringDtypeSupportsStructuralOpsButNotArithmetic) {
    auto s = Tensor::from_string_values({3}, {"a", "b", "c"});
    EXPECT_EQ(s.dtype_name(), "string");
    EXPECT_EQ(s.string_at_flat(0), "a");
    EXPECT_EQ(s.string_at({2}), "c");

    EXPECT_THROW(s.at_flat(0), std::logic_error);
    EXPECT_THROW(s.sum(), std::invalid_argument);
    EXPECT_THROW(s.add(s), std::invalid_argument);
    EXPECT_THROW(s.matmul(s), std::invalid_argument);

    auto reshaped = s.reshape({1, 3});
    EXPECT_EQ(reshaped.string_at({0, 1}), "b");

    auto t = Tensor::from_values({3}, {1, 2, 3});
    EXPECT_THROW(t.string_at_flat(0), std::logic_error);
    EXPECT_THROW(t.set_string_flat(0, "x"), std::logic_error);
}

TEST(Tensor, ArangeAndEye) {
    auto r = Tensor::arange(0.0, 5.0, 1.0);
    EXPECT_EQ(r.shape(), (std::vector<std::size_t>{5}));
    for (std::size_t i = 0; i < 5; ++i) EXPECT_DOUBLE_EQ(r.at_flat(i), static_cast<double>(i));

    auto id = Tensor::eye(3);
    EXPECT_EQ(id.shape(), (std::vector<std::size_t>{3, 3}));
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) EXPECT_DOUBLE_EQ(id.at({i, j}), i == j ? 1.0 : 0.0);
}

TEST(Tensor, ReshapePreservesRowMajorDataAndRejectsSizeMismatch) {
    auto t = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});
    auto r = t.reshape({3, 2});
    EXPECT_DOUBLE_EQ(r.at({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(r.at({0, 1}), 2.0);
    EXPECT_DOUBLE_EQ(r.at({1, 0}), 3.0);
    EXPECT_DOUBLE_EQ(r.at({2, 1}), 6.0);
    EXPECT_THROW(t.reshape({4, 4}), std::invalid_argument);

    auto flat = t.flatten();
    EXPECT_EQ(flat.shape(), (std::vector<std::size_t>{6}));
}

TEST(Tensor, TransposeDefaultReversesAxesAndSupportsExplicitPermutation) {
    auto t = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});
    auto tr = t.transpose();
    EXPECT_EQ(tr.shape(), (std::vector<std::size_t>{3, 2}));
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) EXPECT_DOUBLE_EQ(tr.at({j, i}), t.at({i, j}));

    auto id = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6}).transpose({0, 1});
    EXPECT_EQ(id.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_DOUBLE_EQ(id.at({1, 2}), 6.0);

    EXPECT_THROW(t.transpose({0, 0}), std::invalid_argument);
    EXPECT_THROW(t.transpose({0}), std::invalid_argument);
}

TEST(Tensor, SqueezeAndExpandDims) {
    auto t = Tensor::zeros({1, 3, 1});
    auto sq = t.squeeze();
    EXPECT_EQ(sq.shape(), (std::vector<std::size_t>{3}));

    auto sq_axis = t.squeeze_axis(0);
    EXPECT_EQ(sq_axis.shape(), (std::vector<std::size_t>{3, 1}));
    EXPECT_THROW(t.squeeze_axis(1), std::invalid_argument);

    auto ex = Tensor::zeros({3}).expand_dims(0);
    EXPECT_EQ(ex.shape(), (std::vector<std::size_t>{1, 3}));
    auto ex_end = Tensor::zeros({3}).expand_dims(1);
    EXPECT_EQ(ex_end.shape(), (std::vector<std::size_t>{3, 1}));
    EXPECT_THROW(Tensor::zeros({3}).expand_dims(2), std::out_of_range);
}

TEST(Tensor, SliceSelectsExpectedSubrangeIncludingStep) {
    auto t = Tensor::arange(0.0, 10.0, 1.0);
    auto s = t.slice(0, 2, 8, 2);
    EXPECT_EQ(s.shape(), (std::vector<std::size_t>{3}));
    EXPECT_DOUBLE_EQ(s.at_flat(0), 2.0);
    EXPECT_DOUBLE_EQ(s.at_flat(1), 4.0);
    EXPECT_DOUBLE_EQ(s.at_flat(2), 6.0);

    EXPECT_THROW(t.slice(0, 8, 2), std::out_of_range);
    EXPECT_THROW(t.slice(0, 0, 20), std::out_of_range);
}

TEST(Tensor, IndexSelectGathersRowsAlongAxis) {
    auto t = Tensor::from_values({3, 2}, {1, 2, 3, 4, 5, 6});
    auto g = t.index_select(0, {2, 0});
    EXPECT_EQ(g.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_DOUBLE_EQ(g.at({0, 0}), 5.0);
    EXPECT_DOUBLE_EQ(g.at({1, 0}), 1.0);
    EXPECT_THROW(t.index_select(0, {5}), std::out_of_range);
}

TEST(Tensor, ConcatenateAndStack) {
    auto a = Tensor::from_values({2, 2}, {1, 2, 3, 4});
    auto b = Tensor::from_values({2, 2}, {5, 6, 7, 8});
    auto cat0 = Tensor::concatenate({a, b}, 0);
    EXPECT_EQ(cat0.shape(), (std::vector<std::size_t>{4, 2}));
    EXPECT_DOUBLE_EQ(cat0.at({2, 0}), 5.0);

    auto cat1 = Tensor::concatenate({a, b}, 1);
    EXPECT_EQ(cat1.shape(), (std::vector<std::size_t>{2, 4}));
    EXPECT_DOUBLE_EQ(cat1.at({0, 2}), 5.0);

    auto stacked = Tensor::stack({a, b}, 0);
    EXPECT_EQ(stacked.shape(), (std::vector<std::size_t>{2, 2, 2}));
    EXPECT_DOUBLE_EQ(stacked.at({0, 0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(stacked.at({1, 0, 0}), 5.0);

    auto c = Tensor::from_values({3, 2}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(Tensor::concatenate({a, c}, 1), std::invalid_argument);

    auto str = Tensor::from_string_values({2, 2}, {"a", "b", "c", "d"});
    EXPECT_THROW(Tensor::concatenate({a, str}, 0), std::invalid_argument);
}

TEST(Tensor, ElementwiseArithmeticBroadcasts) {
    auto a = Tensor::from_values({3, 1}, {1, 2, 3});
    auto b = Tensor::from_values({1, 4}, {10, 20, 30, 40});
    auto sum = a.add(b);
    EXPECT_EQ(sum.shape(), (std::vector<std::size_t>{3, 4}));
    EXPECT_DOUBLE_EQ(sum.at({0, 0}), 11.0);
    EXPECT_DOUBLE_EQ(sum.at({2, 3}), 43.0);

    auto c = Tensor::from_values({2, 2}, {1, 2, 3, 4});
    auto d = Tensor::from_values({2, 2}, {10, 10, 10, 10});
    EXPECT_DOUBLE_EQ(c.subtract(d).at({0, 0}), -9.0);
    EXPECT_DOUBLE_EQ(c.multiply(d).at({1, 1}), 40.0);
    EXPECT_DOUBLE_EQ(d.divide(c).at({0, 0}), 10.0);
    EXPECT_DOUBLE_EQ(Tensor::from_values({1}, {2}).power(Tensor::from_values({1}, {10})).at_flat(0), 1024.0);

    auto e = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});
    auto f = Tensor::from_values({4}, {1, 2, 3, 4});
    EXPECT_THROW(e.add(f), std::invalid_argument);
}

TEST(Tensor, ScalarArithmeticAndUnaryMathOps) {
    auto t = Tensor::from_values({4}, {1, -4, 9, 16});
    EXPECT_DOUBLE_EQ(t.add_scalar(10.0).at_flat(0), 11.0);
    EXPECT_DOUBLE_EQ(t.multiply_scalar(2.0).at_flat(1), -8.0);
    EXPECT_DOUBLE_EQ(t.negate().at_flat(1), 4.0);
    EXPECT_DOUBLE_EQ(t.abs().at_flat(1), 4.0);
    EXPECT_DOUBLE_EQ(t.sqrt().at_flat(2), 3.0);
    EXPECT_DOUBLE_EQ(Tensor::from_values({1}, {0}).exp().at_flat(0), 1.0);
    EXPECT_DOUBLE_EQ(Tensor::from_values({1}, {1}).log().at_flat(0), 0.0);

    auto squared = t.apply([](double x) { return x * x; });
    EXPECT_DOUBLE_EQ(squared.at_flat(0), 1.0);
    EXPECT_DOUBLE_EQ(squared.at_flat(1), 16.0);
}

TEST(Tensor, ComparisonsRequireMatchingDtypeAndProduceBoolTensor) {
    auto a = Tensor::from_values({3}, {1, 2, 3});
    auto b = Tensor::from_values({3}, {1, 5, 2});
    auto eq = a.equal(b);
    EXPECT_EQ(eq.dtype_name(), "bool");
    EXPECT_DOUBLE_EQ(eq.at_flat(0), 1.0);
    EXPECT_DOUBLE_EQ(eq.at_flat(1), 0.0);

    auto lt = a.less(b);
    EXPECT_DOUBLE_EQ(lt.at_flat(1), 1.0);
    EXPECT_DOUBLE_EQ(lt.at_flat(2), 0.0);

    auto s1 = Tensor::from_string_values({2}, {"apple", "zebra"});
    auto s2 = Tensor::from_string_values({2}, {"banana", "zebra"});
    auto slt = s1.less(s2);
    EXPECT_DOUBLE_EQ(slt.at_flat(0), 1.0);
    auto seq = s1.equal(s2);
    EXPECT_DOUBLE_EQ(seq.at_flat(1), 1.0);

    EXPECT_THROW(a.equal(s1), std::invalid_argument);
}

TEST(Tensor, GlobalReductionsMatchHandComputedValues) {
    auto t = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_DOUBLE_EQ(t.sum(), 21.0);
    EXPECT_DOUBLE_EQ(t.mean(), 3.5);
    EXPECT_DOUBLE_EQ(t.max(), 6.0);
    EXPECT_DOUBLE_EQ(t.min(), 1.0);
    EXPECT_DOUBLE_EQ(t.prod(), 720.0);
    EXPECT_EQ(t.argmax(), 5u);
    EXPECT_EQ(t.argmin(), 0u);

    auto mask = Tensor::from_bool_values({3}, {1, 1, 1});
    EXPECT_TRUE(mask.all());
    EXPECT_TRUE(mask.any());
    auto mask2 = Tensor::from_bool_values({3}, {0, 0, 0});
    EXPECT_FALSE(mask2.all());
    EXPECT_FALSE(mask2.any());
}

TEST(Tensor, AxisReductionsMatchHandComputedValuesAndRespectKeepdims) {
    auto t = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});

    auto col_sum = t.sum_axis(0);
    EXPECT_EQ(col_sum.shape(), (std::vector<std::size_t>{3}));
    EXPECT_DOUBLE_EQ(col_sum.at_flat(0), 5.0);
    EXPECT_DOUBLE_EQ(col_sum.at_flat(1), 7.0);
    EXPECT_DOUBLE_EQ(col_sum.at_flat(2), 9.0);

    auto row_sum = t.sum_axis(1, true);
    EXPECT_EQ(row_sum.shape(), (std::vector<std::size_t>{2, 1}));
    EXPECT_DOUBLE_EQ(row_sum.at({0, 0}), 6.0);
    EXPECT_DOUBLE_EQ(row_sum.at({1, 0}), 15.0);

    auto row_mean = t.mean_axis(1);
    EXPECT_DOUBLE_EQ(row_mean.at_flat(0), 2.0);
    EXPECT_DOUBLE_EQ(row_mean.at_flat(1), 5.0);

    auto row_max = t.max_axis(1);
    EXPECT_DOUBLE_EQ(row_max.at_flat(0), 3.0);
    auto row_argmax = t.argmax_axis(1);
    EXPECT_DOUBLE_EQ(row_argmax.at_flat(0), 2.0);

    EXPECT_THROW(t.sum_axis(2), std::out_of_range);
}

TEST(Tensor, MatmulDotAndOuter) {
    auto a = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = Tensor::from_values({3, 2}, {7, 8, 9, 10, 11, 12});
    auto m = a.matmul(b);
    EXPECT_EQ(m.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_DOUBLE_EQ(m.at({0, 0}), 1 * 7 + 2 * 9 + 3 * 11);
    EXPECT_DOUBLE_EQ(m.at({0, 1}), 1 * 8 + 2 * 10 + 3 * 12);
    EXPECT_DOUBLE_EQ(m.at({1, 0}), 4 * 7 + 5 * 9 + 6 * 11);

    EXPECT_THROW(a.matmul(a), std::invalid_argument);

    auto v1 = Tensor::from_values({3}, {1, 2, 3});
    auto v2 = Tensor::from_values({3}, {4, 5, 6});
    EXPECT_DOUBLE_EQ(v1.dot(v2), 32.0);
    EXPECT_THROW(a.dot(v1), std::invalid_argument);

    auto o = v1.outer(v2);
    EXPECT_EQ(o.shape(), (std::vector<std::size_t>{3, 3}));
    EXPECT_DOUBLE_EQ(o.at({0, 0}), 4.0);
    EXPECT_DOUBLE_EQ(o.at({2, 2}), 18.0);
}

TEST(Tensor, ToStringMentionsShapeAndDtype) {
    auto t = Tensor::from_values({2, 2}, {1, 2, 3, 4});
    const auto text = t.to_string();
    EXPECT_NE(text.find("shape=[2, 2]"), std::string::npos);
    EXPECT_NE(text.find("float64"), std::string::npos);
}

TEST(Tensor, OutOfRangeAndRankMismatchIndexingThrow) {
    auto t = Tensor::zeros({2, 2});
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
    EXPECT_THROW(t.at({0}), std::invalid_argument);
    EXPECT_THROW(t.at_flat(4), std::out_of_range);
}
