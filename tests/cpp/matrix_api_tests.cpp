/// @file matrix_api_tests.cpp
/// @brief Matrix facade, lazy lifetime, storage, numerical solve and formatting regressions.
#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/format/matrix.hpp>
#include <datamunge/linalg/matrix.hpp>
#include <limits>
#include <locale>
#include <type_traits>

namespace {
using Mat = datamunge::Mat<>;
using datamunge::solve;
using SpMat = datamunge::SpMat<>;

template<class T>
class TypedMatrixApi : public ::testing::Test {};
using FloatingTypes = ::testing::Types<float, double, long double>;
TYPED_TEST_SUITE(TypedMatrixApi, FloatingTypes);

TYPED_TEST(TypedMatrixApi, MatrixPowersAndInverseActionsPreserveScalars) {
  using T                                      = TypeParam;
  const datamunge::Mat<datamunge::Dense, T>  a = {{4, 0}, {0, 9}}, b = {{8}, {18}};
  const datamunge::Mat<datamunge::Sparse, T> sparse = a, rhs = b;
  const auto                                 inverse      = ((sparse ^ -1) * rhs).eval();
  const auto                                 inverse_real = ((sparse ^ -1.0) * rhs).eval();
  const auto                                 square       = (a ^ 2).eval();
  const auto                                 root         = (a ^ 0.5).eval();
  static_assert(std::is_same_v<typename decltype(root)::value_type, T>);
  static_assert(std::is_same_v<typename decltype(inverse)::value_type, T>);
  EXPECT_EQ(inverse(0, 0), 2);
  EXPECT_EQ(inverse(1, 0), 2);
  EXPECT_EQ(inverse_real(0, 0), 2);
  EXPECT_EQ(inverse_real(1, 0), 2);
  EXPECT_EQ(square(0, 0), 16);
  EXPECT_EQ(square(1, 1), 81);
  EXPECT_EQ(root(0, 0), 2);
  EXPECT_EQ(root(1, 1), 3);
  const auto zero = (a ^ 0).eval();
  EXPECT_EQ(zero(0, 0), 1);
  EXPECT_EQ(zero(0, 1), 0);
  const auto cube = (a ^ -3).eval();
  EXPECT_NEAR(cube(1, 1), 1.0 / 729, 10 * std::numeric_limits<T>::epsilon());

  const datamunge::Mat<datamunge::Dense, T> symmetric           = {{5, 4}, {4, 5}};
  const datamunge::Mat<datamunge::Dense, T> jordan              = {{4, 1}, {0, 4}};
  const auto                                symmetric_root      = (symmetric ^ 0.5).eval();
  const auto                                jordan_root         = (jordan ^ 0.5).eval();
  const auto                                jordan_inverse_root = (jordan ^ -0.5).eval();
  const auto                                tolerance = 500 * std::numeric_limits<T>::epsilon();
  EXPECT_NEAR(symmetric_root(0, 0), 2, tolerance);
  EXPECT_NEAR(symmetric_root(0, 1), 1, tolerance);
  EXPECT_NEAR(jordan_root(0, 0), 2, tolerance);
  EXPECT_NEAR(jordan_root(0, 1), 0.25, tolerance);
  EXPECT_NEAR(jordan_inverse_root(0, 0), 0.5, tolerance);
  EXPECT_NEAR(jordan_inverse_root(0, 1), -0.0625, tolerance);
  const auto mixed = (jordan ^ 1.5).eval();
  EXPECT_NEAR(mixed(0, 0), 8, tolerance);
  EXPECT_NEAR(mixed(0, 1), 3, tolerance);
}

TEST(MatrixApi, PowersOwnTemporariesObserveMutationAndRecheckShapes) {
  Mat        a = {{2, 0}, {0, 4}}, b = {{6}, {8}};
  const auto delayed = (a ^ -1) * b;
  a(0, 0)            = 3;
  EXPECT_DOUBLE_EQ(delayed(0, 0), 2);
  const auto temporary = Mat{{4, 1}, {0, 4}} ^ 0.5;
  EXPECT_NEAR(temporary(0, 1), 0.25, 1e-12);
  const auto temporary_inverse = (SpMat{{2, 0}, {0, 4}} ^ -1.0) * Mat{{6}, {8}};
  EXPECT_DOUBLE_EQ(temporary_inverse(0, 0), 3);
  EXPECT_EQ(
      (Mat{{1, 2}, {0, 1}} ^ 3).to_tex(), "\\begin{bmatrix}\n1 & 6 \\\\\n0 & 1\n\\end{bmatrix}");
  a = Mat(2, 3);
  EXPECT_THROW(delayed.eval(), std::invalid_argument);
}

TEST(MatrixApi, PowersDistinguishMatrixPowerFromElementwiseAndExponential) {
  const Mat  a = {{1, 1}, {0, 1}}, b = {{2, 0}, {0, 3}}, c = {{1}, {2}};
  const auto product = ((a ^ 2) * b * c).eval();
  EXPECT_DOUBLE_EQ(product(0, 0), 14);
  EXPECT_DOUBLE_EQ(product(1, 0), 6);
  const Mat  rotation = {{0, -1}, {1, 0}};
  const auto root     = (rotation ^ 0.5).eval();
  EXPECT_NEAR(root(0, 0), std::sqrt(0.5), 1e-12);
  EXPECT_NEAR(root(0, 1), -std::sqrt(0.5), 1e-12);
  const Mat unit = {{1}};
  EXPECT_EQ((unit ^ std::numeric_limits<std::uint64_t>::max())(0, 0), 1);
  EXPECT_EQ((unit ^ std::numeric_limits<std::int64_t>::min())(0, 0), 1);
  EXPECT_EQ((unit ^ std::numeric_limits<signed char>::min())(0, 0), 1);
  const Mat negative = {{-2}};
  EXPECT_EQ((negative ^ 3)(0, 0), -8);
  EXPECT_EQ((negative ^ -1.0)(0, 0), -0.5);
  EXPECT_EQ((Mat{{0, 0}, {0, 4}} ^ 0.5)(0, 0), 0);
  EXPECT_EQ((Mat() ^ 0.5).eval().rows(), 0);
}

TEST(MatrixApi, PowersRejectInvalidDomainsAndOverflow) {
  Mat a = {{1, 0}, {0, 2}};
  EXPECT_THROW((Mat(2, 3) ^ -1), std::invalid_argument);
  EXPECT_THROW((a ^ std::numeric_limits<double>::infinity()), std::invalid_argument);
  EXPECT_THROW((a ^ std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
  EXPECT_THROW((Mat{{0}} ^ -1).eval(), std::runtime_error);
  EXPECT_THROW((Mat{{0}} ^ -0.5).eval(), std::domain_error);
  EXPECT_THROW((Mat{{-1}} ^ 0.5).eval(), std::domain_error);
  EXPECT_THROW((Mat{{-1, 1}, {0, -1}} ^ 0.5).eval(), std::runtime_error);
  EXPECT_THROW((Mat{{1, 1}, {1, 1}} ^ 0.5).eval(), std::runtime_error);
  EXPECT_THROW((Mat{{std::numeric_limits<double>::quiet_NaN()}} ^ 0).eval(), std::invalid_argument);
  EXPECT_THROW((Mat{{1e200}} ^ 2).eval(), std::overflow_error);
  EXPECT_THROW((a ^ 1e100).eval(), std::overflow_error);
}

TEST(MatrixApi, FreeTransposeForwardsOwnershipConstnessAndStorage) {
  Mat         a              = {{1, 2, 3}};
  const Mat   ca             = a;
  const SpMat sparse         = a;
  const auto  borrowed       = datamunge::t(a);
  const auto  const_borrowed = datamunge::t(ca);
  const auto  owned          = datamunge::t(Mat{{4, 5}});
  const auto  const_owned    = datamunge::t(static_cast<const Mat&&>(Mat{{6, 7}}));
  const auto  sparse_owned   = datamunge::t(SpMat{{0, 8}});
  static_assert(std::is_same_v<decltype(datamunge::t(sparse).eval()), SpMat>);
  EXPECT_EQ(borrowed.rows(), 3);
  EXPECT_EQ(const_borrowed(2, 0), 3);
  a(0, 2) = 9;
  EXPECT_EQ(borrowed(2, 0), 9);
  EXPECT_EQ(owned(1, 0), 5);
  EXPECT_EQ(const_owned(1, 0), 7);
  EXPECT_EQ(sparse_owned(1, 0), 8);
  const auto expression = datamunge::t(datamunge::t(ca) * ca);
  EXPECT_EQ(expression(2, 1), 6);
}

TYPED_TEST(TypedMatrixApi, FactoriesAllocateZerosAndSupportCheckedCoefficientAssignment) {
  auto dense  = datamunge::mat<TypeParam>(3, 5);
  auto sparse = datamunge::spmat<TypeParam>(30, 40);
  static_assert(std::is_same_v<decltype(dense), datamunge::Mat<datamunge::Dense, TypeParam>>);
  static_assert(std::is_same_v<decltype(sparse), datamunge::Mat<datamunge::Sparse, TypeParam>>);
  static_assert(std::is_same_v<decltype(dense(0, 0)), TypeParam&>);
  static_assert(std::is_same_v<decltype(std::as_const(sparse)(0, 0)), TypeParam>);
  EXPECT_EQ(dense.nnz(), 0);
  EXPECT_EQ(sparse.nnz(), 0);
  for (std::size_t i = 0; i < dense.rows(); ++i)
    for (std::size_t j = 0; j < dense.cols(); ++j)
      EXPECT_EQ(dense(i, j), 0);
  dense(2, 4) = static_cast<TypeParam>(1.1);
  dense.set(2, 2, static_cast<TypeParam>(1.1));
  sparse(2, 8) = static_cast<TypeParam>(2.3);
  sparse(4, 5) = static_cast<TypeParam>(1.8);
  EXPECT_EQ(dense(2, 4), static_cast<TypeParam>(1.1));
  EXPECT_EQ(sparse(2, 8), static_cast<TypeParam>(2.3));
  EXPECT_EQ(dense.nnz(), 2);
  EXPECT_EQ(sparse.nnz(), 2);
  EXPECT_EQ(std::format("{:.1f}", sparse(4, 5)), "1.8");
  EXPECT_THROW(dense(3, 4) = 1, std::out_of_range);
  EXPECT_THROW(dense(2, 5) = 1, std::out_of_range);
  EXPECT_THROW(sparse(30, 39) = 1, std::out_of_range);
  EXPECT_THROW(sparse(29, 40) = 1, std::out_of_range);
  const auto coefficient = (sparse.t() * sparse)(8, 8);
  static_assert(std::is_same_v<std::decay_t<decltype(coefficient)>, TypeParam>);
  EXPECT_EQ(coefficient, static_cast<TypeParam>(2.3) * static_cast<TypeParam>(2.3));
}

TEST(MatrixApi, FactoriesDefaultToDoubleAndRetainEmptyAndSparseHugeShapes) {
  static_assert(std::is_same_v<decltype(datamunge::mat(3, 5)), Mat>);
  static_assert(std::is_same_v<decltype(datamunge::spmat(30, 40)), SpMat>);
  for (std::size_t rows : {0, 3}) {
    auto dense  = datamunge::mat(rows, 0);
    auto sparse = datamunge::spmat(rows, 0);
    EXPECT_EQ(dense.rows(), rows);
    EXPECT_EQ(sparse.rows(), rows);
    EXPECT_EQ(dense.cols(), 0);
    EXPECT_EQ(sparse.cols(), 0);
    EXPECT_EQ(sparse.nnz(), 0);
  }
  auto wide          = datamunge::spmat(2, 1000000000);
  wide(0, 999999999) = 3;
  EXPECT_EQ(wide.nnz(), 1);
  EXPECT_EQ(wide.cols(), 1000000000);
  EXPECT_EQ(wide.storage().row_ptr, (std::vector<std::size_t>{0, 1, 1}));
  EXPECT_THROW((void)datamunge::mat(std::numeric_limits<std::size_t>::max(), 2), std::length_error);
  EXPECT_THROW((void)datamunge::spmat(std::numeric_limits<std::size_t>::max(), 2), std::length_error);
}

TEST(MatrixApi, WritableSparseCoordinatesPreserveCsrWithoutInsertingOnReads) {
  auto sparse = datamunge::spmat(3, 5);
  auto held   = sparse(2, 4);
  EXPECT_EQ(static_cast<double>(held), 0);
  EXPECT_EQ(std::format("{}", held), "0");
  EXPECT_EQ(sparse.nnz(), 0);
  sparse(2, 1) = 3;
  sparse(0, 4) = 7;
  held         = 9;
  sparse(1, 3) = held;
  sparse(0, 0) = sparse(1, 3) = sparse(2, 1);
  EXPECT_EQ(sparse(0, 0), 3);
  EXPECT_EQ(sparse(1, 3), 3);
  EXPECT_EQ(static_cast<double>(held), 9);
  held         = held;
  sparse(2, 1) = 0;
  held         = 0;
  EXPECT_EQ(sparse.nnz(), 3);
  EXPECT_EQ(sparse.storage().row_ptr, (std::vector<std::size_t>{0, 2, 3, 3}));
  EXPECT_EQ(sparse.storage().col_idx, (std::vector<std::size_t>{0, 4, 3}));
  sparse = SpMat();
  EXPECT_THROW(static_cast<double>(held), std::out_of_range);
  EXPECT_THROW(held = 1, std::out_of_range);
}

TYPED_TEST(TypedMatrixApi, PlaygroundTemplatePoliciesPreserveScalarsThroughSolve) {
  using T = TypeParam;
  using datamunge::Mat;
  using Dense  = datamunge::Dense;
  using Sparse = datamunge::Sparse;
  static_assert(std::is_same_v<Mat<Dense, T>, datamunge::la::Mat<Dense, T>>);
  static_assert(std::is_same_v<Mat<Sparse, T>, datamunge::la::Mat<Sparse, T>>);
  Mat<Dense, T>  x = {{1, 0}, {0, 1}, {1, 1}}, y = {{1}, {2}, {3}};
  Mat<Sparse, T> sx = x, sy = y;
  const auto     dense  = (solve(x.t() * x) * x.t() * y).eval();
  const auto     sparse = (solve(sx.t() * sx) * sx.t() * sy).eval();
  static_assert(std::is_same_v<typename std::decay_t<decltype(dense)>::value_type, T>);
  static_assert(std::is_same_v<typename std::decay_t<decltype(sparse)>::value_type, T>);
  static_assert(std::is_same_v<decltype((sx.t() * sx).eval()), Mat<Sparse, T>>);
  const auto direct    = solve(x, y);
  const auto tolerance = 100 * std::numeric_limits<T>::epsilon();
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_LE(std::abs(dense(i, 0) - static_cast<T>(i + 1)), tolerance);
    EXPECT_LE(std::abs(sparse(i, 0) - static_cast<T>(i + 1)), tolerance);
    EXPECT_LE(std::abs(direct(i, 0) - static_cast<T>(i + 1)), tolerance);
  }
  EXPECT_TRUE(std::format("{}", dense).starts_with("Mat 2 x 1\n"));
  Mat<> defaults = {{2}};
  static_assert(std::is_same_v<Mat<>, Mat<Dense, double>>);
  static_assert(std::is_same_v<Mat<Dense>, Mat<>>);
  static_assert(std::is_same_v<Mat<Sparse>, datamunge::SpMat<>>);
  static_assert(std::is_same_v<datamunge::la::Mat<>, Mat<>>);
  static_assert(std::is_same_v<datamunge::la::Mat<Sparse>, Mat<Sparse>>);
  static_assert(std::is_same_v<Mat<datamunge::la::DMat, T>, Mat<Dense, T>>);
  static_assert(std::is_same_v<Mat<datamunge::la::SpMat, T>, Mat<Sparse, T>>);
  static_assert(std::is_same_v<Mat<>, datamunge::BasicMat<false, double>>);
  static_assert(std::is_same_v<typename decltype(defaults)::value_type, double>);
}

TYPED_TEST(TypedMatrixApi, TexIncludesAllDenseAndSparseCoefficients) {
  const datamunge::Mat<datamunge::Dense, TypeParam> dense = {{1, 0}, {-2.5, 3}};
  const datamunge::Mat<datamunge::Sparse, TypeParam> sparse = dense;
  const std::string body = "\n1 & 0 \\\\\n-2.5 & 3\n";
  EXPECT_EQ(dense.to_tex(), "\\begin{bmatrix}" + body + "\\end{bmatrix}");
  for (const std::string type : {"bmatrix", "matrix", "pmatrix", "Bmatrix", "vmatrix",
                                  "Vmatrix", "smallmatrix", "bmatrix*"}) {
    const auto expected = "\\begin{" + type + "}" + body + "\\end{" + type + "}";
    EXPECT_EQ(dense.to_tex(type), expected);
    EXPECT_EQ(sparse.to_tex(type), expected);
  }
}

TEST(MatrixApi, TexEvaluatesTransposesProductsAndSolvesAtCallTime) {
  Mat         a = {{2, 0}, {0, 4}}, b = {{6}, {8}};
  const SpMat sa = a, sb = b;
  const auto  expression = solve(a) * b;
  EXPECT_EQ(expression.to_tex(), "\\begin{bmatrix}\n3 \\\\\n2\n\\end{bmatrix}");
  EXPECT_EQ((solve(sa) * sb).to_tex(), expression.to_tex());
  EXPECT_EQ(a.t().to_tex("matrix"), Mat(a.t()).to_tex("matrix"));
  EXPECT_EQ((sa.t() * sa).to_tex(), Mat(sa.t() * sa).to_tex());
  EXPECT_EQ(solve(a).to_tex(), "\\begin{bmatrix}\n0.5 & 0 \\\\\n0 & 0.25\n\\end{bmatrix}");
  a.set(0, 0, 3);
  EXPECT_EQ(expression.to_tex(), "\\begin{bmatrix}\n2 \\\\\n2\n\\end{bmatrix}");
  a.set(0, 0, 0);
  EXPECT_THROW(expression.to_tex(), std::runtime_error);
  EXPECT_THROW(expression.to_tex(""), std::invalid_argument);
}

TEST(MatrixApi, TexAnnotationsOwnTheMatrixSnapshot) {
  Mat a = {{1, 2}};
  const auto tex = a.to_tex();
  static_assert(std::is_same_v<std::decay_t<decltype(tex)>, datamunge::Tex>);
  const auto brace = tex.underbrace("X");
  const auto label = tex.underset("X");
  a.set(0, 0, 9);
  EXPECT_EQ(brace, "\\underbrace{\\begin{bmatrix}\n1 & 2\n\\end{bmatrix}}_{X}");
  EXPECT_EQ(label, "\\underset{X}{\\begin{bmatrix}\n1 & 2\n\\end{bmatrix}}");
  EXPECT_NE(a.to_tex(), tex);
  const std::string source = tex;
  EXPECT_EQ(source, "\\begin{bmatrix}\n1 & 2\n\\end{bmatrix}");
}

TEST(MatrixApi, TexHandlesEmptyShapesAndDoesNotTruncate) {
  const std::string empty = "\\begin{bmatrix}\n\\end{bmatrix}";
  for (const auto& matrix : {Mat(), Mat(0, 3), Mat(3, 0)})
    EXPECT_EQ(matrix.to_tex(), empty);
  for (const auto& matrix : {SpMat(), SpMat(0, 3), SpMat(3, 0)})
    EXPECT_EQ(matrix.to_tex(), empty);
  Mat big(12, 10);
  big.set(11, 9, 42);
  const auto tex = big.to_tex();
  EXPECT_EQ(std::count(tex.begin(), tex.end(), '&'), 12 * 9);
  EXPECT_NE(tex.find(" & 42\n\\end{bmatrix}"), std::string::npos);
  for (const auto& type : {"", "*", "matrix**", "matrix\n", "matrix}", "matrix[lr]"})
    EXPECT_THROW(big.to_tex(type), std::invalid_argument);
}

TEST(MatrixApi, TexFormatsScientificNotationAndNonfiniteValues) {
  const Mat a = {
      {1e20, -2e-12, 1.23456789},
      {std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::quiet_NaN()}};
  EXPECT_EQ(
      a.to_tex(),
      "\\begin{bmatrix}\n1 \\times 10^{20} & -2 \\times 10^{-12} & 1.234568 \\\\\n"
      "\\infty & -\\infty & \\mathrm{NaN}\n\\end{bmatrix}");
  const int exponent = std::min(400, std::numeric_limits<long double>::max_exponent10 - 1);
  const datamunge::Mat<datamunge::Dense, long double> precise = {{std::pow(10.0L, exponent)}};
  EXPECT_EQ(precise.to_tex(), "\\begin{bmatrix}\n1 \\times 10^{" +
                                 std::to_string(exponent) + "}\n\\end{bmatrix}");
}

TEST(MatrixApi, TexKeepsDecimalPointsUnderACommaLocale) {
  struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
  };
  struct RestoreLocale {
    std::locale original;
    ~RestoreLocale() { std::locale::global(original); }
  } restore;
  std::locale::global(std::locale(std::locale::classic(), new CommaDecimal));
  EXPECT_EQ((Mat{{1.25, 1e-12}}.to_tex()),
            "\\begin{bmatrix}\n1.25 & 1 \\times 10^{-12}\n\\end{bmatrix}");
}

TEST(MatrixApi, ExplicitScalarAndStorageConversionsDoNotDensifySparseShapes) {
  using SparseFloat = datamunge::la::Mat<datamunge::Sparse, float>;
  SpMat wide(2, 1000000000);
  wide.set(0, 999999999, 3.5);
  const SparseFloat converted = wide;
  EXPECT_EQ(converted.cols(), wide.cols());
  EXPECT_EQ(converted.nnz(), 1);
  EXPECT_FLOAT_EQ(converted(0, 999999999), 3.5f);
  const datamunge::la::Mat<datamunge::Dense, float> dense_float   = Mat{{1, 2}};
  const SpMat                                       sparse_double = dense_float;
  EXPECT_EQ(sparse_double.nnz(), 2);
  const datamunge::la::Mat<datamunge::Dense, long double> precise = sparse_double;
  EXPECT_EQ(precise(0, 1), 2.0L);
}

TEST(MatrixApi, LongDoubleEvaluationKeepsInformationBeyondDoublePrecision) {
  const long double value = 1 + 4 * std::numeric_limits<long double>::epsilon();
  datamunge::la::Mat<datamunge::Dense, long double> a = {{value}}, one = {{1}};
  const auto                                        product = (a * one).eval();
  EXPECT_EQ(product(0, 0), value);
  if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits)
    EXPECT_GT(product(0, 0), static_cast<long double>(static_cast<double>(value)));
}

/// @brief Compare a facade matrix to hand-computed row-major values.
/// @tparam E Matrix/expression type.
/// @param expression Value to materialise once.
/// @param rows Expected rows.
/// @param cols Expected columns.
/// @param values Row-major reference values.
/// @param tolerance Absolute tolerance.
template<class E>
void expect_matrix(
    const E&                      expression,
    std::size_t                   rows,
    std::size_t                   cols,
    std::initializer_list<double> values,
    double                        tolerance = 1e-11) {
  const auto actual = expression.eval();
  ASSERT_EQ(actual.rows(), rows);
  ASSERT_EQ(actual.cols(), cols);
  ASSERT_EQ(values.size(), rows * cols);
  auto expected = values.begin();
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      EXPECT_NEAR(actual(i, j), *expected++, tolerance);
}

TEST(MatrixApi, NestedRowsStorageAndBounds) {
  const Mat   a = {{1, 0, 2}, {0, 3, 0}};
  const SpMat s = {{1, 0, 2}, {0, 3, 0}};
  expect_matrix(a, 2, 3, {1, 0, 2, 0, 3, 0});
  expect_matrix(s, 2, 3, {1, 0, 2, 0, 3, 0});
  EXPECT_EQ(a.nnz(), 3);
  EXPECT_EQ(s.nnz(), 3);
  EXPECT_EQ(s.storage().row_ptr, (std::vector<std::size_t>{0, 2, 3}));
  EXPECT_EQ(s.storage().col_idx, (std::vector<std::size_t>{0, 2, 1}));
  EXPECT_THROW((Mat{{1}, {2, 3}}), std::invalid_argument);
  EXPECT_THROW((SpMat{{1}, {2, 3}}), std::invalid_argument);
  EXPECT_THROW(a(2, 0), std::out_of_range);
  EXPECT_THROW(s(0, 3), std::out_of_range);
  EXPECT_THROW((Mat(std::numeric_limits<std::size_t>::max(), 2)), std::length_error);
}

TEST(MatrixApi, ZeroShapesAndEmptyProducts) {
  Mat   empty;
  SpMat sparse;
  EXPECT_EQ(empty.rows(), 0);
  EXPECT_EQ(sparse.cols(), 0);
  EXPECT_EQ(sparse.storage().row_ptr, (std::vector<std::size_t>{0}));
  expect_matrix(Mat(2, 0) * Mat(0, 3), 2, 3, {0, 0, 0, 0, 0, 0});
  expect_matrix(SpMat(2, 0) * SpMat(0, 3), 2, 3, {0, 0, 0, 0, 0, 0});
  EXPECT_EQ(Mat(2, 0).t().eval().cols(), 2);
  EXPECT_EQ(solve(empty).eval().rows(), 0);
}

TEST(MatrixApi, MutationAndConversionsPreserveSparseInvariants) {
  SpMat s(3, 4);
  s.set(1, 3, 5);
  s.set(1, 0, 2);
  s.set(0, 2, 7);
  s.set(1, 3, 6);
  EXPECT_EQ(s.nnz(), 3);
  EXPECT_EQ(s.storage().row_ptr, (std::vector<std::size_t>{0, 1, 3, 3}));
  s.set(0, 2, 0);
  s.set(2, 1, 0);
  EXPECT_EQ(s.nnz(), 2);
  EXPECT_EQ(s.storage().col_idx, (std::vector<std::size_t>{0, 3}));
  Mat   dense      = s;
  SpMat round_trip = dense;
  expect_matrix(round_trip, 3, 4, {0, 0, 0, 0, 2, 0, 0, 6, 0, 0, 0, 0});
  EXPECT_THROW(s.set(3, 0, 1), std::out_of_range);
  dense.set(0, 1, 4);
  EXPECT_DOUBLE_EQ(dense(0, 1), 4);
  EXPECT_DOUBLE_EQ(s(0, 1), 0);
}

TEST(MatrixApi, RejectsMalformedNativeCsr) {
  datamunge::linalg::SparseCSR<double> csr;
  csr.nrows = 1;
  EXPECT_THROW((SpMat(csr)), std::invalid_argument);
  csr.nrows   = 1;
  csr.ncols   = 2;
  csr.row_ptr = {0, 2};
  csr.col_idx = {1, 0};
  csr.values  = {1, 2};
  EXPECT_THROW((SpMat(csr)), std::invalid_argument);
  csr.col_idx = {0, 2};
  EXPECT_THROW((SpMat(csr)), std::invalid_argument);
  csr.row_ptr = {1, 2};
  EXPECT_THROW((SpMat(csr)), std::invalid_argument);
}

TEST(MatrixApi, TransposeAndAllProductStorageCombinations) {
  const Mat   a = {{1, 0, 2}, {0, 3, 0}}, b = {{4, 0}, {0, 5}, {6, 0}};
  const SpMat sa = a, sb = b;
  expect_matrix(a * b, 2, 2, {16, 0, 0, 15});
  expect_matrix(sa * b, 2, 2, {16, 0, 0, 15});
  expect_matrix(a * sb, 2, 2, {16, 0, 0, 15});
  const auto sparse = (sa * sb).eval();
  static_assert(std::is_same_v<std::decay_t<decltype(sparse)>, SpMat>);
  EXPECT_EQ(sparse.nnz(), 2);
  expect_matrix(sa.t(), 3, 2, {1, 0, 0, 3, 2, 0});
  expect_matrix(a.t().t(), 2, 3, {1, 0, 2, 0, 3, 0});
  expect_matrix(sa.t().t(), 2, 3, {1, 0, 2, 0, 3, 0});
  expect_matrix(a.t() * a, 3, 3, {1, 0, 2, 0, 9, 0, 2, 0, 4});
  expect_matrix(sa.t() * sa, 3, 3, {1, 0, 2, 0, 9, 0, 2, 0, 4});
  expect_matrix(a.t() * sa, 3, 3, {1, 0, 2, 0, 9, 0, 2, 0, 4});
  expect_matrix(sa.t() * a, 3, 3, {1, 0, 2, 0, 9, 0, 2, 0, 4});
  expect_matrix(a * a.t(), 2, 2, {5, 0, 0, 9});
  expect_matrix(a * sa.t(), 2, 2, {5, 0, 0, 9});
  expect_matrix(sa * a.t(), 2, 2, {5, 0, 0, 9});
  expect_matrix(sa * sa.t(), 2, 2, {5, 0, 0, 9});
  expect_matrix(b.t() * sa.t(), 2, 2, {16, 0, 0, 15});
  expect_matrix(sb.t() * a.t(), 2, 2, {16, 0, 0, 15});
  expect_matrix(sb.t() * sa.t(), 2, 2, {16, 0, 0, 15});
}

TEST(MatrixApi, SparseCancellationAndVeryWideOutputStaySparse) {
  const SpMat a = {{1, 1}, {1, -1}}, b = {{1, 0}, {-1, 2}};
  const auto  result = (a * b).eval();
  expect_matrix(result, 2, 2, {0, 2, 2, -2});
  EXPECT_EQ(result.nnz(), 3);
  SpMat wide(2, 1000000000);
  wide.set(0, 999999999, 3);
  wide.set(1, 0, 4);
  const auto huge_shape = (SpMat{{2, 0}, {0, 5}} * wide).eval();
  EXPECT_EQ(huge_shape.cols(), 1000000000);
  EXPECT_EQ(huge_shape.nnz(), 2);
  EXPECT_DOUBLE_EQ(huge_shape(0, 999999999), 6);
  EXPECT_DOUBLE_EQ(huge_shape(1, 0), 20);
}

TEST(MatrixApi, DeferredOperandsObserveMutationUntilMaterialised) {
  Mat  a          = {{1, 2}, {3, 4}};
  auto expression = a.t() * a;
  Mat  snapshot   = expression;
  a.set(0, 0, 5);
  expect_matrix(snapshot, 2, 2, {10, 14, 14, 20});
  expect_matrix(expression, 2, 2, {34, 22, 22, 20});
  Mat        singular = {{0, 0}, {0, 0}}, rhs = {{2}, {4}};
  const auto delayed = solve(singular) * rhs;
  singular.set(0, 0, 1);
  singular.set(1, 1, 2);
  expect_matrix(delayed, 2, 1, {2, 2});
}

TEST(MatrixApi, TemporaryMatricesAndNamedExpressionNodesAreOwned) {
  const auto expression = [] { return Mat{{1, 2}}.t() * Mat{{3, 4}}; }();
  expect_matrix(expression, 2, 2, {3, 4, 6, 8});
  const auto const_temporary = []() -> const Mat { return Mat{{1, 2}}; }().t();
  expect_matrix(const_temporary, 2, 1, {1, 2});
  Mat        x = {{1, 0}, {1, 1}, {1, 2}}, y = {{1}, {3}, {5}};
  const auto returned = [&] {
    const auto gram           = x.t() * x;
    const auto inverse_action = solve(gram);
    const auto first_product  = inverse_action * x.t();
    return first_product * y;
  }();
  expect_matrix(returned, 2, 1, {1, 2});
  const auto sparse = [] { return SpMat{{1, 2}}.t() * SpMat{{3, 4}}; }();
  expect_matrix(sparse, 2, 2, {3, 4, 6, 8});
}

TEST(MatrixApi, AliasedAssignmentIsSafeForBothStorageKinds) {
  Mat a = {{1, 2, 3}, {4, 5, 6}};
  a     = a.t();
  expect_matrix(a, 3, 2, {1, 4, 2, 5, 3, 6});
  a = a.t() * a;
  expect_matrix(a, 2, 2, {14, 32, 32, 77});
  SpMat s = {{1, 2}, {0, 3}};
  s       = s.t() * s;
  expect_matrix(s, 2, 2, {1, 2, 2, 13});
}

TEST(MatrixApi, MovesAndBackendTransfersLeaveUsableEmptySources) {
  Mat a = {{1, 2}};
  Mat b(std::move(a));
  EXPECT_EQ(a.rows(), 0);
  EXPECT_EQ(a.cols(), 0);
  EXPECT_EQ(a.nnz(), 0);
  EXPECT_THROW(a(0, 0), std::out_of_range);
  SpMat s = {{0, 3}}, t;
  t       = std::move(s);
  EXPECT_EQ(s.rows(), 0);
  EXPECT_EQ(s.nnz(), 0);
  EXPECT_THROW(s.set(0, 0, 1), std::out_of_range);
  const auto native = std::move(b).storage();
  EXPECT_EQ(native.cols(), 2);
  EXPECT_EQ(b.cols(), 0);
  auto moved_again = std::move(t).storage();
  EXPECT_EQ(t.cols(), 0);
  EXPECT_EQ(moved_again.nnz(), 1);
  const SpMat copied_empty = s;
  EXPECT_EQ(copied_empty.t().eval().rows(), 0);
  SpMat default_native(datamunge::linalg::SparseCSR<double>{});
  EXPECT_EQ(default_native.rows(), 0);
}

TEST(MatrixApi, ShapesAreCheckedAtConstructionAndAgainBeforeEvaluation) {
  Mat a(2, 3), b(4, 2);
  EXPECT_THROW(a * b, std::invalid_argument);
  EXPECT_THROW(solve(a), std::invalid_argument);
  Mat  left = {{1, 0}, {0, 1}}, rhs = {{1}, {2}};
  auto expression = solve(left) * left * rhs;
  rhs             = Mat(3, 1);
  EXPECT_THROW(expression.eval(), std::invalid_argument);
  auto inverse = solve(left);
  left         = Mat(2, 3);
  EXPECT_THROW(inverse.eval(), std::invalid_argument);
}

TEST(MatrixApi, DenseAndSparsePlaygroundHaveAnalyticCoefficients) {
  Mat         x = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}}, y = {{1.2}, {3.3}, {4.4}};
  const auto  dense = solve(x.t() * x) * x.t() * y;
  const SpMat sx = x, sy = y;
  const auto  sparse = solve(sx.t() * sx) * sx.t() * sy;
  // Exact decimal-input normal equations give beta = (7/33, 17/33).
  expect_matrix(dense, 2, 1, {7.0 / 33, 17.0 / 33});
  expect_matrix(sparse, 2, 1, {7.0 / 33, 17.0 / 33});
  const auto fitted           = (x * dense).eval();
  double     squared_residual = 0;
  for (std::size_t row = 0; row < x.rows(); ++row)
    squared_residual += std::pow(y(row, 0) - fitted(row, 0), 2);
  EXPECT_NEAR(squared_residual, 1.0 / 6, 1e-13);
  for (std::size_t col = 0; col < x.cols(); ++col) {
    double normal_residual = 0;
    for (std::size_t row = 0; row < x.rows(); ++row)
      normal_residual += x(row, col) * (y(row, 0) - fitted(row, 0));
    EXPECT_NEAR(normal_residual, 0, 1e-12);
  }
}

TEST(MatrixApi, MultiRhsDirectSolveAndLongInverseActionChains) {
  const Mat a = {{0, 2}, {1, 3}}, b = {{4, 2}, {7, 4}};
  expect_matrix(solve(a) * b, 2, 2, {1, 1, 2, 1});
  expect_matrix(solve(a, b), 2, 2, {1, 1, 2, 1});
  expect_matrix(solve(a), 2, 2, {-1.5, 1, .5, 0});
  const Mat x = {{1, 0}, {1, 1}, {1, 2}}, y = {{1, 2}, {3, 4}, {5, 6}};
  expect_matrix(solve(x, y), 2, 2, {1, 2, 2, 2});
  expect_matrix(solve(x.t() * x) * x.t() * y * Mat{{1}, {2}}, 2, 1, {5, 6});
  expect_matrix((solve(a) * b).t(), 2, 2, {1, 2, 1, 1});
}

TEST(MatrixApi, ReassociationDoesNotChangeAnUnrelatedProduct) {
  Mat a = {{2, 1}, {1, 2}}, b = {{1, 2}, {3, 4}}, c = {{5}, {6}};
  expect_matrix(a * b * c, 2, 1, {73, 95});
  // A solve in the right factor must not be mistaken for an inverse at the far left.
  expect_matrix(b * (solve(a) * c), 2, 1, {6, 40.0 / 3});
}

TEST(MatrixApi, SingularRankDeficientNonfiniteAndWideInputsAreRejected) {
  Mat singular = {{1, 2}, {2, 4}}, zero_rhs(2, 1);
  EXPECT_THROW((solve(singular) * zero_rhs).eval(), std::runtime_error);
  EXPECT_THROW(solve(singular).eval(), std::runtime_error);
  Mat rank_deficient = {{1, 2}, {2, 4}, {3, 6}}, y = {{1}, {2}, {3}};
  EXPECT_THROW(solve(rank_deficient, y), std::runtime_error);
  EXPECT_THROW(solve(Mat(2, 3), Mat(2, 1)), std::invalid_argument);
  EXPECT_THROW(solve(Mat(2, 2), Mat(3, 1)), std::invalid_argument);
  Mat nonfinite = {{1, 0}, {0, 1}};
  nonfinite.set(0, 0, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW((solve(nonfinite) * zero_rhs).eval(), std::invalid_argument);
  Mat identity = {{1, 0}, {0, 1}};
  zero_rhs.set(0, 0, std::numeric_limits<double>::infinity());
  EXPECT_THROW(solve(identity, zero_rhs), std::invalid_argument);
}

TEST(MatrixApi, FormattingEvaluatesDeferredComputationsAndRetainsPreviewConventions) {
  static_assert(std::formattable<Mat, char>);
  static_assert(std::formattable<SpMat, char>);
  Mat  a = {{1, 2}}, b = {{3}, {4}};
  auto product = a * b;
  EXPECT_EQ(std::format("{}", product), "Mat 1 x 1\n     [,0]\n[0,]   11");
  a.set(0, 0, 2);
  EXPECT_EQ(std::format("{}", product), "Mat 1 x 1\n     [,0]\n[0,]   14");
  const SpMat sparse = {{0, 2}, {3, 0}};
  const auto  text   = std::format("{}", sparse);
  EXPECT_TRUE(text.starts_with("SpMat 2 x 2, nnz=2\n"));
  EXPECT_NE(text.find("."), std::string::npos);
  EXPECT_EQ(
      std::format("{}", (sparse * sparse)),
      "SpMat 2 x 2, nnz=2\n     [,0] [,1]\n[0,]    6    .\n[1,]    .    6");
  const Mat matrix(100, 100);
  EXPECT_NE(std::format("{}", matrix).find("..."), std::string::npos);
  EXPECT_EQ(std::format("{:.3}", matrix), "Mat");
}
} // namespace
