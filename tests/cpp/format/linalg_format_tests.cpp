// std::format / std::print support for datamunge::linalg types.
#include <datamunge/format/linalg.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace la = datamunge::linalg;

static_assert(std::formattable<la::DenseMatrix<double>, char>);
static_assert(std::formattable<la::DenseMatrix<int>, char>);
static_assert(std::formattable<la::SparseCOO<double>, char>);
static_assert(std::formattable<la::SparseCSR<double>, char>);
static_assert(std::formattable<la::SparseCSC<double>, char>);
static_assert(std::formattable<la::SparseELL<double>, char>);
static_assert(std::formattable<la::SparseDIA<double>, char>);
static_assert(std::formattable<la::SparseMatrix<double>, char>);
static_assert(std::formattable<la::SparseMatrix<double>::Format, char>);
static_assert(std::formattable<la::Tensor, char>);
static_assert(std::formattable<la::TensorDType, char>);
static_assert(std::formattable<la::CholeskyDecomposition<double>, char>);
static_assert(std::formattable<la::LUDecomposition<double>, char>);
static_assert(std::formattable<la::QRDecomposition<double>, char>);
static_assert(std::formattable<la::SymmetricEigenResult<double>, char>);
static_assert(std::formattable<la::JacobiPreconditioner<double>, char>);
static_assert(std::formattable<la::SolverResult<double>, char>);
static_assert(std::formattable<la::SolverOptions, char>);
static_assert(std::formattable<la::SparseLinearRegressionOptions, char>);
static_assert(std::formattable<la::LinearRegressionResult<double>, char>);

namespace {
la::SparseCOO<double> small_coo() {
  la::SparseCOO<double> m(3, 3);
  m.set(0, 0, 4);
  m.set(1, 1, 5);
  m.set(2, 0, -1);
  m.set(2, 2, 6);
  m.compress();
  return m;
}
const char* small_grid =
    "     [,0] [,1] [,2]\n"
    "[0,]    4    .    .\n"
    "[1,]    .    5    .\n"
    "[2,]   -1    .    6";
}  // namespace

TEST(LinalgFormat, DenseMatrixHeaderAndGrid) {
  const la::DenseMatrix<double> m(2, 3, {1, 2, 3, 4, 5, 6.5});
  EXPECT_EQ(std::format("{}", m),
            "DenseMatrix<double> 2 x 3\n"
            "     [,0] [,1] [,2]\n"
            "[0,]    1    2    3\n"
            "[1,]    4    5  6.5");
  EXPECT_EQ(std::format("{}", la::DenseMatrix<int>()), "DenseMatrix<int> 0 x 0");
}

TEST(LinalgFormat, DenseMatrixIsCapped) {
  la::DenseMatrix<double> m(20, 12, 0.0);
  m(19, 11) = 7;
  const std::string text = std::format("{}", m);
  EXPECT_TRUE(text.starts_with("DenseMatrix<double> 20 x 12\n"));
  EXPECT_NE(text.find("[,6] ... [,11]"), std::string::npos);
  EXPECT_NE(text.find("\n...  "), std::string::npos);
  EXPECT_TRUE(text.ends_with("0 ...     7"));
  EXPECT_EQ(text.find("[,7]"), std::string::npos);
  EXPECT_EQ(std::format("{}", la::DenseMatrix<double>(1, 1, 1.0 / 3)),
            "DenseMatrix<double> 1 x 1\n          [,0]\n[0,] 0.3333333");
  EXPECT_EQ(text.find("[9,]"), std::string::npos);
}

TEST(LinalgFormat, SparseFormats) {
  const auto coo = small_coo();
  EXPECT_EQ(std::format("{}", coo), std::string("SparseCOO<double> 3 x 3, nnz=4\n") + small_grid);
  EXPECT_EQ(std::format("{}", la::to_csr(coo)), std::string("SparseCSR<double> 3 x 3, nnz=4\n") + small_grid);
  EXPECT_EQ(std::format("{}", la::to_csc(coo)), std::string("SparseCSC<double> 3 x 3, nnz=4\n") + small_grid);
  EXPECT_EQ(std::format("{}", la::to_ell(coo)),
            std::string("SparseELL<double> 3 x 3, nnz=4, max_nnz_per_row=2\n") + small_grid);
  EXPECT_EQ(std::format("{}", la::to_dia(coo)),
            std::string("SparseDIA<double> 3 x 3, nnz=4, offsets=[-2, 0]\n") + small_grid);

  la::SparseMatrix<double> m(coo);
  m.convert_to(la::SparseMatrix<double>::Format::CSR);
  EXPECT_EQ(std::format("{}", m), std::string("SparseMatrix<double> (CSR) 3 x 3, nnz=4\n") + small_grid);
  EXPECT_EQ(std::format("{}", m.format()), "CSR");

  la::SparseCOO<double> pending(2, 2);
  pending.set(0, 1, 1);
  pending.set(0, 1, 2);
  EXPECT_EQ(std::format("{}", pending),
            "SparseCOO<double> 2 x 2, nnz=2, uncompressed\n"
            "     [,0] [,1]\n"
            "[0,]    .    3\n"
            "[1,]    .    .");
}

TEST(LinalgFormat, Tensor) {
  EXPECT_EQ(std::format("{}", la::Tensor::from_values({2, 2}, {1, 2, 3, 4.5})),
            "Tensor(shape=[2, 2], dtype=float64)\n"
            "     [,0] [,1]\n"
            "[0,]    1    2\n"
            "[1,]    3  4.5");
  EXPECT_EQ(std::format("{}", la::Tensor::arange(0, 3)), "Tensor(shape=[3], dtype=float64)\n[0, 1, 2]");
  EXPECT_EQ(std::format("{}", la::Tensor::from_string_values({2}, {"a", "b"})),
            "Tensor(shape=[2], dtype=string)\n[\"a\", \"b\"]");
  EXPECT_EQ(std::format("{}", la::Tensor::from_bool_values({1}, {1})), "Tensor(shape=[1], dtype=bool)\n[true]");
  EXPECT_EQ(std::format("{}", la::Tensor::from_values({2, 1, 2}, {1, 2, 3, 4})),
            "Tensor(shape=[2, 1, 2], dtype=float64)\n"
            "[0, :, :]\n"
            "     [,0] [,1]\n"
            "[0,]    1    2\n"
            "[1, :, :]\n"
            "     [,0] [,1]\n"
            "[0,]    3    4");
  const std::string big = std::format("{}", la::Tensor::zeros({6, 2, 2}));
  EXPECT_TRUE(big.ends_with("... (2 more slices)"));
  EXPECT_NE(std::format("{}", la::Tensor::arange(0, 100)).find("8, ..., 99]"), std::string::npos);
  EXPECT_EQ(std::format("{}", la::TensorDType::Bool), "Bool");
}

TEST(LinalgFormat, Decompositions) {
  const la::DenseMatrix<double> a(2, 2, {4, 2, 2, 3});
  const std::string chol = std::format("{}", la::cholesky(a));
  EXPECT_TRUE(chol.starts_with("CholeskyDecomposition<double>(n=2, ok=true)\nL = DenseMatrix<double> 2 x 2\n"));
  const std::string lu = std::format("{}", la::lu(a));
  EXPECT_TRUE(lu.starts_with("LUDecomposition<double>(n=2, singular=false, piv=[0, 1])\nLU = DenseMatrix<double> 2 x 2"));
  const std::string qr = std::format("{}", la::qr(a));
  EXPECT_NE(qr.find("Q = DenseMatrix<double> 2 x 2"), std::string::npos);
  EXPECT_NE(qr.find("R = DenseMatrix<double> 2 x 2"), std::string::npos);
  const la::SymmetricEigenResult<double> eig{{3, 1}, la::DenseMatrix<double>::identity(2)};
  EXPECT_EQ(std::format("{}", eig),
            "SymmetricEigenResult<double>(eigenvalues=[3, 1])\n"
            "eigenvectors = DenseMatrix<double> 2 x 2\n"
            "     [,0] [,1]\n"
            "[0,]    1    0\n"
            "[1,]    0    1");
}

TEST(LinalgFormat, SolverStructs) {
  EXPECT_EQ(std::format("{}", la::SolverOptions{}), "SolverOptions(tol=1e-08, max_iter=0)");
  EXPECT_EQ(std::format("{}", la::SparseLinearRegressionOptions{}),
            "SparseLinearRegressionOptions(solver=SolverOptions(tol=1e-08, max_iter=0), use_jacobi=true)");
  const la::SolverResult<double> r{{1, 2}, 3, 1e-10, true};
  EXPECT_EQ(std::format("{}", r), "SolverResult<double>(converged=true, iterations=3, residual_norm=1e-10, x=[1, 2])");
  EXPECT_EQ(std::format("{}", la::make_jacobi(small_coo())), "JacobiPreconditioner<double>(inv_diag=[0.25, 0.2, 0.16666666666666666])");
  EXPECT_EQ(std::format("{:>12}", la::SolverOptions{}).size(), std::format("{}", la::SolverOptions{}).size());
}

TEST(LinalgFormat, RegressionResult) {
  la::LinearRegressionResult<double> r;
  r.coefficients = {1, 2};
  r.standard_errors = {0.5, 0.25};
  r.observations = 10;
  r.predictors = 2;
  r.degrees_of_freedom = 8;
  r.r_squared = 0.9;
  EXPECT_EQ(std::format("{}", r),
            "LinearRegressionResult<double>(observations=10, predictors=2, df=8)\n"
            "coefficients=[1, 2], standard_errors=[0.5, 0.25]\n"
            "r_squared=0.9, adjusted_r_squared=0, sigma=0, rss=0, tss=0");
}
