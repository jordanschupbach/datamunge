/// @file strassen.cpp
/// @brief Actual checked seven-product recursion, scalar work and returned-block capture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/strassen.hpp>
#include <limits>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Translation-unit-private checked producing kernels; no shared mutable state.
namespace {
/// @brief Invocation-owned row-major block matrix.
using Matrix = StrassenMatrix;
/// @brief Invocation-owned contiguous scalar row.
using Vector = std::vector<double>;
/// @brief Reject actual nonfinite arithmetic.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Strassen arithmetic overflow");
  return x;
}
/// @brief Checked complete size product.
std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("Strassen capacity");
  return a * b;
}
/// @brief Validate every finite rectangular row before dimension shortcuts.
std::size_t validate(const Matrix& x) {
  auto n = x.empty() ? 0 : x.front().size();
  for (const auto& row : x) {
    if (row.size() != n)
      throw std::invalid_argument("Strassen ragged matrix");
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("Strassen finite operands required");
  }
  return n;
}
/// @brief Actual producer state and checked counters/capture, owned per call.
struct Context {
  StrassenOptions options;     ///< Independent complete invocation policies.
  StrassenRun     run;         ///< Actual producing result and counters.
  bool            recording{}; ///< Retain actual completed blocks.
  std::size_t     copied{};    ///< Actual retained scalar entries.

  /// @brief Count an actual block scalar sum/difference before evaluation.
  void addition() {
    if (run.block_additions == std::numeric_limits<std::size_t>::max() ||
        (options.max_block_additions && run.block_additions >= options.max_block_additions))
      throw std::length_error("Strassen block work");
    ++run.block_additions;
  }
  /// @brief Count actual schoolbook product and accumulator addition.
  void product() {
    ++run.scalar_products;
    ++run.leaf_accumulations;
  }
  /// @brief Capture an actual returned block under complete record/value limits.
  void capture(const std::string& path, std::size_t depth, const Matrix& x) {
    if (!recording)
      return;
    auto n = multiply(x.size(), x.size());
    if (n > std::numeric_limits<std::size_t>::max() - copied ||
        (options.max_records && run.steps.size() >= options.max_records) ||
        (options.max_history_values &&
         (copied > options.max_history_values || n > options.max_history_values - copied)))
      throw std::length_error("Strassen history");
    copied += n;
    run.steps.push_back({path, depth, x, run.scalar_products, run.block_additions});
  }
};
/// @brief Add actual square block entries and count scalar additions.
Matrix add(const Matrix& A, const Matrix& B, Context& ctx) {
  const std::size_t n = A.size();
  Matrix            C(n, Vector(n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      ctx.addition();
      C[i][j] = finite(A[i][j] + B[i][j]);
    }
  return C;
}
/// @brief Subtract actual square block entries and count scalar subtractions.
Matrix sub(const Matrix& A, const Matrix& B, Context& ctx) {
  const std::size_t n = A.size();
  Matrix            C(n, Vector(n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      ctx.addition();
      C[i][j] = finite(A[i][j] - B[i][j]);
    }
  return C;
}
/// @brief Multiply actual square schoolbook leaf entries in i-k-j order.
Matrix naive(const Matrix& A, const Matrix& B, Context& ctx) {
  const std::size_t n = A.size();
  Matrix            C(n, Vector(n, 0.0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k) {
      const double a = A[i][k];
      for (std::size_t j = 0; j < n; ++j) {
        ctx.product();
        C[i][j] = finite(C[i][j] + finite(a * B[k][j]));
      }
    }
  return C;
}

/// @brief Execute actual seven-product recursion and capture real returned blocks.
Matrix strassen_square(
    const Matrix&      A,
    const Matrix&      B,
    Context&           ctx,
    const std::string& path,
    std::size_t        depth) {
  const std::size_t n = A.size();
  ++ctx.run.recursive_calls;
  if (n <= ctx.options.leaf_size) {
    ++ctx.run.leaf_calls;
    auto C = naive(A, B, ctx);
    ctx.capture(path, depth, C);
    return C;
  }
  const std::size_t m = n / 2;

  Matrix A11(m, Vector(m)), A12(m, Vector(m)), A21(m, Vector(m)), A22(m, Vector(m));
  Matrix B11(m, Vector(m)), B12(m, Vector(m)), B21(m, Vector(m)), B22(m, Vector(m));
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < m; ++j) {
      A11[i][j] = A[i][j];
      A12[i][j] = A[i][j + m];
      A21[i][j] = A[i + m][j];
      A22[i][j] = A[i + m][j + m];
      B11[i][j] = B[i][j];
      B12[i][j] = B[i][j + m];
      B21[i][j] = B[i + m][j];
      B22[i][j] = B[i + m][j + m];
    }

  const Matrix M1 =
      strassen_square(add(A11, A22, ctx), add(B11, B22, ctx), ctx, path + "/1", depth + 1);
  const Matrix M2 = strassen_square(add(A21, A22, ctx), B11, ctx, path + "/2", depth + 1);
  const Matrix M3 = strassen_square(A11, sub(B12, B22, ctx), ctx, path + "/3", depth + 1);
  const Matrix M4 = strassen_square(A22, sub(B21, B11, ctx), ctx, path + "/4", depth + 1);
  const Matrix M5 = strassen_square(add(A11, A12, ctx), B22, ctx, path + "/5", depth + 1);
  const Matrix M6 =
      strassen_square(sub(A21, A11, ctx), add(B11, B12, ctx), ctx, path + "/6", depth + 1);
  const Matrix M7 =
      strassen_square(sub(A12, A22, ctx), add(B21, B22, ctx), ctx, path + "/7", depth + 1);

  const Matrix C11 = add(sub(add(M1, M4, ctx), M5, ctx), M7, ctx);
  const Matrix C12 = add(M3, M5, ctx);
  const Matrix C21 = add(M2, M4, ctx);
  const Matrix C22 = add(add(sub(M1, M2, ctx), M3, ctx), M6, ctx);

  Matrix C(n, Vector(n, 0.0));
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < m; ++j) {
      C[i][j]         = C11[i][j];
      C[i][j + m]     = C12[i][j];
      C[i + m][j]     = C21[i][j];
      C[i + m][j + m] = C22[i][j];
    }
  ctx.capture(path, depth, C);
  return C;
}

/// @brief Validate and execute one actual padded multiplication.
StrassenRun execute(const Matrix& a, const Matrix& b, const StrassenOptions& o, bool capture) {
  auto k = validate(a), n = validate(b);
  if (b.size() != k || !o.leaf_size)
    throw std::invalid_argument("Strassen dimensions/crossover");
  Context ctx{o, {}, capture};
  auto    m = a.size();
  if (!m || !k || !n) {
    ctx.run.product = Matrix(m, Vector(n));
    return ctx.run;
  }
  std::size_t s = 1;
  while (s < std::max({m, k, n})) {
    if (s > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("Strassen padding");
    s *= 2;
  }
  auto cells = multiply(s, s);
  if (o.max_padded_cells && cells > o.max_padded_cells)
    throw std::length_error("Strassen square budget");
  std::size_t leaves = 1, order = s;
  while (order > o.leaf_size) {
    leaves = multiply(leaves, 7);
    order /= 2;
  }
  auto work = multiply(leaves, multiply(multiply(order, order), order));
  if (o.max_scalar_products && work > o.max_scalar_products)
    throw std::length_error("Strassen complete leaf work");
  Matrix A(s, Vector(s)), B(s, Vector(s));
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < k; ++j)
      A[i][j] = a[i][j];
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < n; ++j)
      B[i][j] = b[i][j];
  auto C               = strassen_square(A, B, ctx, "root", 0);
  ctx.run.padded_order = s;
  ctx.run.product      = Matrix(m, Vector(n));
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j)
      ctx.run.product[i][j] = C[i][j];
  return ctx.run;
}
} // namespace
StrassenRun
strassen_run(const StrassenMatrix& a, const StrassenMatrix& b, const StrassenOptions& options) {
  return execute(a, b, options, false);
}
StrassenRun
strassen_trace(const StrassenMatrix& a, const StrassenMatrix& b, const StrassenOptions& options) {
  return execute(a, b, options, true);
}
StrassenMatrix strassen_multiply(const StrassenMatrix& a, const StrassenMatrix& b) {
  auto k = validate(a);
  validate(b);
  if (b.size() != k)
    return {};
  return strassen_run(a, b).product;
}
std::vector<StrassenMatrix> strassen_example_matrices() {
  std::mt19937                           rng(1);
  std::uniform_real_distribution<double> u(-1, 1);
  std::vector<StrassenMatrix>            r(2, StrassenMatrix(100, std::vector<double>(100)));
  for (auto& m : r)
    for (auto& row : m)
      for (auto& v : row)
        v = u(rng);
  return r;
}
StrassenRun strassen_schoolbook_run(
    const StrassenMatrix&  a,
    const StrassenMatrix&  b,
    const StrassenOptions& options) {
  auto k = validate(a), n = validate(b);
  if (b.size() != k)
    throw std::invalid_argument("schoolbook dimensions");
  auto work = multiply(multiply(a.size(), k), n);
  if (options.max_scalar_products && work > options.max_scalar_products)
    throw std::length_error("schoolbook work");
  Context ctx{options};
  ctx.run.product = Matrix(a.size(), Vector(n));
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t p = 0; p < k; ++p)
      for (std::size_t j = 0; j < n; ++j) {
        ctx.product();
        ctx.run.product[i][j] = finite(ctx.run.product[i][j] + finite(a[i][p] * b[p][j]));
      }
  return ctx.run;
}
} // namespace datamunge::algorithms
