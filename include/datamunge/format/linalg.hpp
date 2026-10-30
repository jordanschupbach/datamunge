#pragma once

/// @file linalg.hpp
/// @brief std::format / std::print support for datamunge::linalg.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Matrices and tensors print R-style: a header line with the shape and
/// element type, then a column-aligned preview of the values capped at
/// 10 rows x 8 columns (elided rows/columns show as `...`). Sparse matrices
/// show structural zeros as `.`, like R's Matrix package. Floating-point
/// cells use 7 significant digits (R's default); scalar fields of result
/// structs print with `{}` (shortest round-trip).
///
/// @par Example
/// @code
/// std::print("{}\n", dm::linalg::DenseMatrix<double>(2, 3, {1, 2, 3, 4, 5, 6.5}));
/// // DenseMatrix<double> 2 x 3
/// //      [,0] [,1] [,2]
/// // [0,]    1    2    3
/// // [1,]    4    5  6.5
/// @endcode

#include <datamunge/format/text_formatter.hpp>
#include <datamunge/info.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/linalg/linalg.hpp>
#include <datamunge/linalg/lu.hpp>
#include <datamunge/linalg/qr.hpp>
#include <datamunge/linalg/tensor.hpp>

#include <algorithm>
#include <cstddef>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge::format_detail {

/// @brief Column-aligned, capped 2-D value previews shared by the matrix, tensor and grid formatters.
namespace grid_fmt {

/// @brief Rows shown before a grid preview elides the middle.
inline constexpr std::size_t max_rows = 10;
/// @brief Columns shown before a grid preview elides the middle.
inline constexpr std::size_t max_cols = 8;

/// @brief Which of `n` indices a capped preview shows: the first `cap - 1`, an ellipsis (nullopt), then the last.
/// @param n Number of indices.
/// @param cap Maximum number of indices to show.
/// @return The shown indices, with std::nullopt marking the elided run.
inline std::vector<std::optional<std::size_t>> shown(std::size_t n, std::size_t cap) {
  std::vector<std::optional<std::size_t>> out;
  if (n <= cap) {
    for (std::size_t i = 0; i < n; ++i) out.emplace_back(i);
    return out;
  }
  for (std::size_t i = 0; i + 1 < cap; ++i) out.emplace_back(i);
  out.emplace_back(std::nullopt);
  out.emplace_back(n - 1);
  return out;
}

/// @brief Renders an R-style labelled grid (`[,j]` column headers, `[i,]` row labels), right-aligned.
/// @tparam Cell Callable `std::string(std::size_t row, std::size_t col)` returning one cell's text.
/// @param rows Number of rows.
/// @param cols Number of columns.
/// @param cell Produces the text of the cell at (row, col); only called for shown cells.
/// @return The grid lines joined by newlines (no trailing newline); empty when either extent is 0.
template <typename Cell>
std::string grid(std::size_t rows, std::size_t cols, Cell&& cell) {
  if (rows == 0 || cols == 0) return {};
  const auto rs = shown(rows, max_rows);
  const auto cs = shown(cols, max_cols);
  // table[r][c]: row 0 is the header, column 0 the row labels.
  std::vector<std::vector<std::string>> table(rs.size() + 1, std::vector<std::string>(cs.size() + 1));
  for (std::size_t c = 0; c < cs.size(); ++c) table[0][c + 1] = cs[c] ? std::format("[,{}]", *cs[c]) : "...";
  for (std::size_t r = 0; r < rs.size(); ++r) {
    table[r + 1][0] = rs[r] ? std::format("[{},]", *rs[r]) : "...";
    for (std::size_t c = 0; c < cs.size(); ++c)
      table[r + 1][c + 1] = (rs[r] && cs[c]) ? cell(*rs[r], *cs[c]) : std::string("...");
  }
  std::vector<std::size_t> width(cs.size() + 1, 0);
  for (const auto& line : table)
    for (std::size_t c = 0; c < line.size(); ++c) width[c] = std::max(width[c], line[c].size());
  std::string out;
  for (std::size_t r = 0; r < table.size(); ++r) {
    if (r > 0) out += '\n';
    for (std::size_t c = 0; c < table[r].size(); ++c) {
      if (c > 0) out += ' ';
      // Row labels left-aligned, values right-aligned.
      if (c == 0) out += std::format("{:<{}}", table[r][c], width[c]);
      else out += std::format("{:>{}}", table[r][c], width[c]);
    }
  }
  return out;
}

/// @brief Appends `"\n" + body` to `head` when `body` is non-empty.
/// @param head The header line.
/// @param body The (possibly empty) grid.
/// @return The combined text.
inline std::string with_body(std::string head, const std::string& body) {
  if (!body.empty()) head += "\n" + body;
  return head;
}

/// @brief Text of one grid cell: floating-point values with 7 significant digits (R's default), others via `{}`.
/// @tparam T Element type.
/// @param value The value.
/// @return The cell text.
template <typename T>
std::string cell_text(const T& value) {
  if constexpr (std::is_floating_point_v<T>) return std::format("{:.7g}", value);
  else return std::format("{}", value);
}

/// @brief Header + grid preview of a dense matrix.
/// @tparam T Element type.
/// @param m The matrix.
/// @return `DenseMatrix<T> r x c` followed by the capped grid.
template <typename T>
std::string dense(const linalg::DenseMatrix<T>& m) {
  return with_body(std::format("DenseMatrix<{}> {} x {}", info_detail::type_name<T>(), m.rows(), m.cols()),
                   grid(m.rows(), m.cols(), [&](std::size_t i, std::size_t j) { return cell_text(m(i, j)); }));
}

/// @brief Grid preview of a COO matrix with structural zeros shown as `.` (duplicates summed).
/// @tparam T Element type.
/// @param m The matrix in coordinate format.
/// @return The capped grid (no header).
template <typename T>
std::string sparse_grid(const linalg::SparseCOO<T>& m) {
  std::map<std::pair<std::size_t, std::size_t>, T> cells;
  const auto& r = m.row_indices();
  const auto& c = m.col_indices();
  const auto& v = m.values();
  for (std::size_t k = 0; k < v.size(); ++k) {
    auto [it, inserted] = cells.try_emplace({r[k], c[k]}, v[k]);
    if (!inserted) it->second += v[k];
  }
  return grid(m.rows(), m.cols(), [&](std::size_t i, std::size_t j) {
    const auto it = cells.find({i, j});
    return it == cells.end() ? std::string(".") : cell_text(it->second);
  });
}

/// @brief Header + sparse grid for one storage format (`SparseCSR<double> 3 x 3, nnz=4`).
/// @tparam T Element type.
/// @param name The storage type's name, e.g. "SparseCSR".
/// @param coo The same matrix in COO form, used for the preview.
/// @param extra Extra header fields, e.g. ", diags=3" (may be empty).
/// @return The header line and the capped grid.
template <typename T>
std::string sparse(const char* name, const linalg::SparseCOO<T>& coo, const std::string& extra = {}) {
  return with_body(std::format("{}<{}> {} x {}, nnz={}{}", name, info_detail::type_name<T>(), coo.rows(), coo.cols(),
                               coo.nnz(), extra),
                   sparse_grid(coo));
}

/// @brief Text of one tensor element: numbers via `{}`, bools as true/false, strings quoted.
/// @param t The tensor.
/// @param flat Row-major flat index of the element.
/// @return The element's text.
inline std::string tensor_cell(const linalg::Tensor& t, std::size_t flat) {
  switch (t.dtype()) {
    case linalg::TensorDType::String: return std::format("\"{}\"", t.string_at_flat(flat));
    case linalg::TensorDType::Bool: return t.at_flat(flat) != 0.0 ? "true" : "false";
    default: return cell_text(t.at_flat(flat));
  }
}

/// @brief Header + shape-aware preview of a tensor: scalar, 1-D preview, 2-D grid, or the first few 2-D slices.
/// @param t The tensor.
/// @return The tensor's text.
inline std::string tensor(const linalg::Tensor& t) {
  const auto& shape = t.shape();
  std::string out = std::format("Tensor(shape={}, dtype={})", shape, t.dtype_name());
  if (t.size() == 0) return out;
  if (shape.empty()) return out + "\n" + tensor_cell(t, 0);
  if (shape.size() == 1) {
    std::vector<std::string> cells;
    for (const auto i : shown(shape[0], max_rows)) cells.push_back(i ? tensor_cell(t, *i) : "...");
    std::string line = "[";
    for (std::size_t k = 0; k < cells.size(); ++k) line += (k ? ", " : "") + cells[k];
    return out + "\n" + line + "]";
  }
  const std::size_t rows = shape[shape.size() - 2];
  const std::size_t cols = shape.back();
  const std::size_t slices = t.size() / (rows * cols);
  constexpr std::size_t max_slices = 4;
  for (std::size_t s = 0; s < std::min(slices, max_slices); ++s) {
    const auto slice_grid = grid(rows, cols, [&](std::size_t i, std::size_t j) {
      return tensor_cell(t, s * rows * cols + i * cols + j);
    });
    if (shape.size() > 2) {
      // Leading index of this slice, e.g. "[1, 0, :, :]".
      std::vector<std::size_t> lead(shape.size() - 2);
      std::size_t rest = s;
      for (std::size_t d = lead.size(); d-- > 0;) {
        lead[d] = rest % shape[d];
        rest /= shape[d];
      }
      std::string label = "[";
      for (const auto i : lead) label += std::format("{}, ", i);
      out += "\n" + label + ":, :]";
    }
    out += "\n" + slice_grid;
  }
  if (slices > max_slices) out += std::format("\n... ({} more slices)", slices - max_slices);
  return out;
}

}  // namespace grid_fmt
}  // namespace datamunge::format_detail

/// @brief Prints a DenseMatrix as a "DenseMatrix<T> r x c" header plus a capped, column-aligned value grid.
template <typename T>
struct std::formatter<datamunge::linalg::DenseMatrix<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::DenseMatrix<T>> {
  /// @brief The matrix header and value preview.
  /// @param m The matrix to print.
  /// @return Header line, then up to 10 x 8 values (`...` marks elided rows/columns).
  static std::string text(const datamunge::linalg::DenseMatrix<T>& m) {
    return datamunge::format_detail::grid_fmt::dense(m);
  }
};

/// @brief Prints a SparseCOO matrix as a header (shape, nnz) plus a value grid with `.` for structural zeros.
template <typename T>
struct std::formatter<datamunge::linalg::SparseCOO<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SparseCOO<T>> {
  /// @brief The matrix header and value preview.
  /// @param m The matrix to print.
  /// @return `SparseCOO<T> r x c, nnz=n` (plus `, uncompressed` if pending compress()), then the grid.
  static std::string text(const datamunge::linalg::SparseCOO<T>& m) {
    return datamunge::format_detail::grid_fmt::sparse("SparseCOO", m, m.is_compressed() ? "" : ", uncompressed");
  }
};

/// @brief Prints a SparseCSR matrix as a header (shape, nnz) plus a value grid with `.` for structural zeros.
template <typename T>
struct std::formatter<datamunge::linalg::SparseCSR<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SparseCSR<T>> {
  /// @brief The matrix header and value preview.
  /// @param m The matrix to print.
  /// @return `SparseCSR<T> r x c, nnz=n`, then the grid.
  static std::string text(const datamunge::linalg::SparseCSR<T>& m) {
    return datamunge::format_detail::grid_fmt::sparse("SparseCSR", datamunge::linalg::to_coo(m));
  }
};

/// @brief Prints a SparseCSC matrix as a header (shape, nnz) plus a value grid with `.` for structural zeros.
template <typename T>
struct std::formatter<datamunge::linalg::SparseCSC<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SparseCSC<T>> {
  /// @brief The matrix header and value preview.
  /// @param m The matrix to print.
  /// @return `SparseCSC<T> r x c, nnz=n`, then the grid.
  static std::string text(const datamunge::linalg::SparseCSC<T>& m) {
    return datamunge::format_detail::grid_fmt::sparse("SparseCSC", datamunge::linalg::to_coo(m));
  }
};

/// @brief Prints a SparseELL matrix as a header (shape, nnz, row width) plus a value grid with `.` for zeros.
template <typename T>
struct std::formatter<datamunge::linalg::SparseELL<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SparseELL<T>> {
  /// @brief The matrix header and value preview.
  /// @param m The matrix to print.
  /// @return `SparseELL<T> r x c, nnz=n, max_nnz_per_row=k`, then the grid.
  static std::string text(const datamunge::linalg::SparseELL<T>& m) {
    return datamunge::format_detail::grid_fmt::sparse("SparseELL", datamunge::linalg::to_coo(m),
                                                      std::format(", max_nnz_per_row={}", m.max_nnz_per_row));
  }
};

/// @brief Prints a SparseDIA matrix as a header (shape, nnz, diagonal offsets) plus a value grid with `.` for zeros.
template <typename T>
struct std::formatter<datamunge::linalg::SparseDIA<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SparseDIA<T>> {
  /// @brief The matrix header and value preview.
  /// @param m The matrix to print.
  /// @return `SparseDIA<T> r x c, nnz=n, offsets=[...]`, then the grid.
  static std::string text(const datamunge::linalg::SparseDIA<T>& m) {
    return datamunge::format_detail::grid_fmt::sparse(
        "SparseDIA", datamunge::linalg::to_coo(m),
        std::format(", offsets={}", datamunge::format_detail::preview(m.offsets)));
  }
};

/// @brief Prints a SparseMatrix as "SparseMatrix<T> (FMT) r x c, nnz=n" plus a value grid with `.` for zeros.
template <typename T>
struct std::formatter<datamunge::linalg::SparseMatrix<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SparseMatrix<T>> {
  /// @brief The matrix header (with its current storage format) and value preview.
  /// @param m The matrix to print.
  /// @return Header line, then the capped grid.
  static std::string text(const datamunge::linalg::SparseMatrix<T>& m) {
    namespace la = datamunge::linalg;
    la::SparseCOO<T> coo;
    const char* fmt = "COO";
    if (const auto* p = m.as_coo()) coo = *p;
    else if (const auto* p = m.as_csr()) coo = la::to_coo(*p), fmt = "CSR";
    else if (const auto* p = m.as_csc()) coo = la::to_coo(*p), fmt = "CSC";
    else if (const auto* p = m.as_ell()) coo = la::to_coo(*p), fmt = "ELL";
    else if (const auto* p = m.as_dia()) coo = la::to_coo(*p), fmt = "DIA";
    return datamunge::format_detail::grid_fmt::with_body(
        std::format("SparseMatrix<{}> ({}) {} x {}, nnz={}", datamunge::info_detail::type_name<T>(), fmt, m.rows(),
                    m.cols(), m.nnz()),
        datamunge::format_detail::grid_fmt::sparse_grid(coo));
  }
};

/// @brief Prints a SparseMatrix<double>::Format storage tag as its enumerator name (COO, CSR, CSC, ELL, DIA).
template <>
struct std::formatter<datamunge::linalg::SparseMatrix<double>::Format>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SparseMatrix<double>::Format> {
  /// @brief The format's name.
  /// @param f The storage format.
  /// @return "COO", "CSR", "CSC", "ELL" or "DIA" ("Format(<int>)" for an unknown value).
  static std::string text(datamunge::linalg::SparseMatrix<double>::Format f) {
    using F = datamunge::linalg::SparseMatrix<double>::Format;
    switch (f) {
      case F::COO: return "COO";
      case F::CSR: return "CSR";
      case F::CSC: return "CSC";
      case F::ELL: return "ELL";
      case F::DIA: return "DIA";
    }
    return std::format("Format({})", static_cast<int>(f));
  }
};

/// @brief Prints a Tensor as "Tensor(shape=[...], dtype=...)" plus a shape-aware, capped value preview.
template <>
struct std::formatter<datamunge::linalg::Tensor> : datamunge::format_detail::text_formatter<datamunge::linalg::Tensor> {
  /// @brief The tensor header and values: a scalar, a 1-D list, a 2-D grid, or the first 2-D slices of a higher-rank tensor.
  /// @param t The tensor to print.
  /// @return Header line, then the preview.
  static std::string text(const datamunge::linalg::Tensor& t) { return datamunge::format_detail::grid_fmt::tensor(t); }
};

/// @brief Prints a TensorDType as its enumerator name (Float64, Bool, String).
template <>
struct std::formatter<datamunge::linalg::TensorDType>
    : datamunge::format_detail::text_formatter<datamunge::linalg::TensorDType> {
  /// @brief The dtype's name.
  /// @param d The element type tag.
  /// @return "Float64", "Bool" or "String" ("TensorDType(<int>)" for an unknown value).
  static std::string text(datamunge::linalg::TensorDType d) {
    using D = datamunge::linalg::TensorDType;
    switch (d) {
      case D::Float64: return "Float64";
      case D::Bool: return "Bool";
      case D::String: return "String";
    }
    return std::format("TensorDType({})", static_cast<int>(d));
  }
};

/// @brief Prints a Cholesky factorization: status line, then the lower factor L.
template <typename T>
struct std::formatter<datamunge::linalg::CholeskyDecomposition<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::CholeskyDecomposition<T>> {
  /// @brief The decomposition's status and factor.
  /// @param d The decomposition.
  /// @return `CholeskyDecomposition<T>(n=.., ok=..)` then `L =` and its grid.
  static std::string text(const datamunge::linalg::CholeskyDecomposition<T>& d) {
    return std::format("CholeskyDecomposition<{}>(n={}, ok={})\nL = {}", datamunge::info_detail::type_name<T>(),
                       d.L.rows(), d.ok, d.L);
  }
};

/// @brief Prints an LU factorization: status line with pivots, then the packed LU matrix.
template <typename T>
struct std::formatter<datamunge::linalg::LUDecomposition<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::LUDecomposition<T>> {
  /// @brief The decomposition's status, pivots and packed factors.
  /// @param d The decomposition.
  /// @return `LUDecomposition<T>(n=.., singular=.., piv=[...])` then `LU =` and its grid.
  static std::string text(const datamunge::linalg::LUDecomposition<T>& d) {
    return std::format("LUDecomposition<{}>(n={}, singular={}, piv={})\nLU = {}",
                       datamunge::info_detail::type_name<T>(), d.LU.rows(), d.singular,
                       datamunge::format_detail::preview(d.piv), d.LU);
  }
};

/// @brief Prints a QR factorization: shapes, then the Q and R factors.
template <typename T>
struct std::formatter<datamunge::linalg::QRDecomposition<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::QRDecomposition<T>> {
  /// @brief The decomposition's factors.
  /// @param d The decomposition.
  /// @return `QRDecomposition<T>` then `Q =` and `R =` with their grids.
  static std::string text(const datamunge::linalg::QRDecomposition<T>& d) {
    return std::format("QRDecomposition<{}>\nQ = {}\nR = {}", datamunge::info_detail::type_name<T>(), d.Q, d.R);
  }
};

/// @brief Prints a symmetric eigendecomposition: eigenvalues, then the eigenvector matrix.
template <typename T>
struct std::formatter<datamunge::linalg::SymmetricEigenResult<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SymmetricEigenResult<T>> {
  /// @brief The eigenvalues and eigenvectors.
  /// @param r The eigendecomposition.
  /// @return `SymmetricEigenResult<T>(eigenvalues=[...])` then `eigenvectors =` and its grid.
  static std::string text(const datamunge::linalg::SymmetricEigenResult<T>& r) {
    return std::format("SymmetricEigenResult<{}>(eigenvalues={})\neigenvectors = {}",
                       datamunge::info_detail::type_name<T>(), datamunge::format_detail::preview(r.eigenvalues),
                       r.eigenvectors);
  }
};

/// @brief Prints a Jacobi preconditioner as "JacobiPreconditioner<T>(inv_diag=[...])".
template <typename T>
struct std::formatter<datamunge::linalg::JacobiPreconditioner<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::JacobiPreconditioner<T>> {
  /// @brief The inverse diagonal.
  /// @param p The preconditioner.
  /// @return One line with a preview of `inv_diag`.
  static std::string text(const datamunge::linalg::JacobiPreconditioner<T>& p) {
    return std::format("JacobiPreconditioner<{}>(inv_diag={})", datamunge::info_detail::type_name<T>(),
                       datamunge::format_detail::preview(p.inv_diag));
  }
};

/// @brief Prints an iterative-solver result as "SolverResult<T>(converged=.., iterations=.., residual_norm=.., x=[...])".
template <typename T>
struct std::formatter<datamunge::linalg::SolverResult<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SolverResult<T>> {
  /// @brief Convergence status and a preview of the solution.
  /// @param r The solver result.
  /// @return One line.
  static std::string text(const datamunge::linalg::SolverResult<T>& r) {
    return std::format("SolverResult<{}>(converged={}, iterations={}, residual_norm={}, x={})",
                       datamunge::info_detail::type_name<T>(), r.converged, r.iterations, r.residual_norm,
                       datamunge::format_detail::preview(r.x));
  }
};

/// @brief Prints Krylov-solver options as "SolverOptions(tol=.., max_iter=..)".
template <>
struct std::formatter<datamunge::linalg::SolverOptions>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SolverOptions> {
  /// @brief The tolerance and iteration cap.
  /// @param o The options.
  /// @return One line.
  static std::string text(const datamunge::linalg::SolverOptions& o) {
    return std::format("SolverOptions(tol={}, max_iter={})", o.tol, o.max_iter);
  }
};

/// @brief Prints sparse-regression options as "SparseLinearRegressionOptions(solver=SolverOptions(..), use_jacobi=..)".
template <>
struct std::formatter<datamunge::linalg::SparseLinearRegressionOptions>
    : datamunge::format_detail::text_formatter<datamunge::linalg::SparseLinearRegressionOptions> {
  /// @brief The nested solver options and preconditioning flag.
  /// @param o The options.
  /// @return One line.
  static std::string text(const datamunge::linalg::SparseLinearRegressionOptions& o) {
    return std::format("SparseLinearRegressionOptions(solver={}, use_jacobi={})", o.solver, o.use_jacobi);
  }
};

/// @brief Prints a least-squares fit: sizes, coefficients with standard errors, and goodness-of-fit statistics.
template <typename T>
struct std::formatter<datamunge::linalg::LinearRegressionResult<T>>
    : datamunge::format_detail::text_formatter<datamunge::linalg::LinearRegressionResult<T>> {
  /// @brief The fit's key results (fitted values/residuals/covariance are omitted).
  /// @param r The regression result.
  /// @return Three lines: sizes, coefficients/standard errors, then R², sigma and sums of squares.
  static std::string text(const datamunge::linalg::LinearRegressionResult<T>& r) {
    using datamunge::format_detail::preview;
    return std::format(
        "LinearRegressionResult<{}>(observations={}, predictors={}, df={})\n"
        "coefficients={}, standard_errors={}\n"
        "r_squared={}, adjusted_r_squared={}, sigma={}, rss={}, tss={}",
        datamunge::info_detail::type_name<T>(), r.observations, r.predictors, r.degrees_of_freedom,
        preview(r.coefficients), preview(r.standard_errors), r.r_squared, r.adjusted_r_squared, r.sigma,
        r.residual_sum_of_squares, r.total_sum_of_squares);
  }
};
