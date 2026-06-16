#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

namespace datamunge {
namespace linalg {

template <typename T>
class SparseCOO;

// ============================================================
// CRTP expression base
// ============================================================

template <typename Derived>
struct SparseExpr {
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

// ============================================================
// Expression nodes
// ============================================================

/// Lazy scalar-matrix product: alpha * A
template <typename Expr, typename Scalar>
struct ScaleExpr : SparseExpr<ScaleExpr<Expr, Scalar>> {
    using value_type = typename Expr::value_type;
    const Expr& expr;
    Scalar      alpha;
    ScaleExpr(const Expr& e, Scalar a) : expr(e), alpha(a) {}
    std::size_t rows() const { return expr.rows(); }
    std::size_t cols() const { return expr.cols(); }
};

/// Lazy matrix sum: A + B
template <typename LHS, typename RHS>
struct SumExpr : SparseExpr<SumExpr<LHS, RHS>> {
    using value_type = typename LHS::value_type;
    const LHS& lhs;
    const RHS& rhs;
    SumExpr(const LHS& l, const RHS& r) : lhs(l), rhs(r) {}
    std::size_t rows() const { return lhs.rows(); }
    std::size_t cols() const { return lhs.cols(); }
};

// ============================================================
// Free expression builders — work on any SparseExpr subtype
// ============================================================

template <typename LHS, typename RHS>
SumExpr<LHS, RHS> operator+(const SparseExpr<LHS>& lhs, const SparseExpr<RHS>& rhs) {
    return {lhs.derived(), rhs.derived()};
}

template <typename Expr, typename Scalar>
ScaleExpr<Expr, Scalar> operator*(Scalar alpha, const SparseExpr<Expr>& e) {
    return {e.derived(), alpha};
}

template <typename Expr, typename Scalar>
ScaleExpr<Expr, Scalar> operator*(const SparseExpr<Expr>& e, Scalar alpha) {
    return {e.derived(), alpha};
}

// ============================================================
// eval() forward declarations
// ============================================================

template <typename T>
SparseCOO<T> eval(const SparseCOO<T>& m);

template <typename Expr, typename Scalar>
SparseCOO<typename Expr::value_type> eval(const ScaleExpr<Expr, Scalar>& e);

template <typename LHS, typename RHS>
SparseCOO<typename LHS::value_type> eval(const SumExpr<LHS, RHS>& e);

// ============================================================
// SparseCOO — coordinate-format sparse matrix
//
// Storage: three parallel SOA vectors (row, col, val).
// SOA layout is preferred over tuple-of-structs for BLAS operations:
// iterating a single array (e.g. just values for scaling) stays in cache.
//
// Invariants after compress():
//   - No explicit zeros remain.
//   - No duplicate (i,j) pairs; overlapping entries are summed.
//   - Entries are sorted by (row, col) — required by spmv_tiled().
// ============================================================

template <typename T = double>
class SparseCOO : public SparseExpr<SparseCOO<T>> {
public:
    using value_type = T;

    // Block size tuned so the output strip y[r0..r1) fits in a 32 KB L1 cache.
    // Override at call sites for different target cache levels (e.g. L2 = 256 KB).
    static constexpr std::size_t default_block_rows = 32 * 1024 / sizeof(T);

    SparseCOO() = default;

    SparseCOO(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {}

    /// Preallocate storage for nnz_hint non-zeros.
    SparseCOO(std::size_t rows, std::size_t cols, std::size_t nnz_hint)
        : rows_(rows), cols_(cols) {
        row_idx_.reserve(nnz_hint);
        col_idx_.reserve(nnz_hint);
        vals_.reserve(nnz_hint);
    }

    // Implicit construction/assignment from any lazy expression.
    template <typename Derived>
    SparseCOO(const SparseExpr<Derived>& expr) {  // NOLINT(*-explicit-*)
        *this = eval(expr.derived());
    }

    template <typename Derived>
    SparseCOO& operator=(const SparseExpr<Derived>& expr) {
        *this = eval(expr.derived());
        return *this;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t nnz()  const { return vals_.size(); }

    /// True after compress(); cleared by any mutation.
    /// spmv_tiled() requires this to be true.
    bool is_compressed() const { return compressed_; }

    const std::vector<std::size_t>& row_indices() const { return row_idx_; }
    const std::vector<std::size_t>& col_indices() const { return col_idx_; }
    const std::vector<T>&           values()      const { return vals_; }

    // ---- Single-entry mutation ----

    /// Append one entry. Duplicate (row,col) pairs are allowed; call compress() to merge them.
    void set(std::size_t row, std::size_t col, T val) {
        compressed_ = false;
        row_idx_.push_back(row);
        col_idx_.push_back(col);
        vals_.push_back(val);
    }

    /// Mark an existing (row,col) entry as zero without removing it from storage.
    /// Call compress() to actually free the space.
    void set_zero(std::size_t row, std::size_t col) {
        compressed_ = false;
        for (std::size_t k = 0; k < vals_.size(); ++k)
            if (row_idx_[k] == row && col_idx_[k] == col)
                vals_[k] = T{0};
    }

    // ---- Batch mutation ----

    /// Append multiple entries at once. Duplicate pairs are allowed; call compress() to merge.
    void set_batch(const std::vector<std::size_t>& rows,
                   const std::vector<std::size_t>& cols,
                   const std::vector<T>&            vals) {
        compressed_ = false;
        row_idx_.insert(row_idx_.end(), rows.begin(), rows.end());
        col_idx_.insert(col_idx_.end(), cols.begin(), cols.end());
        vals_.insert(vals_.end(), vals.begin(), vals.end());
    }

    /// Mark a batch of (row,col) entries as zero. Call compress() to free space.
    void set_zero_batch(const std::vector<std::size_t>& rows,
                        const std::vector<std::size_t>& cols) {
        compressed_ = false;
        for (std::size_t i = 0; i < rows.size(); ++i)
            set_zero(rows[i], cols[i]);
    }

    // ---- Recompression ----

    /// Sort entries by (row, col), merge duplicates by summing, remove explicit zeros.
    /// Required before calling spmv_tiled().
    void compress() {
        std::size_t n = vals_.size();
        if (n == 0) {
            compressed_ = true;
            return;
        }

        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return row_idx_[a] != row_idx_[b] ? row_idx_[a] < row_idx_[b]
                                              : col_idx_[a] < col_idx_[b];
        });

        std::vector<std::size_t> nr, nc;
        std::vector<T>           nv;
        nr.reserve(n);
        nc.reserve(n);
        nv.reserve(n);

        for (std::size_t i = 0; i < n;) {
            std::size_t r = row_idx_[order[i]];
            std::size_t c = col_idx_[order[i]];
            T           s = T{0};
            std::size_t j = i;
            while (j < n && row_idx_[order[j]] == r && col_idx_[order[j]] == c)
                s += vals_[order[j++]];
            if (s != T{0}) {
                nr.push_back(r);
                nc.push_back(c);
                nv.push_back(s);
            }
            i = j;
        }

        row_idx_     = std::move(nr);
        col_idx_     = std::move(nc);
        vals_        = std::move(nv);
        compressed_  = true;
    }

    // ---- In-place arithmetic (used by eval) ----

    // Scaling preserves sort order, so compressed_ is intentionally left unchanged.
    void scale_inplace(T alpha) {
        for (auto& v : vals_) v *= alpha;
    }

    // ---- Sparse matrix-vector product ----

    /// y = A * x — allocates and returns a fresh output vector.
    std::vector<T> spmv(const std::vector<T>& x) const {
        std::vector<T> y(rows_, T{0});
        spmv(x, y);
        return y;
    }

    /// y = A * x — writes into a caller-provided buffer (must have size rows()).
    /// Zeroes y before accumulating so the buffer can be reused across calls.
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        std::fill(y.begin(), y.end(), T{0});
        for (std::size_t k = 0; k < vals_.size(); ++k)
            y[row_idx_[k]] += vals_[k] * x[col_idx_[k]];
    }

    /// y = A * x — row-strip tiled to keep the active slice of y in L1 cache.
    ///
    /// Requires compress() to have been called (entries must be sorted by row).
    /// block_rows controls the strip height; defaults to fill a 32 KB L1 cache.
    /// Pass a larger value (e.g. 256*1024/sizeof(T)) to target L2 instead.
    ///
    /// Writes into a caller-provided buffer so the same allocation can be
    /// reused across repeated calls (e.g. in an iterative solver).
    void spmv_tiled(const std::vector<T>& x, std::vector<T>& y,
                    std::size_t block_rows = default_block_rows) const {
        assert(compressed_ && "spmv_tiled requires compress() to have been called first");
        std::fill(y.begin(), y.end(), T{0});

        const std::size_t n = vals_.size();
        std::size_t       k = 0;  // advances monotonically through sorted entries

        for (std::size_t r0 = 0; r0 < rows_ && k < n; r0 += block_rows) {
            const std::size_t r1 = std::min(r0 + block_rows, rows_);
            // Entries for rows [r0, r1) are contiguous in sorted order.
            // The slice y[r0..r1) stays in cache for this inner loop.
            while (k < n && row_idx_[k] < r1) {
                y[row_idx_[k]] += vals_[k] * x[col_idx_[k]];
                ++k;
            }
        }
    }

    /// y = A * x — tiled version that allocates and returns the output vector.
    std::vector<T> spmv_tiled(const std::vector<T>& x,
                               std::size_t block_rows = default_block_rows) const {
        std::vector<T> y(rows_, T{0});
        spmv_tiled(x, y, block_rows);
        return y;
    }

private:
    std::size_t              rows_{0}, cols_{0};
    std::vector<std::size_t> row_idx_, col_idx_;
    std::vector<T>           vals_;
    bool                     compressed_{false};
};

// ============================================================
// eval() implementations — materialize lazy expressions
// ============================================================

template <typename T>
SparseCOO<T> eval(const SparseCOO<T>& m) {
    return m;
}

template <typename Expr, typename Scalar>
SparseCOO<typename Expr::value_type> eval(const ScaleExpr<Expr, Scalar>& e) {
    using T = typename Expr::value_type;
    SparseCOO<T> result = eval(e.expr);
    result.scale_inplace(static_cast<T>(e.alpha));
    return result;
}

template <typename LHS, typename RHS>
SparseCOO<typename LHS::value_type> eval(const SumExpr<LHS, RHS>& e) {
    using T = typename LHS::value_type;
    SparseCOO<T>       result  = eval(e.lhs);
    const SparseCOO<T> rhs_mat = eval(e.rhs);
    result.set_batch(rhs_mat.row_indices(), rhs_mat.col_indices(), rhs_mat.values());
    result.compress();
    return result;
}

} // namespace linalg
} // namespace datamunge
