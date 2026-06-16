#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <variant>
#include <vector>

#include <datamunge/linalg/csc.hpp>
#include <datamunge/linalg/csr.hpp>
#include <datamunge/linalg/dia.hpp>
#include <datamunge/linalg/ell.hpp>
#include <datamunge/linalg/sparse_coo.hpp>

namespace datamunge {
namespace linalg {

// ============================================================
// SparseMatrix<T> — format-polymorphic sparse matrix
//
// Holds one of {COO, CSR, CSC, ELL, DIA} internally and dispatches
// spmv() to the appropriate implementation.  convert_to() changes the
// active format in-place; all conversions route through a compressed
// COO as the canonical intermediate.
//
// Typical usage:
//   Build the matrix with SparseCOO (incremental insertion is cheap there),
//   then call convert_to() once before entering a hot loop.
//
//   SparseMatrix<double> mat(std::move(coo));
//   mat.convert_to(SparseMatrix<double>::Format::CSR);
//   for (...) mat.spmv(x, y);
// ============================================================

template <typename T = double>
class SparseMatrix {
public:
    using value_type = T;

    enum class Format { COO, CSR, CSC, ELL, DIA };

    // ---- Construction from any supported storage type ----

    explicit SparseMatrix(SparseCOO<T> m) : storage_(std::move(m)), fmt_(Format::COO) {}
    explicit SparseMatrix(SparseCSR<T> m) : storage_(std::move(m)), fmt_(Format::CSR) {}
    explicit SparseMatrix(SparseCSC<T> m) : storage_(std::move(m)), fmt_(Format::CSC) {}
    explicit SparseMatrix(SparseELL<T> m) : storage_(std::move(m)), fmt_(Format::ELL) {}
    explicit SparseMatrix(SparseDIA<T> m) : storage_(std::move(m)), fmt_(Format::DIA) {}

    // ---- Accessors ----

    Format format() const { return fmt_; }

    std::size_t rows() const {
        return std::visit([](const auto& s) { return s.rows(); }, storage_);
    }
    std::size_t cols() const {
        return std::visit([](const auto& s) { return s.cols(); }, storage_);
    }
    std::size_t nnz() const {
        return std::visit([](const auto& s) { return s.nnz(); }, storage_);
    }

    // Typed accessors — returns nullptr if not currently in that format.
    const SparseCOO<T>* as_coo() const { return std::get_if<SparseCOO<T>>(&storage_); }
    const SparseCSR<T>* as_csr() const { return std::get_if<SparseCSR<T>>(&storage_); }
    const SparseCSC<T>* as_csc() const { return std::get_if<SparseCSC<T>>(&storage_); }
    const SparseELL<T>* as_ell() const { return std::get_if<SparseELL<T>>(&storage_); }
    const SparseDIA<T>* as_dia() const { return std::get_if<SparseDIA<T>>(&storage_); }

    // ---- Format conversion ----

    /// Convert the internal storage to the requested format.
    /// No-op if already in that format.
    /// All cross-format paths go through a compressed COO intermediate.
    void convert_to(Format target) {
        if (target == fmt_) return;

        // Step 1: materialise a compressed COO from whatever we currently hold.
        SparseCOO<T> coo = std::visit(
            [](const auto& s) -> SparseCOO<T> {
                using S = std::decay_t<decltype(s)>;
                if constexpr (std::is_same_v<S, SparseCOO<T>>)
                    return s;            // copy (already in COO)
                else
                    return to_coo(s);   // convert from format
            },
            storage_);
        if (!coo.is_compressed()) coo.compress();

        // Step 2: convert COO to the target format.
        switch (target) {
            case Format::COO: storage_ = std::move(coo); break;
            case Format::CSR: storage_ = to_csr(coo);    break;
            case Format::CSC: storage_ = to_csc(coo);    break;
            case Format::ELL: storage_ = to_ell(coo);    break;
            case Format::DIA: storage_ = to_dia(coo);    break;
        }
        fmt_ = target;
    }

    // ---- SpMV ----

    /// y = A * x — dispatches to the active format's implementation.
    void spmv(const std::vector<T>& x, std::vector<T>& y) const {
        std::visit([&](const auto& s) { s.spmv(x, y); }, storage_);
    }

    std::vector<T> spmv(const std::vector<T>& x) const {
        return std::visit([&](const auto& s) { return s.spmv(x); }, storage_);
    }

private:
    using storage_t =
        std::variant<SparseCOO<T>, SparseCSR<T>, SparseCSC<T>, SparseELL<T>, SparseDIA<T>>;

    storage_t storage_;
    Format    fmt_;
};

} // namespace linalg
} // namespace datamunge
