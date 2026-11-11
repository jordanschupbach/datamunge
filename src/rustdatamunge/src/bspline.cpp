/// @file bspline.cpp
/// @brief Private C ABI bridge from the Rust slice facade to the canonical C++ fitter.

#include <algorithm>
#include <cstddef>
#include <datamunge/bspline_regression.hpp>
#include <vector>

/// @brief Fit a B-spline curve and copy predictions into a caller-owned Rust buffer.
///
/// No pointers are retained. Every C++ exception is caught before crossing the ABI.
/// The safe Rust facade supplies nonempty, equally sized x/y and valid slices;
/// the C++ implementation validates finite coordinates, rank and basis size.
/// Independent calls share no mutable state.
/// @param x Borrowed training coordinates, containing len doubles.
/// @param y Borrowed responses, containing len doubles.
/// @param len Positive training slice length.
/// @param basis Number of cubic basis functions, at least four.
/// @param query Borrowed prediction coordinates, containing query_len doubles.
/// @param query_len Query/output length; zero is valid.
/// @param out Writable buffer of query_len doubles, not overlapping any input.
/// @return Zero on success; one on a fitting, prediction or allocation failure.
/// @par Example
/// @code
/// // Called by rustdatamunge::bspline_predict(&x, &y, 4, &query).
/// double x[] = {0, .25, .5, .75, 1}, y[] = {0, .25, .5, .75, 1};
/// double query[] = {.5}, out[1];
/// const int status = datamunge_rust_bspline_predict(x, y, 5, 4, query, 1, out);
/// // status == 0 and out[0] == .5 (within floating-point tolerance).
/// @endcode
/// @see datamunge::BSplineRegression
extern "C" int datamunge_rust_bspline_predict(
    const double* x,
    const double* y,
    std::size_t   len,
    std::size_t   basis,
    const double* query,
    std::size_t   query_len,
    double*       out) noexcept {
  try {
    const datamunge::BSplineRegression fit(
        std::vector<double>(x, x + len), std::vector<double>(y, y + len), basis);
    const auto predictions = fit.predict(std::vector<double>(query, query + query_len));
    std::copy(predictions.begin(), predictions.end(), out);
    return 0;
  } catch (...) {
    return 1;
  }
}
