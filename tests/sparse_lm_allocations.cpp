/// @file sparse_lm_allocations.cpp
/// @brief Prevent sparse spline designs, model fits and predictions from allocating dense grids.
#include <datamunge/datamunge.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
bool tracking = false; ///< Whether to record ordinary allocation sizes.
std::size_t largest = 0; ///< Largest allocation during the measured operation.
}

/// @brief Record standard vector allocations during the sparse operation.
/// @param size Allocation size in bytes.
/// @return Owned allocation.
/// @throws std::bad_alloc On allocation failure.
void* operator new(std::size_t size) {
  if (tracking) largest = std::max(largest, size);
  if (void* result = std::malloc(size ? size : 1)) return result;
  throw std::bad_alloc();
}
/// @brief Release an ordinary allocation.
/// @param value Previously allocated memory.
void operator delete(void* value) noexcept { std::free(value); }
/// @brief Release an ordinary sized allocation.
/// @param value Previously allocated memory.
/// @param size Unused allocation size.
void operator delete(void* value, std::size_t size) noexcept { (void)size; std::free(value); }

/// @brief Check sparse allocation budgets and fitted coefficients on a large categorical design.
/// @return Zero if both sparse budgets and predictions hold; one otherwise.
int main() {
  namespace dm = datamunge;
  const auto points = dm::seql(0, 1, 20000);
  const auto basis = dm::bs(0, 1, dm::seql(0, 1, 1001));
  tracking = true;
  const auto evaluated = basis.speval(points);
  tracking = false;
  // A dense evaluation would require about 160 MB; sparse growth buffers fit below 2 MB.
  if (largest >= 2 * 1024 * 1024 || evaluated.nnz() > 4 * points.size()) return 1;
  std::printf("Sparse spline evaluation: largest allocation %zu bytes\n", largest);

  dm::linalg::SparseCSR<double> storage;
  storage.nrows = 20000; storage.ncols = 31;
  storage.row_ptr.push_back(0);
  std::vector<double> response;
  for (std::size_t i = 0; i < storage.nrows; ++i) {
    const auto group = i % 32;
    if (group) { storage.col_idx.push_back(group - 1); storage.values.push_back(1); }
    storage.row_ptr.push_back(storage.values.size());
    response.push_back(2 + 0.1 * group + 0.01 * (static_cast<int>(i % 3) - 1));
  }
  const dm::SpMat<> x(std::move(storage));
  largest = 0; tracking = true;
  const dm::LM fit(response, x);
  const auto predicted = fit.predict(x);
  tracking = false;
  // A dense intercept-expanded design would exceed 5 MB. Only sparse arrays,
  // row-sized vectors and 32-by-32 inference matrices belong in this path.
  if (largest >= 1024 * 1024 || predicted.size() != response.size() ||
      std::abs(predicted[0] - 2) > 0.001) return 1;
  std::printf("Sparse LM fit/prediction: largest allocation %zu bytes\n", largest);
}
