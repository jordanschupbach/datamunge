#pragma once
/// @file summed_area_table.hpp
/// @brief Checked finite integral-image build states and actual four-lookup queries.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the SummedAreaOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SummedAreaOptions r{};
/// (void)r;
/// @endcode
/// @see SummedAreaTable()
struct SummedAreaOptions {
  std::size_t max_cells{1000000};   ///< Complete zero-bordered table cell budget; zero disables.
  std::size_t max_records{1000000}; ///< Actual captured prefix-cell records; zero disables.
  bool        capture{};            ///< Retain actual build transitions; defaultfalse.
};
/// @brief Own the SummedAreaStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SummedAreaStep r{};
/// (void)r;
/// @endcode
/// @see SummedAreaTable()
struct SummedAreaStep {
  std::size_t row{};      ///< Actual zero-based input row.
  std::size_t col{};      ///< Actual zero-based input column.
  double      input{};    ///< Actual consumed finite input value.
  double      above{};    ///< Actual prefix value immediately above output.
  double      left{};     ///< Actual prefix value immediately left of output.
  double      diagonal{}; ///< Actual above-left prefix value.
  double      value{};    ///< Actual computed input+above+left-diagonal in that double order.
};
/// @brief Own the SummedAreaQuery contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SummedAreaQuery r{};
/// (void)r;
/// @endcode
/// @see SummedAreaTable()
struct SummedAreaQuery {
  double      bottom_right{}; ///< Actual prefix(r1+1,c1+1).
  double      above{};        ///< Actual prefix(r0,c1+1).
  double      left{};         ///< Actual prefix(r1+1,c0).
  double      diagonal{};     ///< Actual prefix(r0,c0).
  double      value{};        ///< Actual inclusion-exclusion sum.
  std::size_t lookups{4};     ///< Exactly four actual prefix reads, including zero-border reads.
};
/// @brief Own a checked integral image for constant-time rectangle sums.
/// Complete finite rectangular grid validation and an optional actual prefix history. Object owns table storage. Const queries are independently readable, with no shared mutable state. Build is linear; inclusive queries use four reads.
/// @par Example
/// @code
/// dm::algorithms::SummedAreaTable s({{1,2}});
/// std::print("{}\n",s.rect_sum(0,0,0,1)); // 3
/// @endcode
/// @see query()
class SummedAreaTable {
 public:
  /// @brief Build an owned checked zero-bordered integral image.
  /// Validates every finite rectangular row before allocation. Empty rows/columns are retained as empty geometry with a zero border; no nonempty rectangle can be queried. Computes input+above+left-diagonal in double, rejecting nonfinite intermediates. O(rows*cols) build time/storage, optional actual per-cell records. Subsequent reads are const and have no shared mutation. Cancellation/rounding may differ from direct summation; only exact small integer examples promise exact equality.
  /// @param grid Complete borrowed finite rectangular row-major input.
  /// @param options Zero-border cell/capture budgets; defaults1million and capturefalse.
  /// @throws std::invalid_argument Ragged or nonfinite grid.
  /// @throws std::length_error Complete table or actual record capacity/budget exceeded.
  /// @throws std::overflow_error Actual prefix arithmetic is nonfinite.
  /// @par Example
  /// @code
  /// dm::algorithms::SummedAreaTable s({{1,2},{3,4}});
  /// std::print("{}\n",s.rect_sum(0,0,1,1)); // 10
  /// @endcode
  /// @par References
  /// - Crow, Summed-area tables for texture mapping, SIGGRAPH1984,207-212.
  /// @see query()
  explicit SummedAreaTable(
      const std::vector<std::vector<double>>& grid,
      const SummedAreaOptions&                options = {});
  /// @brief Return the actual four prefix values and inclusion-exclusion sum.
  /// Requires nonempty in-range ordered inclusive rectangle. Reads exactly four owned prefix cells, with no mutation. Double subtraction/addition can lose precision and actual nonfinite arithmetic throws.
  /// @param r0 Inclusive first input row.
  /// @param c0 Inclusive first input column.
  /// @param r1 Inclusive last input row, at least r0 and below rows().
  /// @param c1 Inclusive last input column, at least c0 and below cols().
  /// @return Actual owned query diagnostics.
  /// @throws std::out_of_range Reversed or out-of-grid inclusive rectangle, including empty grids.
  /// @throws std::overflow_error Actual query arithmetic nonfinite.
  /// @par Example
  /// @code
  /// dm::algorithms::SummedAreaTable s({{1,2},{3,4}});
  /// std::print("{}\n",s.query(0,0,1,1).lookups); // 4
  /// @endcode
  /// @see SummedAreaTable()
  SummedAreaQuery query(std::size_t r0, std::size_t c0, std::size_t r1, std::size_t c1) const;
  /// @brief Sum an inclusive checked rectangle in constant time.
  /// Requires nonempty in-range ordered inclusive rectangle. Reads exactly four owned prefix cells, with no mutation. Double subtraction/addition can lose precision and actual nonfinite arithmetic throws.
  /// @param r0 Inclusive first input row.
  /// @param c0 Inclusive first input column.
  /// @param r1 Inclusive last input row, at least r0 and below rows().
  /// @param c1 Inclusive last input column, at least c0 and below cols().
  /// @return Actual finite rectangle sum.
  /// @throws std::out_of_range Reversed or out-of-grid inclusive rectangle, including empty grids.
  /// @throws std::overflow_error Actual query arithmetic nonfinite.
  /// @par Example
  /// @code
  /// dm::algorithms::SummedAreaTable s({{1,2},{3,4}});
  /// std::print("{}\n",s.rect_sum(0,0,1,1)); // 10
  /// @endcode
  /// @see SummedAreaTable()
  double rect_sum(std::size_t r0, std::size_t c0, std::size_t r1, std::size_t c1) const;
  /// @brief Average an inclusive checked rectangle in constant time.
  /// Requires nonempty in-range ordered inclusive rectangle. Reads exactly four owned prefix cells, with no mutation. Double subtraction/addition can lose precision and actual nonfinite arithmetic throws.
  /// @param r0 Inclusive first input row.
  /// @param c0 Inclusive first input column.
  /// @param r1 Inclusive last input row, at least r0 and below rows().
  /// @param c1 Inclusive last input column, at least c0 and below cols().
  /// @return Actual finite rectangle mean.
  /// @throws std::out_of_range Reversed or out-of-grid inclusive rectangle, including empty grids.
  /// @throws std::overflow_error Actual query arithmetic nonfinite.
  /// @par Example
  /// @code
  /// dm::algorithms::SummedAreaTable s({{1,2},{3,4}});
  /// std::print("{}\n",s.rect_mean(0,0,1,1)); // 2.5
  /// @endcode
  /// @see SummedAreaTable()
  double rect_mean(std::size_t r0, std::size_t c0, std::size_t r1, std::size_t c1) const;
  /// @brief Borrow or inspect the input row count.
  /// Constant time, const read. Borrowed containers remain valid for this object lifetime and are not mutable through the accessor.
  /// @return Actual input row count.
  /// @par Example
  /// @code
  /// dm::algorithms::SummedAreaTable s({{1,2}});
  /// std::print("{}\n",s.rows()); // 1
  /// @endcode
  /// @see query()
  std::size_t rows() const noexcept;
  /// @brief Borrow or inspect the input column count.
  /// Constant time, const read. Borrowed containers remain valid for this object lifetime and are not mutable through the accessor.
  /// @return Actual input column count.
  /// @par Example
  /// @code
  /// dm::algorithms::SummedAreaTable s({{1,2}});
  /// std::print("{}\n",s.cols()); // 2
  /// @endcode
  /// @see query()
  std::size_t cols() const noexcept;
  /// @brief Borrow or inspect the complete zero-bordered producing table.
  /// Constant time, const read. Borrowed containers remain valid for this object lifetime and are not mutable through the accessor.
  /// @return Actual complete zero-bordered producing table.
  /// @par Example
  /// @code
  /// dm::algorithms::SummedAreaTable s({{1,2}});
  /// std::print("{}\n",s.prefix().size()); // 2
  /// @endcode
  /// @see query()
  const std::vector<std::vector<double>>& prefix() const noexcept;
  /// @brief Borrow or inspect the actual captured build transitions.
  /// Constant time, const read. Borrowed containers remain valid for this object lifetime and are not mutable through the accessor.
  /// @return Actual actual captured build transitions.
  /// @par Example
  /// @code
  /// dm::algorithms::SummedAreaTable s({{1,2}});
  /// std::print("{}\n",s.steps().size()); // 0
  /// @endcode
  /// @see query()
  const std::vector<SummedAreaStep>& steps() const noexcept;

 private:
  std::size_t                      rows_{}; ///< Complete input rows.
  std::size_t                      cols_{}; ///< Complete input columns.
  std::vector<std::vector<double>> s_;      ///< Owned(rows+1) by(cols+1) actual prefix table.
  std::vector<SummedAreaStep>      steps_;  ///< Optional owned actual build states.
};
/// @brief Generate the retained6 by6 row-major1..36 fixture.
/// Owned integer-valued double grid; no shared state.
/// @return Original complete6 by6 grid.
/// @par Example
/// @code
/// auto g=dm::algorithms::summed_area_example_grid();
/// std::print("{}\n",g.back().back()); // 36
/// @endcode
/// @see SummedAreaTable()
std::vector<std::vector<double>> summed_area_example_grid();
} // namespace datamunge::algorithms
