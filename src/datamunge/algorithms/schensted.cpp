/// @file schensted.cpp
/// @brief One checked actual row-bump producer and native paired-tableau renderer.
#include <datamunge/algorithms/schensted.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
/// @brief Canonical row-insertion implementation.
namespace datamunge::algorithms {
/// @brief Private actual row producer.
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Execute actual comparisons, bumps and terminal recording-box appends.
/// @param values Borrowed source integers.
/// @param o Policies.
/// @param capture Retain real row states.
/// @return Owned actual tableaux and selected work.
SchenstedRun solve(const std::vector<int>& values, const SchenstedOptions& o, bool capture) {
  const auto max = std::numeric_limits<std::size_t>::max();
  if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (o.max_values && values.size() > o.max_values))
    throw std::length_error("Schensted: source/label policy");
  SchenstedRun out;
  std::size_t  retained = 0;
  for (std::size_t step = 0; step < values.size(); ++step) {
    int         carried = values[step];
    std::size_t row     = 0;
    while (true) {
      const bool  fresh  = row == out.tableau.p.size();
      const auto  length = fresh ? 0 : out.tableau.p[row].size();
      std::size_t column = 0, comparisons = 0;
      if (!fresh)
        while (column < length) {
          if (out.comparisons == max || (o.max_comparisons && out.comparisons >= o.max_comparisons))
            throw std::length_error("Schensted: comparison policy");
          ++out.comparisons;
          ++comparisons;
          if (out.tableau.p[row][column] > carried)
            break;
          ++column;
        }
      const bool    bump = column < length;
      SchenstedStep event;
      event.input_index = step;
      event.row         = row;
      event.column      = column;
      event.carried     = carried;
      event.bumped      = bump;
      event.comparisons = comparisons;
      if (capture) {
        if (length > (max - 2) / 2)
          throw std::length_error("Schensted: row geometry");
        const auto cost = 2 * length + 1 + !bump;
        if (cost > max - retained ||
            (o.max_trace_entries &&
             (retained > o.max_trace_entries || cost > o.max_trace_entries - retained)))
          throw std::length_error("Schensted: history policy");
        retained += cost;
        if (!fresh)
          event.before = out.tableau.p[row];
      }
      if (fresh) {
        out.tableau.p.emplace_back();
        out.tableau.q.emplace_back();
      }
      if (bump) {
        event.outgoing = out.tableau.p[row][column];
        std::swap(out.tableau.p[row][column], carried);
        ++out.bumps;
      } else {
        out.tableau.p[row].push_back(carried);
        out.tableau.q[row].push_back(static_cast<int>(step + 1));
        ++out.appends;
      }
      if (capture) {
        event.after = out.tableau.p[row];
        out.steps.push_back(std::move(event));
      }
      if (!bump)
        break;
      ++row;
    }
  }
  return out;
}
} // namespace
SchenstedRun schensted_run(const std::vector<int>& values, const SchenstedOptions& o) {
  return solve(values, o, false);
}
SchenstedRun schensted_trace(const std::vector<int>& values, const SchenstedOptions& o) {
  return solve(values, o, true);
}
YoungTableaux rsk_insert(const std::vector<int>& values) {
  return solve(values, {}, false).tableau;
}
graphics::Diagram schensted_diagram(const YoungTableaux& t, std::size_t max_cells) {
  if (t.p.size() != t.q.size())
    throw std::invalid_argument("Schensted diagram: row count");
  const auto  max   = std::numeric_limits<std::size_t>::max();
  std::size_t count = 0, previous = max;
  for (std::size_t row = 0; row < t.p.size(); ++row) {
    const auto size = t.p[row].size();
    if (!size || size != t.q[row].size() || size > previous)
      throw std::invalid_argument("Schensted diagram: partition shape");
    if (size > max / 2 || 2 * size > max - count)
      throw std::length_error("Schensted diagram: cell capacity");
    count += 2 * size;
    if (max_cells && count > max_cells)
      throw std::length_error("Schensted diagram: cell policy");
    previous = size;
  }
  graphics::Diagram d(48, 1100);
  if (t.p.empty()) {
    d.text({0, 0}, "Actual empty P and Q", 24);
    return d;
  }
  graphics::DiagramNodeStyle style;
  style.width         = 1.2;
  style.height        = 1.2;
  style.font_size     = 20;
  const double offset = static_cast<double>(t.p[0].size()) * 1.35 + 3;
  d.text({0, 1.8}, "Actual P", 24).text({offset, 1.8}, "Actual Q", 24);
  for (std::size_t row = 0; row < t.p.size(); ++row)
    for (std::size_t column = 0; column < t.p[row].size(); ++column) {
      const auto   id = std::to_string(row) + "_" + std::to_string(column);
      const double x = column * 1.35, y = -static_cast<double>(row) * 1.35;
      d.node("p" + id, {x, y}, std::to_string(t.p[row][column]), style);
      d.node("q" + id, {x + offset, y}, std::to_string(t.q[row][column]), style);
    }
  return d;
}
} // namespace datamunge::algorithms
