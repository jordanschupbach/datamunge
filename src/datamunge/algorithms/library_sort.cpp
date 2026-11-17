/// @file library_sort.cpp
/// @brief Checked shared historical fixed-capacity gapped insertion with actual predecessor, gap and move evidence.
#include <datamunge/algorithms/library_sort.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete shelf geometry and same-execution producer.
namespace library_detail {
/// @brief Add a selected count without size_t wrap.
/// @param count Mutable selected count.
/// @param amount Actual increment, default1.
/// @throws std::length_error On count overflow.
void add(std::size_t& count, std::size_t amount = 1) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("library sort count overflow");
  count += amount;
}
/// @brief Validate complete signed predecessor/shelf/container geometry.
/// @param n Complete original input length.
/// @return Actual fixed shelf capacity,0 for trivial-copy path.
/// @throws std::length_error On full geometry limits.
std::size_t geometry(std::size_t n) {
  if (n <= 1)
    return 0;
  const auto max = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (n > (max - 2) / 2)
    throw std::length_error("library sort signed shelf geometry");
  const auto cap = 2 * n + 2;
  if (cap > std::vector<int>{}.max_size() || cap > std::vector<char>{}.max_size())
    throw std::length_error("library sort shelf container");
  return cap;
}
/// @brief Produce the historical fixed shelf once with optional admitted state/decision capture.
/// @param input Original complete integers.
/// @param capacity Validated fixed capacity,0 on trivial path.
/// @param trace Optional completely admitted capture.
/// @return Owned actual sorted values and selected work.
/// @throws std::length_error On actual selected count overflow.
/// @throws std::bad_alloc On runtime/capture/output allocation failure.
LibrarySortAnalysis
run(const std::vector<int>& input, std::size_t capacity, LibrarySortTrace* trace) {
  LibrarySortAnalysis out;
  if (input.size() <= 1) {
    out.values = input;
    return out;
  }
  out.work.capacity = capacity;
  std::vector<int>  slots(capacity);
  std::vector<char> occupied(capacity);
  if (trace)
    trace->shelves.push_back({0, slots, occupied});
  const auto cap = static_cast<std::ptrdiff_t>(capacity);
  for (std::size_t e = 0; e < input.size(); ++e) {
    const int            x = input[e];
    LibrarySortInsertion event;
    event.input_index = e;
    event.value       = x;
    if (e == 0) {
      event.target = event.gap = capacity / 2;
      event.kind               = LibrarySortPlacement::Seed;
    } else {
      std::ptrdiff_t lo = 0, hi = cap - 1, answer = -1;
      while (lo <= hi) {
        add(out.work.search_iterations);
        const auto mid = lo + (hi - lo) / 2;
        auto       p   = mid;
        while (p >= lo) {
          add(out.work.occupancy_reads);
          if (occupied[p])
            break;
          --p;
        }
        if (p < lo) {
          lo = mid + 1;
          continue;
        }
        add(out.work.comparisons);
        if (slots[p] <= x) {
          answer = p;
          lo     = p + 1;
        } else
          hi = p - 1;
      }
      event.predecessor = answer;
      const auto start  = answer + 1;
      bool       direct = false;
      if (start < cap) {
        add(out.work.occupancy_reads);
        direct = !occupied[start];
      }
      if (direct) {
        event.target = event.gap = static_cast<std::size_t>(start);
        event.kind               = LibrarySortPlacement::Direct;
      } else {
        auto j = start;
        while (j < cap) {
          add(out.work.occupancy_reads);
          if (!occupied[j])
            break;
          ++j;
        }
        if (j < cap) {
          event.kind   = LibrarySortPlacement::ShiftRight;
          event.target = start;
          event.gap    = j;
          for (auto k = j; k > start; --k) {
            slots[k]    = slots[k - 1];
            occupied[k] = occupied[k - 1];
            add(out.work.shift_writes);
            ++event.moved;
          }
        } else {
          // Unreachable: after e placements, largest slot <=n+1+e<capacity-1.
          throw std::logic_error("library sort guaranteed right-gap invariant");
        }
      }
    }
    slots[event.target]    = x;
    occupied[event.target] = 1;
    add(out.work.placements);
    if (trace) {
      trace->insertions.push_back(event);
      trace->shelves.push_back({e + 1, slots, occupied});
    }
  }
  out.values.reserve(input.size());
  for (std::size_t i = 0; i < capacity; ++i) {
    add(out.work.output_slots);
    if (occupied[i])
      out.values.push_back(slots[i]);
  }
  return out;
}
} // namespace library_detail
std::vector<int> library_sort(std::vector<int> input) {
  return library_sort_analyze(input).values;
}
LibrarySortAnalysis library_sort_analyze(const std::vector<int>& input) {
  return library_detail::run(input, library_detail::geometry(input.size()), nullptr);
}
LibrarySortTrace library_sort_trace(const std::vector<int>& input, std::size_t max_entries) {
  const auto n = input.size(), cap = library_detail::geometry(n),
             max      = std::numeric_limits<std::size_t>::max();
  std::size_t capture = n;
  if (n > 1) {
    library_detail::add(capture, n);
    if (cap > (max - 1) / 2)
      throw std::length_error("library sort capture geometry");
    const auto per = 1 + 2 * cap;
    if (n + 1 > max / per)
      throw std::length_error("library sort complete capture geometry");
    library_detail::add(capture, (n + 1) * per);
  }
  if (max_entries && capture > max_entries)
    throw std::length_error("library sort complete capture quota");
  LibrarySortTrace out;
  out.input            = input;
  out.captured_entries = capture;
  if (n > 1) {
    out.insertions.reserve(n);
    out.shelves.reserve(n + 1);
  }
  out.analysis = library_detail::run(input, cap, &out);
  return out;
}
std::vector<int> library_sort_demo_input(std::size_t count, std::uint32_t seed) {
  if (count > std::vector<int>{}.max_size())
    throw std::length_error("library sort fixture geometry");
  std::vector<int> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    seed = seed * 1103515245u + 12345u;
    out.push_back(static_cast<int>((seed >> 16) % 100));
  }
  return out;
}
} // namespace datamunge::algorithms
