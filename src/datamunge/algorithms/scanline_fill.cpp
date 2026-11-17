/// @file scanline_fill.cpp
/// @brief Single checked rasterizer with optional actual row capture.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/scanline_fill.hpp>
#include <utility>
/// @brief Canonical checked native algorithm producers and owned outputs.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Check an incremental size admission without overflow.
/// @param used Existing units.
/// @param count Next units.
/// @param cap Zero or policy limit.
void admit(std::size_t used, std::size_t count, std::size_t cap) {
  if (count > std::numeric_limits<std::size_t>::max() - used ||
      (cap && (used > cap || count > cap - used)))
    throw std::length_error("scanline: capacity/policy");
}
/// @brief Execute actual crossing, pairing, union and emission work once.
/// @param p Borrowed finite polygon.
/// @param o Independent bounds.
/// @param capture Retain actual rows.
/// @return Owned actual pixels/work/history.
ScanlineRun raster(const std::vector<Point2D>& p, const ScanlineOptions& o, bool capture) {
  if (o.min_x > o.max_x || o.min_y > o.max_y)
    throw std::invalid_argument("scanline: reversed clip");
  admit(0, p.size(), o.max_vertices);
  long double low = 0, high = 0;
  bool        first = true;
  for (const auto& v : p) {
    if (!std::isfinite(v.x) || !std::isfinite(v.y))
      throw std::invalid_argument("scanline: finite vertices required");
    if (first) {
      low = high = v.y;
      first      = false;
    } else {
      low  = std::min(low, static_cast<long double>(v.y));
      high = std::max(high, static_cast<long double>(v.y));
    }
  }
  ScanlineRun out;
  if (p.size() < 3)
    return out;
  const long double begin = std::max(std::ceil(low - 0.5L), static_cast<long double>(o.min_y));
  const long double end   = std::min(std::ceil(high - 0.5L), static_cast<long double>(o.max_y) + 1);
  if (begin >= end)
    return out;
  const auto row_count = static_cast<std::uint64_t>(end - begin);
  if (row_count > std::numeric_limits<std::size_t>::max())
    throw std::length_error("scanline: row capacity");
  admit(0, static_cast<std::size_t>(row_count), o.max_scanlines);
  std::size_t retained = 0;
  for (std::int64_t y = static_cast<std::int64_t>(begin); y < static_cast<std::int64_t>(end); ++y) {
    ScanlineRow row;
    row.y      = static_cast<int>(y);
    row.center = static_cast<long double>(y) + 0.5L;
    ++out.scanlines;
    for (std::size_t i = 0; i < p.size(); ++i) {
      admit(out.edge_tests, 1, o.max_edge_tests);
      ++out.edge_tests;
      const auto&       a  = p[i];
      const auto&       b  = p[(i + 1) % p.size()];
      const long double ya = a.y, yb = b.y, ys = row.center;
      if ((ya <= ys && ys < yb) || (yb <= ys && ys < ya)) {
        long double numerator = ys - ya, denominator = yb - ya, t;
        if (std::isfinite(numerator) && std::isfinite(denominator))
          t = numerator / denominator;
        else {
          const long double scale = std::max({std::abs(ya), std::abs(yb), std::abs(ys)});
          t                       = (ys / scale - ya / scale) / (yb / scale - ya / scale);
        }
        const long double weighted=static_cast<long double>(a.x)*(yb-ys)+static_cast<long double>(b.x)*(ys-ya);
        const auto x=std::isfinite(weighted)&&std::isfinite(denominator)?weighted/denominator:std::lerp(static_cast<long double>(a.x),static_cast<long double>(b.x),t);
        row.crossings.push_back(x);
        ++out.intersections;
      }
    }
    std::sort(row.crossings.begin(), row.crossings.end());
    if (row.crossings.size() % 2)
      throw std::logic_error("scanline: crossing parity");
    for (std::size_t k = 0; k < row.crossings.size(); k += 2) {
      const auto a =
          std::max(std::ceil(row.crossings[k] - 0.5L), static_cast<long double>(o.min_x));
      const auto b =
          std::min(std::floor(row.crossings[k + 1] - 0.5L), static_cast<long double>(o.max_x));
      if (a <= b)
        row.paired.push_back({static_cast<int>(a), static_cast<int>(b)});
    }
    for (const auto& span : row.paired) {
      if (row.merged.empty() || static_cast<std::int64_t>(span.first) >
                                    static_cast<std::int64_t>(row.merged.back().last) + 1)
        row.merged.push_back(span);
      else
        row.merged.back().last = std::max(row.merged.back().last, span.last);
    }
    std::size_t count = 0;
    for (const auto& span : row.merged) {
      const auto n =
          static_cast<std::uint64_t>(static_cast<std::int64_t>(span.last) - span.first) + 1;
      if (n > std::numeric_limits<std::size_t>::max())
        throw std::length_error("scanline: span capacity");
      admit(count, static_cast<std::size_t>(n), 0);
      count += static_cast<std::size_t>(n);
    }
    admit(out.pixels.size(), count, o.max_pixels);
    if (capture) {
      admit(retained, 1, o.max_trace_entries);
      ++retained;
      for (auto n : {row.crossings.size(), row.paired.size(), row.merged.size()}) {
        admit(retained, n, o.max_trace_entries);
        retained += n;
      }
    }
    for (const auto& span : row.merged)
      for (std::int64_t x = span.first; x <= static_cast<std::int64_t>(span.last); ++x)
        out.pixels.push_back({static_cast<int>(x), row.y});
    if (capture)
      out.rows.push_back(std::move(row));
  }
  return out;
}
} // namespace
ScanlineRun scanline_fill_run(const std::vector<Point2D>& p, const ScanlineOptions& o) {
  return raster(p, o, false);
}
ScanlineRun scanline_fill_trace(const std::vector<Point2D>& p, const ScanlineOptions& o) {
  return raster(p, o, true);
}
std::vector<Pixel> scanline_fill(const std::vector<Point2D>& p) {
  return scanline_fill_run(p).pixels;
}
} // namespace datamunge::algorithms
