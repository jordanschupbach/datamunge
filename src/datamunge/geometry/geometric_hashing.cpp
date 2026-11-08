/// @file geometric_hashing.cpp
/// @brief Shared legacy-ordered hashing kernels, checked arithmetic and actual primitive capture.
#include <cmath>
#include <datamunge/geometry/geometric_hashing.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::geometry {
/// @brief Private domain checks and the single build/recognition implementation.
namespace {
/// @brief Validate one finite planar point without mutation.
/// @param p Point to inspect.
/// @throws std::invalid_argument A coordinate is nonfinite.
void point_input(const Point2D& p) {
  if (!std::isfinite(p.x) || !std::isfinite(p.y))
    throw std::invalid_argument("geometric hashing: finite points required");
}
/// @brief Validate a complete vector before any kernel/history work.
/// @param points Ordered input vector.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @throws std::length_error Point count cannot fit legacy int ids.
void points_input(const std::vector<Point2D>& points) {
  if (points.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("geometric hashing: point id capacity");
  for (const auto& p : points)
    point_input(p);
}
/// @brief Validate the positive finite quantization width, including empty calls.
/// @param quant Invariant-coordinate grid width.
/// @throws std::invalid_argument Nonpositive or nonfinite width.
void quant_input(double quant) {
  if (!std::isfinite(quant) || quant <= 0)
    throw std::invalid_argument("geometric hashing: positive finite quant required");
}
/// @brief Reject a nonfinite intermediate before it reaches a cast or comparison.
/// @param value Required finite intermediate.
/// @throws std::overflow_error The intermediate is nonfinite.
void finite_work(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("geometric hashing: nonfinite intermediate");
}
/// @brief Increment one selected actual-work counter without wrapping.
/// @param value Counter in the owned current result.
/// @throws std::length_error size_t count would overflow.
void increment(std::size_t& value) {
  if (value == std::numeric_limits<std::size_t>::max())
    throw std::length_error("geometric hashing: work capacity");
  ++value;
}
/// @brief Execute the actual legacy ordered-double frame recurrence once.
/// @param p Already finite query.
/// @param b0 Already finite first basis point.
/// @param b1 Already finite second basis point.
/// @return Actual coordinates, squared length and cutoff validity.
/// @throws std::overflow_error A required intermediate is nonfinite.
GeoHashFrame frame(const Point2D& p, const Point2D& b0, const Point2D& b1) {
  const double dx = b1.x - b0.x, dy = b1.y - b0.y;
  const double L2 = dx * dx + dy * dy;
  finite_work(L2);
  GeoHashFrame out;
  out.length_squared = L2;
  if (L2 < 1e-18)
    return out;
  const double ax = p.x - b0.x, ay = p.y - b0.y;
  finite_work(ax);
  finite_work(ay);
  const double nu = ax * dx + ay * dy, nv = ay * dx - ax * dy;
  finite_work(nu);
  finite_work(nv);
  out.u = nu / L2;
  out.v = nv / L2;
  finite_work(out.u);
  finite_work(out.v);
  out.valid = true;
  return out;
}
/// @brief Quantize already validated coordinates/width with safe floor casts.
/// @param u Finite invariant u.
/// @param v Finite invariant v.
/// @param quant Positive finite bin width.
/// @return Actual floor cells and packed key.
/// @throws std::overflow_error A floor cell is nonfinite or outside int32.
GeoHashBin bin(double u, double v, double quant) {
  const double fu = std::floor(u / quant), fv = std::floor(v / quant);
  finite_work(fu);
  finite_work(fv);
  constexpr auto low  = std::numeric_limits<std::int32_t>::min(),
                 high = std::numeric_limits<std::int32_t>::max();
  if (fu < low || fu > high || fv < low || fv > high)
    throw std::overflow_error("geometric hashing: signed bin capacity");
  GeoHashBin out;
  out.u   = static_cast<std::int32_t>(fu);
  out.v   = static_cast<std::int32_t>(fv);
  out.key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(out.u)) << 32) |
            static_cast<std::uint32_t>(out.v);
  return out;
}
/// @brief Append one whole actual primitive event under an online optional quota.
/// @param events Null for ordinary/run calls, otherwise owned current history.
/// @param cap Positive event quota, or zero to disable only optional quota.
/// @param event Copied primitive event from the just-executed kernel operation.
/// @throws std::length_error Optional quota or vector capacity exhausted.
void record(std::vector<GeoHashEvent>* events, std::size_t cap, const GeoHashEvent& event) {
  if (!events)
    return;
  if ((cap && events->size() >= cap) || events->size() == events->max_size())
    throw std::length_error("geometric hashing: complete event quota/capacity");
  events->push_back(event);
}
/// @brief Form a copied actual frame record without redoing numerical work.
/// @param kind Model or scene frame operation.
/// @param i First basis id.
/// @param j Second basis id.
/// @param k Other point id.
/// @param f Actual frame output.
/// @return Owned event with only relevant primitive slots populated.
GeoHashEvent frame_event(GeoHashEventKind kind, int i, int j, int k, const GeoHashFrame& f) {
  GeoHashEvent e;
  e.kind             = kind;
  e.first            = i;
  e.second           = j;
  e.point            = k;
  e.accepted         = f.valid;
  e.u                = f.u;
  e.v                = f.v;
  e.squared_distance = f.length_squared;
  return e;
}
/// @brief Build the legacy table with optional capture from the executing loop.
/// @param model Completely validated model.
/// @param quant Positive finite grid width.
/// @param events Optional owned primitive history.
/// @param cap Online optional quota.
/// @return Owned ordinary table and actual work.
/// @throws std::overflow_error Required arithmetic/bin is unrepresentable.
/// @throws std::length_error Actual count, container or history capacity exceeded.
GeoHashBuildRun build(
    const std::vector<Point2D>& model,
    double                      quant,
    std::vector<GeoHashEvent>*  events,
    std::size_t                 cap) {
  GeoHashBuildRun out;
  out.table.quant = quant;
  const int n     = static_cast<int>(model.size());
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;
      for (int k = 0; k < n; ++k) {
        if (k == i || k == j)
          continue;
        const auto f = frame(model[k], model[i], model[j]);
        increment(out.work.frame_attempts);
        record(events, cap, frame_event(GeoHashEventKind::build_frame, i, j, k, f));
        if (!f.valid) {
          increment(out.work.degenerate_frames);
          continue;
        }
        const auto key = bin(f.u, f.v, quant).key;
        out.table.bins[key].emplace_back(i, j);
        increment(out.work.insertions);
        GeoHashEvent e;
        e.kind   = GeoHashEventKind::insert;
        e.first  = i;
        e.second = j;
        e.point  = k;
        e.key    = key;
        e.count  = out.table.bins[key].size();
        record(events, cap, e);
      }
    }
  return out;
}
/// @brief Validate the complete recognition domain before kernel or capture work.
/// @param table Public table, expected to come from the supplied ordered model.
/// @param model Original ordered model.
/// @param scene Ordered scene.
/// @param tol Finite nonnegative distance tolerance with finite square.
/// @param min_votes Positive minimum votes.
/// @throws std::invalid_argument Invalid numeric domain, ids or identical basis ids.
/// @throws std::length_error Point ids cannot fit int.
void recognition_input(
    const GeoHashTable&         table,
    const std::vector<Point2D>& model,
    const std::vector<Point2D>& scene,
    double                      tol,
    int                         min_votes) {
  quant_input(table.quant);
  points_input(model);
  points_input(scene);
  if (!std::isfinite(tol) || tol < 0 || !std::isfinite(tol * tol) || min_votes < 1)
    throw std::invalid_argument(
        "geometric hashing: finite nonnegative squared tolerance and positive votes required");
  for (const auto& [key, entries] : table.bins)
    for (const auto& [i, j] : entries) {
      if (i < 0 || j < 0 || i == j || static_cast<std::size_t>(i) >= model.size() ||
          static_cast<std::size_t>(j) >= model.size())
        throw std::invalid_argument("geometric hashing: invalid table model basis");
    }
}
/// @brief Run the exact legacy vote-selection and first-hit verification order.
/// @param table Completely validated table ids/width, from the supplied model.
/// @param model Completely validated finite model.
/// @param scene Completely validated finite scene.
/// @param tol Validated finite tolerance with finite square.
/// @param min_votes Positive threshold.
/// @param events Optional owned actual history.
/// @param cap Online optional event quota.
/// @return Owned candidate and selected actual work.
/// @throws std::overflow_error Required arithmetic/bin is unrepresentable.
/// @throws std::length_error Vote/work/container/history capacity exhausted.
GeoHashRun recognition(
    const GeoHashTable&         table,
    const std::vector<Point2D>& model,
    const std::vector<Point2D>& scene,
    double                      tol,
    int                         min_votes,
    std::vector<GeoHashEvent>*  events,
    std::size_t                 cap) {
  GeoHashRun out;
  auto&      best = out.match;
  const int  m    = static_cast<int>(scene.size());
  for (int a = 0; a < m; ++a)
    for (int b = 0; b < m; ++b) {
      if (a == b)
        continue;
      std::unordered_map<std::uint64_t, int> tally;
      for (int k = 0; k < m; ++k) {
        if (k == a || k == b)
          continue;
        const auto f = frame(scene[k], scene[a], scene[b]);
        increment(out.work.frame_attempts);
        record(events, cap, frame_event(GeoHashEventKind::scene_frame, a, b, k, f));
        if (!f.valid) {
          increment(out.work.degenerate_frames);
          continue;
        }
        const auto bin_key = bin(f.u, f.v, table.quant).key;
        auto       it      = table.bins.find(bin_key);
        increment(out.work.bin_lookups);
        GeoHashEvent e;
        e.kind     = GeoHashEventKind::lookup;
        e.first    = a;
        e.second   = b;
        e.point    = k;
        e.key      = bin_key;
        e.accepted = it != table.bins.end();
        record(events, cap, e);
        if (it == table.bins.end())
          continue;
        increment(out.work.bin_hits);
        for (const auto& mb : it->second) {
          const auto key =
              (static_cast<std::uint64_t>(static_cast<std::uint32_t>(mb.first)) << 32) |
              static_cast<std::uint32_t>(mb.second);
          auto& votes = tally[key];
          if (votes == std::numeric_limits<int>::max())
            throw std::length_error("geometric hashing: vote count capacity");
          ++votes;
          increment(out.work.vote_updates);
          GeoHashEvent v;
          v.kind    = GeoHashEventKind::vote;
          v.first   = a;
          v.second  = b;
          v.point   = k;
          v.model_i = mb.first;
          v.model_j = mb.second;
          v.key     = key;
          v.count   = static_cast<std::size_t>(votes);
          record(events, cap, v);
        }
      }
      for (const auto& [key, votes] : tally) {
        increment(out.work.candidate_comparisons);
        GeoHashEvent e;
        e.kind     = GeoHashEventKind::candidate;
        e.first    = a;
        e.second   = b;
        e.key      = key;
        e.count    = static_cast<std::size_t>(votes);
        e.accepted = votes > best.votes;
        e.model_i  = static_cast<int>(key >> 32);
        e.model_j  = static_cast<std::int32_t>(static_cast<std::uint32_t>(key));
        record(events, cap, e);
        if (votes <= best.votes)
          continue;
        best.votes   = votes;
        best.scene_a = a;
        best.scene_b = b;
        best.model_i = e.model_i;
        best.model_j = e.model_j;
        increment(out.work.best_updates);
      }
    }
  if (best.votes < min_votes)
    return out;
  const Point2D& mi  = model[best.model_i];
  const Point2D& mj  = model[best.model_j];
  const Point2D& sa  = scene[best.scene_a];
  const Point2D& sb  = scene[best.scene_b];
  const double   mdx = mj.x - mi.x, mdy = mj.y - mi.y, sdx = sb.x - sa.x, sdy = sb.y - sa.y;
  const double   mL2 = mdx * mdx + mdy * mdy;
  finite_work(mL2);
  if (mL2 < 1e-18)
    return out;
  const double nc = sdx * mdx + sdy * mdy, ns = sdy * mdx - sdx * mdy;
  finite_work(nc);
  finite_work(ns);
  const double cs = nc / mL2, sn = ns / mL2;
  finite_work(cs);
  finite_work(sn);
  const double scale2 = cs * cs + sn * sn;
  finite_work(scale2);
  best.scale       = std::sqrt(scale2);
  best.angle       = std::atan2(sn, cs);
  best.translation = {sa.x - (cs * mi.x - sn * mi.y), sa.y - (sn * mi.x + cs * mi.y)};
  finite_work(best.translation.x);
  finite_work(best.translation.y);
  best.found = true;
  increment(out.work.recoveries);
  GeoHashEvent e;
  e.kind        = GeoHashEventKind::recover;
  e.first       = best.scene_a;
  e.second      = best.scene_b;
  e.model_i     = best.model_i;
  e.model_j     = best.model_j;
  e.count       = static_cast<std::size_t>(best.votes);
  e.accepted    = true;
  e.cs          = cs;
  e.sn          = sn;
  e.scale       = best.scale;
  e.angle       = best.angle;
  e.translation = best.translation;
  record(events, cap, e);
  for (std::size_t i = 0; i < model.size(); ++i) {
    const auto& p = model[i];
    increment(out.work.model_points_checked);
    const double tx = cs * p.x - sn * p.y + best.translation.x,
                 ty = sn * p.x + cs * p.y + best.translation.y;
    finite_work(tx);
    finite_work(ty);
    for (int j = 0; j < m; ++j) {
      const auto&  q  = scene[j];
      const double dx = tx - q.x, dy = ty - q.y;
      const double d2 = dx * dx + dy * dy;
      finite_work(d2);
      increment(out.work.distance_checks);
      const bool   accepted = d2 <= tol * tol;
      GeoHashEvent d;
      d.kind             = GeoHashEventKind::distance;
      d.first            = static_cast<int>(i);
      d.second           = j;
      d.x                = tx;
      d.y                = ty;
      d.squared_distance = d2;
      d.accepted         = accepted;
      record(events, cap, d);
      if (accepted) {
        if (best.inliers == std::numeric_limits<int>::max())
          throw std::length_error("geometric hashing: inlier capacity");
        ++best.inliers;
        break;
      }
    }
  }
  return out;
}
} // namespace

GeoHashTable build_geometric_hash(const std::vector<Point2D>& model, double quant) {
  quant_input(quant);
  points_input(model);
  return build(model, quant, nullptr, 0).table;
}
GeoHashBuildRun build_geometric_hash_run(const std::vector<Point2D>& model, double quant) {
  quant_input(quant);
  points_input(model);
  return build(model, quant, nullptr, 0);
}
GeoHashBuildTrace build_geometric_hash_trace(
    const std::vector<Point2D>& model,
    double                      quant,
    std::size_t                 max_records) {
  quant_input(quant);
  points_input(model);
  GeoHashBuildTrace out;
  out.run = build(model, quant, &out.events, max_records);
  return out;
}
GeoMatch recognize(
    const GeoHashTable&         table,
    const std::vector<Point2D>& model,
    const std::vector<Point2D>& scene,
    double                      tol,
    int                         min_votes) {
  recognition_input(table, model, scene, tol, min_votes);
  return recognition(table, model, scene, tol, min_votes, nullptr, 0).match;
}
GeoHashRun recognize_run(
    const GeoHashTable&         table,
    const std::vector<Point2D>& model,
    const std::vector<Point2D>& scene,
    double                      tol,
    int                         min_votes) {
  recognition_input(table, model, scene, tol, min_votes);
  return recognition(table, model, scene, tol, min_votes, nullptr, 0);
}
GeoHashTrace recognize_trace(
    const GeoHashTable&         table,
    const std::vector<Point2D>& model,
    const std::vector<Point2D>& scene,
    double                      tol,
    int                         min_votes,
    std::size_t                 max_records) {
  recognition_input(table, model, scene, tol, min_votes);
  GeoHashTrace out;
  out.run = recognition(table, model, scene, tol, min_votes, &out.events, max_records);
  return out;
}
GeoHashFrame geometric_hash_frame(const Point2D& p, const Point2D& b0, const Point2D& b1) {
  point_input(p);
  point_input(b0);
  point_input(b1);
  return frame(p, b0, b1);
}
GeoHashBin geometric_hash_bin(double u, double v, double quant) {
  quant_input(quant);
  if (!std::isfinite(u) || !std::isfinite(v))
    throw std::invalid_argument("geometric hashing: finite invariant coordinates required");
  return bin(u, v, quant);
}
std::vector<Point2D> geometric_transform(
    const std::vector<Point2D>& points,
    double                      scale,
    double                      angle,
    Point2D                     translation) {
  points_input(points);
  point_input(translation);
  if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(angle))
    throw std::invalid_argument(
        "geometric hashing: positive finite scale and finite angle required");
  const double c = scale * std::cos(angle), d = scale * std::sin(angle);
  finite_work(c);
  finite_work(d);
  std::vector<Point2D> out;
  out.reserve(points.size());
  for (const auto& p : points) {
    const double x = c * p.x - d * p.y + translation.x, y = d * p.x + c * p.y + translation.y;
    finite_work(x);
    finite_work(y);
    out.push_back({x, y});
  }
  return out;
}
namespace detail {
bool gh_frame_coord(const Point2D& p, const Point2D& b0, const Point2D& b1, double& u, double& v) {
  const auto f = geometric_hash_frame(p, b0, b1);
  if (!f.valid)
    return false;
  u = f.u;
  v = f.v;
  return true;
}
std::uint64_t gh_bin_key(double u, double v, double quant) {
  return geometric_hash_bin(u, v, quant).key;
}
} // namespace detail
} // namespace datamunge::geometry
