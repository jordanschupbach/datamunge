/// @file terrain_search.cpp
/// @brief A*, beam, beam-stack and Dijkstra searches on 8-connected elevation grids, and energy edges for Bellman-Ford.

#include <datamunge/algorithms/terrain_search.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {

/// @brief Neighbour enumeration and argument checks.
namespace {

/// @brief Row and column offsets of the 8 neighbours.
constexpr int kDr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
/// @brief Column offsets matching kDr.
constexpr int kDc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

/// @brief Calls @p f(neighbour) for each in-grid neighbour of @p cell.
template <typename F>
void for_each_neighbour(const TerrainGrid& g, std::size_t cell, F&& f) {
    const auto r = static_cast<long>(cell / g.cols), c = static_cast<long>(cell % g.cols);
    for (int k = 0; k < 8; ++k) {
        const long rr = r + kDr[k], cc = c + kDc[k];
        if (rr < 0 || cc < 0 || rr >= static_cast<long>(g.rows) || cc >= static_cast<long>(g.cols)) continue;
        f(static_cast<std::size_t>(rr) * g.cols + static_cast<std::size_t>(cc));
    }
}

/// @brief True when @p candidate beats @p current by more than rounding noise. Grid paths of equal
///        real cost (straight-then-diagonal versus diagonal-then-straight) can differ in the last
///        bit, and treating that as an improvement would reopen closed cells.
bool improves(double candidate, double current) { return candidate < current - 1e-12 * (1.0 + std::abs(candidate)); }

/// @brief Fills out.path by following @p parent back from @p end, and its length, ascent and descent.
void trace_path(const TerrainGrid& g, const std::vector<std::size_t>& parent, std::size_t end, TerrainPath& out) {
    const std::size_t n = g.elevation.size();
    out.path.clear();
    out.length = out.ascent = out.descent = 0.0;
    for (std::size_t v = end; v != n; v = parent[v]) out.path.push_back(v);
    std::reverse(out.path.begin(), out.path.end());
    for (std::size_t k = 1; k < out.path.size(); ++k) {
        const std::size_t a = out.path[k - 1], b = out.path[k];
        const bool diagonal = (a / g.cols != b / g.cols) && (a % g.cols != b % g.cols);
        out.length += g.cell_size * (diagonal ? std::sqrt(2.0) : 1.0);
        const double dz = g.elevation[b] - g.elevation[a];
        (dz > 0 ? out.ascent : out.descent) += std::abs(dz);
    }
}

/// @brief Throws std::invalid_argument naming @p what if @p cell is outside the grid.
/// @brief Throws unless the grid is consistent and @p cell is inside it.
void check(const TerrainGrid& g, std::size_t cell, const char* what) {
    if (g.rows == 0 || g.cols == 0 || g.elevation.size() != g.rows * g.cols)
        throw std::invalid_argument("terrain search: elevation must have rows * cols values");
    if (cell >= g.elevation.size()) throw std::invalid_argument(std::string("terrain search: ") + what + " out of range");
}

}  // namespace

TerrainGrid terrain_from_frame(const dstruct::DataFrame& frame, const std::string& x_column,
                               const std::string& y_column, const std::string& elevation_column,
                               double climb_cost) {
    const std::size_t n = frame.nrows();
    std::map<double, std::size_t> xs, ys;
    for (std::size_t i = 0; i < n; ++i) {
        xs[frame.double_at(x_column, i)] = 0;
        ys[frame.double_at(y_column, i)] = 0;
    }
    if (xs.size() < 2 || ys.size() < 2 || xs.size() * ys.size() != n)
        throw std::invalid_argument("terrain_from_frame: the table must hold one row per cell of a full grid");
    auto index_and_step = [](std::map<double, std::size_t>& m) {
        std::size_t k = 0;
        double      prev = 0.0, step = 0.0;
        for (auto& [v, idx] : m) {
            if (k == 1) step = v - prev;
            if (k > 1 && std::abs((v - prev) - step) > 1e-9 * std::abs(step))
                throw std::invalid_argument("terrain_from_frame: grid spacing must be uniform");
            prev = v;
            idx  = k++;
        }
        return step;
    };
    const double dx = index_and_step(xs), dy = index_and_step(ys);
    if (std::abs(dx - dy) > 1e-9 * std::abs(dx))
        throw std::invalid_argument("terrain_from_frame: x and y spacing must be equal");

    TerrainGrid g;
    g.rows       = xs.size();
    g.cols       = ys.size();
    g.cell_size  = dx;
    g.climb_cost = climb_cost;
    g.elevation.assign(n, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t cell = xs[frame.double_at(x_column, i)] * g.cols + ys[frame.double_at(y_column, i)];
        if (!std::isnan(g.elevation[cell])) throw std::invalid_argument("terrain_from_frame: repeated cell");
        g.elevation[cell] = frame.double_at(elevation_column, i);
    }
    return g;
}

double terrain_step_cost(const TerrainGrid& g, std::size_t from, std::size_t to) {
    check(g, from, "from");
    check(g, to, "to");
    const long dr = std::labs(static_cast<long>(from / g.cols) - static_cast<long>(to / g.cols));
    const long dc = std::labs(static_cast<long>(from % g.cols) - static_cast<long>(to % g.cols));
    if (dr > 1 || dc > 1 || dr + dc == 0) throw std::invalid_argument("terrain_step_cost: cells are not neighbours");
    const double horizontal = g.cell_size * (dr + dc == 2 ? std::sqrt(2.0) : 1.0);
    return horizontal + g.climb_cost * std::max(0.0, g.elevation[to] - g.elevation[from]);
}

double terrain_heuristic(const TerrainGrid& g, std::size_t cell, std::size_t goal, TerrainHeuristic kind) {
    check(g, cell, "cell");
    check(g, goal, "goal");
    const double dr = std::abs(static_cast<double>(cell / g.cols) - static_cast<double>(goal / g.cols));
    const double dc = std::abs(static_cast<double>(cell % g.cols) - static_cast<double>(goal % g.cols));
    switch (kind) {
        case TerrainHeuristic::Zero: return 0.0;
        case TerrainHeuristic::Euclidean: return g.cell_size * std::hypot(dr, dc);
        case TerrainHeuristic::Octile:
            return g.cell_size * (std::max(dr, dc) + (std::sqrt(2.0) - 1.0) * std::min(dr, dc));
        case TerrainHeuristic::OctileClimb:
            return g.cell_size * (std::max(dr, dc) + (std::sqrt(2.0) - 1.0) * std::min(dr, dc)) +
                   g.climb_cost * std::max(0.0, g.elevation[goal] - g.elevation[cell]);
    }
    return 0.0;
}

TerrainPath terrain_a_star(const TerrainGrid& g, std::size_t start, std::size_t goal, TerrainHeuristic kind,
                           double weight, bool reopen) {
    check(g, start, "start");
    check(g, goal, "goal");
    if (!(weight >= 1.0)) throw std::invalid_argument("terrain_a_star: weight must be at least 1");
    const std::size_t   n = g.elevation.size();
    std::vector<double> best(n, std::numeric_limits<double>::infinity()), h(n, -1.0);
    std::vector<std::size_t> parent(n, n);
    std::vector<char>        closed(n, 0);
    auto heuristic = [&](std::size_t c) {
        if (h[c] < 0.0) h[c] = terrain_heuristic(g, c, goal, kind);
        return h[c];
    };
    // Queue entries (f, -g, cell): ties in f go to the deeper node, which tends to reach the goal sooner.
    using Entry = std::tuple<double, double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    TerrainPath out;
    best[start] = 0.0;
    open.emplace(weight * heuristic(start), 0.0, start);
    out.pushes = 1;
    while (!open.empty()) {
        const auto [f, neg_g, u] = open.top();
        open.pop();
        if (-neg_g > best[u] || (!reopen && closed[u])) continue;  // stale entry
        closed[u] = 1;
        out.expansion_order.push_back(u);
        if (u == goal) {
            out.found = true;
            out.cost  = best[u];
            trace_path(g, parent, u, out);
            return out;
        }
        for_each_neighbour(g, u, [&](std::size_t v) {
            if (!reopen && closed[v]) return;
            const double cand = best[u] + terrain_step_cost(g, u, v);
            if (improves(cand, best[v])) {
                best[v]   = cand;
                parent[v] = u;
                open.emplace(cand + weight * heuristic(v), -cand, v);
                ++out.pushes;
            }
        });
    }
    return out;
}

TerrainPath terrain_best_first(const TerrainGrid& g, std::size_t start, std::size_t goal,
                               const std::vector<double>& heuristic, double g_weight) {
    check(g, start, "start");
    check(g, goal, "goal");
    const std::size_t n = g.elevation.size();
    if (heuristic.size() != n) throw std::invalid_argument("terrain_best_first: one heuristic value per cell");
    if (!(g_weight >= 0.0)) throw std::invalid_argument("terrain_best_first: g_weight must be non-negative");
    std::vector<double>      best(n, std::numeric_limits<double>::infinity());
    std::vector<std::size_t> parent(n, n);
    std::vector<char>        closed(n, 0);
    // Queue entries (f, h, g, cell); an entry is stale when its g no longer matches best[cell].
    using Entry = std::tuple<double, double, double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    TerrainPath out;
    best[start] = 0.0;
    open.emplace(heuristic[start], heuristic[start], 0.0, start);
    out.pushes = 1;
    while (!open.empty()) {
        const auto [f, h, gu, u] = open.top();
        open.pop();
        if (closed[u] || gu > best[u]) continue;
        closed[u] = 1;
        out.expansion_order.push_back(u);
        if (u == goal) {
            out.found = true;
            out.cost  = best[u];
            trace_path(g, parent, u, out);
            return out;
        }
        for_each_neighbour(g, u, [&](std::size_t v) {
            if (closed[v]) return;
            const double cand = best[u] + terrain_step_cost(g, u, v);
            if (improves(cand, best[v])) {
                best[v]   = cand;
                parent[v] = u;
                open.emplace(g_weight * cand + heuristic[v], heuristic[v], cand, v);
                ++out.pushes;
            }
        });
    }
    return out;
}

TerrainBeamResult terrain_beam_search(const TerrainGrid& g, std::size_t start, std::size_t goal, TerrainHeuristic kind,
                                      std::size_t width, BeamRanking ranking, bool stop_at_first) {
    check(g, start, "start");
    check(g, goal, "goal");
    if (width == 0) throw std::invalid_argument("terrain_beam_search: width must be at least 1");
    const std::size_t        n   = g.elevation.size();
    const double             inf = std::numeric_limits<double>::infinity();
    std::vector<double>      best(n, inf), h(n);
    std::vector<std::size_t> parent(n, n), stamp(n, 0);
    for (std::size_t c = 0; c < n; ++c) h[c] = terrain_heuristic(g, c, goal, kind);
    TerrainBeamResult out;
    double            incumbent = inf;
    auto below_incumbent = [&](std::size_t v) { return improves(best[v] + h[v], incumbent); };

    best[start] = 0.0;
    std::vector<std::size_t> beam;
    if (start == goal) {
        incumbent = 0.0;
        out.incumbent_costs.push_back(0.0);
        out.incumbent_layers.push_back(0);
    } else {
        beam.push_back(start);
    }
    while (!beam.empty()) {
        ++out.layers;
        out.max_beam = std::max(out.max_beam, beam.size());
        out.layer_start.push_back(out.best.expansion_order.size());
        std::vector<std::size_t> next;
        for (const std::size_t u : beam) {
            out.best.expansion_order.push_back(u);
            for_each_neighbour(g, u, [&](std::size_t v) {
                const double cand = best[u] + terrain_step_cost(g, u, v);
                if (!improves(cand, best[v])) return;
                if (!improves(cand + h[v], incumbent)) {
                    ++out.bound_pruned;
                    return;
                }
                best[v]   = cand;
                parent[v] = u;
                if (v == goal) {
                    incumbent = cand;
                    out.incumbent_costs.push_back(cand);
                    out.incumbent_layers.push_back(out.layers);
                    if (out.incumbent_costs.size() == 1) out.first_layer = out.layers;
                    return;
                }
                if (stamp[v] != out.layers) {
                    stamp[v] = out.layers;
                    next.push_back(v);
                }
            });
        }
        if (out.first_layer == out.layers) out.first_cost = incumbent;
        if (stop_at_first && !out.incumbent_costs.empty()) break;
        // Drop candidates the latest goal cost has made useless, then rank and cut.
        const auto useless = std::ranges::remove_if(next, [&](std::size_t v) { return !below_incumbent(v); });
        out.bound_pruned += static_cast<std::size_t>(std::ranges::distance(useless));
        next.erase(useless.begin(), useless.end());
        auto key = [&](std::size_t v) { return ranking == BeamRanking::Heuristic ? h[v] : best[v] + h[v]; };
        std::ranges::sort(next, [&](std::size_t a, std::size_t b) {
            const double ka = key(a), kb = key(b);
            if (ka != kb) return ka < kb;
            if (best[a] != best[b]) return best[a] < best[b];
            return a < b;
        });
        if (next.size() > width) {
            out.width_pruned += next.size() - width;
            next.resize(width);
        }
        beam.swap(next);
    }
    if (!out.incumbent_costs.empty()) {
        // Parents may have improved since the goal was reached, so cost the path as traced.
        out.best.found = true;
        trace_path(g, parent, goal, out.best);
        out.best.cost = 0.0;
        for (std::size_t k = 1; k < out.best.path.size(); ++k)
            out.best.cost += terrain_step_cost(g, out.best.path[k - 1], out.best.path[k]);
    }
    return out;
}

TerrainBeamStackResult terrain_beam_stack_search(const TerrainGrid& g, std::size_t start, std::size_t goal,
                                                 TerrainHeuristic kind, std::size_t width, double upper_bound,
                                                 bool remember_finished, std::size_t max_expansions) {
    check(g, start, "start");
    check(g, goal, "goal");
    if (width == 0) throw std::invalid_argument("terrain_beam_stack_search: width must be at least 1");
    const std::size_t   n   = g.elevation.size();
    const double        inf = std::numeric_limits<double>::infinity();
    std::vector<double> h(n);
    for (std::size_t c = 0; c < n; ++c) h[c] = terrain_heuristic(g, c, goal, kind);

    struct Node {
        std::size_t cell;
        double      g;
        std::size_t parent;  // index into the previous layer
    };
    using Key = std::pair<double, std::size_t>;  // (f, cell): a total order on a layer's nodes
    const Key lowest{-inf, 0}, highest{inf, n};
    struct Layer {
        std::vector<Node> nodes;
        Key               lo, hi;  // f-range [lo, hi) of this layer's current slice
    };
    std::vector<Layer> layers{{{{start, 0.0, 0}}, lowest, highest}};

    // For each cell, a stack of (layer, smallest g of the cell in layers 0..layer), for duplicate checks.
    std::vector<std::vector<std::pair<std::size_t, double>>> stored(n);
    std::size_t stored_count = 1;
    stored[start].push_back({0, 0.0});
    auto best_stored = [&](std::size_t c) { return stored[c].empty() ? inf : stored[c].back().second; };
    // Smallest g of each cell among nodes whose layer was dropped after all their extensions were explored.
    std::vector<double> finished(remember_finished ? n : 0, inf);
    auto drop_layer  = [&](std::size_t l) {
        for (const Node& nd : layers[l].nodes) {
            stored[nd.cell].pop_back();
            if (remember_finished) finished[nd.cell] = std::min(finished[nd.cell], nd.g);
        }
        stored_count -= layers[l].nodes.size();
    };

    TerrainBeamStackResult out;
    double                 bound = upper_bound;
    std::vector<double>    cand_g(n, inf);
    std::vector<std::size_t> cand_parent(n), cand_stamp(n, 0);
    std::size_t            stamp = 0;

    // Builds the nodes of layer l + 1 within [lo, hi) from layer l; narrows hi if the layer is cut.
    auto generate = [&](std::size_t l, const Key& lo, Key& hi) {
        ++stamp;
        std::vector<std::size_t> cells;
        const auto& above = layers[l].nodes;
        for (std::size_t i = 0; i < above.size(); ++i) {
            const Node& u = above[i];
            out.best.expansion_order.push_back(u.cell);
            for_each_neighbour(g, u.cell, [&](std::size_t v) {
                const double cand = u.g + terrain_step_cost(g, u.cell, v);
                const Key    key{cand + h[v], v};
                if (key < lo || !(key < hi) || !improves(cand + h[v], bound)) return;
                if (!improves(cand, best_stored(v))) return;  // a stored copy is at least as good
                if (remember_finished && !improves(cand, finished[v])) return;
                if (v == goal) {
                    bound = cand;
                    out.incumbent_costs.push_back(cand);
                    out.incumbent_expansions.push_back(out.best.expansion_order.size());
                    out.best.path.clear();
                    out.best.path.push_back(v);
                    std::size_t idx = i;
                    for (std::size_t k = l + 1; k-- > 0;) {
                        out.best.path.push_back(layers[k].nodes[idx].cell);
                        idx = layers[k].nodes[idx].parent;
                    }
                    std::reverse(out.best.path.begin(), out.best.path.end());
                    return;
                }
                if (cand_stamp[v] != stamp) {
                    cand_stamp[v] = stamp;
                    cand_g[v]     = inf;
                    cells.push_back(v);
                }
                if (improves(cand, cand_g[v])) {
                    cand_g[v]      = cand;
                    cand_parent[v] = i;
                }
            });
        }
        std::vector<Node> nodes;
        for (const std::size_t v : cells)
            if (improves(cand_g[v] + h[v], bound)) nodes.push_back({v, cand_g[v], cand_parent[v]});
        std::ranges::sort(nodes, {}, [&](const Node& nd) { return Key{nd.g + h[nd.cell], nd.cell}; });
        if (nodes.size() > width) {
            hi = Key{nodes[width].g + h[nodes[width].cell], nodes[width].cell};
            nodes.resize(width);
        }
        return nodes;
    };
    auto push_layer = [&](std::vector<Node> nodes, const Key& lo, const Key& hi) {
        const std::size_t l = layers.size();
        for (const Node& nd : nodes) stored[nd.cell].push_back({l, std::min(best_stored(nd.cell), nd.g)});
        stored_count += nodes.size();
        layers.push_back({std::move(nodes), lo, hi});
        out.max_stored = std::max(out.max_stored, stored_count);
        out.max_depth  = std::max(out.max_depth, l);
    };

    if (start == goal) {
        bound = 0.0;
        out.incumbent_costs.push_back(0.0);
        out.incumbent_expansions.push_back(0);
        out.best.path = {start};
    }
    const auto over_budget = [&] { return max_expansions != 0 && out.best.expansion_order.size() >= max_expansions; };
    bool budget_hit = false;
    while (bound > 0.0) {
        if (over_budget()) {
            budget_hit = true;
            break;
        }
        Key  hi    = highest;
        auto nodes = generate(layers.size() - 1, lowest, hi);
        if (!nodes.empty()) {
            push_layer(std::move(nodes), lowest, hi);
            continue;
        }
        // Dead end: back up to the deepest layer that still has an unexplored slice below the bound.
        bool resumed = false;
        while (layers.size() > 1 && !resumed) {
            Layer& top = layers.back();
            if (top.hi == highest || !improves(top.hi.first, bound)) {
                drop_layer(layers.size() - 1);
                layers.pop_back();
                continue;
            }
            if (over_budget()) break;
            const Key lo = top.hi;
            drop_layer(layers.size() - 1);
            layers.pop_back();
            ++out.backtracks;
            Key  next_hi = highest;
            auto regen   = generate(layers.size() - 1, lo, next_hi);
            if (!regen.empty()) {
                push_layer(std::move(regen), lo, next_hi);
                resumed = true;
            } else {
                // Empty slice: nothing left in this layer; keep backing up.
                continue;
            }
        }
        if (!resumed) {
            budget_hit = over_budget() && layers.size() > 1;
            break;
        }
    }
    out.exhausted = !budget_hit;
    if (!out.incumbent_costs.empty()) {
        out.best.found = true;
        auto& p = out.best.path;
        out.best.cost = 0.0;
        for (std::size_t k = 1; k < p.size(); ++k) {
            const std::size_t a = p[k - 1], b = p[k];
            out.best.cost += terrain_step_cost(g, a, b);
            const bool diagonal = (a / g.cols != b / g.cols) && (a % g.cols != b % g.cols);
            out.best.length += g.cell_size * (diagonal ? std::sqrt(2.0) : 1.0);
            const double dz = g.elevation[b] - g.elevation[a];
            (dz > 0 ? out.best.ascent : out.best.descent) += std::abs(dz);
        }
    }
    return out;
}

std::vector<double> terrain_cost_to_go(const TerrainGrid& g, std::size_t goal) {
    check(g, goal, "goal");
    std::vector<double> dist(g.elevation.size(), std::numeric_limits<double>::infinity());
    using Entry = std::pair<double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    dist[goal] = 0.0;
    open.emplace(0.0, goal);
    while (!open.empty()) {
        const auto [d, v] = open.top();
        open.pop();
        if (d > dist[v]) continue;
        // Relax the reversed step u -> v for every neighbour u of v.
        for_each_neighbour(g, v, [&](std::size_t u) {
            const double cand = d + terrain_step_cost(g, u, v);
            if (improves(cand, dist[u])) {
                dist[u] = cand;
                open.emplace(cand, u);
            }
        });
    }
    return dist;
}

std::vector<double> terrain_cost_from(const TerrainGrid& g, std::size_t start) {
    check(g, start, "start");
    std::vector<double> dist(g.elevation.size(), std::numeric_limits<double>::infinity());
    using Entry = std::pair<double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    dist[start] = 0.0;
    open.emplace(0.0, start);
    while (!open.empty()) {
        const auto [d, u] = open.top();
        open.pop();
        if (d > dist[u]) continue;
        for_each_neighbour(g, u, [&](std::size_t v) {
            const double cand = d + terrain_step_cost(g, u, v);
            if (improves(cand, dist[v])) {
                dist[v] = cand;
                open.emplace(cand, v);
            }
        });
    }
    return dist;
}

std::vector<std::tuple<std::size_t, std::size_t, double>> terrain_edges(const TerrainGrid& g) {
    check(g, 0, "grid");
    std::vector<std::tuple<std::size_t, std::size_t, double>> edges;
    edges.reserve(g.elevation.size() * 8);
    for (std::size_t u = 0; u < g.elevation.size(); ++u)
        for_each_neighbour(g, u, [&](std::size_t v) { edges.emplace_back(u, v, terrain_step_cost(g, u, v)); });
    return edges;
}

}  // namespace datamunge::algorithms

namespace datamunge::algorithms {

std::vector<std::tuple<std::size_t, std::size_t, double>> terrain_energy_edges(const TerrainGrid& g,
                                                                               double descent_credit) {
    auto edges = terrain_edges(g);
    for (auto& [u, v, w] : edges) w -= descent_credit * std::max(0.0, g.elevation[u] - g.elevation[v]);
    return edges;
}

}  // namespace datamunge::algorithms
