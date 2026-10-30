#pragma once

/// @file terrain_search.hpp
/// @brief A* and beam search path planning on an elevation grid, with admissible terrain heuristics.
///
/// A TerrainGrid is a rectangular grid of elevations. A walker moves to any of the 8
/// neighbouring cells. A step costs its horizontal length (one cell, or \f$\sqrt2\f$ cells on a
/// diagonal) plus `climb_cost` times the height gained, so climbing is penalized and descending
/// is free. terrain_a_star() expands cells in order of \f$f=g+w\,h\f$, where \f$g\f$ is the cost so
/// far and \f$h\f$ a lower bound on the cost to the goal. Every TerrainHeuristic is consistent
/// (\f$h(u)\le c(u,v)+h(v)\f$), so with weight \f$w=1\f$ the first path to reach the goal is
/// optimal; with \f$w>1\f$ its cost is at most \f$w\f$ times the optimum.

#include <datamunge/dstruct/dataframe.hpp>

#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {

/// An elevation grid; cell (i, j) has index i * cols + j.
struct TerrainGrid {
    std::size_t         rows       = 0;    ///< Number of grid lines along the first axis.
    std::size_t         cols       = 0;    ///< Number of grid lines along the second axis.
    std::vector<double> elevation;         ///< Height of each cell, row-major (rows * cols values).
    double              cell_size  = 1.0;  ///< Horizontal distance between adjacent cells.
    double              climb_cost = 0.0;  ///< Extra cost per unit of height gained.
};

/// @brief Builds a grid from a long-format table with one row per cell.
///
/// The distinct values of @p x_column become the rows and those of @p y_column the columns, so
/// cell (i, j) holds the elevation at the i-th smallest x and the j-th smallest y.
/// @throws std::invalid_argument if a cell is missing or repeated, or the x and y spacings are
///         not equal and uniform.
TerrainGrid terrain_from_frame(const dstruct::DataFrame& frame, const std::string& x_column,
                               const std::string& y_column, const std::string& elevation_column,
                               double climb_cost = 0.0);

/// Lower bounds on the cost from a cell to the goal.
enum class TerrainHeuristic {
    Zero,         ///< \f$h=0\f$: A* becomes Dijkstra's algorithm.
    Euclidean,    ///< Straight-line horizontal distance.
    Octile,       ///< Shortest 8-neighbour distance on an empty grid: \f$\max+(\sqrt2-1)\min\f$ of the axis offsets.
    OctileClimb,  ///< Octile plus `climb_cost` times the net height still to gain, \f$\max(0,z_{goal}-z)\f$.
};

/// Result of terrain_a_star().
struct TerrainPath {
    bool                     found = false;    ///< Whether the goal was reached.
    double                   cost  = 0.0;      ///< Total cost of the path.
    std::vector<std::size_t> path;             ///< Cells from start to goal.
    std::vector<std::size_t> expansion_order;  ///< Cells in the order they were expanded (repeats if reopened).
    std::size_t              pushes  = 0;      ///< Entries pushed onto the priority queue.
    double                   length  = 0.0;    ///< Horizontal length of the path.
    double                   ascent  = 0.0;    ///< Total height gained along the path.
    double                   descent = 0.0;    ///< Total height lost along the path.
};

/// @brief Cost of the step from cell @p from to the neighbouring cell @p to.
double terrain_step_cost(const TerrainGrid& grid, std::size_t from, std::size_t to);

/// @brief Heuristic value at @p cell for reaching @p goal.
double terrain_heuristic(const TerrainGrid& grid, std::size_t cell, std::size_t goal, TerrainHeuristic kind);

/// @brief A* (weighted when @p weight > 1) from @p start to @p goal.
/// @param weight  multiplies the heuristic: \f$f=g+w\,h\f$; the path costs at most @p weight times the optimum.
/// @param reopen  re-expand a closed cell when a cheaper route to it turns up. With a consistent
///                heuristic and weight 1 this never happens. With weight > 1, setting it to false
///                avoids repeated expansions and still keeps the weight bound (Likhachev, Gordon
///                and Thrun 2003).
/// @throws std::invalid_argument for out-of-range cells or a weight below 1.
TerrainPath terrain_a_star(const TerrainGrid& grid, std::size_t start, std::size_t goal, TerrainHeuristic kind,
                           double weight = 1.0, bool reopen = true);

/// @brief Best-first search ordered by \f$f=\alpha\,g+h\f$ with any heuristic table.
///
/// With \f$\alpha=0\f$ this is greedy best-first search: the cell that looks closest to the goal is
/// expanded next, whatever it cost to reach. With \f$\alpha=1\f$ it is A* with the given heuristic.
/// Closed cells are never reopened; a cheaper route to an open cell updates its cost and parent.
/// Ties in f go to the cell with the smaller h. The goal is tested when it is expanded.
///
/// @param grid the terrain.
/// @param start the start cell.
/// @param goal the goal cell.
/// @param heuristic one value per cell (for example from terrain_cost_to_go() or terrain_heuristic()).
/// @param g_weight the weight \f$\alpha\ge0\f$ of the cost so far.
/// @return the path found, its cost and the expansions.
/// @throws std::invalid_argument for out-of-range cells, a heuristic of the wrong size or a
///         negative weight.
///
/// @par Example
/// @code
/// const auto h = dm::algorithms::terrain_cost_to_go(grid, goal);
/// const auto greedy = dm::algorithms::terrain_best_first(grid, start, goal, h, 0.0);
/// @endcode
/// @see terrain_a_star()
TerrainPath terrain_best_first(const TerrainGrid& grid, std::size_t start, std::size_t goal,
                               const std::vector<double>& heuristic, double g_weight = 0.0);

/// How terrain_beam_search() ranks the cells competing for a place in the beam.
enum class BeamRanking {
    CostPlusHeuristic,  ///< \f$f=g+h\f$, as in A*.
    Heuristic,          ///< \f$h\f$ alone (greedy beam search): ignores the cost already paid.
};

/// Result of terrain_beam_search().
struct TerrainBeamResult {
    TerrainPath              best;            ///< Cheapest path found; `best.expansion_order` lists every expanded cell.
    double                   first_cost  = 0.0;  ///< Best goal cost at the end of the first layer that reached the goal (0 if none).
    std::size_t              first_layer = 0;    ///< Layer in which the goal was first reached.
    std::size_t              layers      = 0;    ///< Number of layers expanded.
    std::size_t              max_beam    = 0;    ///< Largest beam held at once.
    std::size_t              width_pruned = 0;   ///< Candidates dropped because the beam was full.
    std::size_t              bound_pruned = 0;   ///< Candidates dropped because \f$g+h\f$ reached the best goal cost.
    std::vector<std::size_t> layer_start;     ///< Index into `best.expansion_order` where each layer starts.
    std::vector<double>      incumbent_costs; ///< Goal cost after each improvement, in order.
    std::vector<std::size_t> incumbent_layers;  ///< Layer of each improvement.
};

/// @brief Layered beam search from @p start to @p goal.
///
/// Layer \f$k+1\f$ holds the neighbours of the layer-\f$k\f$ cells whose cost \f$g\f$ improved,
/// ranked by @p ranking and cut to the best @p width. A cell reached again is kept only if its new
/// \f$g\f$ is lower. Once the goal is reached, cells with \f$g+h\f$ at least the best goal cost are
/// dropped, and the search continues until the beam is empty, keeping the cheapest goal path
/// (unless @p stop_at_first). Memory is \f$O(\text{width})\f$ per layer. The search is incomplete
/// and the path need not be optimal for finite width; with width at least the number of cells and
/// an admissible heuristic it is optimal.
/// @param width  Beam width, at least 1.
/// @param stop_at_first  Return as soon as the goal is first reached (classic beam search).
/// @throws std::invalid_argument for out-of-range cells or zero width.
TerrainBeamResult terrain_beam_search(const TerrainGrid& grid, std::size_t start, std::size_t goal,
                                      TerrainHeuristic kind, std::size_t width,
                                      BeamRanking ranking = BeamRanking::CostPlusHeuristic,
                                      bool stop_at_first = false);

/// Result of terrain_beam_stack_search().
struct TerrainBeamStackResult {
    TerrainPath              best;               ///< Optimal path (if one costs less than the bound); `best.expansion_order` lists every expansion.
    std::size_t              backtracks  = 0;    ///< Times a layer was regenerated with its next f-range.
    std::size_t              max_stored  = 0;    ///< Largest number of nodes held in all layers at once.
    std::size_t              max_depth   = 0;    ///< Deepest layer reached.
    std::vector<double>      incumbent_costs;    ///< Goal cost after each improvement, in order.
    std::vector<std::size_t> incumbent_expansions;  ///< Expansions done when each improvement was found.
    bool                     exhausted   = false;   ///< Whether the search finished, proving the result optimal (false if the budget ran out).
};

/// @brief Beam-stack search (Zhou and Hansen 2005): beam search made complete by backtracking.
///
/// Layer \f$l\f$ keeps at most @p width nodes, the best by \f$f=g+h\f$ within a range
/// \f$[f_{\min,l},f_{\max,l})\f$ held on a stack (ties broken by cell index). When a layer is cut,
/// \f$f_{\max,l}\f$ is set to the first node left out. When the search cannot go deeper, it backtracks to
/// the deepest cut layer, moves its range on to \f$[f_{\max,l},\infty)\f$, and regenerates it from the
/// layer above. Nodes with \f$f\f$ at least the best goal cost \f$U\f$ are pruned. A node whose cell
/// already sits in a stored layer with no greater \f$g\f$ is dropped. When the stack empties, every
/// path cheaper than \f$U\f$ has been ruled out, so with an admissible heuristic the result is optimal.
/// Memory is at most @p width nodes per layer, but cells forgotten when a layer is dropped can be
/// re-expanded many times. With @p remember_finished, the smallest \f$g\f$ of every node whose layer
/// was dropped (so all its extensions were explored) is kept in a table of one value per cell, and a
/// later node with no smaller \f$g\f$ is pruned. This removes the re-expansion at the cost of
/// memory proportional to the grid.
/// @param width  Beam width, at least 1.
/// @param upper_bound  Only paths cheaper than this are sought (for example a beam search result).
/// @param remember_finished  Keep the table of finished cells described above.
/// @param max_expansions  Stop after this many expansions (0 for no limit); `exhausted` is then false.
/// @throws std::invalid_argument for out-of-range cells or zero width.
TerrainBeamStackResult terrain_beam_stack_search(const TerrainGrid& grid, std::size_t start, std::size_t goal,
                                                 TerrainHeuristic kind, std::size_t width,
                                                 double upper_bound = std::numeric_limits<double>::infinity(),
                                                 bool remember_finished = false, std::size_t max_expansions = 0);

/// @brief Exact cost-to-go \f$h^*(u)\f$ from every cell to @p goal (Dijkstra on reversed steps).
std::vector<double> terrain_cost_to_go(const TerrainGrid& grid, std::size_t goal);

/// @brief Exact cost \f$g^*(u)\f$ of the cheapest path from @p start to every cell (Dijkstra).
std::vector<double> terrain_cost_from(const TerrainGrid& grid, std::size_t start);

/// @brief Every step of the grid as (from, to, cost), for the generic a_star_search().
std::vector<std::tuple<std::size_t, std::size_t, double>> terrain_edges(const TerrainGrid& grid);

/// @brief Every step of the grid as (from, to, cost) when descending also earns a credit.
///
/// The cost is terrain_step_cost() minus @p descent_credit per unit of height lost, so a step
/// \f$u\to v\f$ costs \f$d(u,v)+\lambda\max(0,\Delta z)-\kappa\max(0,-\Delta z)\f$, with
/// \f$\lambda\f$ = `climb_cost` and \f$\kappa\f$ = @p descent_credit. Steep descents get negative
/// costs. Around any closed loop the ascent equals the descent, so a loop costs its length plus
/// \f$(\lambda-\kappa)\f$ times its ascent: there are no negative cycles when
/// \f$\kappa\le\lambda\f$, and steep up-and-down loops become negative when \f$\kappa>\lambda\f$.
/// Use with bellman_ford().
/// @param grid the terrain.
/// @param descent_credit cost recovered per unit of height lost (for example a fraction of `climb_cost`).
/// @return all directed 8-neighbour steps with their costs.
std::vector<std::tuple<std::size_t, std::size_t, double>> terrain_energy_edges(const TerrainGrid& grid,
                                                                               double descent_credit);

}  // namespace datamunge::algorithms
