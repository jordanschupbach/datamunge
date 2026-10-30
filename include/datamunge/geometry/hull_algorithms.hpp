#pragma once

/// @file hull_algorithms.hpp
/// @brief Classic convex-hull algorithms (Graham scan, Jarvis march, Chan's algorithm,
///        Quickhull), the binary-search tangent Chan's algorithm relies on, convex layers, and
///        random point clouds with controllable hull size.
///
/// They complement the module's default Andrew's-monotone-chain `convex_hull`. Every hull
/// function returns the strict convex hull: vertices counterclockwise, points on an edge
/// excluded, no repeated first/last vertex. They are interchangeable and cross-checkable.
///
/// Chan's algorithm (Chan 1996) is output-sensitive: \f$O(n\log h)\f$ for \f$h\f$ hull
/// vertices. It guesses \f$m\f$, splits the points into groups of at most \f$m\f$, hulls each
/// group with Graham scan, and gift-wraps the groups for at most \f$m\f$ steps, finding each
/// group's candidate by binary search on its hull. When the wrap does not close, it squares
/// \f$m\f$ and starts again. `chans_algorithm_trace` records every round.
///
/// Orientation tests are exact for integer coordinates up to about \f$2^{26}\f$; with general
/// floating-point input near-collinear triples can be misclassified.

#include <datamunge/geometry/point2d.hpp>

#include <cstdint>
#include <vector>

namespace datamunge::geometry {

/// @brief Graham scan: sort by polar angle around the lowest point, then sweep keeping only
///        left turns. \f$O(n\log n)\f$.
/// @param points Input points (duplicates allowed).
/// @return The strict convex hull, counterclockwise.
std::vector<Point2D> graham_scan(std::vector<Point2D> points);

/// @brief Graham scan that also counts the orientation tests (calls of `cross`) it makes,
///        in the angular sort and in the sweep.
/// @param points Input points (duplicates allowed).
/// @param orientation_tests Incremented by the number of orientation tests.
/// @return The strict convex hull, counterclockwise.
std::vector<Point2D> graham_scan(std::vector<Point2D> points, long long& orientation_tests);

/// @brief Gift wrapping (Jarvis march): from the leftmost point, repeatedly pick the most
///        clockwise next point. \f$O(nh)\f$ for \f$h\f$ hull vertices.
/// @param points Input points (duplicates allowed).
/// @return The strict convex hull, counterclockwise.
std::vector<Point2D> jarvis_march(std::vector<Point2D> points);

/// @brief Jarvis march that also counts its orientation tests (\f$n-2\f$ per hull vertex
///        for \f$n\f$ distinct points).
/// @param points Input points (duplicates allowed).
/// @param orientation_tests Incremented by the number of orientation tests.
/// @return The strict convex hull, counterclockwise.
std::vector<Point2D> jarvis_march(std::vector<Point2D> points, long long& orientation_tests);

/// @brief Result of a tangent search from an outside point to a convex polygon.
struct TangentSearch {
    int              index = -1;          ///< Index of the tangent vertex in the polygon.
    std::vector<int> probes;              ///< Vertices examined by the binary search, in order.
    long long        orientation_tests{}; ///< Orientation tests used.
};

/// @brief Find the vertex \f$v_t\f$ of a convex polygon such that every vertex lies left of,
///        or on, the ray from @p p through \f$v_t\f$, by binary search in \f$O(\log k)\f$.
///
/// Seen from an outside point, the directions to the vertices of a convex polygon form a
/// cyclic bitonic sequence: they turn counterclockwise from one tangent vertex to the other,
/// then clockwise back. The wanted vertex is the most clockwise one (the minimum), found by
/// halving the cycle. Of two vertices collinear with @p p, the farther is returned. This is the
/// step that lets Chan's algorithm query a group hull of size \f$m\f$ in \f$O(\log m)\f$.
/// @param polygon Strictly convex polygon, counterclockwise, no repeated vertex.
/// @param p Point strictly outside the polygon.
/// @return The tangent index, the probe sequence and the test count; index -1 if @p polygon
///         is empty.
TangentSearch convex_polygon_tangent(const std::vector<Point2D>& polygon, const Point2D& p);

/// @brief Options for `chans_algorithm_trace`.
struct ChanOptions {
    long long initial_m       = 4;     ///< First guess for the hull size (squared each round).
    bool      square_guess    = true;  ///< Square \f$m\f$ after a failed round; false doubles it.
    bool      binary_tangent  = true;  ///< Binary-search the tangents; false scans each group hull.
    bool      record_geometry = false; ///< Keep each round's group hulls and wrap path.
};

/// @brief One round of Chan's algorithm with guess \f$m\f$.
struct ChanRound {
    long long                         m{};                   ///< Guess for the hull size.
    int                               groups{};              ///< Number of groups, \f$\lceil n/m\rceil\f$.
    int                               group_hull_vertices{}; ///< Vertices over all group hulls.
    int                               wrap_steps{};          ///< Wrap steps taken (at most \f$m\f$).
    bool                              completed{};           ///< Whether the wrap closed.
    long long                         hull_tests{};          ///< Orientation tests in the group hulls.
    long long                         wrap_tests{};          ///< Orientation tests in the wrap.
    std::vector<std::vector<Point2D>> group_hulls;           ///< Group hulls (if recorded).
    std::vector<Point2D>              wrap_path;             ///< Vertices wrapped so far (if recorded).
};

/// @brief Hull plus the per-round trace of Chan's algorithm.
struct ChanResult {
    std::vector<Point2D>   hull;                ///< Strict convex hull, counterclockwise.
    std::vector<ChanRound> rounds;              ///< One entry per guess \f$m\f$.
    long long              orientation_tests{}; ///< Total over all rounds.
    int                    distinct_points{};   ///< Points left after removing duplicates.
};

/// @brief Chan's algorithm with a round-by-round trace.
///
/// Duplicates are removed by hashing (no sort, which would cost \f$O(n\log n)\f$). Groups are
/// consecutive runs of the remaining points in input order. Round \f$t\f$ uses
/// \f$m_t=\min(n, m_0^{2^{t}})\f$ and costs \f$O(n\log m_t)\f$; the first round with
/// \f$m_t\ge h\f$ closes the wrap, so the total is \f$O(n\log h)\f$.
/// Doubling the guess instead (`square_guess = false`) costs \f$O(n\log^2 h)\f$.
/// @param points Input points (duplicates allowed).
/// @param options Initial guess, tangent method and whether to record geometry.
/// @return The hull and the trace.
ChanResult chans_algorithm_trace(const std::vector<Point2D>& points, const ChanOptions& options = {});

/// @brief Chan's output-sensitive convex hull, \f$O(n\log h)\f$.
/// @param points Input points (duplicates allowed).
/// @return The strict convex hull, counterclockwise.
std::vector<Point2D> chans_algorithm(std::vector<Point2D> points);

/// @brief Quickhull: split by the line through the two extreme-x points, then recursively take
///        the farthest point on each side. \f$O(n\log n)\f$ on average, \f$O(n^2)\f$ worst.
/// @param points Input points (duplicates allowed).
/// @return The strict convex hull, counterclockwise.
std::vector<Point2D> quickhull(std::vector<Point2D> points);

/// @brief Convex layers ("onion peeling"): layer 0 is the hull's vertices, layer 1 the hull
///        of the rest, and so on.
///
/// Each layer is computed with Chan's algorithm, so peeling costs
/// \f$O\big(\sum_j n_j\log h_j\big)\f$ for \f$n_j\f$ points left and \f$h_j\f$ vertices on
/// layer \f$j\f$. Points on a hull edge but not at a vertex go to a deeper layer. Duplicate
/// points get the same layer.
/// @param points Input points.
/// @return The layer of each input point, in input order.
std::vector<int> convex_layers(const std::vector<Point2D>& points);

/// @brief Shapes for `random_point_cloud`, chosen for their different expected hull sizes.
enum class PointCloud {
    Square,   ///< Uniform in \f$[0,1]^2\f$: \f$E[h]=\Theta(\log n)\f$.
    Disk,     ///< Uniform in the unit disk: \f$E[h]=\Theta(n^{1/3})\f$.
    Gaussian, ///< Standard bivariate normal: \f$E[h]=\Theta(\sqrt{\log n})\f$.
    Parabola  ///< On \f$y=x^2\f$ at distinct integers \f$x\f$ (convex position): \f$h=n\f$.
};

/// @brief Draw @p n random points from one of the `PointCloud` shapes.
///
/// For Square, Disk and Gaussian the coordinates are rounded to multiples of \f$2^{-20}\f$, so
/// orientation tests on them are exact. Parabola points are \f$(x,x^2)\f$ for a random
/// permutation of the integers \f$x\in[-n/2,n/2)\f$; their orientation signs stay correct for
/// \f$n\f$ up to several million, because a triple's true orientation \f$(b-a)(c-a)(c-b)\f$ is
/// far larger than the rounding error whenever the products exceed \f$2^{53}\f$.
/// @param n Number of points.
/// @param cloud Shape to sample.
/// @param seed Random seed.
/// @return The points.
std::vector<Point2D> random_point_cloud(int n, PointCloud cloud, std::uint64_t seed);

} // namespace datamunge::geometry
