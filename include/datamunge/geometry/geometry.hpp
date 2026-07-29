#pragma once

// Convenience umbrella header -- includes the whole computational-geometry module: 2D point
// primitives, convex hull, closest pair of points, polygon operations, segment intersection, a
// k-d tree for nearest-neighbor/range search, Delaunay triangulation and its dual Voronoi
// diagram, and two curve/sequence-similarity measures (dynamic time warping, discrete Frechet
// distance).

#include <datamunge/geometry/bounding_box.hpp>
#include <datamunge/geometry/closest_pair.hpp>
#include <datamunge/geometry/convex_hull.hpp>
#include <datamunge/geometry/delaunay.hpp>
#include <datamunge/geometry/dtw.hpp>
#include <datamunge/geometry/frechet.hpp>
#include <datamunge/geometry/geometric_hashing.hpp>
#include <datamunge/geometry/icp.hpp>
#include <datamunge/geometry/jump_and_walk.hpp>
#include <datamunge/geometry/kdtree.hpp>
#include <datamunge/geometry/laplacian_smoothing.hpp>
#include <datamunge/geometry/min_enclosing_circle.hpp>
#include <datamunge/geometry/point2d.hpp>
#include <datamunge/geometry/polygon.hpp>
#include <datamunge/geometry/polygon_clip.hpp>
#include <datamunge/geometry/polygon_triangulation.hpp>
#include <datamunge/geometry/rotating_calipers.hpp>
#include <datamunge/geometry/segment_intersection.hpp>
#include <datamunge/geometry/simplify_polyline.hpp>
#include <datamunge/geometry/voronoi.hpp>
