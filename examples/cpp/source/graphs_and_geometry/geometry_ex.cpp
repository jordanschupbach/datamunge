#include <datamunge/geometry/geometry.hpp>

#include <iomanip>
#include <iostream>

using namespace datamunge::geometry;

namespace {
void print_point(const Point2D& p) { std::cout << "(" << p.x << ", " << p.y << ")"; }

void print_points(const std::vector<Point2D>& pts) {
    for (std::size_t i = 0; i < pts.size(); ++i) {
        if (i) std::cout << ", ";
        print_point(pts[i]);
    }
}
} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "=================== Convex hull, closest pair, polygon basics ===================\n";
    const std::vector<Point2D> cloud = {{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}, {1, 1}, {3, 3}, {2, 5}};
    const auto hull = convex_hull(cloud);
    std::cout << "convex hull of " << cloud.size() << " points (" << hull.size() << " on the hull): ";
    print_points(hull);
    std::cout << "\n";

    const auto closest = closest_pair(cloud);
    std::cout << "closest pair: ";
    print_point(closest.a);
    std::cout << " and ";
    print_point(closest.b);
    std::cout << ", distance = " << closest.distance << "\n";

    const std::vector<Point2D> square = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
    std::cout << "square area = " << polygon_area(square) << ", centroid = ";
    print_point(polygon_centroid(square));
    std::cout << ", is convex = " << std::boolalpha << is_convex_polygon(square) << "\n";
    std::cout << "point (2,2) inside square: " << point_in_polygon({2, 2}, square) << ", (5,5) inside: "
              << point_in_polygon({5, 5}, square) << "\n\n";

    std::cout << "=================== Segment/line intersection ===================\n";
    Point2D hit;
    if (segment_intersection_point({0, 0}, {4, 4}, {0, 4}, {4, 0}, hit)) {
        std::cout << "the two diagonals of the square cross at ";
        print_point(hit);
        std::cout << "\n";
    }
    std::cout << "segments {0,0}-{1,0} and {0,1}-{1,1} intersect: " << segments_intersect({0, 0}, {1, 0}, {0, 1}, {1, 1}) << "\n\n";

    std::cout << "=================== k-d tree nearest-neighbor search ===================\n";
    KDTree2D tree(cloud);
    const auto nearest = tree.nearest({1.9, 1.9});
    std::cout << "point nearest to (1.9, 1.9): ";
    print_point(nearest);
    std::cout << "\n";
    const auto k_nearest = tree.k_nearest({0, 0}, 3);
    std::cout << "3 nearest points to (0, 0): ";
    print_points(k_nearest);
    std::cout << "\n\n";

    std::cout << "=================== Delaunay triangulation and its dual Voronoi diagram ===================\n";
    const auto triangles = delaunay_triangulation(cloud);
    std::cout << "Delaunay triangulation of " << cloud.size() << " points produced " << triangles.size() << " triangles\n";
    const auto voronoi = voronoi_diagram(cloud);
    std::cout << "Voronoi diagram: " << voronoi.vertices.size() << " vertices, " << voronoi.cells.size() << " cells\n\n";

    std::cout << "=================== Dynamic time warping and discrete Frechet distance ===================\n";
    const std::vector<double> series_a = {1, 2, 3, 4, 5};
    const std::vector<double> series_b = {1, 1, 2, 3, 4, 5, 5}; // same shape, stretched at the ends
    const auto dtw = dynamic_time_warping(series_a, series_b);
    std::cout << "DTW distance between a stretched copy of the same shape: " << dtw.distance << " (expect a small value)\n";

    const std::vector<Point2D> curve_a = {{0, 0}, {1, 1}, {2, 0}};
    const std::vector<Point2D> curve_b = {{0, 0.1}, {1, 1.1}, {2, 0.1}};
    std::cout << "discrete Frechet distance between two near-identical curves: " << discrete_frechet_distance(curve_a, curve_b)
              << " (expect ~0.1)\n\n";

    std::cout << "=================== Bounding box, rotating calipers, min enclosing circle ===================\n";
    const auto bbox = bounding_box(cloud);
    std::cout << "bounding box: min=";
    print_point(bbox.min);
    std::cout << " max=";
    print_point(bbox.max);
    std::cout << "\n";

    const auto diameter = polygon_diameter(cloud);
    std::cout << "point-set diameter: " << diameter.distance << " between ";
    print_point(diameter.a);
    std::cout << " and ";
    print_point(diameter.b);
    std::cout << "\n";

    const std::vector<Point2D> diamond = {{2, 0}, {4, 2}, {2, 4}, {0, 2}};
    const auto mbr = minimum_bounding_rectangle(diamond);
    std::cout << "diamond's minimum-area bounding rectangle: area = " << mbr.area << " (its axis-aligned bbox would be 16)\n";

    const auto circle = min_enclosing_circle(cloud);
    std::cout << "minimum enclosing circle: center=";
    print_point(circle.center);
    std::cout << " radius=" << circle.radius << "\n\n";

    std::cout << "=================== Simplification, clipping, triangulation ===================\n";
    const std::vector<Point2D> noisy_line = {{0, 0}, {1, 0.1}, {2, 5}, {3, 0.1}, {4, 0}};
    const auto simplified = simplify_polyline(noisy_line, 1.0);
    std::cout << "Douglas-Peucker simplified a " << noisy_line.size() << "-point line down to " << simplified.size()
              << " points: ";
    print_points(simplified);
    std::cout << "\n";

    const std::vector<Point2D> clip_window = {{2, 2}, {6, 2}, {6, 6}, {2, 6}};
    const auto clipped = clip_polygon(square, clip_window);
    std::cout << "Sutherland-Hodgman clip of the square against an overlapping window: area = " << polygon_area(clipped) << "\n";

    const std::vector<Point2D> l_shape = {{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}};
    const auto ears = triangulate_polygon(l_shape);
    std::cout << "ear-clipping triangulated an L-shape (area " << polygon_area(l_shape) << ") into " << ears.size()
              << " triangles\n";

    return 0;
}
