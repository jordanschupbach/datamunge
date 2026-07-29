#pragma once
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
struct TourResult{std::vector<std::size_t> tour;double cost{0};};
TourResult nearest_neighbor_tsp(const std::vector<std::vector<double>>& distance,std::size_t start=0);
TourResult christofides_tsp(const std::vector<std::vector<double>>& distance,std::size_t start=0);
struct VehicleRoutes{std::vector<std::vector<std::size_t>> routes;double total_distance{0};};
VehicleRoutes clarke_wright_savings(const std::vector<std::pair<double,double>>& points,
 const std::vector<double>& demand,double capacity,std::size_t depot=0);
std::vector<std::pair<int,int>> warnsdorff_knight_tour(int rows,int columns,int start_row=0,int start_column=0);
struct SearchResult{double cost{0};std::vector<std::size_t> path;std::size_t expanded{0};bool found{false};};
SearchResult a_star_search(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
 std::size_t start,std::size_t goal,const std::vector<double>& heuristic);
SearchResult b_star_search(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
 std::size_t start,const std::vector<std::size_t>& goals,const std::vector<double>& goal_set_lower_bound);
}
