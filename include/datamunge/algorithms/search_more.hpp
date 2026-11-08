#pragma once
#include <datamunge/algorithms/general_problem_solver.hpp>
#include <datamunge/algorithms/basic_path_result.hpp>
#include <datamunge/algorithms/d_star.hpp>
#include <datamunge/algorithms/depth_first_search.hpp>
#include <datamunge/algorithms/brute_force_search.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
BasicPathResult iterative_deepening_dfs(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,std::size_t max_depth);
GridPathResult jump_point_search(const std::vector<std::vector<bool>>& blocked,std::pair<int,int> start,std::pair<int,int> goal);

}
