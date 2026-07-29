#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
struct BasicPathResult{std::vector<std::size_t> path;std::size_t expanded{0};bool found{false};};
BasicPathResult brute_force_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal);
BasicPathResult depth_first_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal);
BasicPathResult iterative_deepening_dfs(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,std::size_t max_depth);
struct GridPathResult{std::vector<std::pair<int,int>> path;double cost{0};bool found{false};};
GridPathResult d_star_replan(const std::vector<std::vector<bool>>& blocked,std::pair<int,int> start,std::pair<int,int> goal);
GridPathResult jump_point_search(const std::vector<std::vector<bool>>& blocked,std::pair<int,int> start,std::pair<int,int> goal);
struct PlanningAction{std::string name;std::uint64_t require{0},add{0},remove{0};};
struct PlanningResult{std::vector<std::string> actions;std::uint64_t final_state{0};bool found{false};};
PlanningResult general_problem_solver(std::uint64_t initial,std::uint64_t goals,const std::vector<PlanningAction>& actions,std::size_t max_steps=32);
}
