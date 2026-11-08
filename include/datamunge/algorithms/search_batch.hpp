#pragma once
#include <datamunge/algorithms/breadth_first_search.hpp>
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
UnweightedSearchResult backtracking_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,bool directed=true);
UnweightedSearchResult best_first_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,const std::vector<double>& heuristic,bool directed=true);
UnweightedSearchResult beam_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,const std::vector<double>& heuristic,std::size_t width,bool directed=true);
UnweightedSearchResult beam_stack_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal,const std::vector<double>& heuristic,std::size_t width,bool directed=true);
UnweightedSearchResult bidirectional_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t start,std::size_t goal);
}
