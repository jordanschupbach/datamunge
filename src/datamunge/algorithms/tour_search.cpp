#include <datamunge/algorithms/tour_search.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
namespace datamunge::algorithms {
namespace {
void matrix_ok(const std::vector<std::vector<double>>&d){for(auto&r:d)if(r.size()!=d.size())throw std::invalid_argument("TSP: distance matrix must be square");for(std::size_t i=0;i<d.size();++i)for(std::size_t j=0;j<d.size();++j)if(d[i][j]<0||!std::isfinite(d[i][j]))throw std::invalid_argument("TSP: invalid distance");}
double length(const std::vector<std::size_t>&t,const std::vector<std::vector<double>>&d){double x=0;for(std::size_t i=1;i<t.size();++i)x+=d[t[i-1]][t[i]];return x;}

}
TourResult nearest_neighbor_tsp(const std::vector<std::vector<double>>&d,std::size_t s){matrix_ok(d);if(d.empty())return{};if(s>=d.size())throw std::invalid_argument("nearest_neighbor_tsp: start");std::vector<bool>used(d.size());TourResult r;r.tour={s};used[s]=true;while(r.tour.size()<d.size()){auto u=r.tour.back(),best=d.size();for(std::size_t v=0;v<d.size();++v)if(!used[v]&&(best==d.size()||d[u][v]<d[u][best]))best=v;used[best]=true;r.tour.push_back(best);}r.tour.push_back(s);r.cost=length(r.tour,d);return r;}


}
