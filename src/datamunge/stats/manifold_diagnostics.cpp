/// @file manifold_diagnostics.cpp
/// @brief Validated raw-metric scales and aligned centroid summaries.
#include <datamunge/stats/manifold_diagnostics.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Stateless validation helpers.
namespace {
/// @brief Validates a nonempty finite matrix.
/// @param x Borrowed matrix.
/// @throws std::invalid_argument Empty shape or nonfinite entry.
void validate(const linalg::DenseMatrix<double>& x){
    if(!x.rows() || !x.cols())throw std::invalid_argument("manifold diagnostics: empty matrix");
    for(std::size_t i=0;i<x.rows();++i)for(std::size_t j=0;j<x.cols();++j)
        if(!std::isfinite(x(i,j)))throw std::invalid_argument("manifold diagnostics: nonfinite coordinate");
}
/// @brief Converts a nonnegative long-double sum to finite double.
/// @param x Intermediate sum.
/// @return Representable double.
/// @throws std::overflow_error Nonfinite or too large.
double checked(long double x){
    if(!std::isfinite(x) || x>std::numeric_limits<double>::max())throw std::overflow_error("manifold diagnostics: squared sum overflow");
    return static_cast<double>(x);
}
}
ManifoldDistanceScales manifold_distance_scales(const linalg::DenseMatrix<double>& x,std::size_t neighbor){
    validate(x);const auto n=x.rows();
    if(!neighbor || neighbor>=n)throw std::invalid_argument("manifold diagnostics: neighbor must be in [1,n)");
    if(n>std::vector<double>().max_size()/(n-1))throw std::length_error("manifold diagnostics: pair capacity");
    std::vector<double> pairs;pairs.reserve(n*(n-1)/2);
    std::vector<double> kth(n);long double total=0;
    for(std::size_t i=0;i<n;++i){
        std::vector<double> distances;distances.reserve(n-1);
        for(std::size_t j=0;j<n;++j){if(i==j)continue;
            long double distance=0;
            for(std::size_t c=0;c<x.cols();++c){const long double delta=static_cast<long double>(x(i,c))-x(j,c);distance+=delta*delta;}
            const double d=checked(distance);distances.push_back(d);if(i<j){pairs.push_back(d);total+=d;}
        }
        std::nth_element(distances.begin(),distances.begin()+neighbor-1,distances.end());kth[i]=distances[neighbor-1];
    }
    const auto mid=pairs.size()/2;std::nth_element(pairs.begin(),pairs.begin()+mid,pairs.end());
    std::nth_element(kth.begin(),kth.begin()+n/2,kth.end());
    return {pairs[mid],checked(total/pairs.size()),kth[n/2]};
}
EmbeddingSeparation embedding_separation(const linalg::DenseMatrix<double>& x,const std::vector<std::string>& labels){
    validate(x);if(labels.size()!=x.rows())throw std::invalid_argument("embedding separation: label alignment");
    std::map<std::string,std::vector<long double>> centers;std::map<std::string,std::size_t> counts;
    std::vector<long double> grand(x.cols(),0);
    for(std::size_t i=0;i<x.rows();++i){auto& c=centers[labels[i]];if(c.empty())c.resize(x.cols(),0);++counts[labels[i]];
        for(std::size_t j=0;j<x.cols();++j){c[j]+=x(i,j);grand[j]+=x(i,j);}}
    for(auto& [label,c]:centers)for(auto& v:c)v/=counts.at(label);
    for(auto& v:grand)v/=x.rows();long double within=0,between=0;
    for(std::size_t i=0;i<x.rows();++i)for(std::size_t j=0;j<x.cols();++j){const auto delta=x(i,j)-centers.at(labels[i])[j];within+=delta*delta;}
    for(const auto& [label,c]:centers)for(std::size_t j=0;j<x.cols();++j){const auto delta=c[j]-grand[j];between+=counts.at(label)*delta*delta;}
    const double w=checked(within),b=checked(between);
    return {w,b,w ? b/w : (b ? std::numeric_limits<double>::infinity():std::numeric_limits<double>::quiet_NaN())};
}
}
