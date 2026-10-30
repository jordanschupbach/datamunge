#pragma once
// Metadata-only adapter to canonical tensor assignment.
#include "datamunge/nn/tensor_assignment.hpp"
namespace hungarian {
inline std::vector<int> HungarianAssignment(const std::vector<std::vector<double>>& cost){const auto R=std::int64_t(cost.size()),C=R?std::int64_t(cost[0].size()):0;std::vector<double> v;for(const auto& row:cost){if(std::int64_t(row.size())!=C)throw std::invalid_argument("ragged assignment matrix");v.insert(v.end(),row.begin(),row.end());}return datamunge::nn::models::TensorHungarianAssignment(datamunge::nn::from_vector(v,{R,C})).to_vector<int>();}
}
