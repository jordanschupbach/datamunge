#pragma once
#include "datamunge/nn/segnet_models.hpp"
namespace segnet {
namespace nn = datamunge::nn;
struct IndexedPool {
  nn::Tensor values;
  std::vector<std::int64_t> indices;
  nn::Shape shape;
};
inline IndexedPool PoolWithIndices(const nn::Tensor &input) {
  const auto r = nn::models::SegNetPoolWithIndices(input);
  return {r.values, r.indices.to_vector<std::int64_t>(), r.shape};
}
inline nn::Tensor IndexUnpool(const nn::Tensor &values, const IndexedPool &record) {
  return nn::models::SegNetIndexUnpool(
      values, {record.values, nn::from_vector(record.indices, values.shape()), record.shape});
}
using nn::models::SegNetConfig;
struct SegNetOutputs {
  nn::Tensor logits;
  std::vector<IndexedPool> pooling;
  std::vector<nn::Tensor> unpooled;
};
inline SegNetOutputs SegNet(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &images, const SegNetConfig &cfg = {},
                            bool training = false) {
  const auto r = nn::models::SegNet(p, state, images, cfg, training);
  SegNetOutputs out;
  out.logits = r.logits;
  out.unpooled = r.unpooled;
  for (const auto &pool : r.pooling)
    out.pooling.push_back({pool.values, pool.indices.to_vector<std::int64_t>(), pool.shape});
  return out;
}
} // namespace segnet
