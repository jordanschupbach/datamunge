/// @file convert.cpp
/// @brief Copy-based conversion between autodiff tensors and the general linear-algebra tensor type.

#include "datamunge/nn/convert.hpp"

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

Tensor from_linalg(const linalg::Tensor& tensor) {
  Shape shape;
  for (auto d : tensor.shape()) shape.push_back(static_cast<std::int64_t>(d));
  const std::int64_t n = shape_numel(shape);
  switch (tensor.dtype()) {
    case linalg::TensorDType::Float64: {
      Tensor  out = empty(shape, DType::Float64);
      double* p   = out.data_ptr<double>();
      for (std::int64_t i = 0; i < n; ++i) p[i] = tensor.at_flat(static_cast<std::size_t>(i));
      return out;
    }
    case linalg::TensorDType::Bool: {
      Tensor out = empty(shape, DType::Bool);
      bool*  p   = out.data_ptr<bool>();
      for (std::int64_t i = 0; i < n; ++i) p[i] = tensor.at_flat(static_cast<std::size_t>(i)) != 0.0;
      return out;
    }
    case linalg::TensorDType::String: break;
  }
  throw std::invalid_argument("from_linalg: string tensors cannot be converted");
}

linalg::Tensor to_linalg(const Tensor& tensor) {
  std::vector<std::size_t> shape;
  for (auto d : tensor.shape()) shape.push_back(static_cast<std::size_t>(d));
  if (tensor.dtype() == DType::Bool) {
    const auto       v = tensor.to_vector<int>();
    return linalg::Tensor::from_bool_values(shape, std::vector<int>(v.begin(), v.end()));
  }
  return linalg::Tensor::from_values(shape, tensor.to_vector<double>());
}

}  // namespace datamunge::nn
