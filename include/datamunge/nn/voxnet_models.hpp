#pragma once
/// @file voxnet_models.hpp
/// @brief Full-width occupancy CNN,explicit regularization and tensor voxel descriptors.
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/training.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Rotate an occupancy tensor around its third spatial axis by quarter turns.
///
/// Index-only geometric rotation;four applications return the original tensor exactly.
///
/// @param x NDHWC occupancy tensor,channel count arbitrary.
/// @param turn Integer0..3,each rotates axes1/2 by transpose then reverse axis1.
/// @return Live tensor graph with the rotated first two spatial axes.
/// @throws std::invalid_argument For rank other than5 or turn outside0..3.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::VoxnetRotate(n::ones({1, 32, 32, 32, 1}), 1);
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
inline Tensor VoxnetRotate(Tensor x, int turn) {
  if (turn < 0 || turn > 3 || x.dim() != 5)
    throw std::invalid_argument("quarter turn0..3");
  for (int i = 0; i < turn; ++i)
    x = nn::reverse(nn::transpose(x, {0, 2, 1, 3, 4}), {1});
  return x;
}
/// @brief Add the exact coupled L2 derivative to every named weight gradient.
///
/// No optimizer update or hidden averaging. Penalty is lambda*sum(W squared),
/// not decoupled weight decay;bias gradients are unchanged.
///
/// @param p Learned store,unchanged by this helper.
/// @param ce One categorical-loss gradient per p.parameters() in store order.
/// @param lambda Finite nonnegative sum-of-squares coefficient,default.001.
/// @return New tensor gradient handles,ce+2*lambda*W for names ending/W;other leaves unchanged.
/// @throws std::invalid_argument For gradient count mismatch,negative/nonfinite coefficient or tensor incompatibility.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p;
/// p.get("head/W", {2, 2}, n::init::ones());
/// const auto gs = m::VoxnetRegularized(p, {n::zeros({2, 2})});
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
inline std::vector<Tensor> VoxnetRegularized(const nn::ParameterStore &p,
                                             const std::vector<Tensor> &ce,
                                             double lambda = .001) {
  if (ce.size() != p.size() || !std::isfinite(lambda) || lambda < 0)
    throw std::invalid_argument("VoxNet regularized gradients");
  auto out = ce;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const auto &name = p.names()[i];
    if (name.size() >= 2 && name.substr(name.size() - 2) == "/W")
      out[i] = ce[i] + p.at(name) * (2 * lambda);
  }
  return out;
}
/// @brief Own all six complete VoxNet forward endpoints.
///
/// Owning tensor handles retain their graph/storage. All widths are original;
/// only the categorical head changes from40to3in the prepared teaching data.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(907);
/// const auto out = m::Voxnet(p, n::zeros({1, 32, 32, 32, 1}), 3);
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
struct VoxnetOutput {
  Tensor conv1;  ///< First LeakyReLU activation[N,14,14,14,32],before dropout.
  Tensor conv2;  ///< Second LeakyReLU activation[N,12,12,12,32].
  Tensor pool;   ///< Spatial2/stride2 max pool[N,6,6,6,32],before dropout.
  Tensor flat;   ///< Flattened post-pool/dropout features[N,6912].
  Tensor fc1;    ///< ReLU dense features[N,128],before head dropout.
  Tensor logits; ///< Unnormalized categorical scores[N,classes].
};
/// @brief Own the three independent mutable seeded VoxNet dropout streams.
///
/// Seeds881/883/887 reproduce source stream order. Training consumes each stream
/// once per forward;inference consumes none. Synchronize concurrent access.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VoxnetDropout dropout;
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
struct VoxnetDropout {
  nn::Generator first{881};  ///< Mutable first convolution dropout stream,rate.2.
  nn::Generator second{883}; ///< Mutable pooled dropout stream,rate.3.
  nn::Generator last{887};   ///< Mutable hidden dropout stream,rate.4.
};
/// @brief Classify binary32cube occupancy with the complete original-width VoxNet graph.
///
/// Input occupancy*2-1;VALID5/stride2/32LeakyReLU.1,drop.2,
/// VALID3/stride1/32LeakyReLU.1,maxpool2,drop.3,flatten6912,
/// FC128ReLU,drop.4,linearhead. ConvolutionsHeNormal,denseNormal.01,bias0.
/// No BN,hidden fitting,L2 update or rotation averaging. Caller explicitly applies
/// CE+lambda sum(W squared),tensor MomentumSGD and four-rotated-LOGIT inference.
///
/// @param p Mutable Float32 store;warm all8leaves before constructing a tensor optimizer.
/// @param occupancy Float32[N,32,32,32,1],Npositive,normally binary0/1.
/// @param classes Categorical count>=2,default40;prepared customsubset uses3.
/// @param training Apply dropout.2/.3/.4iftrue,otherwise deterministic.
/// @param dropout Required mutable three-stream state for training,unused at inference.
/// @return Raw logits and all six convolution/pool/dense endpoints.
/// @throws std::invalid_argument For invalid shape/dtype/classes,missing training dropout or incompatible stored tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(907);
/// const auto out = m::Voxnet(p, n::zeros({1, 32, 32, 32, 1}), 3);
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
inline VoxnetOutput Voxnet(nn::ParameterStore &p, const Tensor &occupancy, int classes = 40,
                           bool training = false, VoxnetDropout *dropout = nullptr) {
  if (classes < 2 || (training && !dropout) || occupancy.dim() != 5 ||
      occupancy.shape() != nn::Shape{occupancy.size(0), 32, 32, 32, 1} ||
      occupancy.size(0) < 1 || occupancy.dtype() != nn::DType::Float32 ||
      p.dtype() != occupancy.dtype())
    throw std::invalid_argument("VoxNet configuration");
  VoxnetOutput out;
  nn::ConvLayerOptions o;
  o.padding = nn::Padding::Valid;
  o.stride = 2;
  o.init = nn::init::he_normal();
  out.conv1 = nn::leaky_relu(nn::conv3d(p, "conv1", occupancy * 2. - 1., 32, 5, o), .1);
  auto x = out.conv1;
  if (training)
    x = nn::dropout(x, .2, true, dropout->first);
  o.stride = 1;
  out.conv2 = nn::leaky_relu(nn::conv3d(p, "conv2", x, 32, 3, o), .1);
  out.pool = nn::max_pool3d(out.conv2, 2, 2, nn::Padding::Valid);
  x = out.pool;
  if (training)
    x = nn::dropout(x, .3, true, dropout->second);
  out.flat = nn::reshape(x, {x.size(0), 6912});
  nn::DenseOptions dense;
  dense.init = nn::init::normal(.01);
  dense.activation = nn::Activation::Relu;
  out.fc1 = nn::dense(p, "fc1", out.flat, 128, dense);
  x = out.fc1;
  if (training)
    x = nn::dropout(x, .4, true, dropout->last);
  dense.activation = nn::Activation::None;
  out.logits = nn::dense(p, "head", x, classes, dense);
  return out;
}

/// @brief Rasterize normalized surface points into binary32cube occupancy with an empty border.
///
/// Map floor((xyz+1)*14.5)+1,clamp1..30,then tensor segment-sum and threshold.
/// This is sampled-SURFACE occupancy,not solid filling or triangle intersection.
/// It has no learnable parameters and is nondifferentiable through discrete bins.
///
/// @param points Float32[N,P,3],N/Ppositive,finite coordinates within[-1,1].
/// @return Float32[N,32,32,32,1]binary counts>0,indices1..30 only.
/// @throws std::invalid_argument For invalid rank,dtype,empty axes,nonfinite or out-of-range coordinates.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto occupancy = m::VoxnetOccupancy(n::zeros({1, 8, 3}));
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
inline Tensor VoxnetOccupancy(const Tensor &points) {
  if (points.dim() != 3 || points.size(0) < 1 || points.size(1) < 1 || points.size(2) != 3 ||
      points.dtype() != nn::DType::Float32 || !nn::all(nn::is_finite(points)).item<bool>() ||
      nn::max(nn::abs(points)).item() > 1.00001)
    throw std::invalid_argument("finite normalized Float32[N,P,3]required");
  const auto bins =
      nn::clip(nn::floor((points.detach() + 1.) * 14.5) + 1., 1., 30.).to(nn::DType::Int64);
  const auto local =
      nn::select(bins, 2, 0) * 1024 + nn::select(bins, 2, 1) * 32 + nn::select(bins, 2, 2);
  const auto offsets =
      nn::arange(0, points.size(0), 1, nn::DType::Int64).reshape({points.size(0), 1}) * 32768;
  const auto counts =
      nn::unsorted_segment_sum(nn::ones({points.size(0) * points.size(1)}),
                               (local + offsets).reshape({-1}), points.size(0) * 32768);
  return nn::greater(counts, 0.)
      .to(nn::DType::Float32)
      .reshape({points.size(0), 32, 32, 32, 1});
}
/// @brief Internal local-archive and prepared-occupancy validation utilities.
namespace voxel_detail {
/// @brief Reject an invalid prepared occupancy contract.
/// @param ok Required condition.
/// @param message Failure context.
inline void require_voxel(bool ok, const std::string &message) {
  if (!ok)
    throw std::invalid_argument(message);
}
/// @brief Index an archive by its original names.
/// @param values Owned archive entries.
/// @return Tensor map retaining all arrays.
inline std::map<std::string, Tensor> archive_map(const nn::NamedTensors &values) {
  return {values.begin(), values.end()};
}
} // namespace voxel_detail
using voxel_detail::archive_map;
using voxel_detail::require_voxel;
/// @brief Own aligned genuine CAD occupancy,labels and source metadata.
///
/// No fitting or automatic network access. Source IDs identify distinct original
/// CAD objects;the prepared balanced subset is custom,not a ModelNet benchmark.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::VoxnetRows({n::zeros({2, 32, 32, 32, 1}), n::zeros({2}, n::DType::Int32),
///                               n::arange(0, 2, 1, n::DType::Int32)},
///                              {1});
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
struct VoxnetData {
  Tensor voxels; ///< Binary Float32[N,32,32,32,1] sampled-surface occupancy.
  Tensor labels; ///< Int32[N]chair0/sofa1/table2.
  Tensor source; ///< Int32[N]unique original object ID across all splits.
};
/// @brief Gather aligned occupancy rows and metadata using tensor indexing.
///
/// Preserves repeated row order;source tensors unchanged.
///
/// @param d Source occupancy/label/ID handles.
/// @param rows Host row-index metadata in requested order.
/// @return Owning gathered aligned tensor handles.
/// @throws std::invalid_argument For invalid row indices or incompatible input axes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::VoxnetRows({n::zeros({2, 32, 32, 32, 1}), n::zeros({2}, n::DType::Int32),
///                               n::arange(0, 2, 1, n::DType::Int32)},
///                              {1});
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
inline VoxnetData VoxnetRows(const VoxnetData &d, const std::vector<std::int64_t> &rows) {
  return {nn::take_rows(d.voxels, rows), nn::take_rows(d.labels, rows),
          nn::take_rows(d.source, rows)};
}
/// @brief Load and validate all three pinned genuine-CAD occupancy pools.
///
/// RequiresFloat32binary32cube,empty one-voxel border,nonempty/nonfull objects,
/// Int32aligned labels/IDs,three balanced classes and unique IDs/voxelbytes
/// across every split. Reads local archive only;never downloads or fits.
///
/// @param root Directory containing locally prepared voxnet.npz.
/// @return Fitting24,validation6,held-out12balanced chair/sofa/table object pools.
/// @throws std::invalid_argument For missing arrays/files,invalid shapes/dtypes/values/balance or duplicate objects.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models; // Prepare with datamunge-prepare-data voxel before loading.
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
inline std::vector<VoxnetData> LoadVoxnet(const std::filesystem::path &root) {
  const auto arrays = archive_map(nn::load_npz((root / "voxnet.npz").string()));
  std::vector<VoxnetData> pools;
  std::set<int> ids;
  std::set<std::string> hashes;
  for (const std::string split : {"fit", "validation", "test"}) {
    VoxnetData d{arrays.at(split + "_voxels"), arrays.at(split + "_label"),
                 arrays.at(split + "_source_id")};
    const auto n = d.voxels.size(0);
    require_voxel(n == (split == "fit"          ? 24
                        : split == "validation" ? 6
                                                : 12) &&
                      d.voxels.dtype() == nn::DType::Float32 &&
                      d.voxels.shape() == nn::Shape({n, 32, 32, 32, 1}),
                  "published 32-cube occupancy/object count");
    require_voxel(d.labels.dtype() == nn::DType::Int32 &&
                      d.source.dtype() == nn::DType::Int32 &&
                      d.labels.shape() == nn::Shape({n}) && d.source.shape() == nn::Shape({n}),
                  "voxel metadata alignment");
    std::vector<int> counts(3);
    const auto labels = d.labels.to_vector<int>(), source = d.source.to_vector<int>();
    const auto values = d.voxels.to_vector<float>();
    for (int i = 0; i < n; ++i) {
      require_voxel(labels[i] >= 0 && labels[i] < 3 && ids.insert(source[i]).second,
                    "disjoint held-out object IDs");
      ++counts[labels[i]];
      double occupied = 0;
      for (int x = 0; x < 32; ++x)
        for (int y = 0; y < 32; ++y)
          for (int z = 0; z < 32; ++z) {
            float v = values[i * 32768 + (x * 32 + y) * 32 + z];
            require_voxel(v == 0 || v == 1, "binary occupancy");
            if (x == 0 || x == 31 || y == 0 || y == 31 || z == 0 || z == 31)
              require_voxel(v == 0, "empty one-voxel border");
            occupied += v;
          }
      require_voxel(occupied > 0 && occupied < 27000, "nonempty genuine CAD occupancy");
      require_voxel(
          hashes
              .insert(std::string(reinterpret_cast<const char *>(values.data() + i * 32768),
                                  32768 * sizeof(float)))
              .second,
          "no duplicate occupancy across objects/splits");
    }
    for (int count : counts)
      require_voxel(count == n / 3, "balanced pinned classes");
    std::cout << "ModelNet10 " << split << ":" << n
              << " distinct objects,balanced chair/sofa/table\n";
    pools.push_back(d);
  }
  return pools;
}

/// @brief Compute four-rotation averaged4cube-block occupancy descriptors with tensors.
///
/// Each8cube block mean contributes1/4perrotation. No learned parameters.
/// The four discrete rotations use exactly the same axes as model voting.
///
/// @param d Aligned occupancy data.
/// @return Float32[N,64]features,rotation-averaged in source accumulation order.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VoxnetData d{n::zeros({1, 32, 32, 32, 1}), {}, {}};
/// const auto f = m::VoxnetFeatures(d);
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
inline Tensor VoxnetFeatures(const VoxnetData &d) {
  auto out = nn::zeros({d.voxels.size(0), 64});
  for (int k = 0; k < 4; ++k) {
    const auto x = VoxnetRotate(d.voxels, k).reshape({d.voxels.size(0), 4, 8, 4, 8, 4, 8});
    const auto f = (nn::sum(x.to(nn::DType::Float64), {2, 4, 6}) / 2048.)
                       .reshape({d.voxels.size(0), 64})
                       .to(nn::DType::Float32);
    out = out + f;
  }
  return out;
}

/// @brief Predict classes by fitting-only standardized retained-memory voxel1NN.
///
/// Parameter-free retained-memory method,no optimizer. Fit-feature population SD
/// usesFloat64/clamp1e-6;source Float32 pairwise subtraction occurs before
/// Float64standardization/squared distance. Allnumericoperations use tensors.
///
/// @param fit Genuine fitting occupancy/labels.
/// @param test Query occupancy;query labels and IDs unused.
/// @return Float32[Nquery,3]one-hot selected-neighbor scores,first fitting row wins ties.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::VoxnetData d{n::zeros({2, 32, 32, 32, 1}), n::zeros({2}, n::DType::Int32), {}};
/// const auto z = m::VoxnetNearest(d, d);
/// @endcode
/// @par References
/// - Maturana and Scherer(2015),VoxNet:A3DConvolutional Neural Network for Real-Time Object Recognition, https://doi.org/10.1109/IROS.2015.7353481. Source full-width32cube graph,LeakyReLU.1,three dropout rates and four-logit rotation vote. Surface-point occupancy preparation is an explicitly declared teaching adaptation,not the published voxelization benchmark.
/// @see Voxnet(),VoxnetFeatures()
inline Tensor VoxnetNearest(const VoxnetData &fit, const VoxnetData &test) {
  const auto a = VoxnetFeatures(fit), b = VoxnetFeatures(test);
  const auto wide = a.to(nn::DType::Float64);
  const auto scale =
      nn::maximum(nn::sqrt(nn::mean(nn::square(wide - nn::mean(wide, {0})), {0})),
                  nn::full({64}, 1e-6, nn::DType::Float64));
  const auto delta =
      (nn::expand_dims(a, 0) - nn::expand_dims(b, 1)).to(nn::DType::Float64) / scale;
  const auto ids = nn::argmin(nn::sum(nn::square(delta), {2}), 1);
  return nn::one_hot(nn::gather(fit.labels, ids, 0), 3);
}
} // namespace datamunge::nn::models
