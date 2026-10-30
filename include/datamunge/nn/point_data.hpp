#pragma once
/// @file point_data.hpp
/// @brief Genuine sampled ModelNet surface ingress with tensor validity and source-disjoint metadata checks.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/training.hpp"
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own genuine sampled CAD surface XYZ and aligned object identities.
///
/// Tensor handles own storage. Compact observations are first256of1024actual
/// area-sampled mesh points,not generated abstract primitives. Numerical
/// area/barycentric sampling and normalization are reproducible in preparation.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadPoints("build/cnn-datasets/modelnet");
/// @endcode
/// @par References
/// - Wu et al.(2015),3DShapeNets:A Deep Representation for Volumetric Shapes, https://arxiv.org/abs/1406.5670;official ModelNet10 source https://modelnet.cs.princeton.edu/. Source teaching subset chair/sofa/table uses42originalOFFmeshes,area-weighted sampled1024points,first256forcompact fits;custom selection/normalization,not full ModelNet40benchmark.
/// @see LoadPoints()
struct PointData {
  Tensor xyz;    ///< Float32[N,256,3],first256source surface samples in unit ball.
  Tensor labels; ///< Int32[N],chair0/sofa1/table2.
  Tensor source; ///< Int32[N],unique original OFF mesh identity.
};
/// @brief Select aligned CAD observations by metadata row indices.
///
/// No resampling,normalization fitting or metadata reordering.
///
/// @param d Point container.
/// @param rows Bounded row identities,duplicates allowed.
/// @return Gathered owning XYZ/label/source handles.
/// @throws std::invalid_argument For invalid row indices.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadPoints("build/cnn-datasets/modelnet");
/// const auto d = m::PointRows(pools[0], {0, 1});
/// @endcode
/// @par References
/// - Wu et al.(2015),3DShapeNets:A Deep Representation for Volumetric Shapes, https://arxiv.org/abs/1406.5670;official ModelNet10 source https://modelnet.cs.princeton.edu/. Source teaching subset chair/sofa/table uses42originalOFFmeshes,area-weighted sampled1024points,first256forcompact fits;custom selection/normalization,not full ModelNet40benchmark.
/// @see LoadPoints()
inline PointData PointRows(const PointData &d, const std::vector<std::int64_t> &rows) {
  return {nn::take_rows(d.xyz, rows), nn::take_rows(d.labels, rows),
          nn::take_rows(d.source, rows)};
}
/// @brief Load genuine unit-ball mesh points with disjoint original object identities.
///
/// Require sourceFloat32[N,1024,3]finite XYZ with Euclidean norm<=1.00001,
/// then keepfirst256points without relabeling. Original Int32label/source shapes
/// and all distinct source identities checked;per-class counts are tensors.
/// No download,new points,learned transform or held-out fitting. Disjoint object
/// IDs do not certify absence of near-duplicate CAD geometry.
///
/// @param root Directory containing point_clouds.npz and original meshes/sampling manifest.
/// @return 24fit/6validation/12test balanced three-class compact point pools.
/// @throws std::invalid_argument For missing arrays,invalid shape/dtype/range,class balance or source leakage.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadPoints("build/cnn-datasets/modelnet");
/// @endcode
/// @par References
/// - Wu et al.(2015),3DShapeNets:A Deep Representation for Volumetric Shapes, https://arxiv.org/abs/1406.5670;official ModelNet10 source https://modelnet.cs.princeton.edu/. Source teaching subset chair/sofa/table uses42originalOFFmeshes,area-weighted sampled1024points,first256forcompact fits;custom selection/normalization,not full ModelNet40benchmark.
/// @see LoadPoints()
inline std::vector<PointData> LoadPoints(const std::filesystem::path &root) {
  nn::NoGradGuard guard;
  std::map<std::string, Tensor> a;
  for (const auto &[key, value] : nn::load_npz((root / "point_clouds.npz").string()))
    if (!a.emplace(key, value).second)
      throw std::invalid_argument("unique mesh archive keys required");
  std::vector<PointData> pools;
  std::set<int> ids;
  for (const std::string split : {"fit", "validation", "test"}) {
    const int n = split == "fit" ? 24 : split == "validation" ? 6 : 12;
    const auto full = a.at(split + "_xyz");
    if (full.dtype() != nn::DType::Float32 || full.shape() != nn::Shape{n, 1024, 3} ||
        !nn::all(nn::is_finite(full)).item<bool>() ||
        !nn::all(nn::less_equal(
                     nn::sqrt(nn::sum(nn::square(full.to(nn::DType::Float64)), {2})), 1.00001))
             .item<bool>())
      throw std::invalid_argument("finite original unit-ball1024point surfaces required");
    PointData d{nn::slice_axis(full, 1, 0, 256), a.at(split + "_label"),
                a.at(split + "_source_id")};
    if (d.labels.dtype() != nn::DType::Int32 || d.source.dtype() != nn::DType::Int32 ||
        d.labels.shape() != nn::Shape{n} || d.source.shape() != nn::Shape{n} ||
        !nn::all(nn::logical_and(nn::greater_equal(d.labels, 0.), nn::less(d.labels, 3.)))
             .item<bool>())
      throw std::invalid_argument("aligned bounded CAD label/source metadata required");
    for (int id : d.source.to_vector<int>())
      if (id < 0 || !ids.insert(id).second)
        throw std::invalid_argument("unique disjoint original mesh IDs required");
    if (!nn::all(nn::equal(nn::sum(nn::one_hot(d.labels, 3), {0}), n / 3.)).item<bool>())
      throw std::invalid_argument("balanced selected original CAD classes required");
    pools.push_back(d);
  }
  return pools;
}
} // namespace datamunge::nn::models
