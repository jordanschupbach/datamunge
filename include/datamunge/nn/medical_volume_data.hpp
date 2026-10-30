#pragma once
/// @file medical_volume_data.hpp
/// @brief Genuine official CT array ingress,identity checks and tensor block-mean memory references.
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
/// @brief Own genuine CT intensity cubes,original binary labels and released array identities.
///
/// Source0benign malignancy ratings1/2,source1malignant ratings4/5;rating3
/// excluded by dataset creators. Released array records do not certify patient
/// independence;no clinical-use claim. Tensor handles own their storage.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadMedical("build/cnn-datasets/medical_volume");
/// const auto fit = pools[0];
/// @endcode
/// @par References
/// - Yang et al.(2023),MedMNIST v2:A large-scale lightweight benchmark for2Dand3Dbiomedical image classification, https://doi.org/10.1038/s41597-022-01721-8;corrected NoduleMNIST3D v2.1 archive https://zenodo.org/records/6496656,official original label/split description https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py. The three-stage residual model is a custom source teaching CNN,not a named published medical architecture or clinical predictor.
/// @see MedicalVolume()
struct MedicalData {
  Tensor voxels; ///< Float32[N,28,28,28,1] source byte/255 intensities.
  Tensor labels; ///< Int32[N],sourcebenign0/malignant1.
  Tensor source; ///< Int32[N],official array offset rowID0..1632.

  /// @brief Return released CT record count.
  /// @return Voxel tensor leading axis.
  std::int64_t size() const {
    return voxels.size(0);
  }
};
/// @brief Gather original CT rows with labels and released source IDs aligned.
///
/// No normalization fitting,augmentation or synthetic voxels.
///
/// @param d CT container.
/// @param rows Bounded metadata indices,duplicates allowed.
/// @return Gathered original intensity/label/source handles.
/// @throws std::invalid_argument For invalid row indices.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadMedical("build/cnn-datasets/medical_volume");
/// const auto batch = m::MedicalRows(pools[0], {0, 1});
/// @endcode
/// @par References
/// - Yang et al.(2023),MedMNIST v2:A large-scale lightweight benchmark for2Dand3Dbiomedical image classification, https://doi.org/10.1038/s41597-022-01721-8;corrected NoduleMNIST3D v2.1 archive https://zenodo.org/records/6496656,official original label/split description https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py. The three-stage residual model is a custom source teaching CNN,not a named published medical architecture or clinical predictor.
/// @see MedicalVolume()
inline MedicalData MedicalRows(const MedicalData &d, const std::vector<std::int64_t> &rows) {
  return {nn::take_rows(d.voxels, rows), nn::take_rows(d.labels, rows),
          nn::take_rows(d.source, rows)};
}
/// @brief Load genuine balanced official-split CT records with tensor intensity scaling.
///
/// Require integral bytes0..255,official source-ID ranges0..1157/1158..1322/
/// 1323..1632,balanced binary labels,unique records and no byte-identical
/// selected cubes. Duplicate serialization checks are identity metadata only;
/// all range checks and/255scaling use tensors. No auto-download,clinical
/// interpretation or population-moment fitting.
///
/// @param root Directory containing medical_volume.npz and original archive/manifest.
/// @return 64fit/16validation/64testFloat32intensity cubes and aligned original metadata.
/// @throws std::invalid_argument For missing/duplicate array keys,invalid shapes/dtypes/ranges,source leakage or byte-identical cubes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadMedical("build/cnn-datasets/medical_volume");
/// @endcode
/// @par References
/// - Yang et al.(2023),MedMNIST v2:A large-scale lightweight benchmark for2Dand3Dbiomedical image classification, https://doi.org/10.1038/s41597-022-01721-8;corrected NoduleMNIST3D v2.1 archive https://zenodo.org/records/6496656,official original label/split description https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py. The three-stage residual model is a custom source teaching CNN,not a named published medical architecture or clinical predictor.
/// @see MedicalVolume()
inline std::vector<MedicalData> LoadMedical(const std::filesystem::path &root) {
  nn::NoGradGuard guard;
  std::map<std::string, Tensor> a;
  for (const auto &[key, value] : nn::load_npz((root / "medical_volume.npz").string()))
    if (!a.emplace(key, value).second)
      throw std::invalid_argument("unique CT array names required");
  std::vector<MedicalData> pools;
  std::set<int> ids;
  std::set<std::vector<int>> unique;
  for (const std::string split : {"fit", "validation", "test"}) {
    const int n = split == "validation" ? 16 : 64;
    const auto bytes = a.at(split + "_voxels");
    if (bytes.shape() != nn::Shape{n, 28, 28, 28, 1} ||
        !nn::all(nn::is_finite(bytes)).item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(bytes, 0.), nn::less_equal(bytes, 255.)))
             .item<bool>() ||
        !nn::all(nn::equal(bytes, bytes.to(nn::DType::Int32))).item<bool>())
      throw std::invalid_argument("integral bounded original28cube CT intensities required");
    MedicalData d{bytes.to(nn::DType::Float32) / 255., a.at(split + "_labels"),
                  a.at(split + "_source_ids")};
    if (d.labels.shape() != nn::Shape{n} || d.source.shape() != nn::Shape{n} ||
        d.labels.dtype() != nn::DType::Int32 || d.source.dtype() != nn::DType::Int32 ||
        !nn::all(nn::logical_and(nn::greater_equal(d.labels, 0.), nn::less(d.labels, 2.)))
             .item<bool>())
      throw std::invalid_argument("aligned original CT label/ID metadata required");
    const auto labels = d.labels.to_vector<int>(), source = d.source.to_vector<int>();
    const auto raw = bytes.to(nn::DType::Int32).to_vector<int>();
    for (int row = 0; row < n; ++row) {
      const int id = source[row];
      if (!ids.insert(id).second ||
          !(split == "fit"          ? id >= 0 && id < 1158
            : split == "validation" ? id >= 1158 && id < 1323
                                    : id >= 1323 && id < 1633) ||
          !unique
               .insert(std::vector<int>(raw.begin() + row * 21952,
                                        raw.begin() + (row + 1) * 21952))
               .second)
        throw std::invalid_argument(
            "disjoint official split IDs and unique CT byte cubes required");
    }
    if (!nn::all(nn::equal(nn::sum(nn::one_hot(d.labels, 2), {0}), n / 2.)).item<bool>())
      throw std::invalid_argument("balanced selected CT classes required");
    pools.push_back(d);
  }
  return pools;
}
/// @brief Extract source4x4x4block-mean intensity features for a CT memory reference.
///
/// Preserves source Float32 division by343 BEFORE Double block summation.
/// No fitted parameters,normalization estimation or input mutation.
///
/// @param voxels Floating[N,28,28,28,1] intensities.
/// @return Double[N,64] source order(depthblock,heightblock,widthblock).
/// @throws std::invalid_argument For invalid volume shape.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadMedical("build/cnn-datasets/medical_volume");
/// const auto x = m::MedicalFeatures(pools[0].voxels);
/// @endcode
/// @par References
/// - Yang et al.(2023),MedMNIST v2:A large-scale lightweight benchmark for2Dand3Dbiomedical image classification, https://doi.org/10.1038/s41597-022-01721-8;corrected NoduleMNIST3D v2.1 archive https://zenodo.org/records/6496656,official original label/split description https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py. The three-stage residual model is a custom source teaching CNN,not a named published medical architecture or clinical predictor.
/// @see MedicalVolume()
inline Tensor MedicalFeatures(const Tensor &voxels) {
  if (voxels.dim() != 5 || voxels.size(0) < 1 ||
      voxels.shape() != nn::Shape{voxels.size(0), 28, 28, 28, 1})
    throw std::invalid_argument("positive28cube intensity features required");
  return nn::sum((voxels.to(nn::DType::Float32) / 343.)
                     .to(nn::DType::Float64)
                     .reshape({voxels.size(0), 4, 7, 4, 7, 4, 7, 1}),
                 {2, 4, 6, 7})
      .reshape({voxels.size(0), 64});
}
/// @brief Predict a parameter-free CT block-mean nearest-neighbor memory reference.
///
/// All64feature distances and first-argmin use tensors. Retained fitting records
/// are memory,not learned model leaves;no optimizer or held-out fitting.
///
/// @param fit Fitting intensity cubes and binary labels.
/// @param test Query intensity cubes;labels/source IDs ignored.
/// @return Float32[M,2] onehot logits from first fitting row at minimum squared distance.
/// @throws std::invalid_argument For invalid volume geometry or missing fitting rows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto pools = m::LoadMedical("build/cnn-datasets/medical_volume");
/// const auto z = m::MedicalBaseline(pools[0], pools[2]);
/// @endcode
/// @par References
/// - Yang et al.(2023),MedMNIST v2:A large-scale lightweight benchmark for2Dand3Dbiomedical image classification, https://doi.org/10.1038/s41597-022-01721-8;corrected NoduleMNIST3D v2.1 archive https://zenodo.org/records/6496656,official original label/split description https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py. The three-stage residual model is a custom source teaching CNN,not a named published medical architecture or clinical predictor.
/// @see MedicalVolume()
inline Tensor MedicalBaseline(const MedicalData &fit, const MedicalData &test) {
  nn::NoGradGuard guard;
  const auto a = MedicalFeatures(fit.voxels), b = MedicalFeatures(test.voxels);
  const auto distance =
      nn::sum(nn::square(nn::expand_dims(b, 1) - nn::expand_dims(a, 0)), {2});
  return nn::one_hot(nn::gather(fit.labels, nn::argmin(distance, 1), 0), 2);
}
} // namespace datamunge::nn::models
