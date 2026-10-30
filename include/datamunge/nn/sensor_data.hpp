#pragma once
/// @file sensor_data.hpp
/// @brief Original ECG200/BasicMotions tensors and fitting-only channel normalization/class weights.
#include "datamunge/nn/sensor_models.hpp"
#include "datamunge/nn/time_series_data.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own original signal partitions and explicit fitting-only normalization/class weights.
///
/// Pools are fitting,validation,test. ECG200 shape96x1/classes2 and
/// BasicMotions100x6/classes4 retain original channel/class order. Mean/std
/// and class weights derive only from fitting tensors,roundedFloat32 once.
/// No actor/patient identity information is inferred from row IDs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::LoadSensor("build/cnn-datasets/ecg_sensor/ECG200");
/// @endcode
/// @par References
/// - Official ECG200 source, https://www.timeseriesclassification.com/description.php?Dataset=ECG200;official BasicMotions source, https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions. Preserve original source class indices;custom20% training-only validation split does not establish patient/participant-disjoint evaluation.
/// @see LoadSensor(), SensorCnn()
struct SensorSplits {
  std::vector<SeriesData>
      pools;           ///< Ordered fitting/validation/test original observed records.
  SensorConfig config; ///< Original channels/classes and compact residual width8.
  nn::Tensor mean;     ///< Float32[C] fitting channel population mean.
  nn::Tensor stddev;   ///< Positive Float32[C] fitting channel population standard deviation.
  nn::Tensor weights;  ///< Float32[K] N/(K*class_count),from fitting labels only.
  int time = 0;        ///< Original source length96ECG or100BasicMotions.
};
/// @brief Load checked original ECG/smartwatch records and tensor fitting statistics.
///
/// Require fixed source geometry,finite Float32 raw signal values,Int32 labels/
/// source IDs and no record overlap. Official test rows remain at original
/// source offset100ECG/40BasicMotions;fit/validation stay inside original train.
/// Fitting Double channel population mean/std and inverse-frequency weights
/// roundFloat32 once. Each fitting class must exist and each channel must have
/// variance>1e-12. No inference inputs or held-out labels estimate statistics.
/// Original BasicMotions channel order is accelerometer xyz,gyroscope xyz;
/// class indices follow the original TS classLabel header,not guessed semantics.
///
/// @param root Dataset subdirectory containing signals.npz and original-source manifest.
/// @return Three original source-row-disjoint pools,compact configuration and fitting-only tensors.
/// @throws std::invalid_argument For invalid source metadata,geometry,values,leakage or degenerate fitting support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::LoadSensor("build/cnn-datasets/ecg_sensor/BasicMotions");
/// @endcode
/// @par References
/// - Official ECG200 source, https://www.timeseriesclassification.com/description.php?Dataset=ECG200;official BasicMotions source, https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions. Preserve original source class indices;custom20% training-only validation split does not establish patient/participant-disjoint evaluation.
/// @see SensorSplits, SensorCE()

inline SensorSplits LoadSensor(const std::filesystem::path &root) {
  nn::NoGradGuard guard;
  std::map<std::string, nn::Tensor> arrays;
  for (const auto &[name, value] : nn::load_npz((root / "signals.npz").string()))
    if (!arrays.emplace(name, value).second)
      throw std::invalid_argument("unique original sensor array names required");
  const auto &cfg = arrays.at("config");
  if (cfg.dtype() != nn::DType::Int32 || cfg.shape() != nn::Shape{4})
    throw std::invalid_argument("original sensor shape metadata required");
  const auto c = cfg.to_vector<int>();
  SensorSplits out;
  out.time = c[0];
  out.config = {c[1], c[2], 8};
  const int channels = c[1], classes = c[2], train = c[3];
  if (!((out.time == 96 && channels == 1 && classes == 2 && train == 100) ||
        (out.time == 100 && channels == 6 && classes == 4 && train == 40)))
    throw std::invalid_argument("original ECG200/BasicMotions dimensions required");
  std::set<int> seen;
  for (const std::string name : {"fit", "validation", "test"}) {
    SeriesData d{arrays.at(name + "_signals"), arrays.at(name + "_labels"),
                 arrays.at(name + "_source_ids")};
    if (d.signals.dim() != 3 || d.size() < 1 || d.signals.dtype() != nn::DType::Float32 ||
        d.signals.shape() != nn::Shape{d.size(), out.time, channels} ||
        d.labels.dtype() != nn::DType::Int32 || d.labels.shape() != nn::Shape{d.size()} ||
        d.source_ids.dtype() != nn::DType::Int32 || d.source_ids.shape() != d.labels.shape() ||
        !nn::all(nn::is_finite(d.signals)).item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(d.labels, 0.),
                                 nn::less(d.labels, double(classes))))
             .item<bool>())
      throw std::invalid_argument(
          "finite original aligned multichannel sensor records required");
    for (int id : d.source_ids.to_vector<int>())
      if (id < 0 || id >= 2 * train || (name == "test" ? id < train : id >= train) ||
          !seen.insert(id).second)
        throw std::invalid_argument("unique official source-row boundaries required");
    out.pools.push_back(d);
  }
  const auto &fit = out.pools[0];
  const auto x = fit.signals.to(nn::DType::Float64), mean = nn::mean(x, {0, 1}),
             variance = nn::mean(nn::square(x - mean), {0, 1}),
             counts =
                 nn::sum(nn::one_hot(fit.labels, classes, 1., 0., nn::DType::Float64), {0});
  if (!nn::all(nn::greater(variance, 1e-12)).item<bool>() ||
      !nn::all(nn::greater(counts, 0.)).item<bool>())
    throw std::invalid_argument(
        "nonconstant channels and every fitting source class required");
  out.mean = mean.to(nn::DType::Float32);
  out.stddev = nn::sqrt(variance).to(nn::DType::Float32);
  out.weights = (double(fit.size()) / (double(classes) * counts)).to(nn::DType::Float32);
  return out;
}
} // namespace datamunge::nn::models
