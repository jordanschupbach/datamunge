/// @file
/// @brief Nearest-neighbor episodic novelty for discrete-state embeddings.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::rl {

/// @brief Score novelty relative to a memory of states visited this episode.
///
/// An episode uses a frozen embedding table. Each observation contributes
/// its nearest squared distances to a running mean that persists through
/// resets. A kernel sum over the normalized distances estimates local
/// density. The bonus is the inverse square root of that density with
/// a small offset in the denominator.
/// @par Example
/// @code
/// dm::rl::EpisodicNovelty novelty;
/// novelty.reset(dm::nn::eye(2),0);
/// const auto fresh=novelty.observe(1);
/// std::print("{}\n",fresh>novelty.observe(1)); // true
/// @endcode
/// @par References
/// - Badia et al. (2020), Never Give Up: Learning Directed Exploration Strategies.
///   https://arxiv.org/abs/2002.06038
class EpisodicNovelty {
 public:
  /// @brief Configure the episodic neighbor count and density kernel.
  /// @param neighbors Maximum number of nearest visits, default 5.
  /// @param kernel Finite positive kernel constant, default 0.001.
  /// @param cluster Finite nonnegative cluster threshold, default 0.008.
  /// @param offset Finite positive denominator offset, default 0.001.
  /// @throws std::invalid_argument Invalid count or constants.
  /// @par Example
  /// @code
  /// dm::rl::EpisodicNovelty novelty(3);
  /// @endcode
  explicit EpisodicNovelty(std::size_t neighbors=5,double kernel=.001,double cluster=.008,
                           double offset=.001)
      : neighbors_(neighbors),kernel_(kernel),cluster_(cluster),offset_(offset) {
    if (neighbors==0 || !(kernel>0 && std::isfinite(kernel)) ||
        !(cluster>=0 && std::isfinite(cluster)) || !(offset>0 && std::isfinite(offset))) {
      throw std::invalid_argument("EpisodicNovelty: invalid kernel configuration");
    }
  }

  /// @brief Freeze embeddings and replace episodic memory with the start state.
  /// @param embeddings Finite Float32 CPU rows [states,features], with fixed positive feature width.
  /// @param start Valid start state index, default zero.
  /// @throws std::invalid_argument Invalid embeddings or start ID.
  /// @par Example
  /// @code
  /// dm::rl::EpisodicNovelty novelty;
  /// novelty.reset(dm::nn::eye(3),1);
  /// @endcode
  void reset(const nn::Tensor& embeddings,std::int64_t start=0) {
    if (!embeddings.defined() || embeddings.dim()!=2 || embeddings.size(1)<1 ||
        embeddings.dtype()!=nn::DType::Float32 || embeddings.device().backend!=compute::Backend::CPU ||
        start<0 || start>=embeddings.size(0) || (width_!=0 && width_!=embeddings.size(1))) {
      throw std::invalid_argument("EpisodicNovelty: invalid embeddings or start");
    }
    auto values=embeddings.to_vector<float>();
    if (!std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);})) {
      throw std::invalid_argument("EpisodicNovelty: embeddings must be finite");
    }
    memory_.assign(1,start);
    embeddings_=std::move(values);
    width_=embeddings.size(1);
    states_=embeddings.size(0);
  }

  /// @brief Score a state against memory, then remember the visit.
  /// @param state Valid state ID in the current snapshot.
  /// @return Positive density-based novelty.
  /// @throws std::logic_error No episode has been initialized.
  /// @throws std::invalid_argument State is outside the snapshot.
  /// @par Example
  /// @code
  /// dm::rl::EpisodicNovelty novelty;
  /// novelty.reset(dm::nn::eye(2));
  /// std::print("{}\n",novelty.observe(1)>1.); // true
  /// @endcode
  double observe(std::int64_t state) {
    if (memory_.empty()) throw std::logic_error("EpisodicNovelty: reset before observing");
    if (state<0 || state>=states_) throw std::invalid_argument("EpisodicNovelty: invalid state");
    std::vector<double> distances;
    distances.reserve(memory_.size());
    for (const auto previous:memory_) {
      double squared=0;
      for (std::int64_t feature=0;feature<width_;++feature) {
        const double difference=double(embeddings_[state*width_+feature])-
            embeddings_[previous*width_+feature];
        squared+=difference*difference;
      }
      distances.push_back(squared);
    }
    const auto count=std::min(neighbors_,distances.size());
    std::partial_sort(distances.begin(),distances.begin()+count,distances.end());
    memory_.push_back(state);
    for (std::size_t i=0;i<count;++i) {
      ++count_;
      mean_+=(distances[i]-mean_)/count_;
    }
    const double scale=std::max(1e-6,mean_);
    double density=0;
    for (std::size_t i=0;i<count;++i) {
      const double normalized=std::max(0.,distances[i]/scale-cluster_);
      density+=kernel_/(normalized+kernel_);
    }
    return 1./(std::sqrt(density)+offset_);
  }

 private:
  std::size_t neighbors_;           ///< Maximum number of neighbors.
  double kernel_;                   ///< Kernel smoothing constant.
  double cluster_;                  ///< Subtracted normalized distance threshold.
  double offset_;                   ///< Offset in the reward denominator.
  std::int64_t states_=0;           ///< Current snapshot row count.
  std::int64_t width_=0;            ///< Fixed feature width after first reset.
  std::vector<float> embeddings_;   ///< Owned copy of the current embeddings.
  std::vector<std::int64_t> memory_;///< Visited state IDs, including repeats.
  double count_=0;                  ///< Number of neighbors in the running scale.
  double mean_=0;                   ///< Running mean of selected squared distances.
};

} // namespace datamunge::rl
