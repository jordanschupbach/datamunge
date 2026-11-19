/// @file
/// @brief Feed-forward neuroevolution with shared historical markings.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// @brief Evolve feed-forward, single-output tanh networks with NEAT.
///
/// Networks share innovation IDs for connections and split nodes. The
/// population starts with only input and bias connections. Speciation
/// uses the first compatible representative and unnormalized counts of
/// nonmatching genes. Each step preserves one global elite and draws the
/// other offspring from species weighted by mean fitness.
/// @par Example
/// @code
/// std::mt19937 rng(1);
/// dm::nn::Neat population(2,8,rng);
/// const auto predictions=population.predict(dm::nn::zeros({4,2}));
/// std::print("{}\n",predictions.size(0)); // 8
/// @endcode
/// @par References
/// - Stanley and Miikkulainen (2002), Evolving Neural Networks through Augmenting Topologies.
///   doi:10.1162/106365602320169811
class Neat {
 public:
  /// @brief Directed connection indexed by its population-wide innovation ID.
  /// @par Example
  /// @code
  /// const dm::nn::Neat::Gene gene{0,3,0.5f,true};
  /// std::print("{}\n",gene.to); // 3
  /// @endcode
  struct Gene {
    int from;      ///< Source node index.
    int to;        ///< Destination node index.
    float weight;  ///< Signed multiplier of the source activation.
    bool enabled;  ///< Whether the connection contributes to inference.
  };

  /// @brief Connections keyed by ascending innovation ID.
  /// @par Example
  /// @code
  /// const dm::nn::Neat::Genome genome{{0,{0,3,0.5f,true}}};
  /// std::print("{}\n",genome.size()); // 1
  /// @endcode
  using Genome=std::map<int,Gene>;

  /// @brief Initialize one fully connected input layer and an output.
  /// @param inputs Positive feature count. The bias and output have indices inputs and inputs+1.
  /// @param population Positive number of networks.
  /// @param rng Generator advanced by the initial normal weight draws.
  /// @throws std::invalid_argument Invalid network or population size.
  /// @par Example
  /// @code
  /// std::mt19937 rng(1);
  /// const dm::nn::Neat population(2,8,rng);
  /// std::print("{}\n",population.genomes().front().size()); // 3
  /// @endcode
  Neat(int inputs,int population,std::mt19937& rng):inputs_(inputs) {
    if (inputs<1 || inputs>=std::numeric_limits<int>::max()-2 || population<1) {
      throw std::invalid_argument("Neat: invalid input or population size");
    }
    depth_.resize(inputs+2);
    depth_[inputs+1]=1;
    genomes_.resize(population);
    std::normal_distribution<float> initial(0.f,.6f);
    for (auto& genome:genomes_) {
      for (int node=0;node<=inputs;++node) connect(genome,node,inputs+1,initial(rng));
    }
  }

  /// @brief Evaluate all networks on the same rows of features.
  /// @param features Finite Float32 CPU tensor [samples,inputs].
  /// @return Float32 network responses [population,samples].
  /// @throws std::invalid_argument Incompatible or nonfinite features.
  /// @par Example
  /// @code
  /// std::mt19937 rng(1);
  /// const dm::nn::Neat population(2,8,rng);
  /// std::print("{}\n",population.predict(dm::nn::zeros({4,2})).size(1)); // 4
  /// @endcode
  [[nodiscard]] Tensor predict(const Tensor& features) const {
    if (!features.defined() || features.dim()!=2 || features.size(1)!=inputs_ ||
        features.dtype()!=DType::Float32 || features.device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("Neat: expected a Float32 CPU feature matrix");
    }
    const auto data=features.to_vector<float>();
    if (!std::all_of(data.begin(),data.end(),[](float x){return std::isfinite(x);})) {
      throw std::invalid_argument("Neat: nonfinite features");
    }
    const auto samples=features.size(0);
    const int output=inputs_+1;
    std::vector<int> order(depth_.size()-output);
    std::iota(order.begin(),order.end(),output);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return depth_[a]<depth_[b];});
    std::vector<float> predictions;
    for (const auto& genome:genomes_) {
      std::vector<std::map<int,float>> incoming(depth_.size());
      for (const auto& [id,gene]:genome) {
        if (gene.enabled) incoming[gene.to].emplace(gene.from,gene.weight);
      }
      std::vector<float> activation(depth_.size());
      activation[inputs_]=1.f;
      for (std::int64_t row=0;row<samples;++row) {
        std::copy_n(data.begin()+row*inputs_,inputs_,activation.begin());
        for (const int node:order) {
          float sum=0;
          for (const auto& [from,weight]:incoming[node]) sum+=activation[from]*weight;
          activation[node]=std::tanh(sum);
        }
        predictions.push_back(activation[output]);
      }
    }
    return from_vector(predictions,{static_cast<std::int64_t>(genomes_.size()),samples});
  }

  /// @brief Assign each genome to the first compatible species.
  /// @param threshold Positive finite strict upper bound on compatibility distance, default 3.4.
  /// @return Species in creation order, with member indices in population order.
  /// @throws std::invalid_argument Invalid threshold.
  /// @par Example
  /// @code
  /// std::mt19937 rng(1);
  /// const dm::nn::Neat population(2,8,rng);
  /// std::print("{}\n",population.species().size()); // 1
  /// @endcode
  [[nodiscard]] std::vector<std::vector<int>> species(double threshold=3.4) const {
    if (!(threshold>0 && std::isfinite(threshold))) {
      throw std::invalid_argument("Neat: invalid species threshold");
    }
    std::vector<std::vector<int>> groups;
    for (std::size_t i=0;i<genomes_.size();++i) {
      auto group=std::find_if(groups.begin(),groups.end(),[&](const auto& members) {
        return distance(genomes_[i],genomes_[members.front()])<threshold;
      });
      if (group==groups.end()) groups.push_back({static_cast<int>(i)});
      else group->push_back(static_cast<int>(i));
    }
    return groups;
  }

  /// @brief Replace the population using selection, crossover and structural mutation.
  /// @param fitness Nonnegative finite scores, with at least one positive score. Larger is better.
  /// @param rng Generator advanced by selection and mutation.
  /// @param threshold Compatibility cutoff passed to species(), default 3.4.
  /// @return Parent IDs [population,2] in offspring order. -1 denotes no second parent.
  /// @throws std::invalid_argument Invalid scores or species threshold.
  /// @par Example
  /// @code
  /// std::mt19937 rng(1);
  /// dm::nn::Neat population(2,2,rng);
  /// const std::array fitness{0.5f,1.f};
  /// const auto parents=population.evolve(fitness,rng);
  /// std::print("{}\n",parents.to_vector<int>()[0]); // 1
  /// @endcode
  Tensor evolve(std::span<const float> fitness,std::mt19937& rng,double threshold=3.4) {
    if (fitness.size()!=genomes_.size() ||
        !std::all_of(fitness.begin(),fitness.end(),[](float x){return x>=0 && std::isfinite(x);})) {
      throw std::invalid_argument("Neat: invalid fitness vector");
    }
    const auto elite=std::max_element(fitness.begin(),fitness.end());
    if (*elite==0) throw std::invalid_argument("Neat: all fitness values are zero");
    const int best=static_cast<int>(elite-fitness.begin());
    const auto groups=species(threshold);
    std::vector<double> scores;
    for (const auto& group:groups) {
      double sum=0;
      for (int index:group) sum+=fitness[index]/group.size();
      scores.push_back(sum);
    }
    std::discrete_distribution<std::size_t> pick(scores.begin(),scores.end());
    std::vector<Genome> next{genomes_[best]};
    std::vector<int> parents{best,-1};
    while (next.size()<genomes_.size()) {
      const auto& group=groups[pick(rng)];
      const auto tournament=[&] {
        const int a=group[rng()%group.size()], b=group[rng()%group.size()];
        return fitness[a]>fitness[b]?a:b;
      };
      int a=tournament(), b=tournament();
      if (fitness[a]<fitness[b]) std::swap(a,b);
      const bool crosses=uniform(rng)<.75;
      next.push_back(crosses?cross(genomes_[a],genomes_[b],rng):genomes_[a]);
      mutate(next.back(),rng);
      parents.insert(parents.end(),{a,crosses?b:-1});
    }
    genomes_=std::move(next);
    return from_vector(parents,{static_cast<std::int64_t>(genomes_.size()),2});
  }

  /// @brief Borrow the current genomes until the next evolution step.
  /// @return Genome sequence in population order.
  [[nodiscard]] const std::vector<Genome>& genomes() const { return genomes_; }

  /// @brief Borrow the historical node depths.
  /// @return Depths indexed by node, extended during evolution.
  [[nodiscard]] const std::vector<double>& depths() const { return depth_; }

  /// @brief Borrow the historical split records.
  /// @return Original connection innovation mapped to its inserted node.
  [[nodiscard]] const std::map<int,int>& splits() const { return splits_; }

  /// @brief Count historical connections.
  /// @return Number of distinct innovation IDs assigned so far.
  [[nodiscard]] std::size_t innovations() const { return connections_.size(); }

 private:
  /// @brief Assign or reuse an edge innovation and enable the gene.
  /// @param genome Destination genome.
  /// @param from Source node.
  /// @param to Destination node.
  /// @param weight Connection weight.
  void connect(Genome& genome,int from,int to,float weight) {
    const int id=connections_.try_emplace({from,to},connections_.size()).first->second;
    genome[id]={from,to,weight,true};
  }

  /// @brief Draw an open-interval uniform variate from the seeded generator.
  /// @param rng Generator advanced once.
  /// @return Variate in (0,1).
  static double uniform(std::mt19937& rng) { return (rng()+.5)/4294967296.; }

  /// @brief Count structural differences and mean matching weight difference.
  /// @param a First genome.
  /// @param b Second genome.
  /// @return Small-genome compatibility distance.
  static double distance(const Genome& a,const Genome& b) {
    int matches=0;
    double delta=0;
    for (const auto& [id,edge]:a) {
      if (auto match=b.find(id);match!=b.end()) {
        ++matches;
        delta+=std::abs(edge.weight-match->second.weight);
      }
    }
    return a.size()+b.size()-2*matches+.4*delta/std::max(1,matches);
  }

  /// @brief Combine matching genes, keeping unmatched genes from the fitter parent.
  /// @param fitter Higher-ranked genome.
  /// @param other Second genome.
  /// @param rng Generator for inheritance decisions.
  /// @return Offspring before mutation.
  static Genome cross(const Genome& fitter,const Genome& other,std::mt19937& rng) {
    auto child=fitter;
    for (auto& [id,gene]:child) {
      auto match=other.find(id);
      if (match==other.end()) continue;
      const bool disabled=!gene.enabled || !match->second.enabled;
      if (rng()%2==0) gene=match->second;
      if (disabled) gene.enabled=uniform(rng)>=.75;
    }
    return child;
  }

  /// @brief Perturb weights and optionally split an edge or add a forward connection.
  /// @param genome Offspring to mutate in place.
  /// @param rng Generator for all mutation decisions.
  void mutate(Genome& genome,std::mt19937& rng) {
    std::normal_distribution<float> offset(0.f,.55f);
    for (auto& [id,gene]:genome) {
      if (uniform(rng)<.8) gene.weight+=offset(rng);
    }
    if (uniform(rng)<.55) {
      std::vector<int> eligible;
      for (const auto& [id,gene]:genome) {
        if (!gene.enabled) continue;
        const auto split=splits_.find(id);
        if (split==splits_.end() || !genome.contains(connections_.at({gene.from,split->second}))) {
          const double midpoint=std::midpoint(depth_[gene.from],depth_[gene.to]);
          if (midpoint>depth_[gene.from] && midpoint<depth_[gene.to]) eligible.push_back(id);
        }
      }
      if (!eligible.empty()) {
        const int id=eligible[rng()%eligible.size()];
        const auto edge=genome.at(id);
        const auto [entry,added]=splits_.try_emplace(id,depth_.size());
        const int node=entry->second;
        if (added) depth_.push_back(std::midpoint(depth_[edge.from],depth_[edge.to]));
        genome.at(id).enabled=false;
        connect(genome,edge.from,node,1.f);
        connect(genome,node,edge.to,edge.weight);
      }
    }
    if (uniform(rng)<.45) {
      std::vector<int> nodes(inputs_+2);
      std::iota(nodes.begin(),nodes.end(),0);
      for (const auto& [id,gene]:genome) {
        nodes.push_back(gene.from);
        nodes.push_back(gene.to);
      }
      std::sort(nodes.begin(),nodes.end());
      nodes.erase(std::unique(nodes.begin(),nodes.end()),nodes.end());
      std::vector<std::pair<int,int>> valid;
      for (int from:nodes) {
        for (int to:nodes) {
          if (depth_[from]>=depth_[to]) continue;
          const auto innovation=connections_.find({from,to});
          if (innovation==connections_.end() || !genome.contains(innovation->second)) {
            valid.emplace_back(from,to);
          }
        }
      }
      if (!valid.empty()) {
        const auto [from,to]=valid[rng()%valid.size()];
        connect(genome,from,to,offset(rng));
      }
    }
  }

  int inputs_;                                   ///< Number of feature inputs.
  std::vector<double> depth_;                    ///< Topological depths including bias and output.
  std::map<std::pair<int,int>,int> connections_;  ///< Innovation IDs per ordered connection.
  std::map<int,int> splits_;                     ///< Inserted node ID per split connection.
  std::vector<Genome> genomes_;                   ///< Current generation.
};

} // namespace datamunge::nn
