#pragma once

/// @file halving_search.hpp
/// @brief Resource allocation for synchronous and asynchronous model search.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::nn {

/// @brief Allocate resources across three Hyperband brackets with budgets 1, 2 and 4.
/// @param start Callback taking a zero-based candidate position and returning a unique trial ID.
/// @param advance Callback taking trial ID, cumulative resource, bracket and completion time
/// (-1 for synchronous runs). Returns finite validation loss. Preserve trial state across calls.
/// @throws std::invalid_argument If a returned score is nonfinite.
/// @par Example
/// @code
/// dm::nn::hyperband_search([](int i){return i;}, [](int i,int,int,double){return double(i);});
/// @endcode
/// @par References
/// - Li et al. (2018), Hyperband. https://jmlr.org/papers/v18/16-558.html
/// @see asha_search()
inline void hyperband_search(const std::function<int(int)>& start,
                             const std::function<double(int,int,int,double)>& advance) {
  int cursor=0;
  for (int bracket=2;bracket>=0;--bracket) {
    const int count=(3*(1<<bracket)+bracket)/(bracket+1);
    std::vector<int> active;
    for (int i=0;i<count;++i) active.push_back(start(cursor++));
    for (int rung=0;rung<=bracket;++rung) {
      std::vector<std::pair<double,int>> ranked;
      const int resource=4>>(bracket-rung);
      for (int id:active) {
        const auto loss=advance(id,resource,bracket,-1);
        if (!std::isfinite(loss)) throw std::invalid_argument("Nonfinite trial score");
        ranked.emplace_back(loss,id);
      }
      std::sort(ranked.begin(),ranked.end());
      active.clear();
      for (std::size_t i=0;i<ranked.size()/2;++i) active.push_back(ranked[i].second);
    }
  }
}

/// @brief Simulate asynchronous successive halving with two logical workers.
/// Promote from the best half of completed observations, checking higher
/// rungs first. Ties in completion time favor the lower worker index.
/// @param starts Maximum number of fresh trials.
/// @param start Callback mapping candidate position to a unique nonnegative trial ID.
/// @param advance Callback taking trial ID, cumulative resource, bracket (zero),
/// and virtual completion time. Returns finite validation loss.
/// @param duration Callback taking trial ID and additional resource, returning finite positive virtual duration.
/// @throws std::invalid_argument For a negative start budget or nonfinite score or duration.
/// @par Example
/// @code
/// dm::nn::asha_search(10,[](int i){return i;},
///     [](int i,int,int,double){return double(i);},[](int,int units){return double(units);});
/// @endcode
/// @par References
/// - Li et al. (2020), A System for Massively Parallel Hyperparameter Tuning. https://arxiv.org/abs/1810.05934
/// @see hyperband_search()
inline void asha_search(int starts,const std::function<int(int)>& start,
    const std::function<double(int,int,int,double)>& advance,
    const std::function<double(int,int)>& duration) {
  if (starts<0) throw std::invalid_argument("Negative ASHA trial budget");
  /// @brief A dispatched tranche on one logical worker.
  struct Job {
    int id=-1; ///< Trial ID or -1 for idle.
    int resource=0; ///< Cumulative resource upon completion.
    double finish=0; ///< Virtual completion timestamp.
  };
  std::array<Job,2> workers;
  std::array<std::vector<std::pair<double,int>>,2> rungs;
  std::array<std::vector<int>,2> promoted;
  double time=0;
  int cursor=0;
  /// Schedule a tranche on one worker at the current virtual time.
  const auto schedule=[&](Job& worker,int id,int resource,int additional) {
    const auto elapsed=duration(id,additional);
    if (!(elapsed>0) || !std::isfinite(time+elapsed))
      throw std::invalid_argument("Invalid ASHA duration");
    worker={id,resource,time+elapsed};
  };
  /// Choose an available promotion or dispatch a fresh trial.
  const auto dispatch=[&](Job& worker) {
    for (int rung=1;rung>=0;--rung) {
      auto ranked=rungs[rung];
      std::sort(ranked.begin(),ranked.end());
      for (std::size_t i=0;i<ranked.size()/2;++i) {
        const int id=ranked[i].second;
        if (std::find(promoted[rung].begin(),promoted[rung].end(),id)!=promoted[rung].end()) continue;
        promoted[rung].push_back(id);
        schedule(worker,id,2<<rung,1<<rung);
        return;
      }
    }
    if (cursor<starts) schedule(worker,start(cursor++),1,1);
  };
  for (auto& worker:workers) dispatch(worker);
  while (true) {
    int next=-1;
    for (int i=0;i<2;++i)
      if (workers[i].id>=0 && (next<0 || workers[i].finish<workers[next].finish)) next=i;
    if (next<0) break;
    const auto job=workers[next];
    workers[next]={};
    time=job.finish;
    const auto loss=advance(job.id,job.resource,0,time);
    if (!std::isfinite(loss)) throw std::invalid_argument("Nonfinite trial score");
    if (job.resource<4) rungs[job.resource-1].emplace_back(loss,job.id);
    for (auto& worker:workers) if (worker.id<0) dispatch(worker);
  }
}

}  // namespace datamunge::nn
