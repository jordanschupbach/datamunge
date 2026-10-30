// CMD: just run-nn vsim_envs
// Native CPU physics rollouts; datamunge task definitions use their own solver.
// Jolt/reference returns and renderer output are not asserted here.
#include <datamunge/sim/tasks.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "common/checks.hpp"
namespace {
double episode(datamunge::sim::Task kind,bool feedback){
  constexpr std::size_t count=8;
  datamunge::sim::TaskBatch task(kind,count,datamunge::compute::Context({},2),42,200);
  std::vector<double> actions(count*task.action_size(),0),returns(count,0);
  std::vector<std::uint8_t> active(count,1);
  std::mt19937_64 random(7);std::uniform_real_distribution<double> force(-3,3);
  for(int tick=0;tick<200&&std::any_of(active.begin(),active.end(),[](auto v){return v!=0;});++tick){
    const auto& obs=task.output().observations;
    if(kind==datamunge::sim::Task::InvertedPendulum)for(std::size_t i=0;i<count;++i){
      const auto* row=obs.data()+i*4;
      actions[i]=feedback?std::clamp(.08*row[0]+.15*row[2]-1.5*row[1]-.3*row[3],-3.,3.):force(random);
    }
    const auto& result=task.step(actions,active);
    for(std::size_t i=0;i<count;++i)if(active[i]){returns[i]+=result.rewards[i];active[i]=!result.terminated[i]&&!result.truncated[i];}
  }
  double total=0;for(auto value:returns){check::Require(std::isfinite(value),"native rollout reward must be finite");total+=value;}
  return total/count;
}
}
int main(){
  try{
    const double random=episode(datamunge::sim::Task::InvertedPendulum,false);
    const double feedback=episode(datamunge::sim::Task::InvertedPendulum,true);
    const double hopper=episode(datamunge::sim::Task::Hopper,false);
    std::cout<<"Native CPU InvertedPendulum: random mean return="<<random<<", feedback="<<feedback<<'\n';
    std::cout<<"Native CPU Hopper: zero-torque mean return="<<hopper<<" (three physical hinge joints)\n";
    check::Require(random>0&&feedback>0,"pendulum must advance real episodes");
    std::cout<<"checks passed (native physics rollouts; explicit termination and truncation)\n";
  }catch(const std::exception& error){std::cerr<<"FAILED: "<<error.what()<<'\n';return 1;}
}
