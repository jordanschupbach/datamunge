// Standalone allocation and throughput check. The executable interposes C++ allocation
// so fixture construction and output formatting stay outside the measured interval.
#include <datamunge/sim/tasks.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
#include <algorithm>
namespace {
std::atomic<bool> recording{false};
std::atomic<std::size_t> allocations{0};
}
void* operator new(std::size_t size){
  if(recording.load(std::memory_order_relaxed))allocations.fetch_add(1,std::memory_order_relaxed);
  if(auto* p=std::malloc(size?size:1))return p;throw std::bad_alloc();
}
void* operator new[](std::size_t size){return ::operator new(size);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}
int main(){
  constexpr std::size_t environments=32,iterations=200;
  datamunge::sim::TaskBatch task(datamunge::sim::Task::Hopper,environments,datamunge::compute::Context({},4),42);
  std::vector<double> actions(environments*3,0);
  std::vector<std::uint8_t> mask(environments,1);
  task.step(actions);task.reset(mask);
  const auto start=std::chrono::steady_clock::now();
  allocations=0;recording=true;
  try{
    for(std::size_t iteration=0;iteration<iterations;++iteration){
      const auto& result=task.step(actions);
      bool done=false;
      for(std::size_t i=0;i<environments;++i){mask[i]=result.terminated[i]||result.truncated[i];done=done||mask[i];}
      if(done)task.reset(mask);
    }
  }catch(const std::exception& error){recording=false;std::cerr<<error.what()<<'\n';return 1;}
  recording=false;
  const auto count=allocations.load();
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::cout<<"{\"backend\":\"cpu\",\"task\":\"hopper\",\"environments\":"<<environments
           <<",\"workers\":4,\"transitions\":"<<environments*iterations<<",\"seconds\":"<<seconds
           <<",\"transitions_per_second\":"<<environments*iterations/seconds<<",\"steady_state_allocations\":"<<count<<"}\n";
  return count?1:0;
}
