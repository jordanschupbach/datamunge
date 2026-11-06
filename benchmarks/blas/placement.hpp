#pragma once

// Thread placement for the comparison drivers. The in-house pool binds its
// threads to last-level cache domains (parallel.cpp; DATAMUNGE_BIND_THREADS=0
// turns that off), the caller's domain first, and binds the thread that
// submits work there too. OpenBLAS, BLIS and LAPACK do not bind (the nixpkgs
// builds are NO_AFFINITY), but the threads they start inherit the affinity of
// the thread that starts them. place_main_thread() therefore binds the main
// thread the way the pool would before an external library starts its
// threads, so both sides run on the same die(s). When the thread count needs
// more physical cores than that domain has, the main thread is freed again,
// so an external library's threads can spread over every die.

#include <datamunge/nn/parallel.hpp>

#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

namespace bench {

inline void place_main_thread(int threads) {
#if defined(__linux__)
  if (threads <= 1) return;
  cpu_set_t before;
  if (sched_getaffinity(0, sizeof before, &before) != 0) return;
  datamunge::nn::set_num_threads(threads);
  datamunge::nn::parallel_for(0, 2, 1, [](std::int64_t, std::int64_t) {});  // binds this thread, if the pool does
  cpu_set_t now;
  if (sched_getaffinity(0, sizeof now, &now) != 0 || CPU_EQUAL(&before, &now)) return;
  std::set<std::pair<int, int>> cores;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &now)) continue;
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream     pkg(base + "physical_package_id"), core(base + "core_id");
    int               p = 0, c = 0;
    if (pkg >> p && core >> c) cores.emplace(p, c);
  }
  if (static_cast<int>(cores.size()) < threads) sched_setaffinity(0, sizeof before, &before);
#else
  (void)threads;
#endif
}

}  // namespace bench
