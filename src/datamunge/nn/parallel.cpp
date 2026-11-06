/// @file parallel.cpp
/// @brief Persistent thread pool and serial fallback for neural network kernels.
///
/// One job runs at a time: the caller and the workers each run their own block
/// of its chunks, then take what is left of the others' blocks, and the caller
/// waits for all of them. Concurrent submitters and nested parallel_for calls
/// use serial fallback instead of waiting on the pool.
///
/// On Linux with several last-level cache domains (the CCDs of a Ryzen, for
/// example), the workers are bound to domains in order: the caller's domain
/// gets the first workers, one per physical core beside the caller's, then the
/// next domain, and so on; a thread that submits work is bound to the first
/// domain when it first does (unless its affinity is no longer the process's).
/// Left to the scheduler, eight threads on a two-die 16-core part were often
/// spread over both dies, and GEMM, Cholesky, LU and QR at n = 256-512 ran
/// 1.4-1.8x slower than on one; with only the workers bound, the scheduler
/// moved the caller to the idle die. A job with fewer chunks than threads,
/// which runs on the lowest participants, also stays on the caller's die.
/// Binding is to a domain's CPU set, not to single CPUs, so the scheduler
/// still balances within it. DATAMUNGE_BIND_THREADS=0 turns it off (several
/// processes sharing the machine, or the caller's own affinity scheme).

#include "datamunge/nn/parallel.hpp"

#include "datamunge/nn/blas.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace datamunge::nn {

/// @brief Process-wide pool ownership and per-thread nesting state.
namespace {

/// @brief Physical cores on Linux (distinct package/core pairs in sysfs), else 0.
int physical_cores() {
#if defined(__linux__)
  std::set<std::pair<int, int>> cores;
  for (int cpu = 0;; ++cpu) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream     pkg(base + "physical_package_id"), core(base + "core_id");
    int               p = 0, c = 0;
    if (!(pkg >> p) || !(core >> c)) break;
    cores.emplace(p, c);
  }
  return static_cast<int>(cores.size());
#else
  return 0;
#endif
}

/// @brief Uses a positive DATAMUNGE_NUM_THREADS value, else one thread per physical core
/// (the kernels are vector and memory bound, so SMT siblings only contend), else hardware
/// concurrency with a fallback of one.
int default_threads() {
  if (const char* env = std::getenv("DATAMUNGE_NUM_THREADS")) {
    const int n = std::atoi(env);
    if (n > 0) return n;
  }
  if (const int cores = physical_cores(); cores > 0) return cores;
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

#if defined(__linux__)
/// @brief CPUs in a sysfs list such as "0-7,16-23".
std::vector<int> parse_cpu_list(const std::string& text) {
  std::vector<int> cpus;
  std::size_t      pos = 0;
  while (pos < text.size()) {
    const std::size_t end   = std::min(text.find(',', pos), text.size());
    const std::string item  = text.substr(pos, end - pos);
    const std::size_t dash  = item.find('-');
    const int         first = std::atoi(item.c_str());
    const int         last  = dash == std::string::npos ? first : std::atoi(item.c_str() + dash + 1);
    for (int c = first; c <= last; ++c) cpus.push_back(c);
    pos = end + 1;
  }
  return cpus;
}

/// @brief Where the pool's threads go: one CPU set per worker (index i for participant i;
/// index 0, the caller, and participants past the last core stay empty, i.e. unbound), the
/// caller's domain and the CPUs the process may use.
struct Binding {
  std::vector<cpu_set_t> sets;
  cpu_set_t              home, allowed;
};

/// @brief The last-level cache domains of the CPUs this process may use, the caller's current
/// domain first, each taking as many participants as it has physical cores. Empty sets when
/// binding is off, the topology is unknown or there is only one domain.
Binding binding(int participants) {
  Binding out;
  if (const char* env = std::getenv("DATAMUNGE_BIND_THREADS"); env && std::string(env) == "0") return out;
  if (participants <= 1) return out;
  CPU_ZERO(&out.allowed);
  const int here = sched_getcpu();
  if (here < 0 || sched_getaffinity(0, sizeof out.allowed, &out.allowed) != 0) return out;
  // Domains by the shared CPU list of the highest readable cache level (index3: L3 on x86).
  struct Domain {
    cpu_set_t                                cpus;
    std::vector<std::pair<int, int>>         keys;   // (package, core) of each core, in CPU order
    std::vector<cpu_set_t>                   cores;  // its logical CPUs
  };
  std::vector<std::string> lists;
  std::vector<Domain>      domains;
  int                      home = -1;
  std::size_t              here_core = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &out.allowed)) continue;
    std::string list;
    for (int index = 4; index >= 2 && list.empty(); --index) {
      std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index" + std::to_string(index) +
                      "/shared_cpu_list");
      std::getline(f, list);
    }
    if (list.empty()) return out;
    const auto d = static_cast<std::size_t>(std::find(lists.begin(), lists.end(), list) - lists.begin());
    if (d == lists.size()) {
      lists.push_back(list);
      domains.emplace_back();
      CPU_ZERO(&domains.back().cpus);
    }
    Domain& dom = domains[d];
    CPU_SET(cpu, &dom.cpus);
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream     pkg(base + "physical_package_id"), core(base + "core_id");
    int               p = 0, c = 0;
    if (!(pkg >> p && core >> c)) return out;
    auto k = static_cast<std::size_t>(std::find(dom.keys.begin(), dom.keys.end(), std::make_pair(p, c)) -
                                      dom.keys.begin());
    if (k == dom.keys.size()) {
      dom.keys.emplace_back(p, c);
      dom.cores.emplace_back();
      CPU_ZERO(&dom.cores.back());
    }
    CPU_SET(cpu, &dom.cores[k]);
    if (cpu == here) home = static_cast<int>(d), here_core = k;
  }
  if (domains.size() < 2 || home < 0) return out;
  std::rotate(domains.begin(), domains.begin() + home, domains.begin() + home + 1);
  out.home = domains[0].cpus;
  out.sets.resize(static_cast<std::size_t>(participants));
  for (auto& set : out.sets) CPU_ZERO(&set);
  // Workers one per core, the caller's core left out (the caller stays free within its domain).
  int q = 1;
  for (std::size_t d = 0; d < domains.size(); ++d)
    for (std::size_t k = 0; k < domains[d].cores.size() && q < participants; ++k)
      if (d != 0 || k != here_core) out.sets[static_cast<std::size_t>(q++)] = domains[d].cores[k];
  return out;
}
#endif

/// @brief Whether this thread is already in a parallel region, preventing recursive pool submission.
thread_local bool tls_in_parallel = false;

/// @brief Pauses the core briefly inside a spin loop.
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#else
  std::this_thread::yield();
#endif
}

/// How long a thread spins for the next job (a worker) or for the last chunk
/// (the submitter) before sleeping on a condition variable. A training step
/// issues hundreds of short parallel regions, and a futex wake-up costs more
/// than many of them; a bounded spin keeps idle cores from burning power.
constexpr std::chrono::microseconds kSpinBudget{50};

/// @brief Spins until `ready()` holds or the budget passes; returns the final predicate value.
template <typename Ready>
bool spin_until(Ready&& ready) {
  const auto deadline = std::chrono::steady_clock::now() + kSpinBudget;
  while (true) {
    for (int i = 0; i < 64; ++i) {
      if (ready()) return true;
      cpu_relax();
    }
    if (std::chrono::steady_clock::now() >= deadline) return ready();
  }
}

/// Chunk claims are packed in one word per participant: the job's tag (the low
/// 40 bits of its generation), the next unclaimed chunk and the end of that
/// participant's home range (12 bits each). One compare-exchange checks that
/// the job is still the current one and takes a chunk, so a worker that wakes
/// late cannot run a chunk of a newer job, and no lock or allocation is needed
/// per job.
constexpr int           kIndexBits = 12;
constexpr std::uint64_t kIndexMask = (std::uint64_t{1} << kIndexBits) - 1;
constexpr std::uint64_t kTagMask   = (std::uint64_t{1} << (64 - 2 * kIndexBits)) - 1;
/// Most chunks one job may have (the largest index the claim word holds).
constexpr std::int64_t  kMaxChunks = static_cast<std::int64_t>(kIndexMask);

constexpr std::uint64_t pack_claim(std::uint64_t tag, std::int64_t next, std::int64_t end) {
  return (tag << (2 * kIndexBits)) | (static_cast<std::uint64_t>(next) << kIndexBits) |
         static_cast<std::uint64_t>(end);
}

/// One participant's claim word, on its own cache line.
struct alignas(64) Claim {
  std::atomic<std::uint64_t> word{0};
};

/// @brief Takes the next chunk of `claim` for the job tagged `tag`; -1 when its range is
/// used up or the job is no longer current.
std::int64_t take(Claim& claim, std::uint64_t tag) {
  std::uint64_t w = claim.word.load(std::memory_order_relaxed);
  while (true) {
    if ((w >> (2 * kIndexBits)) != tag) return -1;
    const std::int64_t next = static_cast<std::int64_t>((w >> kIndexBits) & kIndexMask);
    if (next >= static_cast<std::int64_t>(w & kIndexMask)) return -1;
    if (claim.word.compare_exchange_weak(w, w + (std::uint64_t{1} << kIndexBits), std::memory_order_acquire,
                                         std::memory_order_relaxed))
      return next;
  }
}

/// @brief Serializes submitted jobs and retains worker threads between kernel calls.
///
/// Participant p (the caller is 0, worker i is i + 1) owns the p-th contiguous
/// block of a job's chunks: it runs its own block first, then takes leftover
/// chunks from the others' blocks. With every thread present, a given range
/// lands on the same thread in every call, so data that one call left in a
/// core's cache is still there for the next call over the same range; a thread
/// that is late (descheduled, asleep) only shifts its block to the others.
/// Which thread runs a chunk never changes what the chunk computes.
class Pool {
 public:
  /// @brief Starts the configured worker count minus one; the caller supplies the remaining thread.
  Pool() { resize(threads_ - 1); }
  /// @brief Signals all workers to stop and joins them before pool state is destroyed.
  ~Pool() { resize(0); }

  /// @brief Returns the configured total parallelism, including the submitting thread.
  int  threads() const { return threads_; }
  /// @brief Waits for the active submission and rebuilds the pool for a positive total thread count.
  void set_threads(int n) {
    std::lock_guard<std::mutex> submit(submit_mutex_);
    threads_ = n;
    resize(n - 1);
  }

  /// Runs body(0) ... body(chunks - 1) (chunks <= kMaxChunks) and returns true,
  /// or returns false without running anything when the pool is busy.
  bool run(std::int64_t chunks, const std::function<void(std::int64_t)>& body) {
    std::unique_lock<std::mutex> submit(submit_mutex_, std::try_to_lock);
    if (!submit.owns_lock()) return false;
#if defined(__linux__)
    if (bind_callers_) bind_caller();
#endif
    // The previous job has finished: no participant can still claim from it,
    // and none touches body_ or remaining_ without a claim.
    const std::uint64_t gen = generation_.load(std::memory_order_relaxed) + 1, tag = gen & kTagMask;
    body_.store(&body, std::memory_order_relaxed);
    remaining_.store(chunks, std::memory_order_relaxed);
    // A job with fewer chunks than participants runs on the first `chunks`
    // ones (the caller and the lowest workers), one home chunk each; the others
    // sit it out. Letting them steal raced them against the owners, so which
    // core wrote a range changed from call to call: an AXPY of 2^16 floats at
    // 16 threads (8 chunks) moved its cache lines between dies and ran 2.6x
    // slower than on one thread.
    const int  active      = static_cast<int>(std::min<std::int64_t>(parts_, chunks));
    const auto block_start = [&](std::int64_t p) { return chunks * p / active; };
    for (int p = 0; p < parts_; ++p) {
      const std::int64_t lo = p < active ? block_start(p) : 0, hi = p < active ? block_start(p + 1) : 0;
      claims_[p].word.store(pack_claim(tag, lo, hi), std::memory_order_relaxed);
    }
    active_.store(active, std::memory_order_relaxed);
    {
      // Under the mutex, so a worker cannot miss the change between checking
      // the generation and going to sleep. Release: a worker that sees the new
      // generation also sees the claims, body_ and remaining_.
      std::lock_guard<std::mutex> lock(mutex_);
      generation_.store(gen, std::memory_order_release);
    }
    wake_.notify_all();  // no system call when every worker is still spinning
    work(0, tag);
    if (!spin_until([&] { return finished(); })) {
      std::unique_lock<std::mutex> lock(done_mutex_);
      done_.wait(lock, [&] { return finished(); });
    }
    if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
    return true;
  }

 private:
  /// @brief Whether every chunk has finished (acquire: the chunks' writes are visible after it).
  bool finished() const { return remaining_.load(std::memory_order_acquire) == 0; }

  /// Runs chunks of the job tagged `tag` for participant `part` until none are left: its own
  /// block, then what is left of the other active participants' blocks.
  void work(int part, std::uint64_t tag) {
    const int active = active_.load(std::memory_order_relaxed);  // published with the generation
    if (part >= active) return;
    const bool was  = tls_in_parallel;
    tls_in_parallel = true;
    std::int64_t       completed = 0;
    std::exception_ptr failure;
    for (int k = 0; k < active; ++k) {
      Claim& claim = claims_[(part + k) % active];
      for (std::int64_t i; (i = take(claim, tag)) >= 0;) {
        // A claim made under `tag` keeps the job unfinished, so body_ is still its body.
        try {
          (*body_.load(std::memory_order_relaxed))(i);
        } catch (...) {
          if (!failure) failure = std::current_exception();
        }
        ++completed;
      }
    }
    // Publish completion once per participant rather than once per chunk.
    if (completed != 0) {
      if (failure) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (!error_) error_ = failure;
      }
      if (remaining_.fetch_sub(completed, std::memory_order_acq_rel) == completed) {
        // Last participant: wake the submitter if it gave up spinning.
        std::lock_guard<std::mutex> lock(done_mutex_);
        done_.notify_all();
      }
    }
    tls_in_parallel = was;
  }

  /// @brief Stops and joins current workers, then creates the requested number of worker threads.
  /// The caller must exclude active submissions, except during initial construction/destruction.
  void resize(int workers) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_.store(true, std::memory_order_release);
    }
    wake_.notify_all();
    for (auto& t : workers_) t.join();
    workers_.clear();
    stop_.store(false, std::memory_order_release);
    parts_  = workers + 1;
    claims_ = std::make_unique<Claim[]>(static_cast<std::size_t>(parts_));
    for (int i = 0; i < workers; ++i) workers_.emplace_back([this, i] { loop(i + 1); });
#if defined(__linux__)
    const Binding b = binding(parts_);
    for (std::size_t i = 0; i < workers_.size() && i + 1 < b.sets.size(); ++i)
      if (CPU_COUNT(&b.sets[i + 1]) > 0)
        pthread_setaffinity_np(workers_[i].native_handle(), sizeof(cpu_set_t), &b.sets[i + 1]);
    bind_callers_ = !b.sets.empty();
    if (bind_callers_) {
      home_    = b.home;
      allowed_ = b.allowed;
      ++bind_epoch_;
    }
#endif
  }

#if defined(__linux__)
  /// @brief Binds the submitting thread to the first domain, once per pool configuration, unless
  /// its affinity was changed from the process's (then it is the caller's to manage). Left
  /// free, the scheduler moved it off its workers' die, which undid the binding's gain.
  void bind_caller() {
    static thread_local std::uint64_t bound_epoch = 0;
    if (bound_epoch == bind_epoch_) return;
    bound_epoch = bind_epoch_;
    cpu_set_t current;
    if (pthread_getaffinity_np(pthread_self(), sizeof current, &current) == 0 && CPU_EQUAL(&current, &allowed_))
      pthread_setaffinity_np(pthread_self(), sizeof home_, &home_);
  }
#endif

  /// @brief Whether a worker that last saw `seen` has something new to do.
  bool wanted(std::uint64_t seen) const {
    return stop_.load(std::memory_order_acquire) || generation_.load(std::memory_order_acquire) != seen;
  }

  /// @brief Worker entry point: waits for new job generations and contributes chunks until stopped.
  void loop(int part) {
    tls_in_parallel    = true;
    std::uint64_t seen = generation_.load(std::memory_order_acquire);
    while (true) {
      // Spin briefly for the next job before sleeping on the condition variable.
      if (!spin_until([&] { return wanted(seen); })) {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return wanted(seen); });
      }
      if (stop_.load(std::memory_order_acquire)) return;
      seen = generation_.load(std::memory_order_acquire);
      work(part, seen & kTagMask);
    }
  }

  /// @brief Configured total threads, counting both workers and a submitting caller.
  int                      threads_{default_threads()};
  /// @brief Participants (workers + the caller), the size of claims_.
  int                      parts_{1};
  /// @brief Participants the current job runs on (the first ones); set before its generation.
  std::atomic<int>         active_{1};
#if defined(__linux__)
  /// @brief Whether submitting threads are bound (the workers are), to home_ when their
  /// affinity is still allowed_; bind_epoch_ counts configurations that bind.
  bool          bind_callers_{false};
  cpu_set_t     home_{}, allowed_{};
  std::uint64_t bind_epoch_{0};
#endif
  /// @brief One claim word per participant, replaced only while no worker runs.
  std::unique_ptr<Claim[]> claims_;
  /// @brief Owned persistent workers, joined before replacement or destruction.
  std::vector<std::thread> workers_;
  /// @brief Serializes whole submissions and pool reconfiguration.
  std::mutex               submit_mutex_;
  /// @brief Pairs with wake_: generation and stop changes are made under it.
  std::mutex               mutex_;
  /// @brief Notifies idle workers of a new job generation or a stop request.
  std::condition_variable  wake_;
  /// @brief The current job's body, borrowed from the submitter until the job finishes.
  std::atomic<const std::function<void(std::int64_t)>*> body_{nullptr};
  /// @brief The current job's unfinished chunks; zero means done.
  alignas(64) std::atomic<std::int64_t> remaining_{0};
  /// @brief Job generation, incremented under mutex_ on publication and polled lock-free by spinning workers.
  alignas(64) std::atomic<std::uint64_t> generation_{0};
  /// @brief Stop signal, set under mutex_ and polled lock-free by spinning workers.
  std::atomic<bool>        stop_{false};
  /// @brief First failure of the current job's chunks, rethrown by the submitter.
  std::exception_ptr       error_;
  /// @brief Protects error_.
  std::mutex               error_mutex_;
  /// @brief Pairs with done_ for a submitter that stopped spinning.
  std::mutex               done_mutex_;
  /// @brief Signals a sleeping submitter after all chunks have finished.
  std::condition_variable  done_;
};

/// @brief Returns the lazily constructed process-wide kernel pool.
Pool& pool() {
  /// @brief Persistent workers and submission state, joined when static destruction runs.
  static Pool instance;
  return instance;
}

}  // namespace

int get_num_threads() { return pool().threads(); }

void set_num_threads(int threads) { pool().set_threads(threads < 1 ? default_threads() : threads); }

void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t grain,
                  const std::function<void(std::int64_t, std::int64_t)>& fn) {
  if (end <= begin) return;
  const std::int64_t n = end - begin;
  grain                = std::max<std::int64_t>(grain, 1);
  const int threads    = tls_in_parallel ? 1 : get_num_threads();
  if (threads <= 1 || n <= grain) {
    fn(begin, end);
    return;
  }
  // A few chunks per thread balances uneven work without much overhead.
  const std::int64_t chunks =
      std::min({(n + grain - 1) / grain, std::int64_t{threads} * 4, kMaxChunks});
  const std::int64_t step   = (n + chunks - 1) / chunks;
  // Translate a claimed chunk number to a disjoint nonempty part of the caller's range.
  const std::function<void(std::int64_t)> body = [&](std::int64_t c) {
    const std::int64_t lo = begin + c * step;
    const std::int64_t hi = std::min(end, lo + step);
    if (lo < hi) fn(lo, hi);
  };
  if (!pool().run(chunks, body)) fn(begin, end);
}

std::string gemm_backend() {
#if defined(DATAMUNGE_HAVE_CBLAS) && defined(__APPLE__)
  return "accelerate";
#elif defined(DATAMUNGE_HAVE_OPENBLAS)
  return "openblas";
#elif defined(DATAMUNGE_HAVE_CBLAS)
  return "cblas";
#else
  return "inhouse-" + blas::isa();
#endif
}

}  // namespace datamunge::nn
