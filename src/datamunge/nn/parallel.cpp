/// @file parallel.cpp
/// @brief Persistent thread pool and serial fallback for neural network kernels.
///
/// One job runs at a time: workers and the caller claim chunks from a shared
/// counter, and the caller waits for all claimed work. Concurrent submitters
/// and nested parallel_for calls use serial fallback instead of waiting on the pool.

#include "datamunge/nn/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace datamunge::nn {

/// @brief Process-wide pool ownership and per-thread nesting state.
namespace {

/// @brief Uses a positive DATAMUNGE_NUM_THREADS value, else hardware concurrency with a fallback of one.
int default_threads() {
  if (const char* env = std::getenv("DATAMUNGE_NUM_THREADS")) {
    const int n = std::atoi(env);
    if (n > 0) return n;
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

/// @brief Whether this thread is already in a parallel region, preventing recursive pool submission.
thread_local bool tls_in_parallel = false;

/// One parallel_for call. Workers hold a shared_ptr to the job they joined,
/// so a worker that wakes late can only touch that (finished) job's counters,
/// never the next job's.
struct Job {
  /// @brief Borrowed callback; the submitting caller keeps it alive until every chunk finishes.
  const std::function<void(std::int64_t)>* body{nullptr};
  /// @brief Total chunk indices available to be claimed, in [0, chunks).
  std::int64_t                             chunks{0};
  /// @brief Atomic index for the next chunk claimed by any participating thread.
  std::atomic<std::int64_t>                next{0};
  /// @brief Unfinished chunk count, published once per participant while holding mutex.
  std::int64_t                             remaining{0};
  /// @brief First published callback failure, guarded by mutex and rethrown by the submitter.
  std::exception_ptr                       error;
  /// @brief Protects completion accounting and the stored exception.
  std::mutex                               mutex;
  /// @brief Signals the submitter after all chunks have finished.
  std::condition_variable                  done;

  /// Runs chunks until none are left.
  void work() {
    const bool was  = tls_in_parallel;
    tls_in_parallel = true;
    std::int64_t completed = 0;
    std::exception_ptr failure;
    while (true) {
      const std::int64_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= chunks) break;
      try {
        (*body)(i);
      } catch (...) {
        if (!failure) failure = std::current_exception();
      }
      ++completed;
    }
    // Publish completion once per participant rather than once per chunk.
    // The caller still waits for every claimed chunk before body can expire.
    if (completed != 0) {
      std::lock_guard<std::mutex> lock(mutex);
      if (failure && !error) error = failure;
      remaining -= completed;
      if (remaining == 0) done.notify_all();
    }
    tls_in_parallel = was;
  }
};

/// @brief Serializes submitted jobs and retains worker threads between kernel calls.
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

  /// Returns false (without running anything) when the pool is busy.
  bool run(std::int64_t chunks, const std::function<void(std::int64_t)>& body) {
    std::unique_lock<std::mutex> submit(submit_mutex_, std::try_to_lock);
    if (!submit.owns_lock()) return false;
    auto job       = std::make_shared<Job>();
    job->body      = &body;
    job->chunks    = chunks;
    job->remaining = chunks;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_ = job;
      ++generation_;
    }
    wake_.notify_all();
    job->work();
    {
      std::unique_lock<std::mutex> lock(job->mutex);
      job->done.wait(lock, [&] { return job->remaining == 0; });
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_.reset();
    }
    if (job->error) std::rethrow_exception(job->error);
    return true;
  }

 private:
  /// @brief Stops and joins current workers, then creates the requested number of worker threads.
  /// The caller must exclude active submissions, except during initial construction/destruction.
  void resize(int workers) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : workers_) t.join();
    workers_.clear();
    stop_ = false;
    for (int i = 0; i < workers; ++i) workers_.emplace_back([this] { loop(); });
  }

  /// @brief Worker entry point: waits for new job generations and contributes chunks until stopped.
  void loop() {
    tls_in_parallel = true;
    std::uint64_t seen;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      seen = generation_;
    }
    while (true) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        job  = job_;
      }
      if (job) job->work();
    }
  }

  /// @brief Configured total threads, counting both workers and a submitting caller.
  int                      threads_{default_threads()};
  /// @brief Owned persistent workers, joined before replacement or destruction.
  std::vector<std::thread> workers_;
  /// @brief Serializes whole submissions and pool reconfiguration.
  std::mutex               submit_mutex_;
  /// @brief Protects the published job, generation counter, and worker stop signal.
  std::mutex               mutex_;
  /// @brief Notifies idle workers of a new job generation or a stop request.
  std::condition_variable  wake_;
  /// @brief Published job retained by participating workers, guarded by mutex_.
  std::shared_ptr<Job>     job_;
  /// @brief Job generation observed by workers, incremented under mutex_ on publication.
  std::uint64_t            generation_{0};
  /// @brief Stop signal checked by workers while holding mutex_.
  bool                     stop_{false};
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
  const std::int64_t chunks = std::min<std::int64_t>((n + grain - 1) / grain, std::int64_t{threads} * 4);
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
  return "builtin";
#endif
}

}  // namespace datamunge::nn
