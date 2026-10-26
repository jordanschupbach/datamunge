// A small persistent thread pool for nn kernels.
//
// One job runs at a time: the caller splits [begin, end) into chunks, the
// workers and the caller take chunks from a shared counter, and the caller
// waits until every chunk is done. A thread that finds the pool busy (another
// user thread, or a nested parallel_for) runs its range serially instead.

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

namespace {

int default_threads() {
  if (const char* env = std::getenv("DATAMUNGE_NUM_THREADS")) {
    const int n = std::atoi(env);
    if (n > 0) return n;
  }
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

thread_local bool tls_in_parallel = false;

/// One parallel_for call. Workers hold a shared_ptr to the job they joined,
/// so a worker that wakes late can only touch that (finished) job's counters,
/// never the next job's.
struct Job {
  const std::function<void(std::int64_t)>* body{nullptr};
  std::int64_t                             chunks{0};
  std::atomic<std::int64_t>                next{0};
  std::int64_t                             remaining{0};  // guarded by mutex
  std::exception_ptr                       error;         // guarded by mutex
  std::mutex                               mutex;
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

class Pool {
 public:
  Pool() { resize(threads_ - 1); }
  ~Pool() { resize(0); }

  int  threads() const { return threads_; }
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

  int                      threads_{default_threads()};
  std::vector<std::thread> workers_;
  std::mutex               submit_mutex_, mutex_;
  std::condition_variable  wake_;
  std::shared_ptr<Job>     job_;  // guarded by mutex_
  std::uint64_t            generation_{0};
  bool                     stop_{false};
};

Pool& pool() {
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
