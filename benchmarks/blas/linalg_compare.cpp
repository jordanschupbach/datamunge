// Dense decompositions of datamunge::nn (linalg.hpp): the built-in kernels
// against LAPACK, float64, square matrices, one matrix per call.
//
//   linalg_compare [--threads 1,8] [--sizes 16,128,1024] [--repeat 5] [--json out.json] [--quick]
//
// Both backends live in one library built with -DDATAMUNGE_USE_LAPACK=ON
// (`just bench-linalg`); set_linalg_backend switches between them, so the
// calls, conversions and allocations around the kernels are identical. The
// default sizes (16-96, then 128-1024) straddle the built-in kernels' switch
// points: Cholesky is blocked around the in-house GEMM from n = 65, LU from 96
// and the triangular solve from 16; the blocked Householder QR starts at 48,
// the symmetric eigensolver at 24 (16 for values only) and the SVD at 16.
//
// Threads: every count runs in its own process (this program re-runs itself)
// with OPENBLAS_NUM_THREADS, GOTO_NUM_THREADS and OMP_NUM_THREADS set to it,
// because the LAPACK is loaded with the library and OpenBLAS's OpenMP build
// follows OMP_NUM_THREADS, not a later openblas_set_num_threads (with the
// variable unset, its "1-thread" rows ran on several cores). The library's
// pool gets the same count. `just bench-linalg` loads the AVX-512 OpenBLAS of
// the blas-bench shell in place of the linked one. The main thread is placed
// as the library's pool places it before LAPACK starts any thread, so both
// run on the same die (placement.hpp).
// Each time is the median over --repeat rounds (default 5) of the median of
// at least three calls after a warm-up, the rounds
// cycling through every op and backend so drift hits both alike. On a busy or
// throttling machine the 8-thread rows still vary by 20-40 % between runs.
// max_err is the relative residual of the result (e.g. |L L^T - A| / |A|).

#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/parallel.hpp>
#include "placement.hpp"
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>


namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

// OpenBLAS's thread controls, if the LAPACK in use is an OpenBLAS. Looked up at
// run time: the library, not this driver, links it.
template <typename Fn>
Fn openblas_fn(const char* name) {
  return reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, name));
}
const auto openblas_set_num_threads = openblas_fn<void (*)(int)>("openblas_set_num_threads");
const auto openblas_get_num_threads = openblas_fn<int (*)()>("openblas_get_num_threads");
const auto openblas_get_config      = openblas_fn<char* (*)()>("openblas_get_config");

std::vector<std::int64_t> parse_list(const std::string& s) {
  std::vector<std::int64_t> out;
  std::stringstream         in(s);
  for (std::string item; std::getline(in, item, ',');)
    if (!item.empty()) out.push_back(std::stoll(item));
  return out;
}

double median_seconds(const std::function<void()>& fn) {
  fn();
  std::vector<double> t;
  const auto          start = std::chrono::steady_clock::now();
  for (int r = 0; r < 200; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (r >= 2 && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 0.5) break;
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

double rel(const Tensor& got, const Tensor& want) {
  const double scale = std::max(1.0, nn::max(nn::abs(want)).item());
  return nn::max(nn::abs(nn::sub(got, want))).item() / scale;
}

/// Runs this program once per thread count with the thread variables set
/// (see the top of the file) and joins the JSON parts into `json`.
int run_per_thread_count(const std::vector<std::int64_t>& threads, int argc, char** argv, const std::string& json) {
  std::vector<std::string> parts;
  for (std::size_t i = 0; i < threads.size(); ++i) {
    const std::string        t = std::to_string(threads[i]);
    std::vector<std::string> args{"linalg_compare", "--child", "--threads", t};
    for (int a = 1; a < argc; ++a) {  // forward everything but --threads and --json
      const std::string arg = argv[a];
      if (arg == "--threads" || arg == "--json") ++a;
      else args.push_back(arg);
    }
    if (i > 0) args.emplace_back("--no-header");
    if (!json.empty()) {
      parts.push_back(json + ".threads" + t);
      args.emplace_back("--json");
      args.push_back(parts.back());
    }
    for (const char* var : {"OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS"}) setenv(var, t.c_str(), 1);
    std::vector<char*> child_argv;
    for (auto& a : args) child_argv.push_back(a.data());
    child_argv.push_back(nullptr);
    std::fflush(stdout);
    pid_t pid = 0;
    if (posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, child_argv.data(), environ) != 0) {
      std::perror("posix_spawn");
      return 1;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  }
  if (json.empty()) return 0;
  std::ofstream out(json);
  out << "[\n";
  bool first = true;
  for (const auto& part : parts) {
    std::ifstream     in(part);
    std::stringstream body;
    body << in.rdbuf();
    std::string rows = body.str();
    const auto  open = rows.find('['), close = rows.rfind(']');
    rows = open == std::string::npos || close == std::string::npos ? "" : rows.substr(open + 1, close - open - 1);
    while (!rows.empty() && (rows.front() == '\n' || rows.front() == ' ')) rows.erase(rows.begin());
    while (!rows.empty() && (rows.back() == '\n' || rows.back() == ' ')) rows.pop_back();
    if (!rows.empty()) {
      out << (first ? "" : ",\n") << rows;
      first = false;
    }
    std::remove(part.c_str());
  }
  out << "\n]\n";
  return 0;
}

/// One operation: its label, the call, and the residual of one result.
struct Op {
  std::string                    name;
  std::function<void()>          call;
  std::function<double()>        residual;
};

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::int64_t> threads{1, 8}, sizes{16, 32, 48, 64, 96, 128, 256, 512, 1024};
  int                       repeat   = 5;
  std::string               json;
  bool                      child = false, header = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg  = argv[i];
    auto              next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (arg == "--threads") threads = parse_list(next());
    else if (arg == "--sizes") sizes = parse_list(next());
    else if (arg == "--repeat") repeat = std::max(1, std::stoi(next()));
    else if (arg == "--json") json = next();
    else if (arg == "--quick") sizes = {256}, threads = {1};
    else if (arg == "--child") child = true;
    else if (arg == "--no-header") header = false;
    else {
      std::fprintf(stderr, "usage: linalg_compare [--threads 1,8] [--sizes 128,512] [--repeat 5] "
                           "[--json out.json] [--quick]\n");
      return 2;
    }
  }
  if (!child) return run_per_thread_count(threads, argc, argv, json);
  std::vector<nn::LinalgBackend> backends{nn::LinalgBackend::Builtin};
  if (nn::linalg_has_lapack()) backends.push_back(nn::LinalgBackend::Lapack);
  else std::fprintf(stderr, "# built without LAPACK (-DDATAMUNGE_USE_LAPACK=ON): built-in rows only\n");
  if (header) {
    if (openblas_get_config) std::printf("# LAPACK: OpenBLAS %s\n", openblas_get_config());
    std::printf("%-22s %6s %4s %-8s %12s %10s\n", "op", "n", "thr", "backend", "ms/call", "max_err");
  }

  std::ofstream out;
  if (!json.empty()) out.open(json), out << "[\n";
  bool       first = true;
  nn::NoGradGuard guard;
  for (const std::int64_t t : threads) {
    bench::place_main_thread(static_cast<int>(t));  // before LAPACK starts its threads (placement.hpp)
    nn::set_num_threads(static_cast<int>(t));
    if (openblas_set_num_threads) openblas_set_num_threads(static_cast<int>(t));
    if (openblas_get_num_threads && openblas_get_num_threads() != t)
      std::fprintf(stderr, "# warning: OpenBLAS runs %d threads, %lld requested\n", openblas_get_num_threads(),
                   static_cast<long long>(t));
    for (const std::int64_t n : sizes) {
      nn::Generator gen(static_cast<std::uint64_t>(n));
      const Tensor  x   = nn::normal({n, n}, 0.0, 1.0, gen, DType::Float64);
      const Tensor  eye = nn::eye(n, DType::Float64);
      const Tensor  spd = nn::add(nn::div(nn::matmul(x, x, false, true), double(n)), eye);
      const Tensor  g   = nn::add(x, nn::mul(eye, 4.0));
      const Tensor  b1 = nn::normal({n, 1}, 0.0, 1.0, gen, DType::Float64), bn = nn::normal({n, n}, 0.0, 1.0, gen, DType::Float64);
      const Tensor  l   = [&] {
        nn::set_linalg_backend(nn::LinalgBackend::Builtin);
        return nn::cholesky(spd);
      }();
      const std::vector<Op> ops{
          {"cholesky", [&] { (void)nn::cholesky(spd); },
           [&] { const Tensor c = nn::cholesky(spd); return rel(nn::matmul(c, c, false, true), spd); }},
          {"solve k=1", [&] { (void)nn::solve(g, b1); }, [&] { return rel(nn::matmul(g, nn::solve(g, b1)), b1); }},
          {"solve k=n", [&] { (void)nn::solve(g, bn); }, [&] { return rel(nn::matmul(g, nn::solve(g, bn)), bn); }},
          {"slogdet", [&] { (void)nn::slogdet(g); }, [] { return 0.0; }},
          {"triangular_solve k=1", [&] { (void)nn::triangular_solve(l, b1); },
           [&] { return rel(nn::matmul(l, nn::triangular_solve(l, b1)), b1); }},
          {"triangular_solve k=n", [&] { (void)nn::triangular_solve(l, bn); },
           [&] { return rel(nn::matmul(l, nn::triangular_solve(l, bn)), bn); }},
          {"qr", [&] { (void)nn::qr(g); },
           [&] { const auto r = nn::qr(g); return rel(nn::matmul(r.q, r.r), g); }},
          {"eigh", [&] { (void)nn::eigh(spd); },
           [&] { const auto r = nn::eigh(spd); return rel(nn::matmul(spd, r.v), nn::mul(r.v, nn::expand_dims(r.e, -2))); }},
          {"svd", [&] { (void)nn::svd(g); },
           [&] {
             const auto r = nn::svd(g);
             return rel(nn::matmul(nn::mul(r.u, nn::expand_dims(r.s, -2)), r.v, false, true), g);
           }},
          {"eigvalsh", [&] { (void)nn::eigvalsh(spd); }, [&] { return rel(nn::eigvalsh(spd), nn::eigh(spd).e); }},
          {"svd values", [&] { (void)nn::svd(g, false); }, [&] { return rel(nn::svd(g, false).s, nn::svd(g).s); }},
      };
      // Rounds over all (op, backend) pairs, so slow drift (temperature,
      // other processes) hits both backends alike; each row reports the
      // median of its rounds' medians.
      std::vector<std::vector<double>> seconds(ops.size() * backends.size());
      for (int round = 0; round < repeat; ++round)
        for (std::size_t o = 0; o < ops.size(); ++o) {
          for (std::size_t b = 0; b < backends.size(); ++b) {
            nn::set_linalg_backend(backends[b]);
            seconds[o * backends.size() + b].push_back(median_seconds(ops[o].call));
          }
        }
      for (std::size_t o = 0; o < ops.size(); ++o) {
        const Op& op = ops[o];
        for (std::size_t b = 0; b < backends.size(); ++b) {
          nn::set_linalg_backend(backends[b]);
          auto& runs = seconds[o * backends.size() + b];
          std::sort(runs.begin(), runs.end());
          const double      sec  = runs[runs.size() / 2];
          const double      err  = op.residual();
          const std::string name = backends[b] == nn::LinalgBackend::Lapack ? "lapack" : "inhouse";
          std::printf("%-22s %6lld %4lld %-8s %12.3f %10.1e\n", op.name.c_str(), (long long)n, (long long)t,
                      name.c_str(), sec * 1e3, err);
          std::fflush(stdout);
          if (out) {
            out << (first ? "" : ",\n") << "  {\"op\": \"linalg\", \"case\": \"" << op.name << " n=" << n
                << "\", \"n\": " << n << ", \"dtype\": \"f64\", \"threads\": " << t << ", \"backend\": \"" << name
                << "\", \"seconds\": " << sec << ", \"ms\": " << sec * 1e3 << ", \"rounds\": " << runs.size()
                << ", \"max_err\": " << err << "}";
            first = false;
          }
        }
      }
    }
  }
  if (out) out << "\n]\n";
  return 0;
}
