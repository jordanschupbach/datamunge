// SqueezeNet CIFAR-10 training-throughput benchmark.
//
// Reproduces the training loop of the SqueezeNet report
// (docs/mepml/examples/nn/.../squeezenet.mepml): CIFAR-10 batch 1, 8000 fit rows,
// batch 32, per-image pad/crop/flip augmentation, Adam with warmup + cosine
// schedule, dropout, kernel L2 term. The headline metric is epochs per minute:
// how many full passes over the 8000 rows fit into a fixed time budget.
//
// Build: `just bench-squeezenet` (target squeezenet_train in benchmarks/).
// Typical runs:
//   squeezenet_train --threads 16 --seconds 60
//   squeezenet_train --threads 16 --no-augment --steps 200
//   squeezenet_train --threads 16 --profile build/squeezenet.trace.json
//
// Every phase of a step is timed separately: augmentation (batch assembly),
// forward (model + loss), backward (autograd) and the optimizer update.

#include <datamunge/nn/augment.hpp>
#include <datamunge/nn/cifar.hpp>
#include <datamunge/nn/losses.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/profiler.hpp>
#include <datamunge/nn/schedules.hpp>
#include <datamunge/nn/squeezenet_models.hpp>
#include <datamunge/nn/training.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace n = datamunge::nn;
using clock_type = std::chrono::steady_clock;

namespace {

struct Options {
  int         threads       = 0;  // 0: library default
  int         batch         = 32;
  int         version       = 10;
  bool        bypass        = false;
  double      seconds       = 30;
  int         steps         = 0;  // >0 overrides seconds
  int         warmup        = 10;
  bool        augment       = true;
  bool        synthetic     = false;
  std::string profile;
  int         profile_steps = 5;
  std::string json;
  std::string label         = "native";
};

[[noreturn]] void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--threads N] [--batch N] [--version 10|11] [--bypass] [--seconds S | --steps N]\n"
               "          [--warmup N] [--no-augment] [--synthetic] [--profile TRACE.json] [--profile-steps N]\n"
               "          [--json OUT.json] [--label NAME]\n",
               argv0);
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", what);
        usage(argv[0]);
      }
      return argv[++i];
    };
    if (a == "--threads") o.threads = std::atoi(value("--threads"));
    else if (a == "--batch") o.batch = std::atoi(value("--batch"));
    else if (a == "--version") o.version = std::atoi(value("--version"));
    else if (a == "--bypass") o.bypass = true;
    else if (a == "--seconds") o.seconds = std::atof(value("--seconds"));
    else if (a == "--steps") o.steps = std::atoi(value("--steps"));
    else if (a == "--warmup") o.warmup = std::atoi(value("--warmup"));
    else if (a == "--no-augment") o.augment = false;
    else if (a == "--synthetic") o.synthetic = true;
    else if (a == "--profile") o.profile = value("--profile");
    else if (a == "--profile-steps") o.profile_steps = std::atoi(value("--profile-steps"));
    else if (a == "--json") o.json = value("--json");
    else if (a == "--label") o.label = value("--label");
    else usage(argv[0]);
  }
  if (o.batch < 1 || (o.version != 10 && o.version != 11) || o.seconds <= 0 || o.warmup < 0) usage(argv[0]);
  return o;
}

double seconds_since(clock_type::time_point t) {
  return std::chrono::duration<double>(clock_type::now() - t).count();
}

/// The 8000 fit rows of CIFAR-10 training batch 1 (or random data of the same shape).
struct Data {
  n::Tensor   x, y;  // [8000, 32, 32, 3] in [0, 1]; [8000] Int64
  std::string source;
};

Data load(bool synthetic) {
  const std::int64_t rows = 8000;
  std::filesystem::path cache;
  if (const char* p = std::getenv("DATAMUNGE_DATA_DIR")) cache = p;
  else if (const char* p = std::getenv("XDG_CACHE_HOME")) cache = std::filesystem::path(p) / "datamunge/datasets";
  else if (const char* p = std::getenv("HOME")) cache = std::filesystem::path(p) / ".cache/datamunge/datasets";
  const auto file = cache / "cifar10/data_batch_1.bin";
  if (!synthetic && std::filesystem::exists(file)) {
    const auto raw = n::data::LoadCifar({file.string()}, n::data::Cifar::k10);
    return {n::slice_axis(n::cast(raw.images, n::DType::Float32) / 255., 0, 0, rows).contiguous(),
            n::slice_axis(n::cast(raw.labels, n::DType::Int64), 0, 0, rows).contiguous(), file.string()};
  }
  n::Generator gen(7);
  return {n::uniform({rows, 32, 32, 3}, 0., 1., gen), n::uniform({rows}, 0., 10., gen).to(n::DType::Int64),
          "synthetic"};
}

/// Per-image zero pad 4, random 32x32 crop, random horizontal flip, then [0,1] -> [-1,1];
/// exactly the report's augmentation.
n::Tensor augment_batch(const n::Tensor& batch, n::Generator& rng) {
  std::vector<n::Tensor> transformed;
  transformed.reserve(static_cast<std::size_t>(batch.size(0)));
  for (std::int64_t row = 0; row < batch.size(0); ++row) {
    auto image = n::pad(n::slice_axis(batch, 0, row, row + 1), {{0, 0}, {4, 4}, {4, 4}, {0, 0}});
    image      = n::augment::random_crop(image, 32, 32, rng);
    transformed.push_back(n::augment::random_flip_left_right(image, rng));
  }
  return 2 * n::concat(transformed, 0) - 1;
}

struct Totals {
  double augment = 0, forward = 0, backward = 0, update = 0;
  int    steps   = 0;
  double loss    = 0;
};

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  if (opt.threads > 0) n::set_num_threads(opt.threads);
  const int threads = n::get_num_threads();

  const Data data = load(opt.synthetic);
  const std::int64_t rows = data.x.size(0);
  const int          epoch_steps = static_cast<int>((rows + opt.batch - 1) / opt.batch);

  n::models::SqueezeNetConfig cfg;
  cfg.version       = opt.version;
  cfg.simple_bypass = opt.bypass;
  cfg.cifar_stem    = true;
  cfg.image_size    = 32;
  cfg.classes       = 10;
  n::Generator      dropout_rng(101), augmentation(151);
  n::ParameterStore p(17);
  {
    n::NoGradGuard guard;
    n::models::SqueezeNet(p, n::zeros({1, 32, 32, 3}), cfg, false);
  }
  const auto params = p.parameters();

  constexpr double  peak = .0005;
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer  optimizer(p, peak, config);
  n::BatchOrder batches(static_cast<std::size_t>(rows), static_cast<std::size_t>(opt.batch), 2026);
  batches.begin_epoch();
  // The schedule length only changes learning-rate values, not the work per step.
  const std::int64_t schedule_steps = 10 * epoch_steps;

  // A fixed batch for the model-only measurement.
  n::Tensor fixed_x, fixed_y;
  if (!opt.augment) {
    auto rows_ = batches.next();
    fixed_x    = 2 * n::take_rows(data.x, rows_) - 1;
    fixed_y    = n::take_rows(data.y, rows_);
  }

  auto step = [&](Totals& t) {
    n::Tensor x, target;
    auto      t0 = clock_type::now();
    {
      n::ProfileScope scope("augment");
      if (opt.augment) {
        auto rows_ = batches.next();
        if (rows_.empty()) {
          batches.begin_epoch();
          rows_ = batches.next();
        }
        x      = augment_batch(n::take_rows(data.x, rows_), augmentation);
        target = n::take_rows(data.y, rows_);
      } else {
        x      = fixed_x;
        target = fixed_y;
      }
    }
    auto t1 = clock_type::now();
    n::Tensor objective, nll;
    {
      n::ProfileScope scope("forward");
      const auto logits = n::models::SqueezeNet(p, x, cfg, true, &dropout_rng).logits;
      nll               = n::sparse_softmax_cross_entropy(logits, target);
      objective         = nll;
      for (const auto& w : params)
        if (w.dim() >= 2) objective = objective + .00005 * n::sum(n::square(w));
    }
    auto t2 = clock_type::now();
    std::vector<n::Tensor> grads;
    {
      n::ProfileScope scope("backward");
      n::GradOptions  options;
      options.allow_unused = true;
      grads                = n::grad({objective}, params, {}, options);
    }
    auto t3 = clock_type::now();
    {
      n::ProfileScope scope("update");
      optimizer.set_learning_rate(n::schedule::warmup_cosine(peak, 0, 100, schedule_steps, optimizer.iterations()));
      optimizer.step(grads);
    }
    auto t4 = clock_type::now();
    t.augment += std::chrono::duration<double>(t1 - t0).count();
    t.forward += std::chrono::duration<double>(t2 - t1).count();
    t.backward += std::chrono::duration<double>(t3 - t2).count();
    t.update += std::chrono::duration<double>(t4 - t3).count();
    t.loss = nll.item();
    ++t.steps;
  };

  Totals warm;
  for (int i = 0; i < opt.warmup; ++i) step(warm);

  Totals     timed;
  const auto start = clock_type::now();
  if (opt.steps > 0) {
    for (int i = 0; i < opt.steps; ++i) step(timed);
  } else {
    while (seconds_since(start) < opt.seconds) step(timed);
  }
  const double wall = seconds_since(start);

  const double step_s     = wall / timed.steps;
  const double img_s      = opt.batch / step_s;
  const double epoch_s    = step_s * epoch_steps;
  const double epochs_min = 60. / epoch_s;
  const double phases     = timed.augment + timed.forward + timed.backward + timed.update;
  auto         share      = [&](double s) { return 100. * s / phases; };

  std::printf("%s: SqueezeNet v%d.%d%s, batch %d, %d threads, %s, data %s\n", opt.label.c_str(), opt.version / 10,
              opt.version % 10, opt.bypass ? " + bypass" : "", opt.batch, threads,
              opt.augment ? "augmented" : "fixed batch", data.source.c_str());
  std::printf("  %d timed steps in %.1f s: %.1f ms/step, %.0f img/s, %.1f s/epoch (%d steps), %.2f epochs/min\n",
              timed.steps, wall, 1e3 * step_s, img_s, epoch_s, epoch_steps, epochs_min);
  std::printf("  per step: augment %.1f ms (%.0f%%), forward %.1f ms (%.0f%%), backward %.1f ms (%.0f%%), update %.1f ms (%.0f%%); last NLL %.4f\n",
              1e3 * timed.augment / timed.steps, share(timed.augment), 1e3 * timed.forward / timed.steps,
              share(timed.forward), 1e3 * timed.backward / timed.steps, share(timed.backward),
              1e3 * timed.update / timed.steps, share(timed.update), timed.loss);

  if (!opt.json.empty()) {
    std::ofstream out(opt.json);
    out << "{\n  \"label\": \"" << json_escape(opt.label) << "\",\n  \"framework\": \"datamunge\",\n"
        << "  \"version\": " << opt.version << ",\n  \"bypass\": " << (opt.bypass ? "true" : "false") << ",\n"
        << "  \"batch\": " << opt.batch << ",\n  \"threads\": " << threads << ",\n"
        << "  \"augment\": " << (opt.augment ? "true" : "false") << ",\n"
        << "  \"data\": \"" << json_escape(data.source) << "\",\n  \"gemm_backend\": \"" << n::gemm_backend() << "\",\n"
        << "  \"timed_steps\": " << timed.steps << ",\n  \"wall_s\": " << wall << ",\n"
        << "  \"step_ms\": " << 1e3 * step_s << ",\n  \"images_per_s\": " << img_s << ",\n"
        << "  \"epoch_steps\": " << epoch_steps << ",\n  \"epoch_s\": " << epoch_s << ",\n"
        << "  \"epochs_per_min\": " << epochs_min << ",\n"
        << "  \"phase_ms\": {\"augment\": " << 1e3 * timed.augment / timed.steps << ", \"forward\": "
        << 1e3 * timed.forward / timed.steps << ", \"backward\": " << 1e3 * timed.backward / timed.steps
        << ", \"update\": " << 1e3 * timed.update / timed.steps << "},\n"
        << "  \"last_nll\": " << timed.loss << "\n}\n";
  }

  if (!opt.profile.empty()) {
    n::Profiler profiler;
    profiler.start();
    Totals profiled;
    for (int i = 0; i < opt.profile_steps; ++i) {
      n::ProfileScope scope("step");
      step(profiled);
    }
    profiler.stop();
    std::ofstream(opt.profile) << profiler.to_chrome_trace();
    std::printf("\nProfile of %d steps (%s):\n%s", opt.profile_steps, opt.profile.c_str(),
                profiler.to_markdown(40).c_str());
  }
  return std::isfinite(timed.loss) ? 0 : 1;
}
