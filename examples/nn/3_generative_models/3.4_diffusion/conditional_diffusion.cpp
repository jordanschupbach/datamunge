// CMD: just run-nn conditional_diffusion
// Direct conditional MNIST diffusion, paired label counterfactuals and fit-only
// class Gaussian baselines.
//
// Port of tflow's conditional_diffusion (common/ml/conditional_diffusion.h ->
// examples/nn/common/conditional_diffusion.hpp). Reuses, with NO additional
// training, the frozen 5856-parameter NULL-dropout conditional epsilon MLP of
// diffusion_guidance: guidance=auto (our diffusion_guidance port's output in
// output/nn/models/diffusion_guidance when present, else tflow's recorded
// output/models/diffusion_guidance; guidance=ours|tf|DIR), together with that
// run's data.npz. Samples 10 classes x 8 paired latents with the correct labels,
// all-NULL labels and labels shifted by +3; held-out correct / wrong / NULL
// epsilon queries; fit-only per-class and global independent Gaussian pixel
// priors as analytic epsilon baselines; counterfactual single-step denoising.
//
// Checks (as tflow): restore bit-exact vs the source model_arrays.npz; finite
// fields; last step ignores noise; NULL rows repeat bit-exactly; the +3 label
// permutation permutes every paired trajectory (< 5e-6); labels change only the
// conditioning, not the query noise/targets; 100 fit sources per class; weights
// immutable. With TF's source weights the records are compared with TF's.
// Deviations: no independent_audit.json. Options: guidance=auto|ours|tf|DIR
// out=DIR tf=DIR threads=4. Output: output/nn/models/conditional_diffusion.
#include "common/conditional_diffusion.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using namespace diffusion;

Map Sample(ConditionalDiffusionModel& model, int mode) {
  const int                 n = 80;
  std::vector<std::int32_t> label_v(n);
  for (int b = 0; b < n; ++b) label_v[b] = mode == 1 ? 10 : mode == 2 ? (b / 8 + 3) % 10 : b / 8;
  const Tensor        labels  = Ints(label_v);
  const Tensor        initial = Cycle(Gaussian8(8, 16201), n);
  Tensor              current = initial;
  std::vector<Tensor> trajectory{initial}, innovations;
  StepStack           stack;
  for (int step = 0; step < 32; ++step) {
    const Tensor time = FullInt(n, 31 - step), noise = Cycle(Gaussian8(8, 16202 + step), n);
    innovations.push_back(Cycle(Gaussian8(8, 16202 + step, true), n));
    const auto out = model.Step(current, time, labels, noise);
    Finite(out);
    stack.add(out);
    if (step == 31)
      check::Require(Bits(model.Step(current, time, labels, nn::full(noise.shape(), 123.0)).at("next")) == Bits(out.at("next")),
                     "conditional last DDPM step ignores noise");
    current = out.at("next");
    trajectory.push_back(current);
  }
  if (mode == 1) {
    const auto v = current.to_vector<float>();
    for (int b = 8; b < n; ++b)
      for (int j = 0; j < 64; ++j)
        check::Require(v[b * 64 + j] == v[(b % 8) * 64 + j], "NULL conditional sampler ignores row class with paired noise");
  }
  Map saved = stack.stacked();
  saved["initial"]     = initial;
  saved["labels"]      = labels;
  saved["trajectory"]  = nn::stack(trajectory, 0);
  saved["innovations"] = nn::stack(innovations, 0);
  saved["images"]      = current;
  return saved;
}

void Run(const fs::path& source, const fs::path& root, const fs::path& tf) {
  const fs::path frozen = FrozenFile(source / "conditional");
  check::Require(!frozen.empty() && fs::exists(source / "data.npz"),
                 "conditional diffusion requires checkpoint: run diffusion_guidance first, or guidance=PATH (" + source.string() + ")");
  std::cout << "frozen conditional epsilon weights: " << frozen.string() << "\n";
  Map data = Load(source / "data.npz");
  for (const std::string key : {"validation_noise", "validation_time", "test_noise", "test_time"}) data.erase(key);
  const Tensor              noise  = Gaussian8(200, 16103);
  const auto                labels = data.at("test_labels").to_vector<std::int32_t>();
  std::vector<std::int32_t> time_v(200), wrong_v(200);
  for (int b = 0; b < 200; ++b) time_v[b] = b % 32, wrong_v[b] = (labels[b] + 3) % 10;
  const Tensor time = Ints(time_v), wrong = Ints(wrong_v), nulls = FullInt(200, 10);
  data["query_noise"]        = noise;
  data["query_time"]         = time;
  data["query_wrong_labels"] = wrong;
  data["query_null_labels"]  = nulls;
  Save(root / "data.npz", data);
  for (const auto& [from, to] : std::vector<std::pair<fs::path, fs::path>>{{source / "data_manifest.json", root / "data_manifest.json"},
                                                                           {source / "conditional/metrics.json", root / "training_metrics.json"}})
    if (fs::exists(from)) fs::copy_file(from, to, fs::copy_options::overwrite_existing);
  ConditionalDiffusionModel model;
  replay::LoadInto(model.store, frozen);
  const auto arrays = model.Arrays();
  check::Require(model.store.count() == 5856 && arrays.size() == 6, "frozen 5856 conditional weights 6 tensors");
  if (fs::exists(source / "conditional/model_arrays.npz")) {
    const auto expected = Load(source / "conditional/model_arrays.npz");
    for (const auto& [key, value] : arrays) check::Require(Bits(value) == Bits(expected.at(key)), "direct conditional restore BITEXACT source");
  }
  Save(root / "model_arrays.npz", arrays);
  Save(root / "schedule.npz", ScheduleArrays());
  const auto                 before = model.Snapshot();
  std::vector<std::string>   names{"conditional", "null", "shifted_labels"};
  std::map<std::string, Map> samples;
  for (int mode = 0; mode < 3; ++mode) {
    const auto sample = Sample(model, mode);
    Save(root / (names[mode] + "_samples.npz"), sample);
    Png(root / (names[mode] + "_samples.png"), sample.at("images"), 8);
    samples.emplace(names[mode], sample);
    std::cout << names[mode] << ":10rowsx8pairedLatents/32reverseSteps PASS" << std::endl;
  }
  double     permutation_error = 0;
  const auto shifted = samples.at("shifted_labels").at("trajectory").to_vector<float>(),
             plain   = samples.at("conditional").at("trajectory").to_vector<float>();
  for (int step = 0; step <= 32; ++step)
    for (int b = 0; b < 80; ++b)
      for (int j = 0; j < 64; ++j) {
        const int mapped  = ((b / 8 + 3) % 10) * 8 + b % 8;
        permutation_error = std::max(permutation_error, std::abs(double(shifted[(step * 80 + b) * 64 + j]) - plain[(step * 80 + mapped) * 64 + j]));
      }
  check::Require(permutation_error < 5e-6, "label permutation permutes ALL paired conditional trajectories");
  std::vector<std::string> query_names{"conditional", "shifted_labels", "null"};
  std::vector<Tensor>      conditions{data.at("test_labels"), wrong, nulls};
  std::vector<Map>         queries;
  for (int mode = 0; mode < 3; ++mode) {
    const auto query = model.Query(data.at("test_images"), noise, time, conditions[mode]);
    Finite(query);
    Save(root / (query_names[mode] + "_queries.npz"), query);
    const auto probe = model.Step(query.at("noised"), time, conditions[mode], noise);
    Finite(probe);
    Save(root / (query_names[mode] + "_query_probe.npz"), probe);
    queries.push_back(query);
  }
  check::Require(Bits(queries[0].at("noised")) == Bits(queries[1].at("noised")) && Bits(queries[0].at("noised")) == Bits(queries[2].at("noised")),
                 "labels change condition NOT query noise or targets");

  // Fit-only class and global independent Gaussian pixel priors.
  const auto          fit = data.at("fit_images").to_vector<float>();
  const auto          fit_labels = data.at("fit_labels").to_vector<std::int32_t>();
  std::vector<double> means(640, 0), variances(640, 0), global_mean(64, 0), global_variance(64, 0), class_baseline(200 * 64),
      global_baseline(200 * 64);
  std::vector<std::int32_t> counts(10, 0);
  for (int b = 0; b < 1000; ++b) {
    const int c = fit_labels[b];
    ++counts[c];
    for (int j = 0; j < 64; ++j) means[c * 64 + j] += double(fit[b * 64 + j]) / 100, global_mean[j] += double(fit[b * 64 + j]) / 1000;
  }
  for (int b = 0; b < 1000; ++b) {
    const int c = fit_labels[b];
    for (int j = 0; j < 64; ++j) {
      variances[c * 64 + j] += std::pow(double(fit[b * 64 + j]) - means[c * 64 + j], 2) / 100;
      global_variance[j] += std::pow(double(fit[b * 64 + j]) - global_mean[j], 2) / 1000;
    }
  }
  for (int c = 0; c < 10; ++c) check::Require(counts[c] == 100, "all class Gaussian priors FIT ONLY 100 sources each");
  const auto& bars = DdpmSchedule().at("alpha_bar");
  const auto  qx = queries[0].at("noised").to_vector<float>(), nv = noise.to_vector<float>();
  double      class_mse = 0, global_mse = 0;
  for (int b = 0; b < 200; ++b) {
    const int    c   = labels[b];
    const double bar = bars[time_v[b]];
    for (int j = 0; j < 64; ++j) {
      const int    k = b * 64 + j;
      const double x = qx[k];
      const double a = std::sqrt(1 - bar) / (bar * variances[c * 64 + j] + 1 - bar) * (x - std::sqrt(bar) * means[c * 64 + j]);
      const double g = std::sqrt(1 - bar) / (bar * global_variance[j] + 1 - bar) * (x - std::sqrt(bar) * global_mean[j]);
      class_baseline[k] = a, global_baseline[k] = g;
      class_mse += std::pow(a - nv[k], 2) / (200 * 64);
      global_mse += std::pow(g - nv[k], 2) / (200 * 64);
    }
  }
  Save(root / "fit_gaussian_baselines.npz", {{"class_mean", nn::from_vector(means, {10, 64})},
                                             {"class_variance", nn::from_vector(variances, {10, 64})},
                                             {"class_counts", Ints(counts)},
                                             {"global_mean", nn::from_vector(global_mean, {64})},
                                             {"global_variance", nn::from_vector(global_variance, {64})},
                                             {"class_epsilon", nn::from_vector(class_baseline, {200, 8, 8, 1})},
                                             {"global_epsilon", nn::from_vector(global_baseline, {200, 8, 8, 1})}});
  std::ofstream per_class(root / "per_class_queries.tsv");
  per_class.precision(17);
  per_class << "class\tcount\tcorrect_epsilon_mse\twrong_epsilon_mse\tnull_epsilon_mse\tclass_gaussian_epsilon_mse\n";
  std::vector<std::vector<float>> qe;
  for (const auto& q : queries) qe.push_back(q.at("epsilon").to_vector<float>());
  for (int c = 0; c < 10; ++c) {
    int    count   = 0;
    double loss[4] = {};
    for (int b = 0; b < 200; ++b)
      if (labels[b] == c) {
        ++count;
        for (int j = 0; j < 64; ++j) {
          const int k = b * 64 + j;
          for (int mode = 0; mode < 3; ++mode) loss[mode] += std::pow(double(qe[mode][k]) - nv[k], 2);
          loss[3] += std::pow(class_baseline[k] - nv[k], 2);
        }
      }
    per_class << c << '\t' << count;
    for (double value : loss) per_class << '\t' << value / (count * 64);
    per_class << '\n';
  }

  // Counterfactual single-step denoising of one held-out source per class.
  std::vector<std::int64_t> rows;
  for (int c = 0; c < 10; ++c)
    for (int b = 0; b < 200; ++b)
      if (labels[b] == c) {
        rows.push_back(b);
        break;
      }
  const Tensor cases = Rows(data.at("test_images"), rows);
  std::vector<std::int32_t> case_labels(10), case_wrong(10), case_ids(10);
  const auto                source_ids = data.at("test_source_ids").to_vector<std::int32_t>();
  for (int c = 0; c < 10; ++c) case_labels[c] = c, case_wrong[c] = (c + 3) % 10, case_ids[c] = source_ids[rows[c]];
  const Tensor case_noise = Gaussian8(10, 16107);
  Map          probes{{"truth", cases}, {"labels", Ints(case_labels)}, {"source_ids", Ints(case_ids)}, {"noise", case_noise}};
  std::vector<std::vector<Tensor>> cells(10);
  for (int c = 0; c < 10; ++c) cells[c].push_back(Row(cases, c));
  const std::vector<Tensor> label_sets{Ints(case_labels), Ints(case_wrong), FullInt(10, 10)};
  for (int level = 0; level < 3; ++level) {
    const Tensor tt      = FullInt(10, level * 8 + 7);
    const auto   correct = model.Query(cases, case_noise, tt, label_sets[0]);
    probes["time" + std::to_string(level)]   = tt;
    probes["noised" + std::to_string(level)] = correct.at("noised");
    for (int c = 0; c < 10; ++c) cells[c].push_back(Row(correct.at("noised"), c));
    for (int mode = 0; mode < 3; ++mode) {
      const auto out = model.Step(correct.at("noised"), tt, label_sets[mode], nn::zeros({10, 8, 8, 1}));
      probes["x0_" + query_names[mode] + std::to_string(level)]      = out.at("x0_clipped");
      probes["epsilon_" + query_names[mode] + std::to_string(level)] = out.at("epsilon");
      for (int c = 0; c < 10; ++c) cells[c].push_back(Row(out.at("x0_clipped"), c));
    }
  }
  const Tensor gallery = Gallery(cells);
  probes["gallery"]    = gallery;
  Save(root / "counterfactual_denoising.npz", probes);
  Png(root / "counterfactual_denoising.png", gallery, 13);
  check::Require(before == model.Snapshot(), "ALL direct conditional queries-sampling-controls leave 6 weights immutable");
  const double loss0 = queries[0].at("loss").item<float>(), loss1 = queries[1].at("loss").item<float>(),
               loss2 = queries[2].at("loss").item<float>();
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters_frozen\":5856,\"parameter_tensors\":6,\"additional_training_updates\":0,\"samples_each\":80,"
          "\"sampler_modes\":3,\"class_shift_permutation_max_error\":"
       << permutation_error << ",\"null_row_invariance_bitexact\":true,\"test_correct_epsilon_mse\":" << loss0
       << ",\"test_wrong_epsilon_mse\":" << loss1 << ",\"test_null_epsilon_mse\":" << loss2
       << ",\"fit_class_gaussian_epsilon_mse\":" << class_mse << ",\"fit_global_gaussian_epsilon_mse\":" << global_mse
       << ",\"frozen_weights\":\"" << frozen.string() << "\"}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Class-conditional MNIST diffusion</title><style>body{font:16px system-ui;"
          "margin:30px;max-width:1150px}img{max-width:100%;image-rendering:pixelated}</style><h1>Direct class-conditional MNIST "
          "diffusion</h1><p>Frozen three-epoch 5856-parameter conditional epsilon MLP from diffusion_guidance, NO additional "
          "training, direct class-conditioned DDPM posterior. Datamunge port of tflow's conditional_diffusion.</p><h2>First 8 "
          "paired initial latents per class</h2><img src='conditional_samples.png'><h2>NULL control</h2><img "
          "src='null_samples.png'><h2>Counterfactual label+3 control</h2><img src='shifted_labels_samples.png'><h2>Held-out "
          "correct/wrong/NULL single-step x0 predictions</h2><p>Columns truth, then noisy/correct/wrong(+3)/NULL groups at "
          "physical levels 8, 16, 24.</p><img src='counterfactual_denoising.png'><p>See metrics.json, per_class_queries.tsv, "
          "fit_gaussian_baselines.npz.</p>";
  check::Require(bool(meta) && bool(html) && bool(per_class), "complete direct conditional study records");
  std::cout << "test epsilon MSE correct/wrong/NULL = " << loss0 << "/" << loss1 << "/" << loss2 << ", fit class/global Gaussian = "
            << class_mse << "/" << global_mse
            << "\ndirectConditional/NULLinvariance/ALLlabelPermutation/queryTargetsFixed/fitOnlyClassBaselines/restore/immutable "
               "PASS;classPermutationError="
            << permutation_error << "\n";
  if (fs::weakly_canonical(source) == fs::weakly_canonical(replay::TfModels("diffusion_guidance")) && fs::exists(tf / "metrics.json")) {
    std::cout << "TF parity (same frozen TF weights):\n";
    double worst = 0;
    for (const auto& name : names) worst = std::max(worst, bpix::Compare(tf, name + "_samples.npz", samples.at(name)));
    for (int mode = 0; mode < 3; ++mode) worst = std::max(worst, bpix::Compare(tf, query_names[mode] + "_queries.npz", queries[mode]));
    bpix::Compare(tf, "fit_gaussian_baselines.npz", Load(root / "fit_gaussian_baselines.npz"));
    check::Require(worst < 1e-3, "conditional samplers on TF's weights reproduce TF's records");
  }
  std::cout << "Artifacts: " << root.string() << "\nAll conditional_diffusion checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = diffusion::ParseOptions(argc, argv, "conditional_diffusion", {"guidance"},
                                           "guidance=auto|ours|tf|DIR out=PATH tf=DIR threads=N");
    const fs::path ours = paths::OutputRoot() / "models" / "diffusion_guidance", tf = replay::TfModels("diffusion_guidance");
    const auto     choice = o.extra.at("guidance");
    const fs::path source = choice == "auto" ? (diffusion::FrozenFile(ours / "conditional").empty() ? tf : ours)
                                             : diffusion::FrozenDir(choice, ours, tf);
    Run(source, o.out, o.tf);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
