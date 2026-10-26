// CMD: just run-nn ddim
// DDIM sampler comparison on a frozen short-trained DDPM, without additional optimization.
//
// Port of tflow's ddim (common/{nn,ml}/ddim.h -> examples/nn/common/ddim.hpp).
// Restores the 5504 DDPM epsilon weights (ddpm=auto: our ddpm port's final.npz
// in output/nn/models/ddpm when present, else tflow's recorded
// output/models/ddpm/model_arrays.npz; ddpm=ours|tf|DIR) and compares 8/16/32
// respaced levels round(31 j / (K - 1)) with eta 0 and 1, on the same 32 initial
// Gaussian draws and shared innovation streams. sigma_eta = eta sqrt((1 - barprev)
// / (1 - bar) (1 - bar / barprev)); x_prev = sqrt(barprev) x0clipped + sqrt(1 -
// barprev - sigma^2) eps_consistent + sigma z with eps_consistent recomputed
// AFTER x0 clipping, so eta 1 on all 32 levels equals the DDPM posterior.
//
// Checks (as tflow): restore matches the frozen weights bit-exactly; eta 0 ignores
// every innovation bit-exactly; eta 1 / 32 steps reproduces the DDPM sample
// trajectory (samples.npz of the frozen run) within 5e-5; eta and the initial
// latent change samples; all samplers leave the weights immutable. When the
// frozen weights are TF's, every configuration is also compared with TF's
// recorded eta*_steps*.npz. Deviations: no independent_audit.json. Options:
// ddpm=auto|ours|tf|DIR out=DIR threads=4. Output: output/nn/models/ddim.
#include "common/ddim.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using namespace diffusion;

Map Sample(DdimModel& model, int steps, float eta, std::uint32_t initial_seed = 11201, std::uint32_t innovation_seed = 11202) {
  const int        n = 32;
  std::vector<int> levels;
  for (int j = steps - 1; j >= 0; --j) levels.push_back(int(std::round(31. * j / (steps - 1))));
  std::vector<std::int32_t> times(steps), previous(steps);
  const Tensor              initial = Gaussian8(n, initial_seed);
  Tensor                    current = initial;
  std::vector<Tensor>       trajectory{initial}, innovations;
  StepStack                 stack;
  for (int step = 0; step < steps; ++step) {
    const int time = levels[step], prev = step + 1 < steps ? levels[step + 1] : -1;
    times[step] = time, previous[step] = prev;
    const Tensor z = Gaussian8(n, innovation_seed + step);
    innovations.push_back(Gaussian8(n, innovation_seed + step, true));
    const auto out = model.Reverse(current, time, prev, eta, z);
    Finite(out);
    stack.add(out);
    current = out.at("next");
    trajectory.push_back(current);
  }
  Map saved = stack.stacked();
  saved["initial"]     = initial;
  saved["images"]      = current;
  saved["trajectory"]  = nn::stack(trajectory, 0);
  saved["innovations"] = nn::stack(innovations, 0);
  saved["time"]        = Ints(times);
  saved["previous"]    = Ints(previous);
  saved["eta"]         = nn::scalar(eta, nn::DType::Float32);
  return saved;
}

void Run(const fs::path& trained, const fs::path& root, const fs::path& tf) {
  const fs::path frozen = FrozenFile(trained);
  check::Require(!frozen.empty(), "DDIM requires frozen DDPM checkpoint: run ddpm first, or ddpm=PATH (" + trained.string() + ")");
  const bool tf_weights = fs::weakly_canonical(trained) == fs::weakly_canonical(replay::TfModels("ddpm"));
  std::cout << "frozen DDPM weights: " << frozen.string() << "\n";
  DdimModel model;
  replay::LoadInto(model.store, frozen);
  const auto arrays = model.Arrays();
  check::Require(model.store.count() == 5504 && arrays.size() == 6, "same DDIM 5504 weights 6 tensors");
  if (fs::exists(trained / "model_arrays.npz")) {
    const auto expected = Load(trained / "model_arrays.npz");
    for (const auto& [key, value] : arrays)
      check::Require(Bits(value) == Bits(expected.at(key)), "DDIM restore matches frozen DDPM weights BITEXACT");
  }
  Save(root / "model_arrays.npz", arrays);
  Save(root / "schedule.npz", ScheduleArrays());
  if (fs::exists(trained / "metrics.json"))
    fs::copy_file(trained / "metrics.json", root / "training_metrics.json", fs::copy_options::overwrite_existing);
  const auto before = model.Snapshot();
  check::Require(fs::exists(trained / "samples.npz"), "frozen DDPM run has samples.npz");
  const auto ddpm = Load(trained / "samples.npz");
  std::vector<std::string>   names;
  std::map<std::string, Map> results;
  std::ofstream              summary(root / "summary.tsv");
  summary.precision(17);
  summary << "configuration\tsteps\teta\tepsilon_evaluations\tmean_abs_vs_ddpm\n";
  double ddpm_error = 0;
  std::vector<std::vector<Tensor>> comparison(8);
  for (int b = 0; b < 8; ++b) comparison[b].push_back(Row(ddpm.at("images"), b));
  const auto ddpm_images = ddpm.at("images").to_vector<float>();
  for (int steps : {8, 16, 32})
    for (int eta : {0, 1}) {
      const std::string name   = "eta" + std::to_string(eta) + "_steps" + std::to_string(steps);
      auto              sample = Sample(model, steps, float(eta));
      if (!eta) {
        const auto changed = Sample(model, steps, 0, 11201, 12202);
        check::Require(Bits(sample.at("trajectory")) == Bits(changed.at("trajectory")), "eta ZERO ignores ALL innovations BITEXACT");
      }
      Save(root / (name + ".npz"), sample);
      Png(root / (name + ".png"), sample.at("images"), 8);
      std::vector<std::vector<Tensor>> cells(8);
      std::vector<std::int32_t>        gallery_levels(5);
      const auto                       prev = sample.at("previous").to_vector<std::int32_t>();
      for (int j = 0; j < 5; ++j) {
        const int position = j * steps / 4;
        gallery_levels[j]  = position ? prev[position - 1] + 1 : 32;
        for (int b = 0; b < 8; ++b) cells[b].push_back(Row(At(sample.at("trajectory"), position), b));
      }
      const Tensor gallery = Gallery(cells);
      Save(root / (name + "_gallery.npz"), {{"images", gallery}, {"physical_levels", Ints(gallery_levels)}});
      Png(root / (name + "_trajectory.png"), gallery, 5);
      const auto images   = sample.at("images").to_vector<float>();
      double     distance = 0;
      for (int i = 0; i < 32 * 64; ++i) distance += std::abs(double(images[i]) - ddpm_images[i]) / (32 * 64);
      summary << name << '\t' << steps << '\t' << eta << '\t' << steps << '\t' << distance << '\n';
      for (int b = 0; b < 8; ++b) comparison[b].push_back(Row(sample.at("images"), b));
      if (steps == 32 && eta == 1) {
        check::Require(Bits(sample.at("initial")) == Bits(ddpm.at("initial")), "common initial Gaussian DDIM-DDPM");
        ddpm_error = MaxAbs(sample.at("trajectory"), ddpm.at("trajectory"));
        check::Require(ddpm_error < 5e-5, "eta ONE full 32 steps matches DDPM posterior WITH clipping-consistent epsilon");
      }
      names.push_back(name);
      results.emplace(name, sample);
      std::cout << name << ":epsilonEvaluations=" << steps << ",meanAbsVsDDPM=" << distance << std::endl;
    }
  const double eta_effect      = MaxAbs(results.at("eta0_steps32").at("images"), results.at("eta1_steps32").at("images"));
  const auto   changed_initial = Sample(model, 32, 0, 11299, 11202);
  Save(root / "changed_initial.npz", changed_initial);
  const double initial_effect = MaxAbs(results.at("eta0_steps32").at("images"), changed_initial.at("images"));
  check::Require(eta_effect > 1e-3 && initial_effect > 1e-3, "eta AND initial latent change actual DDIM samples");
  check::Require(before == model.Snapshot(), "ALL DDIM samplers-controls-gradients leave 6 weights immutable");
  const Tensor comparison_images = Gallery(comparison);
  Save(root / "comparison.npz", {{"images", comparison_images}});
  Png(root / "comparison.png", comparison_images, 7);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters_frozen\":5504,\"parameter_tensors\":6,\"additional_training_updates\":0,"
          "\"configurations\":6,\"samples_each\":32,\"initial_seed\":11201,\"innovation_base_seed\":11202,"
          "\"eta_zero_innovation_canary_bitexact\":true,\"eta_one_full32_ddpm_trajectory_max_error\":"
       << ddpm_error << ",\"eta_effect_max\":" << eta_effect << ",\"initial_latent_effect_max\":" << initial_effect
       << ",\"frozen_weights\":\"" << frozen.string() << "\"}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>DDIM sampler comparison</title><style>body{font:16px system-ui;"
          "margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>DDIM: fewer sampling steps on a "
          "frozen DDPM</h1><p><a href='https://arxiv.org/abs/2010.02502'>DDIM</a>, NO additional training. 8/16/32 respaced "
          "levels, eta 0/1, same 32 initial Gaussian draws, shared innovation streams; epsilon_consistent recomputed after x0 "
          "clipping so eta 1 / full 32 matches the trained DDPM posterior. Datamunge port of tflow's ddim.</p><h2>Paired first 8 "
          "initial latents</h2><p>Columns DDPM32, DDIM eta0/1 steps8, eta0/1 steps16, eta0/1 steps32.</p><img src='comparison.png'>";
  for (const auto& name : names)
    html << "<h2>" << name << "</h2><img src='" << name << ".png'><p>First 8 trajectories at quarter sampler progress.</p><img src='"
         << name << "_trajectory.png'>";
  check::Require(bool(meta) && bool(summary) && bool(html), "complete DDIM records");
  std::cout << "eta0innovationcanary/eta1full32DDPMparity/initialControl/immutable PASS;DDPMtrajectoryerror=" << ddpm_error << "\n";
  if (tf_weights && fs::exists(tf / "summary.tsv")) {
    std::cout << "TF parity (same frozen TF weights):\n";
    double worst = 0;
    for (const auto& name : names) worst = std::max(worst, bpix::Compare(tf, name + ".npz", results.at(name)));
    check::Require(worst < 1e-3, "DDIM samplers on TF's weights reproduce TF's recorded runs");
  }
  std::cout << "Artifacts: " << root.string() << "\nAll ddim checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = diffusion::ParseOptions(argc, argv, "ddim", {"ddpm"}, "ddpm=auto|ours|tf|DIR out=PATH tf=DIR threads=N");
    const auto trained =
        diffusion::FrozenDir(o.extra.at("ddpm"), paths::OutputRoot() / "models" / "ddpm", replay::TfModels("ddpm"));
    Run(trained, o.out, o.tf);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
