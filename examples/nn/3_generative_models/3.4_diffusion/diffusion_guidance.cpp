// CMD: just run-nn diffusion_guidance
// Classifier and classifier-free guidance: two cheap trained conditioners plus a
// frozen unconditional DDPM.
//
// Port of tflow's diffusion_guidance (common/{nn,ml}/diffusion_guidance.h ->
// examples/nn/common/diffusion_guidance.hpp). Data: tflow's saliency MNIST pools
// (paths::Dataset("saliency")), continuous 8x8 [-1, 1] images, 32 cosine VP
// levels. Trains a noisy-image classifier (3722 parameters, CE on x_t and time,
// labels are targets only) and a conditional epsilon MLP (5856 parameters, 11-way
// one-hot input with NULL = 10, 10% independent label dropout, epsilon MSE),
// three epochs / 96 Adam(.003) steps each, tflow's BatchOrder(1000, 32, 14019).
// Frozen unconditional DDPM: ddpm=auto (our ddpm port's final.npz when present,
// else tflow's recorded model_arrays.npz; ddpm=ours|tf|DIR). Samples 10 classes x
// 4 common latents with CG and CFG at scales 0, 1, 3 (32 reverse steps each).
//
// Checks (as tflow): census, finite fields, restore bit-exact, classifier labels
// are not inputs (zero effect), CFG class changes epsilon, CG scale 0 equals the
// frozen DDPM step bit-exactly, CFG scale 1 equals the conditional epsilon
// (2e-6), zero guidance ignores class with paired latents, last step adds no
// noise, target-class effects, all weights immutable. TF parity: init=auto
// replays TF's classifier/ and conditional/ initial_arrays.npz and compares the
// records with TF's; TF's final weights must reproduce TF's test outputs.
// Deviations: final.npz checkpoints; no independent_audit.json. Options:
// epochs=3 data=DIR ddpm=auto|ours|tf|DIR out=DIR init=auto|tf|nn tf=DIR threads=4.
// Output: output/nn/models/diffusion_guidance.
#include "common/diffusion_guidance.hpp"

namespace {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using namespace diffusion;

Tensor LabelRows(const Tensor& labels, const std::vector<std::int64_t>& rows, bool dropout, std::uint32_t seed) {
  const auto                l = labels.to_vector<std::int32_t>();
  std::vector<std::int32_t> out(rows.size());
  std::mt19937              rng(seed);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int label = l[rows[i]];
    out[i]          = dropout && (double(rng()) + .5) / 4294967296. < .1 ? 10 : label;
  }
  return Ints(out);
}

int Argmax10(const std::vector<float>& logits, int b) {
  int best = 0;
  for (int c = 1; c < 10; ++c)
    if (logits[b * 10 + c] > logits[b * 10 + best]) best = c;
  return best;
}

struct Banks {
  Tensor validation_noise, validation_time, test_noise, test_time;
};

double Scalar(const Map& m, const std::string& k) { return m.at(k).item<float>(); }

void Train(const DigitData& d, const Banks& k, const fs::path& root, const fs::path& tf, const Options& o, bool classifier) {
  fs::create_directories(root);
  const auto& images = d.images;
  const auto& pools  = d.pools;
  GuidanceTrainingModel model(classifier);
  check::Require(model.store.count() == (classifier ? 3722 : 5856) && model.store.size() == 6,
                 "guidance classifier 3722 or CFG 5856 parameters 6 tensors");
  const fs::path init = bpix::InitFile(o.init, tf);
  if (!init.empty()) {
    replay::LoadInto(model.store, init);
    std::cout << "replaying TF's initial arrays " << init.string() << "\n";
  }
  Save(root / "initial_arrays.npz", model.Arrays());
  Save(root / "initial_outputs.npz", model.Evaluate(images[2], k.test_noise, k.test_time, pools[2].labels));
  nn::BatchOrder order(1000, 32, 14019);
  Map            randomness;
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_loss\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto source  = pools[0].source.to_vector<int>();
  int        updates = 0, null_count = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = Rows(images[0], rows), e = Gaussian8(n, 1400051 + updates), t = Times(n, 2400021 + updates),
                   condition = LabelRows(pools[0].labels, rows, !classifier, 3400401 + updates);
      const std::string s = "step" + std::to_string(updates);
      randomness[s + "_noise"]     = Gaussian8(n, 1400051 + updates, true);
      randomness[s + "_time"]      = t;
      randomness[s + "_condition"] = condition;
      for (int c : condition.to_vector<std::int32_t>()) null_count += c == 10;
      if (!updates) {
        Save(root / "first_batch.npz", {{"images", x}, {"noise", e}, {"time", t}, {"condition", condition}});
        Save(root / "first_forward.npz", model.Evaluate(x, e, t, condition));
        Save(root / "initial_gradients.npz", model.Gradient(x, e, t, condition));
      }
      model.Update(x, e, t, condition);
      if (!updates) Save(root / "first_after.npz", model.Arrays());
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\t' << source[row] << '\n';
      ++updates;
    }
    const auto out = model.Evaluate(images[1], k.validation_noise, k.validation_time, pools[1].labels);
    Finite(out);
    history << epoch << '\t' << Scalar(out, "loss") << '\n';
    Save(root / ("epoch" + std::to_string(epoch) + "_arrays.npz"), model.Arrays());
    Save(root / ("epoch" + std::to_string(epoch) + "_validation.npz"), out);
    std::cout << (classifier ? "classifier" : "conditional") << "epoch" << epoch << ":validationLoss=" << Scalar(out, "loss")
              << std::endl;
  }
  Save(root / "training_randomness.npz", randomness);
  model.store.save((root / "final.npz").string());
  const auto first = Load(root / "first_batch.npz");
  Save(root / "final_gradients.npz", model.Gradient(first.at("images"), first.at("noise"), first.at("time"), first.at("condition")));
  GuidanceTrainingModel deployed(classifier, false);
  deployed.store.load((root / "final.npz").string(), true);
  check::Require(model.Snapshot() == deployed.Snapshot(), "all guidance training weights BITEXACT restore");
  const auto before = deployed.Snapshot();
  const auto out    = deployed.Evaluate(images[2], k.test_noise, k.test_time, pools[2].labels);
  Finite(out);
  Save(root / "outputs.npz", out);
  Save(root / "model_arrays.npz", deployed.Arrays());
  Save(root / "input_probe.npz", deployed.Predict(out.at("noised"), k.test_time, pools[2].labels));
  const auto                test_labels = pools[2].labels.to_vector<std::int32_t>();
  std::vector<std::int32_t> changed_v(200);
  for (int i = 0; i < 200; ++i) changed_v[i] = (test_labels[i] + 1) % 10;
  const Tensor changed   = Ints(changed_v);
  const auto   alternate = deployed.Evaluate(images[2], k.test_noise, k.test_time, changed);
  const double effect    = MaxAbs(out.at("prediction"), alternate.at("prediction"));
  if (classifier)
    check::Require(effect == 0, "classifier labels are targets NOT input features");
  else
    check::Require(effect > 1e-5, "CFG class condition changes actual epsilon");
  Save(root / "label_control.npz", {{"condition", changed}, {"prediction", alternate.at("prediction")}, {"loss", alternate.at("loss")}});
  double null_loss = 0, accuracy = 0;
  if (!classifier) {
    const auto unconditioned = deployed.Evaluate(images[2], k.test_noise, k.test_time, FullInt(200, 10));
    Save(root / "null_outputs.npz", unconditioned);
    null_loss = Scalar(unconditioned, "loss");
  } else {
    const auto logits = out.at("prediction").to_vector<float>();
    const auto tt     = k.test_time.to_vector<std::int32_t>();
    int        correct = 0;
    for (int b = 0; b < 200; ++b) correct += Argmax10(logits, b) == test_labels[b];
    accuracy = correct / 200.;
    std::ofstream buckets(root / "noise_bucket_accuracy.tsv");
    buckets << "first_time\tlast_time\tcount\tcorrect\n";
    for (int q = 0; q < 4; ++q) {
      int count = 0, ok = 0;
      for (int b = 0; b < 200; ++b)
        if (tt[b] / 8 == q) ++count, ok += Argmax10(logits, b) == test_labels[b];
      buckets << 8 * q << '\t' << 8 * q + 7 << '\t' << count << '\t' << ok << '\n';
    }
  }
  check::Require(before == deployed.Snapshot(), "all guidance eval-gradient-label controls immutable");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"mode\":\"" << (classifier ? "noisy_classifier" : "conditional_epsilon") << "\",\"parameters\":" << (classifier ? 3722 : 5856)
       << ",\"parameter_tensors\":6,\"epochs\":" << o.epochs << ",\"optimizer_updates\":" << updates
       << ",\"null_condition_training_count\":" << null_count << ",\"test_loss\":" << Scalar(out, "loss")
       << ",\"test_classifier_accuracy\":" << accuracy << ",\"test_null_epsilon_mse\":" << null_loss
       << ",\"class_condition_epsilon_max_effect\":" << effect << "}\n";
  check::Require(bool(meta) && bool(history) && bool(batches), "complete guidance training records");
  std::cout << (classifier ? "classifier" : "conditional") << ": test loss " << Scalar(out, "loss")
            << (classifier ? ", accuracy " + std::to_string(accuracy) : ", NULL epsilon MSE " + std::to_string(null_loss))
            << ", NULL training count " << null_count << "\n";

  if (!fs::exists(tf / "model_arrays.npz")) return;
  std::cout << "TF parity (" << tf.string() << "):\n";
  if (!init.empty()) {
    const double fwd = bpix::Compare(tf, "first_forward.npz", Load(root / "first_forward.npz"));
    const double grd = bpix::Compare(tf, "initial_gradients.npz", Load(root / "initial_gradients.npz"));
    const double aft = bpix::Compare(tf, "first_after.npz", Load(root / "first_after.npz"));
    check::Require(fwd < 1e-5 && grd < 1e-5 && aft < 1e-5, "replayed first forward/gradients/update match TF");
    bpix::Compare(tf, "model_arrays.npz", deployed.Arrays());
    history.flush();
    const auto ours = replay::History(root / "history.tsv"), theirs = replay::History(tf / "history.tsv");
    for (std::size_t i = 0; i < std::min(ours.size(), theirs.size()); ++i)
      std::printf("  epoch %zu validation loss ours %.6f TF %.6f\n", i + 1, ours[i], theirs[i]);
  }
  GuidanceTrainingModel tf_model(classifier, false);
  replay::LoadInto(tf_model.store, tf / "model_arrays.npz");
  const auto   tf_out = tf_model.Evaluate(images[2], k.test_noise, k.test_time, pools[2].labels);
  const double diff   = bpix::Compare(tf, "outputs.npz", tf_out);
  check::Require(diff < 1e-4, "TF's final weights reproduce TF's test outputs");
}

Map Sample(GuidedDiffusionModel& model, bool cfg, float weight) {
  const int    n            = 40;
  const Tensor base_initial = Gaussian8(4, 15201);
  std::vector<std::int32_t> label_v(n);
  for (int b = 0; b < n; ++b) label_v[b] = b / 4;
  const Tensor labels  = Ints(label_v);
  const Tensor initial = Cycle(base_initial, n);
  Tensor       current = initial;
  std::vector<Tensor> trajectory{initial}, innovations;
  StepStack           stack;
  const std::string   next_key = cfg ? "cfg_next" : "cg_next";
  for (int step = 0; step < 32; ++step) {
    const Tensor t     = FullInt(n, 31 - step);
    const Tensor noise = Cycle(Gaussian8(4, 15202 + step), n);
    innovations.push_back(Cycle(Gaussian8(4, 15202 + step, true), n));
    const auto out = model.Evaluate(current, t, labels, weight, noise);
    Finite(out);
    if (!cfg && weight == 0) check::Require(Bits(out.at("cg_next")) == Bits(out.at("base_next")), "CG scale ZERO equals FROZEN DDPM step BITEXACT");
    if (cfg && weight == 1)
      check::Require(MaxAbs(out.at("cfg_epsilon"), out.at("conditional_epsilon")) <= 2e-6, "CFG scale ONE equals conditional epsilon");
    stack.add(out);
    const Tensor next = out.at(next_key);
    if (step == 31) {
      const auto alternate = model.Evaluate(current, t, labels, weight, nn::full(noise.shape(), 123.0));
      check::Require(Bits(next) == Bits(alternate.at(next_key)), "guided DDPM last step adds NO noise");
    }
    current = next;
    trajectory.push_back(current);
  }
  if (weight == 0) {
    const auto v = current.to_vector<float>();
    for (int b = 4; b < n; ++b)
      for (int j = 0; j < 64; ++j)
        check::Require(v[b * 64 + j] == v[(b % 4) * 64 + j], "zero guidance output IGNORES class with paired latents and noise");
  }
  Map saved = stack.stacked();
  saved["initial"]     = initial;
  saved["labels"]      = labels;
  saved["trajectory"]  = nn::stack(trajectory, 0);
  saved["innovations"] = nn::stack(innovations, 0);
  saved["images"]      = current;
  saved["scale"]       = nn::scalar(weight, nn::DType::Float32);
  return saved;
}

void Run(const Options& o, const fs::path& ddpm_root) {
  const fs::path root = o.out, tf = o.tf;
  const fs::path frozen = FrozenFile(ddpm_root);
  check::Require(!frozen.empty(), "guidance requires frozen DDPM: run ddpm first, or ddpm=PATH (" + ddpm_root.string() + ")");
  std::cout << "frozen DDPM weights: " << frozen.string() << "\n";
  auto d    = LoadDigits(o.data);
  Map  data = d.record;
  Banks k{Gaussian8(200, 14101), Times(200, 0, true), Gaussian8(200, 14103), Times(200, 0, true)};
  data["validation_noise"] = k.validation_noise;
  data["test_noise"]       = k.test_noise;
  data["validation_time"]  = k.validation_time;
  data["test_time"]        = k.test_time;
  Save(root / "data.npz", data);
  if (fs::exists(tf / "data.npz")) {
    const auto theirs = Load(tf / "data.npz");
    for (const std::string key : {"fit_images", "test_images", "test_noise", "validation_noise", "test_time"})
      check::Require(Bits(theirs.at(key)) == Bits(data.at(key)), "data/noise/time bit-identical to TF's data.npz: " + key);
  }
  ae::CopyManifests(o.data, root);
  Save(root / "schedule.npz", ScheduleArrays());
  Train(d, k, root / "classifier", tf / "classifier", o, true);
  Train(d, k, root / "conditional", tf / "conditional", o, false);
  GuidedDiffusionModel model;
  replay::LoadInto(model.unconditional, frozen);
  model.classifier.load((root / "classifier/final.npz").string(), true);
  model.conditional.load((root / "conditional/final.npz").string(), true);
  const auto uncond = replay::Arrays(model.unconditional);
  if (fs::exists(ddpm_root / "model_arrays.npz")) {
    const auto original = Load(ddpm_root / "model_arrays.npz");
    for (const auto& [key, value] : uncond) check::Require(Bits(value) == Bits(original.at(key)), "CG uses original DDPM weights BITEXACT");
  }
  Save(root / "unconditional_arrays.npz", uncond);
  const auto                       before = model.Snapshot();
  std::vector<std::string>         names;
  std::map<std::string, Map>       results;
  std::vector<std::vector<Tensor>> comparison(10);
  for (bool cfg : {false, true})
    for (int scale : {0, 1, 3}) {
      const std::string name   = std::string(cfg ? "cfg" : "classifier_guidance") + "_scale" + std::to_string(scale);
      const auto        sample = Sample(model, cfg, float(scale));
      Save(root / (name + ".npz"), sample);
      Png(root / (name + ".png"), sample.at("images"), 4);
      std::vector<std::vector<Tensor>> cells(10);
      for (int cls = 0; cls < 10; ++cls) {
        comparison[cls].push_back(Row(sample.at("images"), cls * 4));
        for (int j = 0; j < 5; ++j) cells[cls].push_back(Row(At(sample.at("trajectory"), j * 8), cls * 4));
      }
      const Tensor gallery = Gallery(cells);
      Save(root / (name + "_gallery.npz"), {{"images", gallery}});
      Png(root / (name + "_trajectory.png"), gallery, 5);
      names.push_back(name);
      results.emplace(name, sample);
      std::cout << name << ":32reverseSteps/10classesx4pairedLatents PASS" << std::endl;
    }
  double     cg_target_effect = 0, cfg_target_effect = 0;
  const auto cg  = results.at("classifier_guidance_scale1").at("cg_epsilon").to_vector<float>();
  const auto cfe = results.at("cfg_scale1").at("conditional_epsilon").to_vector<float>();
  for (int cls = 1; cls < 10; ++cls)
    for (int j = 0; j < 4 * 64; ++j) {
      cg_target_effect  = std::max(cg_target_effect, std::abs(double(cg[cls * 4 * 64 + j]) - cg[j]));
      cfg_target_effect = std::max(cfg_target_effect, std::abs(double(cfe[cls * 4 * 64 + j]) - cfe[j]));
    }
  check::Require(cg_target_effect > 1e-6 && cfg_target_effect > 1e-6, "actual classifier- and CFG target class effects");
  check::Require(before == model.Snapshot(), "ALL 18 weights across three guidance models IMMUTABLE");
  const Tensor comparison_images = Gallery(comparison);
  Save(root / "comparison.npz", {{"images", comparison_images}});
  Png(root / "comparison.png", comparison_images, 6);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"classifier_parameters\":3722,\"conditional_parameters\":5856,\"unconditional_parameters_frozen\":5504,\"epochs_each\":"
       << o.epochs << ",\"optimizer_updates_each\":" << o.epochs * 32
       << ",\"null_dropout_probability\":0.1,\"guidance_scales\":[0,1,3],\"samples_per_configuration\":40,"
          "\"configurations\":6,\"cg_zero_ddpm_step_parity_bitexact\":true,\"zero_guidance_class_invariance_bitexact\":true,"
          "\"cg_target_epsilon_max_effect\":"
       << cg_target_effect << ",\"cfg_target_epsilon_max_effect\":" << cfg_target_effect << ",\"frozen_ddpm\":\"" << frozen.string()
       << "\"}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Diffusion guidance</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Classifier and classifier-free guidance</h1>"
          "<p><a href='https://arxiv.org/abs/2105.05233'>Classifier guidance</a> and <a href='https://arxiv.org/abs/2207.12598'>"
          "classifier-free guidance</a> on a frozen 5504-parameter DDPM; noisy classifier (3722) and NULL-dropout conditional "
          "epsilon MLP (5856), three epochs each. CG eps = eps_DDPM - w sqrt(1 - alpha_bar) grad_x log p(y|x_t,t); CFG eps = "
          "eps_NULL + w (eps_CLASS - eps_NULL). Datamunge port of tflow's diffusion_guidance.</p><h2>Paired first latent "
          "comparison</h2><p>Rows classes 0..9; columns CG w0/1/3, CFG w0/1/3.</p><img src='comparison.png'>";
  for (const auto& name : names)
    html << "<h2>" << name << "</h2><img src='" << name << ".png'><p>First latent trajectory for each class, physical levels "
         << "32/24/16/8/0.</p><img src='" << name << "_trajectory.png'>";
  check::Require(bool(meta) && bool(html), "complete guidance study records");
  std::cout << "CGactualLogpGrad/CFGNULLdropout/scale0-1controls/targeteffects/frozenRestore/immutable PASS; CG target effect "
            << cg_target_effect << ", CFG target effect " << cfg_target_effect << "\n";
  const bool all_tf = fs::weakly_canonical(ddpm_root) == fs::weakly_canonical(replay::TfModels("ddpm"));
  if (all_tf && o.init != "nn" && fs::exists(tf / "cfg_scale1.npz")) {
    std::cout << "TF parity of guided samplers (TF frozen DDPM, replayed conditioners):\n";
    for (const auto& name : names) bpix::Compare(tf, name + ".npz", results.at(name));
  }
  std::cout << "Artifacts: " << root.string() << "\nAll diffusion_guidance checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto o = diffusion::ParseOptions(argc, argv, "diffusion_guidance", {"ddpm"},
                                           "epochs=N data=PATH ddpm=auto|ours|tf|DIR out=PATH init=auto|tf|nn tf=DIR threads=N");
    Run(o, diffusion::FrozenDir(o.extra.at("ddpm"), paths::OutputRoot() / "models" / "ddpm", replay::TfModels("ddpm")));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
