// CMD: just run-nn deep_belief
// Greedy DBN: freeze a pretrained bottom RBM, learn a top RBM, then generate
// through the directed lower links.
//
// Port of tflow's deep_belief (common/{nn,ml}/deep_belief.h ->
// examples/nn/common/deep_belief.hpp). Source: the frozen 3-epoch 8 -> 4 bottom RBM
// and its data.npz from restricted_boltzmann, chosen by source=auto|ours|tf|DIR
// (auto: our port's output/nn/models/restricted_boltzmann/final.npz when present,
// else tflow's recorded output/models/restricted_boltzmann/model_arrays.npz). Each
// source row gets ONE sampled recognition state h1 = [u < sigmoid(vW + c)] (tflow's
// uniform banks 22001+2k). The upper 4 -> 3 RBM (19 parameters) trains three epochs
// of 48 CD1 SGD(.05) updates (BatchOrder(512, 32, 22019), uniforms 2200051+step ...).
// Generative DBN p(v,h1,h2) = p_top(h1,h2) p(v|h1): bottom hidden_bias is
// recognition-only; 63 stored = 59 generative + 4 recognition parameters.
//
// Checks (as tflow): 19 top parameters; 63 stored weights; recognition bias
// unreachable from the generative likelihood (nn::grad allow_unused returns no
// gradient, TF's "unreachable" error); top restore bit-exact; hybrid restore from
// all-zero initialization bit-exact; DBN visible marginal sums to 1; top Gibbs +
// directed decode repeats bit-exactly; changing the recognition bias leaves every
// generative probability bit-exact but changes recognition; bottom unchanged by
// greedy training; inference immutable. TF parity: init=auto (default) replays TF's
// initial_top_arrays.npz from tflow's output/models/deep_belief; first top
// forward/gradients/update, history, final top weights and DBN outputs compared
// with TF's; independently TF's final 63 weights must reproduce TF's outputs and
// sampled chains. Deviations: final.npz / top_final.npz instead of SaveV2 bundles;
// no independent_audit.json. Options: epochs=3 source=auto|ours|tf|DIR out=DIR
// init=auto|tf|nn|PATH tf=DIR threads=4. Output: output/nn/models/deep_belief.
#include <chrono>

#include "common/deep_belief.hpp"

namespace {

using namespace ebm;

Map Sample(const DeepBeliefModel& model) {
  const std::int64_t n     = 1024;
  Tensor             state = nn::cast(nn::less(BinaryUniform(n, 4, 22201), .5), nn::DType::Float32);
  Map                saved{{"initial", state}};
  std::vector<Tensor> path{state};
  std::map<std::string, std::vector<Tensor>> steps;
  for (int k = 0; k < 100; ++k) {
    const auto fields = model.TopStep(state, BinaryUniform(n, 3, 2220051 + k), BinaryUniform(n, 4, 3220051 + k));
    for (const auto& [key, value] : fields) steps[key].push_back(value);
    state = fields.at("visible");
    path.push_back(state);
  }
  for (const auto& [key, values] : steps) saved[key] = nn::stack(values, 0);
  const auto decode              = model.Decode(state, BinaryUniform(n, 8, 22203));
  saved["path"]                  = nn::stack(path, 0);
  saved["top_states"]            = state;
  saved["decoder_probability"]   = decode.at("probability");
  saved["samples"]               = decode.at("visible");
  return saved;
}

Map Zeros(const Map& m) {
  Map out;
  for (const auto& [k, v] : m) out[k] = nn::zeros_like(v);
  return out;
}

void Run(const Options& o) {
  const fs::path root = o.out, tf = o.tf;
  const auto     started = std::chrono::steady_clock::now();
  const fs::path source =
      FrozenDir(o.extra.at("source"), paths::OutputRoot() / "models" / "restricted_boltzmann", replay::TfModels("restricted_boltzmann"));
  const fs::path frozen = FrozenFile(source);
  check::Require(!frozen.empty() && fs::exists(source / "data.npz"),
                 "restore three-epoch bottom RBM 8x4: no final.npz/model_arrays.npz + data.npz in " + source.string() +
                     " (run restricted_boltzmann first or pass source=tf)");
  std::cout << "bottom RBM: " << frozen.string() << "\n";
  RestrictedBoltzmannModel lower(false);
  replay::LoadInto(lower.store, frozen);
  const auto bottom_before = lower.Snapshot();
  const Map  bottom_arrays = lower.Arrays();
  Map        data          = Load(source / "data.npz");
  std::vector<Tensor> original, pools;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    const Tensor      x     = data.at(split + "_points");
    original.push_back(x);
    const Tensor hp      = lower.Evaluate(x).at("hidden_probability");
    const Tensor sampled = nn::cast(nn::less(BinaryUniform(x.size(0), 4, 22001 + 2 * k), hp), nn::DType::Float32);
    data[split + "_hidden_probability"] = hp;
    data[split + "_hidden_uniform"]     = BinaryUniform(x.size(0), 4, 22001 + 2 * k, true);
    data[split + "_hidden"]             = sampled;
    pools.push_back(sampled);
  }
  Save(root / "data.npz", data);
  CheckData(tf, data, {"fit_points", "validation_points", "test_points", "fit_hidden_uniform"});
  if (fs::exists(tf / "data.npz") && fs::exists(tf / "bottom_arrays.npz"))
    std::printf("bottom vs TF's bottom_arrays.npz: max diff %.3g; sampled h1 codes agreement with TF's data.npz %.6f\n",
                replay::MaxDiff(bottom_arrays, tf / "bottom_arrays.npz"),
                Agreement(tf, "data.npz", data, {"fit_hidden", "validation_hidden", "test_hidden"}));
  Save(root / "bottom_arrays.npz", bottom_arrays);
  if (fs::exists(source / "metrics.json"))
    fs::copy_file(source / "metrics.json", root / "source_bottom_metrics.json", fs::copy_options::overwrite_existing);

  RestrictedBoltzmannModel upper(true, 4, 3);
  check::Require(upper.store.count() == 19, "19 top RBM parameters");
  fs::path init;
  if (o.init == "auto" || o.init == "tf") {
    if (fs::exists(tf / "initial_top_arrays.npz")) init = tf / "initial_top_arrays.npz";
    check::Require(o.init == "auto" || !init.empty(), "init=tf needs " + (tf / "initial_top_arrays.npz").string());
  } else if (o.init != "nn") {
    init = o.init;
  }
  const bool replaying = !init.empty();
  if (replaying) {
    replay::LoadInto(upper.store, init);
    std::cout << "replaying TF's initial top arrays " << init.string() << "\n";
  } else {
    std::cout << "top weights from nn's generator (statistical parity with TF only)\n";
  }
  Save(root / "initial_top_arrays.npz", upper.Arrays());
  {
    const DeepBeliefModel initial(bottom_arrays, upper.Arrays());
    Save(root / "initial_dbn_outputs.npz", initial.Evaluate(original[2]));
    Save(root / "initial_dbn_gradients.npz", initial.Gradient(original[2]));
    check::Require(initial.RecognitionUnreachable(original[2]), "recognition bias unreachable from generative likelihood");
  }
  nn::BatchOrder order(512, 32, 22019);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tvalidation_top_exact_nll\tvalidation_dbn_exact_nll\n";
  batches << "epoch\tstep\trow\n";
  Map randomness;
  int updates = 0;
  for (int epoch = 1; epoch <= o.epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto   n = std::int64_t(rows.size());
      const Tensor x = Rows(pools[0], rows), u = BinaryUniform(n, 3, 2200051 + updates), v = BinaryUniform(n, 4, 3200051 + updates);
      randomness["step" + std::to_string(updates) + "_hidden"]  = BinaryUniform(n, 3, 2200051 + updates, true);
      randomness["step" + std::to_string(updates) + "_visible"] = BinaryUniform(n, 4, 3200051 + updates, true);
      if (!updates) {
        Save(root / "first_batch.npz", {{"points", x}, {"hidden_uniform", u}, {"visible_uniform", v}});
        Save(root / "first_forward.npz", upper.Gibbs(x, u, v));
        Save(root / "initial_top_gradients.npz", upper.Gradient(x, u, v));
      }
      upper.Update(x, u, v);
      if (!updates) Save(root / "first_after.npz", upper.Arrays());
      for (auto row : rows) batches << epoch << '\t' << updates << '\t' << row << '\n';
      ++updates;
    }
    const auto            valid = upper.Evaluate(pools[1]);
    const DeepBeliefModel assembled(bottom_arrays, upper.Arrays());
    const auto            dbn_valid = assembled.Evaluate(original[1]);
    history << epoch << '\t' << Scalar(valid, "nll") << '\t' << Scalar(dbn_valid, "nll") << '\n';
    Save(root / ("epoch" + std::to_string(epoch) + "_top_arrays.npz"), upper.Arrays());
    Save(root / ("epoch" + std::to_string(epoch) + "_top_validation.npz"), valid);
    Save(root / ("epoch" + std::to_string(epoch) + "_dbn_validation.npz"), dbn_valid);
    std::cout << "epoch" << epoch << " topExactNLL=" << Scalar(valid, "nll") << " DBNvisibleNLL=" << Scalar(dbn_valid, "nll") << std::endl;
  }
  history.flush();
  batches.flush();
  const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  Save(root / "training_randomness.npz", randomness);
  check::Require(SameAsTf(tf, "training_randomness.npz", randomness), "CD1 uniform banks bit-identical to TF's training_randomness.npz");
  const auto first = Load(root / "first_batch.npz");
  Save(root / "final_top_gradients.npz", upper.Gradient(first.at("points"), first.at("hidden_uniform"), first.at("visible_uniform")));
  Save(root / "final_top_first_gibbs.npz", upper.Gibbs(first.at("points"), first.at("hidden_uniform"), first.at("visible_uniform")));
  upper.store.save((root / "top_final.npz").string());
  RestrictedBoltzmannModel restored_top(false, 4, 3);
  restored_top.store.load((root / "top_final.npz").string(), true);
  check::Require(upper.Snapshot() == restored_top.Snapshot(), "top 19 weights bitexact restore");
  const Map top_arrays = restored_top.Arrays();
  Save(root / "top_arrays.npz", top_arrays);
  const DeepBeliefModel assembled(bottom_arrays, top_arrays);
  Save(root / "final.npz", assembled.Arrays());
  DeepBeliefModel deployed(Zeros(bottom_arrays), Zeros(top_arrays));
  deployed.Restore(root / "final.npz");
  check::Require(assembled.Snapshot() == deployed.Snapshot(), "all 63 weights bitexact hybrid restore from zero initialization");
  const auto before = deployed.Snapshot();
  Save(root / "model_arrays.npz", deployed.Arrays());
  const auto output = deployed.Evaluate(original[2]);
  Save(root / "outputs.npz", output);
  Save(root / "final_dbn_gradients.npz", deployed.Gradient(original[2]));
  const Tensor states = BinaryStates(8);
  auto         all    = deployed.Evaluate(states);
  all["states"]       = states;
  double mass         = 0;
  for (float lp : all.at("log_probability").to_vector<float>()) mass += std::exp(double(lp));
  check::Near(mass, 1., 2e-6, "normalized DBN marginal from directed conditional and top prior");
  Save(root / "exact_distribution.npz", all);
  const auto sample = Sample(deployed), repeat = Sample(deployed);
  for (const auto& [k, v] : sample) check::Require(Bits(v) == Bits(repeat.at(k)), "DBN top Gibbs + directed decode repeat bitexact");
  Save(root / "samples.npz", sample);
  BinaryTiles(root / "samples.png", Rows(sample.at("samples"), Range(64)), 8);
  BinaryTiles(root / "decoder_means.png", Rows(sample.at("decoder_probability"), Range(64)), 8);

  // Exact top prior categorical control: no finite-chain approximation, same bottom decoder uniforms.
  const auto top_dist = restored_top.Evaluate(BinaryStates(4));
  const auto uniforms = BinaryUniform(1024, 1, 22301, true);
  const auto lp = top_dist.at("log_probability").to_vector<float>();
  const auto uv = uniforms.to_vector<double>();
  const auto top_states = BinaryStates(4).to_vector<float>();
  double     probabilities[16], total = 0;
  for (int k = 0; k < 16; ++k) total += probabilities[k] = std::exp(double(lp[k]));
  std::vector<float> exact_top(1024 * 4);
  for (int b = 0; b < 1024; ++b) {
    const double u = uv[b] * total;
    double       cumulative = 0;
    int          selected   = 15;
    for (int k = 0; k < 16; ++k) {
      cumulative += probabilities[k];
      if (u < cumulative) {
        selected = k;
        break;
      }
    }
    std::copy_n(top_states.data() + selected * 4, 4, exact_top.data() + b * 4);
  }
  const Tensor exact_top_t     = nn::from_vector(exact_top, {1024, 4});
  auto         exact_decode    = deployed.Decode(exact_top_t, BinaryUniform(1024, 8, 22203));
  exact_decode["top_states"]   = exact_top_t;
  exact_decode["uniform"]      = uniforms;
  Save(root / "exact_prior_samples.npz", exact_decode);
  BinaryTiles(root / "exact_prior_samples.png", Rows(exact_decode.at("visible"), Range(64)), 8);

  // Changing the recognition-only bias must change recognition but leave ALL generative probabilities bit-exact.
  Map changed                = bottom_arrays;
  changed.at("hidden_bias")  = nn::add(bottom_arrays.at("hidden_bias"), nn::full({4}, .7f));
  const DeepBeliefModel control(changed, top_arrays);
  const auto            altered = control.Evaluate(original[2]);
  check::Require(Bits(altered.at("log_probability")) == Bits(output.at("log_probability")), "recognition-only bias cannot affect DBN marginal");
  check::Require(Bits(altered.at("recognition_probability")) != Bits(output.at("recognition_probability")),
                 "recognition control actually changes recognition");
  Save(root / "recognition_control.npz", altered);
  check::Require(bottom_before == lower.Snapshot(), "pretrained bottom unchanged across greedy top training");
  check::Require(before == deployed.Snapshot(), "hybrid inference all 63 weights immutable");
  const double source_nll = Scalar(lower.Evaluate(original[2]), "nll");
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"stored_parameters\":63,\"generative_parameters\":59,\"recognition_only_bias_parameters\":4,\"top_parameters\":19,"
          "\"bottom_frozen_parameters\":44,\"top_epochs\":"
       << o.epochs << ",\"top_optimizer_updates\":" << updates << ",\"bottom_source_epochs\":3,\"test_exact_dbn_nll\":" << Scalar(output, "nll")
       << ",\"source_bottom_exact_nll\":" << source_nll << ",\"sampling_top_sweeps\":100,\"samples\":1024,\"replayed_tf_initial_arrays\":"
       << (replaying ? "true" : "false") << ",\"training_seconds\":" << train_seconds << "}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Deep belief network</title><style>body{font:16px system-ui;margin:30px;"
          "max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Greedy deep belief network: undirected top, "
          "directed bottom</h1><p>Frozen 3-epoch 8->4 Bernoulli RBM from restricted_boltzmann; one sampled recognition state per "
          "source; upper 4->3 RBM (19 parameters) trained three epochs of 48 CD1 SGD .05 updates. p(v,h1,h2) = p_topRBM(h1,h2) "
          "p_directed(v|h1); the bottom hidden bias is recognition only (63 stored = 59 generative + 4 recognition). Exact "
          "normalized DBN marginal by enumerating 16 h1 / 256 visible states. Datamunge port of tflow's deep_belief.</p><h2>First "
          "64 generated samples</h2><p>1024 Bernoulli .5 top states, 100 top RBM Gibbs sweeps, then one directed Bernoulli "
          "visible draw.</p><img src='samples.png'><h2>Corresponding directed decoder probabilities</h2><img src='decoder_means.png'>"
          "<h2>Exact top prior categorical control, then directed decode</h2><img src='exact_prior_samples.png'><p><a "
          "href='https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf'>Hinton, Osindero, Teh 2006</a>.</p>";
  check::Require(bool(meta) && bool(html) && bool(history) && bool(batches), "complete deep belief records");
  std::cout << "exactDBNtestNLL=" << Scalar(output, "nll") << " sourceBottom=" << source_nll
            << ";frozenBottom/greedyTop/hybridNormalization/recognitionOnly/restore/chainRepeat/immutable PASS (training "
            << train_seconds << " s)\n";

  // ---- TF parity ----
  if (!fs::exists(tf / "model_arrays.npz")) {
    std::cout << "no TF record at " << tf.string() << "; parity checks skipped\nArtifacts: " << root.string() << "\n";
    return;
  }
  std::cout << "TF parity (" << tf.string() << "):\n";
  const bool same_codes = SameAsTf(tf, "data.npz", {{"fit_hidden", data.at("fit_hidden")}});
  if (replaying && same_codes) {
    const double fwd = bpix::Compare(tf, "first_forward.npz", Load(root / "first_forward.npz"));
    const double grd = bpix::Compare(tf, "initial_top_gradients.npz", Load(root / "initial_top_gradients.npz"));
    const double aft = bpix::Compare(tf, "first_after.npz", Load(root / "first_after.npz"));
    check::Require(fwd < 1e-5 && grd < 1e-5 && aft < 1e-5, "replayed first top forward/gradients/update match TF");
    if (o.epochs == 3) {
      const double fin = bpix::Compare(tf, "top_arrays.npz", top_arrays);
      bpix::Compare(tf, "outputs.npz", output);
      check::Require(fin < 1e-4, "replayed run's final top weights match TF's");
    }
    CompareHistory(root, tf, 1, "validation top exact NLL");
    CompareHistory(root, tf, 2, "validation DBN exact NLL");
  } else if (replaying) {
    std::cout << "  (sampled h1 codes differ from TF's: replay comparisons skipped)\n";
  }
  DeepBeliefModel tf_model(Zeros(bottom_arrays), Zeros(top_arrays));
  tf_model.Restore(tf / "model_arrays.npz");
  const Tensor tf_test  = Load(tf / "data.npz").at("test_points");
  const double out_diff = bpix::Compare(tf, "outputs.npz", tf_model.Evaluate(tf_test));
  const auto   chains   = Sample(tf_model);
  const double prb_diff = bpix::Compare(tf, "samples.npz", chains);
  const double agree    = Agreement(tf, "samples.npz", chains, {"path", "hidden", "visible", "samples", "top_states"});
  std::printf("  TF final weights: top chain + decode sampled states agreement with TF's samples.npz %.6f\n", agree);
  check::Require(out_diff < 1e-5 && prb_diff < 1e-5 && agree > .999, "TF's final weights reproduce TF's DBN outputs and samples");
  std::cout << "Artifacts: " << root.string() << "\nAll deep_belief checks passed.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Run(ParseOptions(argc, argv, "deep_belief", {{"source", "auto"}},
                     "epochs=N source=auto|ours|tf|DIR out=PATH init=auto|tf|nn|PATH tf=DIR threads=N"));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
