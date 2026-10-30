// CMD: just run-nn posterior_collapse
// THREE epochs EACH, constant KL weight versus linear warm-up, on the generated-sentence GRU VAE.
//
// Port of tflow's posterior_collapse. Preparation (in ~/projects/tflow_examples): just
// sequence-vae-assets; data are read from $DATAMUNGE_DATA_DIR/sequence_vae. Outputs go to
// output/nn/models/posterior_collapse (constant/ and warmup/ subdirectories, synthetic control
// json/npz, index.html). Checkpoints are best.npz instead of SaveV2 bundles; metrics.json gains
// training_seconds. Options: epochs=N data=PATH out=PATH init=PATH (shared initial parameters of
// both schedules, e.g. tflow's constant/initial_arrays.npz, to follow its trajectories exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/posterior_collapse.hpp"

namespace {

namespace nn = datamunge::nn;

std::vector<double> Score(ae::AnnealedSequenceModel& m, const nn::Tensor& x, const ae::VaeNoiseBank& bank) {
  auto s = ae::SequenceVaeScore(m, x, bank);
  return {s[0], s[1], s[2], s[3]};
}

double TokenNll(const nn::Tensor& tokens, const nn::Tensor& probabilities) {
  const auto t = tokens.to_vector<int>();
  const auto p = probabilities.to_vector<float>();
  double     sum = 0;
  for (std::int64_t b = 0; b < tokens.size(0); ++b)
    for (int s = 0; s < 7; ++s) {
      const int id = t[b * 7 + s];
      if (id) sum -= std::log(double(p[(b * 7 + s) * 19 + id]));
    }
  return sum / tokens.size(0);
}

void Add(nn::NamedTensors& out, const std::string& prefix, const nn::NamedTensors& fields) {
  for (const auto& [key, value] : fields) out.emplace_back(prefix + key, value);
}

void Controls(const std::filesystem::path& root) {
  auto               mu = nn::zeros({8, 4}), lv = nn::zeros({8, 4});
  std::vector<float> s(32, 0);
  for (int b = 0; b < 8; ++b) s[b * 4] = b % 2 ? 2 : -2;
  auto                    shifted = nn::from_vector(s, {8, 4});
  auto                    noise   = ae::BetaVaeNoise(8, 4, 4081);
  std::vector<nn::Tensor> collapsed, active;
  nn::NamedTensors        arrays{{"mu", mu}, {"log_variance", lv}, {"active_mu", shifted}, {"standard", noise.standard},
                                 {"epsilon", noise.epsilon}};
  for (int k = 0; k < 4; ++k) {
    collapsed.push_back(ae::BetaVaeNoiseAt(noise, k));
    auto z = (ae::BetaVaeNoiseAt(noise, k) + shifted).contiguous();
    active.push_back(z);
    arrays.emplace_back("active_z" + std::to_string(k), z);
  }
  auto c = ae::CollapseStatistics(mu, lv, collapsed), a = ae::CollapseStatistics(shifted, lv, active);
  check::Near(c.analytic_kl, 0, 0, "known q=N0I collapsedcontrol KLexactzero");
  check::Require(c.active_units == 0 && a.active_units == 1, "known collapsed/active meanvariance controls");
  check::Near(a.analytic_kl, 2, 0, "known separatedmeans conditionalKL2");
  check::Require(std::abs(c.mixture_information) < 1e-12 && std::abs(c.aggregate_kl) < 1e-12 &&
                     a.mixture_information > .1,
                 "empirical mixture information identifies synthetic controls");
  ae::SaveCollapseStatistics(root / "collapsed_control.json", c);
  ae::SaveCollapseStatistics(root / "active_control.json", a);
  nn::save_npz((root / "diagnostic_controls.npz").string(), arrays);
}

void Run(const std::vector<ae::SequenceVaeData>& pools, const std::filesystem::path& root, int epochs, bool warm,
         const std::string& init) {
  std::filesystem::create_directories(root);
  nn::save_npz((root / "data.npz").string(), ae::SequenceDataArrays(pools));
  nn::NamedTensors              bank;
  std::vector<ae::VaeNoiseBank> noise;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    noise.push_back(ae::BetaVaeNoise(pools[k].tokens.size(0), 4, 3137 + (k == 0 ? 0 : k == 1 ? 26 : 44)));
    bank.emplace_back(split + "_standard", noise[k].standard);
    bank.emplace_back(split + "_epsilon", noise[k].epsilon);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), bank);
  ae::AnnealedSequenceModel model;
  if (!init.empty()) model.Load(nn::load_npz(init));
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(288, 32, 3119);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv"), schedule(root / "schedule.tsv");
  history.precision(17);
  schedule.precision(17);
  history << "epoch\tvalidation_nll\tvalidation_kl\tvalidation_negative_elbo\tvalidation_teacher_reconstruction_token_ppl\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  schedule << "step\tepoch\tbeta\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, steps = 0;
  const int  ramp    = 18;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    auto training = ae::BetaVaeNoise(288, 1, 3191 + epoch);
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", training.standard}, {"epsilon", training.epsilon}});
    auto e = ae::BetaVaeNoiseAt(training, 0);
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto        x = nn::take_rows(pools[0].tokens, rows), epsilon = nn::take_rows(e, rows);
      const float beta = ae::CollapseBeta(warm, steps, ramp);
      schedule << steps << '\t' << epoch << '\t' << beta << '\n';
      if (!steps) {
        nn::save_npz((root / "first_batch.npz").string(),
                     {{"tokens", x}, {"epsilon", epsilon}, {"beta", nn::full({}, beta)}});
        nn::save_npz((root / "first_forward.npz").string(), ae::SequenceFields(model.Evaluate(x, epsilon)));
        auto             gs = model.Gradient(x, epsilon, beta);
        nn::NamedTensors gradients;
        for (std::size_t i = 0; i < gs.size(); ++i) gradients.emplace_back(model.parameters.names()[i], gs[i]);
        nn::save_npz((root / "first_gradients.npz").string(), gradients);
      }
      model.Update(x, epsilon, beta);
      if (!steps) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << fit_source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 288, "all fit sentences each epoch");
    auto score = Score(model, pools[1].tokens, noise[1]);
    history << epoch;
    for (auto v : score) history << '\t' << v;
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[2] < best) {
      best     = score[2];
      selected = epoch;
      model.parameters.save(best_path);  // COMMON beta1 validation negative ELBO selects both schedules
    }
    std::cout << (warm ? "warmup" : "constant") << ",epoch" << epoch << ":valNLL=" << score[0] << ",KL=" << score[1]
              << ",COMMONnegativeELBO=" << score[2] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  ae::AnnealedSequenceModel deployed(false);
  deployed.parameters.load(best_path);  // fresh annealed sentence model-only restore
  check::Near(Score(deployed, pools[1].tokens, noise[1])[2], best, 1e-8, "restored commonselection score");
  const auto              before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors        outputs;
  std::vector<nn::Tensor> fit_samples;
  ae::SequenceVaeOutput   fit_base;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    for (int draw = 0; draw < 4; ++draw) {
      auto f = deployed.Evaluate(pools[k].tokens, ae::BetaVaeNoiseAt(noise[k], draw));
      if (!k) {
        fit_samples.push_back(f.z);
        if (!draw) fit_base = f;
      }
      Add(outputs, split + "_draw" + std::to_string(draw) + "_", ae::SequenceFields(f));
    }
  }
  auto stats = ae::CollapseStatistics(fit_base.mu, fit_base.log_variance, fit_samples);
  ae::SaveCollapseStatistics(root / "fit_posterior_diagnostics.json", stats);
  auto               test = deployed.Evaluate(pools[2].tokens, ae::BetaVaeNoiseAt(noise[2], 0));
  const auto         zv   = test.z.to_vector<float>();
  std::vector<float> sh(zv.size());
  for (int b = 0; b < 72; ++b)
    for (int j = 0; j < 4; ++j) sh[b * 4 + j] = zv[((b + 1) % 72) * 4 + j];
  std::map<std::string, nn::Tensor> codes{{"posterior", test.z},
                                          {"zero", nn::zeros({72, 4})},
                                          {"shuffled", nn::from_vector(sh, {72, 4})},
                                          {"prior", ae::BetaVaeNoiseAt(noise[2], 0)}};
  nn::NamedTensors ablation;
  for (const auto& name : {"posterior", "zero", "shuffled", "prior"}) ablation.emplace_back(std::string(name) + "_z", codes.at(name));
  ablation.emplace_back("tokens", pools[2].tokens);
  ablation.emplace_back("inputs", pools[2].inputs);
  std::vector<double>               nll;
  std::map<std::string, nn::Tensor> probability;
  for (const std::string name : {"posterior", "zero", "shuffled", "prior"}) {
    auto p = deployed.Teacher(codes.at(name), pools[2].inputs).second;
    probability[name] = p;
    ablation.emplace_back(name + "_probability", p);
    nll.push_back(TokenNll(pools[2].tokens, p));
    if (name == "posterior")
      check::Require(ae::Bytes(p) == ae::Bytes(test.probability), "posterior fixed-z teacher decode BITEXACT");
  }
  const std::int64_t count         = 72 * 7 * 19;
  const double       zero_effect   = ae::MaxAbsDiff(test.probability, probability.at("zero"), count),
               shuffle_effect = ae::MaxAbsDiff(test.probability, probability.at("shuffled"), count);
  check::Require(zero_effect > 0 && shuffle_effect > 0,
                 "latent path structurally affects probabilities,not proof of useful semantics");
  nn::save_npz((root / "ablations.npz").string(), ablation);
  auto mean_free = deployed.Generate(test.mu), posterior_free = deployed.Generate(test.z);
  Add(outputs, "test_mean_greedy_",
      {{"ids", mean_free.ids}, {"logits", mean_free.logits}, {"probability", mean_free.probability}, {"ended", mean_free.ended}});
  Add(outputs, "test_posterior_greedy_",
      {{"ids", posterior_free.ids},
       {"logits", posterior_free.logits},
       {"probability", posterior_free.probability},
       {"ended", posterior_free.ended}});
  check::Require(before == ae::Snapshot(deployed.parameters), "all diagnostics and ablations immutable");
  nn::save_npz((root / "outputs.npz").string(), outputs);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  auto          score = Score(deployed, pools[2].tokens, noise[2]);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":3243,\"tensors\":22,\"epochs\":" << epochs << ",\"updates\":" << steps
       << ",\"warmup\":" << (warm ? "true" : "false") << ",\"ramp_steps\":18,\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"test_nll\":" << score[0] << ",\"test_kl\":" << score[1]
       << ",\"test_negative_elbo\":" << score[2] << ",\"test_teacher_reconstruction_token_ppl\":" << score[3]
       << ",\"posterior_draw_teacher_nll\":" << nll[0] << ",\"zero_code_teacher_nll\":" << nll[1]
       << ",\"shuffled_code_teacher_nll\":" << nll[2] << ",\"prior_code_teacher_nll\":" << nll[3]
       << ",\"zero_code_probability_max_effect\":" << zero_effect
       << ",\"shuffled_code_probability_max_effect\":" << shuffle_effect << ",\"training_seconds\":" << seconds
       << "}\n";

  std::ofstream html(root / "index.html"), table(root / "examples.tsv");
  html << "<!doctype html><meta charset='utf-8'><title>Posterior collapse diagnostics</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1200px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid "
          "#bbb}</style><h1>Posterior collapse diagnostics: "
       << (warm ? "linear warmup" : "constant beta1")
       << "</h1><p>Same masked GRU16/separateembeddings8/Gaussian4 sentence VAE,3243params. " << epochs << " epochs,"
       << steps
       << " Adam updates. Constantbeta1 versus linear min(1,zero-basedstep/17),first18updates ramp0to1,then1. "
          "Matched initialization,data,order,noise. BOTHcheckpoints selected by SAMEfourdraw unweighted "
          "validationnegativeELBO,beta1. Warmed-up training losses at different beta are not compared as ELBOs. "
          "Generated432sentences288fit72validation72test,not a language benchmark.</p><p>Fit meanKL="
       << stats.analytic_kl << ",activeunits=" << stats.active_units
       << "/4 using prespecified var(mu)>.01. Empirical fit-mixture information estimate=" << stats.mixture_information
       << ",aggregateKL estimate=" << stats.aggregate_kl << ",sampledconditionalKL=" << stats.sampled_conditional_kl
       << ". Component Gaussian mixture uses SAME288fitposteriors and four aligned draws each;this finite-sample "
          "diagnostic has self-inclusion bias and is not a population MI estimator or a semantic score. Sampled "
          "information+aggregateKL decomposes sampledconditionalKL exactly;analyticKL differs by MonteCarlo error. "
          "Negative MonteCarlo values retained.</p><p>TestcommonnegativeELBO="
       << score[2] << ",NLL=" << score[0] << ",KL=" << score[1] << ",TEACHERreconstructiontokenPPL=" << score[3]
       << ". Not unconditional language-model PPL. Fixedtrueprefix teacherNLL with posterior/zero/shuffled/prior code="
       << nll[0] << "/" << nll[1] << "/" << nll[2] << "/" << nll[3]
       << ". Labels show one matched posterior draw,not predictive means. Prefixes can predict tokens even with weak "
          "latent use;latent connectivity alone does not establish usefulness. Structural probabilityeffects "
          "zero/shuffle="
       << zero_effect << "/" << shuffle_effect
       << ".</p><p>Known syntheticcollapsed q=N0I has exactzeroKL/zeroactiveunits/zeromixtureinformation;separatedmeans "
          "control hasKL2/oneactiveunit. Neither substitutes for diagnosing actual trained model. SmallKL alone is "
          "insufficient for a definitive collapse diagnosis;warmup is not assumed to cure it.</p><table><tr><th>"
          "Original</th><th>FREEgreedy(mu)</th><th>FREEposteriorgreedy</th></tr>";
  table << "case\tsource_id\toriginal\tfree_mean_greedy\tfree_posterior_greedy\n";
  const auto test_source = pools[2].source.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int b     = k * 71 / 9;
    auto      truth = ae::SequenceText(pools[2].tokens, b), mean = ae::SequenceText(mean_free.ids, b),
         post = ae::SequenceText(posterior_free.ids, b);
    html << "<tr><td>" << truth << "</td><td>" << mean << "</td><td>" << post << "</td></tr>";
    table << k << '\t' << test_source[b] << '\t' << truth << '\t' << mean << '\t' << post << '\n';
  }
  html << "</table><p>Alltenheldoutcases unfiltered,EOSstop/PADsuffix/max7steps. "
          "MissingEOS/repetition/invalidgrammar retained. No warmup-superiority,semanticrepresentation or fluency "
          "claim from this small single seed.</p>";
  check::Require(bool(meta) && bool(html) && bool(table) && bool(history) && bool(schedule),
                 "complete collapse diagnostics");
  std::cout << (warm ? "warmup" : "constant") << ",selected" << selected << ",testnegativeELBO=" << score[2]
            << ",KL=" << score[1] << ",fitactive=" << stats.active_units << ",mixtureInfo=" << stats.mixture_information
            << ",aggregateKL=" << stats.aggregate_kl << ",posterior/zero/shuffle/prior teacherNLL=" << nll[0] << "/"
            << nll[1] << "/" << nll[2] << "/" << nll[3] << ";controls/restore/ablations/immutable PASS\ntraining time "
            << seconds << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("sequence_vae"), out = paths::OutputRoot() / "models/posterior_collapse";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N data=PATH out=PATH init=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "three default epochs or explicit1..100");
    auto pools = ae::LoadSequenceVaeData(data);
    std::filesystem::create_directories(out);
    Controls(out);
    Run(pools, out / "constant", epochs, false, init);
    Run(pools, out / "warmup", epochs, true, init);
    auto a = nn::load_npz((out / "constant" / "initial_arrays.npz").string()),
         b = nn::load_npz((out / "warmup" / "initial_arrays.npz").string());
    for (const auto& [key, value] : a)
      check::Require(ae::Bytes(value) == ae::Bytes(ae::Find(b, key)), "matched initial weights EXACT");
    std::ofstream html(out / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>Paired collapse diagnostics and KL warmup</title><h1>Posterior "
            "collapse diagnostics and KL annealing</h1><p>Matched autoregressive sentence VAE,"
         << epochs
         << " epochs EACH. <a href='constant/index.html'>Constantbeta1</a> / <a href='warmup/index.html'>Linear0to1warmup "
            "over18updates</a>. COMMONbeta1validationselection. KL byunit,activeunits,empirical Gaussian-mixture "
            "decomposition and fixed-prefix latent ablations;knowncollapsed/active controls. No assumption warmup fixes "
            "collapse.</p>";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
