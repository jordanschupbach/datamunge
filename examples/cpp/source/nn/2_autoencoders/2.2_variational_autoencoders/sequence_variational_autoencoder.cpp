// CMD: just run-nn sequence_variational_autoencoder
// Masked GRU sentence VAE with EOS-aware loss and separate teacher-forced/greedy/ancestral diagnostics.
// THREE default epochs, 27 Adam updates.
//
// Port of tflow's sequence_variational_autoencoder. Preparation (in ~/projects/tflow_examples):
// just sequence-vae-assets; data are read from $DATAMUNGE_DATA_DIR/sequence_vae. Outputs go to
// output/nn/models/sequence_variational_autoencoder. Checkpoints are best.npz instead of SaveV2
// bundles; metrics.json gains training_seconds. Options: epochs=N data=PATH out=PATH init=PATH
// (initial parameters, e.g. tflow's initial_arrays.npz, to follow its trajectory exactly).

#include <chrono>
#include <iostream>
#include <limits>

#include "common/sequence_variational_autoencoder.hpp"

namespace {

namespace nn = datamunge::nn;

nn::Tensor Uniforms(std::int64_t n, std::uint32_t seed) {
  std::vector<double> out(n * 7);
  std::mt19937        random(seed);
  for (auto& v : out) v = (double(random()) + .5) / 4294967296.;
  return nn::from_vector(out, {n, 7});
}

std::vector<int> GenerationStats(const nn::Tensor& ids) {
  int                   ended = 0, valid = 0;
  std::set<std::string> unique;
  const auto            v = ids.to_vector<int>();
  for (std::int64_t b = 0; b < ids.size(0); ++b) {
    bool done = false;
    for (int t = 0; t < 7; ++t) {
      const int token = v[b * 7 + t];
      check::Require(token >= 0 && token < 19, "generated token in vocabulary");
      if (done) check::Require(token == 0, "EOS output suffix must be PAD");
      if (token == 2) done = true;
    }
    ended += done;
    valid += ae::SequenceGrammarValid(ids, b);
    unique.insert(ae::SequenceText(ids, b));
  }
  return {ended, valid, int(unique.size())};
}

void Add(nn::NamedTensors& out, const std::string& prefix, const nn::NamedTensors& fields) {
  for (const auto& [key, value] : fields) out.emplace_back(prefix + key, value);
}
nn::NamedTensors Fields(const ae::SequenceGeneration& g) {
  return {{"ids", g.ids}, {"logits", g.logits}, {"probability", g.probability}, {"ended", g.ended}};
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto pools = ae::LoadSequenceVaeData(data);
  std::filesystem::create_directories(root);
  std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json",
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::copy_file(data / "vocabulary.txt", root / "vocabulary.txt",
                             std::filesystem::copy_options::overwrite_existing);
  nn::save_npz((root / "data.npz").string(), ae::SequenceDataArrays(pools));
  std::vector<ae::VaeNoiseBank> noise{ae::BetaVaeNoise(288, 4, 3137), ae::BetaVaeNoise(72, 4, 3163),
                                      ae::BetaVaeNoise(72, 4, 3181)},
      training;
  auto             prior_noise = ae::BetaVaeNoise(20, 1, 3203);
  nn::NamedTensors fixed{{"prior_standard", prior_noise.standard}, {"prior_epsilon", prior_noise.epsilon}};
  for (int split = 0; split < 3; ++split) {
    const std::string name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    fixed.emplace_back(name + "_standard", noise[split].standard);
    fixed.emplace_back(name + "_epsilon", noise[split].epsilon);
  }
  nn::save_npz((root / "fixed_noise.npz").string(), fixed);
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    training.push_back(ae::BetaVaeNoise(288, 1, 3191 + epoch));
    nn::save_npz((root / ("train_noise_epoch" + std::to_string(epoch) + ".npz")).string(),
                 {{"standard", training.back().standard}, {"epsilon", training.back().epsilon}});
  }
  ae::SequenceVariationalModel model;
  if (!init.empty()) model.Load(nn::load_npz(init));
  check::Require(model.parameters.count() == 3243 && model.parameters.size() == 22,
                 "small embedding8/GRU16/latent4 sentence VAE census");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::BatchOrder order(288, 32, 3119);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_preupdate_negative_elbo\tvalidation_sequence_nll\tvalidation_kl\tvalidation_negative_elbo\t"
             "validation_teacher_reconstruction_token_perplexity\tvalidation_teacher_token_accuracy\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, step = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    auto   epsilon = ae::BetaVaeNoiseAt(training[epoch - 1], 0);
    double sum     = 0;
    int    seen    = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto input = nn::take_rows(pools[0].tokens, rows), e = nn::take_rows(epsilon, rows);
      if (step == 0) {
        nn::save_npz((root / "first_batch.npz").string(), {{"tokens", input}, {"epsilon", e}});
        auto gradients = ae::CheckGradients(model.parameters, model.Gradient(input, e), "all GRU BPTT gradients finite",
                                            "all22sentence VAE tensors receive derivatives");
        for (const auto& [name, row] : std::vector<std::pair<std::string, int>>{
                 {"encoder_embedding", 0}, {"encoder_embedding", 1}, {"decoder_embedding", 0}, {"decoder_embedding", 2}})
          for (float v : nn::select(ae::Find(gradients, name), 0, row).to_vector<float>())
            check::Near(v, 0, 0, "PAD/unused BOS or post-EOS embedding gradients exactly zero");
        nn::save_npz((root / "first_gradients.npz").string(), gradients);
        nn::save_npz((root / "first_forward.npz").string(), ae::SequenceFields(model.Evaluate(input, e)));
      }
      sum += double(rows.size()) * model.Update(input, e);
      seen += int(rows.size());
      for (auto row : rows) batches << epoch << '\t' << step << '\t' << row << '\t' << fit_source[row] << '\n';
      if (step == 0) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      ++step;
    }
    check::Require(seen == 288, "all288fit sentences once per epoch");
    auto score = ae::SequenceVaeScore(model, pools[1].tokens, noise[1]);
    history << epoch << '\t' << sum / seen;
    for (double value : score) history << '\t' << value;
    history << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (score[2] < best) {
      best     = score[2];
      selected = epoch;
      model.parameters.save(best_path);  // four-draw validation negative ELBO selects
    }
    std::cout << "epoch" << epoch << ":validation sequenceNLL=" << score[0] << ",KL=" << score[1]
              << ",negativeELBO=" << score[2] << ",TEACHERreconstruction tokenPPL=" << score[3]
              << ",teacheraccuracy=" << score[4] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  ae::SequenceVariationalModel deployed(false);
  deployed.parameters.load(best_path);  // fresh sentence VAE model-only restore
  check::Near(ae::SequenceVaeScore(deployed, pools[1].tokens, noise[1])[2], best, 1e-8,
              "restored selected sequence score");
  const auto       before = ae::Snapshot(deployed.parameters);
  nn::NamedTensors output;
  for (int split = 0; split < 3; ++split) {
    const std::int64_t n    = pools[split].tokens.size(0);
    const std::string  name = split == 0 ? "fit" : split == 1 ? "validation" : "test";
    auto               mean = deployed.Evaluate(pools[split].tokens, nn::zeros({n, 4}));
    check::Require(ae::Bytes(mean.z) == ae::Bytes(mean.mu), "epsilon0 gives z=mu");
    check::Require(ae::Bytes(mean.decoder_inputs) == ae::Bytes(pools[split].inputs),
                   "graph decoder input exactly BOS plus previous target");
    Add(output, name + "_mean_greedy_", Fields(deployed.Generate(mean.z)));
    Add(output, name + "_mean_", ae::SequenceFields(mean));
    const auto lengths = pools[split].lengths.to_vector<int>(), tokens = pools[split].tokens.to_vector<int>();
    for (int k = 0; k < 4; ++k) {
      auto f = deployed.Evaluate(pools[split].tokens, ae::BetaVaeNoiseAt(noise[split], k));
      check::Require(ae::Bytes(f.mu) == ae::Bytes(mean.mu) && ae::Bytes(f.encoder_state) == ae::Bytes(mean.encoder_state),
                     "Gaussian noise does not change sentence encoding");
      const auto valid = f.valid_tokens.to_vector<float>(), token_nll = f.token_nll.to_vector<float>();
      for (std::int64_t b = 0; b < n; ++b) {
        check::Near(valid[b], lengths[b], 0, "mask includes EOS,excludes PAD");
        for (int t = 0; t < 7; ++t)
          if (tokens[b * 7 + t] == 0) check::Near(token_nll[b * 7 + t], 0, 0, "padding has exactly zero reconstruction loss");
      }
      Add(output, name + "_draw" + std::to_string(k) + "_", ae::SequenceFields(f));
    }
  }
  auto test   = deployed.Evaluate(pools[2].tokens, ae::BetaVaeNoiseAt(noise[2], 0)),
       repeat = deployed.Evaluate(pools[2].tokens, ae::BetaVaeNoiseAt(noise[2], 0)),
       other  = deployed.Evaluate(pools[2].tokens, ae::BetaVaeNoiseAt(noise[2], 1));
  check::Require(ae::Bytes(test.probability) == ae::Bytes(repeat.probability),
                 "fixed-noise teacher probabilities repeat bitexact");
  check::Require(ae::Bytes(test.z) != ae::Bytes(other.z) && ae::Bytes(test.probability) != ae::Bytes(other.probability),
                 "Gaussian samples affect latent and teacher decoder");
  check::Require(ae::Bytes(deployed.Teacher(test.z, pools[2].inputs).second) == ae::Bytes(test.probability),
                 "fixed-z prefix-only decoder exactly matches teacher evaluation");
  auto single = deployed.Evaluate(nn::take_rows(pools[2].tokens, {0}),
                                  nn::take_rows(ae::BetaVaeNoiseAt(noise[2], 0), {0}));
  const double parity = ae::MaxAbsDiff(single.probability, test.probability, 7 * 19);
  check::Require(parity < 2e-6, "sequence single/batch parity");
  std::vector<int> padded(72 * 8, 0);
  {
    const auto v = pools[2].tokens.to_vector<int>();
    for (int b = 0; b < 72; ++b)
      for (int t = 0; t < 7; ++t) padded[b * 8 + t] = v[b * 7 + t];
  }
  auto padded_tokens = nn::from_vector(padded, {72, 8});
  auto extra         = deployed.ExtraPaddingState(padded_tokens);
  check::Require(ae::Bytes(extra) == ae::Bytes(test.encoder_state), "appending an extra PAD leaves encoded state BITEXACT");
  nn::save_npz((root / "padding_probe.npz").string(), {{"tokens8", padded_tokens}, {"encoder_state8", extra}});

  auto probe_z = nn::take_rows(test.z, {0}), prefix = nn::take_rows(pools[2].inputs, {0});
  auto changed_v = prefix.to_vector<int>();
  changed_v[4]   = changed_v[4] == 5 ? 6 : 5;
  auto changed   = nn::from_vector(changed_v, {1, 7});
  auto original = deployed.Teacher(probe_z, prefix).first.to_vector<float>(),
       altered  = deployed.Teacher(probe_z, changed).first.to_vector<float>();
  double earlier = 0, later = 0;
  for (int t = 0; t < 7; ++t)
    for (int w = 0; w < 19; ++w) {
      const double delta = std::abs(double(original[t * 19 + w]) - altered[t * 19 + w]);
      (t < 4 ? earlier : later) = std::max(t < 4 ? earlier : later, delta);
    }
  check::Require(earlier == 0 && later > 1e-6, "future decoder input cannot affect earlier token logits at FIXED z");
  nn::save_npz((root / "causality_probe.npz").string(),
               {{"z", probe_z},
                {"inputs", prefix},
                {"changed_inputs", changed},
                {"logits", nn::from_vector(original, {1, 7, 19})},
                {"changed_logits", nn::from_vector(altered, {1, 7, 19})}});

  auto prior_z = ae::BetaVaeNoiseAt(prior_noise, 0), token_uniforms = Uniforms(20, 3217);
  auto prior_greedy = deployed.Generate(prior_z), prior_sampled = deployed.Generate(prior_z, true, token_uniforms);
  auto zero_gen     = deployed.Generate(nn::zeros({20, 4}));
  const double latent_effect = ae::MaxAbsDiff(prior_greedy.logits, zero_gen.logits, 20 * 7 * 19);
  check::Require(latent_effect > 1e-5, "free decoder logits structurally depend on sentence latent");
  auto posterior_greedy = deployed.Generate(test.z);
  Add(output, "test_posterior_greedy_", Fields(posterior_greedy));
  nn::NamedTensors prior{{"z", prior_z}, {"token_uniforms", token_uniforms}};
  Add(prior, "greedy_", Fields(prior_greedy));
  Add(prior, "sampled_", Fields(prior_sampled));
  Add(prior, "zero_", Fields(zero_gen));
  nn::save_npz((root / "prior.npz").string(), prior);
  auto greedy_stats = GenerationStats(prior_greedy.ids), sampled_stats = GenerationStats(prior_sampled.ids);
  GenerationStats(ae::Find(output, "test_mean_greedy_ids"));
  GenerationStats(posterior_greedy.ids);
  check::Require(before == ae::Snapshot(deployed.parameters),
                 "teacher/padding/causality/free generation leave all parameters unchanged");
  nn::save_npz((root / "outputs.npz").string(), output);
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));

  std::vector<int>    unigram_count(19, 0);
  std::vector<double> unigram_probability(19);
  int                 fit_tokens = 0;
  for (int token : pools[0].tokens.to_vector<int>())
    if (token) {
      ++unigram_count[token];
      ++fit_tokens;
    }
  for (int v = 0; v < 19; ++v) unigram_probability[v] = (unigram_count[v] + .5) / (fit_tokens + 9.5);
  double baseline_nll = 0;
  int    test_tokens  = 0;
  for (int token : pools[2].tokens.to_vector<int>())
    if (token) {
      baseline_nll -= std::log(unigram_probability[token]);
      ++test_tokens;
    }
  nn::save_npz((root / "unigram_baseline.npz").string(),
               {{"fit_counts", nn::from_vector(unigram_count, {19})},
                {"probabilities", nn::from_vector(unigram_probability, {19})}});
  auto          score = ae::SequenceVaeScore(deployed, pools[2].tokens, noise[2]);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":3243,\"tensors\":22,\"epochs\":" << epochs << ",\"updates\":" << step
       << ",\"selected_epoch\":" << selected
       << ",\"mc_samples\":4,\"vocabulary\":19,\"maximum_steps\":7,\"test_sequence_nll\":" << score[0]
       << ",\"test_kl\":" << score[1] << ",\"test_negative_elbo\":" << score[2]
       << ",\"test_teacher_reconstruction_token_perplexity\":" << score[3]
       << ",\"test_teacher_token_accuracy\":" << score[4]
       << ",\"fit_only_unigram_test_sequence_nll\":" << baseline_nll / 72
       << ",\"fit_only_unigram_test_token_perplexity\":" << std::exp(baseline_nll / test_tokens)
       << ",\"prior_greedy_eos_count\":" << greedy_stats[0] << ",\"prior_greedy_grammar_valid_count\":" << greedy_stats[1]
       << ",\"prior_greedy_unique_count\":" << greedy_stats[2] << ",\"prior_sampled_eos_count\":" << sampled_stats[0]
       << ",\"prior_sampled_grammar_valid_count\":" << sampled_stats[1]
       << ",\"prior_sampled_unique_count\":" << sampled_stats[2] << ",\"single_batch_max_error\":" << parity
       << ",\"causal_earlier_max_effect\":" << earlier << ",\"causal_later_max_effect\":" << later
       << ",\"latent_generation_logit_max_effect\":" << latent_effect << ",\"training_seconds\":" << seconds << "}\n";

  std::ofstream html(root / "index.html"), table(root / "examples.tsv");
  html << "<!doctype html><meta charset='utf-8'><title>Generated-sentence GRU VAE</title><style>body{font:16px "
          "system-ui;max-width:1300px;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid "
          "#aaa;vertical-align:top}</style><h1>Sequence / sentence variational autoencoder</h1><p>Generated432unique "
          "sentences,288fit72validation72test;fixed19token vocabulary,PAD0/BOS1/EOS2,6or7targets including EOS. No "
          "language-corpus benchmark. GRU encoder/decoder hidden16,separate embeddings8,Gaussian global sentence "
          "code4,decoder initial state from z.3243parameters22tensors,"
       << epochs << " epochs," << step
       << " Adam updates. <a href='https://arxiv.org/abs/1511.06349'>Sentence VAE reference</a>;compact GRU teaching "
          "variant,not the paper's LSTM/dropout/warm-up setup.</p><p>Encode complete sentence with PAD-masked state "
          "updates. Decode causally from BOS using PREVIOUS target during fitting;loss SUMvalidtokenCE INCLUDING "
          "EOS,EXCLUDING PAD+analyticGaussianKL,mean sentences. Four validation draws select negativeELBO. No "
          "KLannealing or word dropout. Teacher reconstruction perplexity conditions on full-sentence posterior and "
          "true prefixes;it is NOT unconditional language-model perplexity. Free generation uses only z and its own "
          "previous outputs,EOS stops,PAD/BOSbeforeEOS and7step truncation remain visible.</p><p>Test sequenceNLL="
       << score[0] << ",KL=" << score[1] << ",negativeELBO=" << score[2] << ",TEACHER reconstruction tokenPPL="
       << score[3] << ",teacheraccuracy=" << score[4] << ". FIT-only smoothedunigram baseline sequenceNLL="
       << baseline_nll / 72 << ",tokenPPL=" << std::exp(baseline_nll / test_tokens)
       << ". These diagnostics do not establish language fluency or latent semantics.</p><h2>Ten evenly spaced "
          "heldout cases,unfiltered</h2><p>Teacher argmax uses TRUE prefixes and z=mu. Free greedy from mu uses no "
          "target prefixes. Posterior greedy uses one explicit z draw;neither greedy output is a posterior predictive "
          "mean.</p><table><tr><th>Source</th><th>Original</th><th>Teacher argmax,true prefixes</th><th>FREE "
          "greedy(mu)</th><th>FREE posterior draw greedy</th></tr>";
  table << "case\tsource_id\toriginal\tteacher_argmax\tfree_mean_greedy\tfree_posterior_greedy\n";
  const auto test_source = pools[2].source.to_vector<int>();
  for (int k = 0; k < 10; ++k) {
    const int b = k * 71 / 9;
    auto truth = ae::SequenceText(pools[2].tokens, b), teacher = ae::SequenceText(ae::Find(output, "test_mean_predicted_ids"), b),
         free = ae::SequenceText(ae::Find(output, "test_mean_greedy_ids"), b),
         post = ae::SequenceText(posterior_greedy.ids, b);
    html << "<tr><td>" << test_source[b] << "</td><td>" << truth << "</td><td>" << teacher << "</td><td>" << free
         << "</td><td>" << post << "</td></tr>";
    table << k << '\t' << test_source[b] << '\t' << truth << '\t' << teacher << '\t' << free << '\t' << post << '\n';
  }
  html << "</table><h2>20unfiltered PRIOR cases</h2><p>z~N0I. Greedy decoding is a diagnostic,not ancestral token "
          "sampling. The sampled column draws EACH next token using explicit uniforms from its OWN prefix. EOS is "
          "the only stop;special tokens before EOS are not filtered. Greedy EOS="
       << greedy_stats[0] << "/20,grammarvalid=" << greedy_stats[1] << ",unique=" << greedy_stats[2]
       << ". Token-sampled EOS=" << sampled_stats[0] << ",grammarvalid=" << sampled_stats[1]
       << ",unique=" << sampled_stats[2]
       << ".</p><table><tr><th>Case</th><th>Prior greedy</th><th>Actual ancestral token sample</th></tr>";
  std::ofstream prior_table(root / "prior_examples.tsv");
  prior_table << "case\tgreedy\tgreedy_eos\tgreedy_grammar_valid\tsampled\tsampled_eos\tsampled_grammar_valid\n";
  const auto greedy_ended = prior_greedy.ended.to_vector<int>(), sampled_ended = prior_sampled.ended.to_vector<int>();
  for (int b = 0; b < 20; ++b) {
    auto g = ae::SequenceText(prior_greedy.ids, b), s = ae::SequenceText(prior_sampled.ids, b);
    html << "<tr><td>" << b << "</td><td>" << g << (greedy_ended[b] ? "" : " [MAX7,NOEOS]") << "</td><td>" << s
         << (sampled_ended[b] ? "" : " [MAX7,NOEOS]") << "</td></tr>";
    prior_table << b << '\t' << g << '\t' << greedy_ended[b] << '\t' << ae::SequenceGrammarValid(prior_greedy.ids, b)
                << '\t' << s << '\t' << sampled_ended[b] << '\t' << ae::SequenceGrammarValid(prior_sampled.ids, b)
                << '\n';
  }
  html << "</table><p>Free-run errors,repetition,missing EOS and invalid grammar are retained. A structurally active "
          "latent path is not proof that the model learns useful sentence representations;small KL and repeated "
          "generations can indicate underuse.</p>";
  check::Require(bool(meta) && bool(html) && bool(table) && bool(prior_table) && bool(history) && bool(batches),
                 "complete sentence VAE artifacts");
  std::cout << "selected" << selected << ",test sequenceNLL=" << score[0] << ",KL=" << score[1]
            << ",negativeELBO=" << score[2] << ",TEACHERtokenPPL=" << score[3] << ",accuracy=" << score[4]
            << ",prior greedy grammar=" << greedy_stats[1] << "/20,sampledgrammar=" << sampled_stats[1]
            << "/20;restore/PAD/EOS/causality/target-freegeneration/BPTT/batch/immutable passed\ntraining time "
            << seconds << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("sequence_vae"),
                          out  = paths::OutputRoot() / "models/sequence_variational_autoencoder";
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
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
