// CMD: just run-nn encoder_pretraining
// Prepare:just autoencoder-assets;just run undercomplete_autoencoder(THREEdefault epochs).
// THREEepochs of supervised HEADtraining EACH,frozenpretrained vs originalrandomencoder.
//
// Port of tflow's encoder_pretraining (model: examples/nn/common/encoder_pretraining.hpp).
// Preparation: tflow's `just autoencoder-assets` (data from $DATAMUNGE_DATA_DIR/saliency) and our
// undercomplete_autoencoder port (`just run-nn undercomplete_autoencoder`), whose
// model_arrays.npz (selected encoder) and initial_arrays.npz (its original random encoder) are
// read from encoder=DIR (default output/nn/models/undercomplete_autoencoder; tflow's
// output/models/undercomplete_autoencoder works too). Outputs (pretrained/ and random/ training
// records, index.html) go to output/nn/models/encoder_pretraining. Checkpoints are best.npz
// instead of TF SaveV2 bundles; metrics.json gains "training_seconds". Options: epochs=N
// data=PATH encoder=PATH out=PATH init=DIR (load the classifier head initialisation from
// DIR/pretrained/initial_arrays.npz, e.g. tflow's output dir, to replay its runs exactly when
// encoder= is tflow's undercomplete_autoencoder too; by default nn's initialiser is used, the
// same head for both variants).

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/encoder_pretraining.hpp"

namespace {

namespace nn = datamunge::nn;

double Run(const std::vector<ae::SaliencyData>& pools, const nn::NamedTensors& source, const std::filesystem::path& root, int epochs,
           const std::filesystem::path& init) {
  std::filesystem::create_directories(root);
  auto fit_code        = ae::TransferFeatures(pools[0].images, source);
  auto [center, scale] = ae::TransferScaling(fit_code);
  ae::EncoderClassifierModel model(source, center, scale);
  if (!init.empty()) model.LoadHead(init);
  auto frozen = model.EncoderSnapshot();
  check::Require(model.encoder_w.numel() + model.encoder_b.numel() == 25120 && model.head.count() == 330, "frozenencoder25120/headONLY330");
  nn::save_npz((root / "initial_arrays.npz").string(), model.Arrays());
  nn::NamedTensors data{{"feature_center", center}, {"feature_scale", scale}};
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    data.emplace_back(split + "_images", pools[k].images);
    data.emplace_back(split + "_labels", pools[k].labels);
    data.emplace_back(split + "_source_ids", pools[k].source);
  }
  nn::save_npz((root / "data.npz").string(), data);
  nn::BatchOrder order(1000, 32, 4721);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\tfit_nll\tfit_accuracy\tvalidation_nll\tvalidation_accuracy\n";
  batches << "epoch\tstep\trow\tsource_id\n";
  const auto fit_source = pools[0].source.to_vector<int>();
  const auto best_path  = (root / "best.npz").string();
  double     best       = std::numeric_limits<double>::infinity();
  int        selected = 0, steps = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(pools[0].images, rows), y = nn::take_rows(pools[0].labels, rows);
      if (!steps) {
        nn::save_npz((root / "first_batch.npz").string(), {{"images", x}, {"labels", y}});
        auto f = model.Predict(x);
        nn::save_npz((root / "first_forward.npz").string(), {{"code", f[0]}, {"features", f[1]}, {"logits", f[2]}, {"probability", f[3]}});
        auto               grad = model.Gradient(x, y);
        std::vector<int>   shifted_labels;
        for (int v : y.to_vector<int>()) shifted_labels.push_back((v + 1) % 10);
        auto             wrong = model.Gradient(x, nn::from_vector(shifted_labels, y.shape()));
        nn::NamedTensors gradients, wrong_gradients;
        for (std::size_t i = 0; i < grad.size(); ++i) {
          gradients.emplace_back("head/" + model.head.names()[i], grad[i]);
          wrong_gradients.emplace_back("head/" + model.head.names()[i], wrong[i]);
        }
        nn::save_npz((root / "first_gradients.npz").string(), gradients);
        nn::save_npz((root / "wrong_label_gradients.npz").string(), wrong_gradients);
      }
      model.Update(x, y);
      if (!steps) nn::save_npz((root / "first_update_arrays.npz").string(), model.Arrays());
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << fit_source[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 1000, "allfittingimages+labelsONCEperepoch");
    check::Require(frozen == model.EncoderSnapshot(), "ENCODERweightsBITEXACTthroughALLheadupdates");
    auto fit = ae::TransferScore(model.Predict(pools[0].images)[3], pools[0].labels),
         val = ae::TransferScore(model.Predict(pools[1].images)[3], pools[1].labels);
    history << epoch << '\t' << fit[0] << '\t' << fit[1] << '\t' << val[0] << '\t' << val[1] << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), model.Arrays());
    if (val[0] < best) {
      best     = val[0];
      selected = epoch;
      nn::save_npz(best_path, model.Arrays());  // validation LABEL NLL selects HEAD checkpoint
    }
    std::cout << root.filename() << ",epoch" << epoch << ":fitacc=" << fit[1] << ",validationNLL=" << val[0] << ",valacc=" << val[1]
              << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::EncoderClassifierModel deployed(source, center, scale, false);
  deployed.LoadHead(best_path);  // fresh transferred encoder + head model-only restore
  check::Near(ae::TransferScore(deployed.Predict(pools[1].images)[3], pools[1].labels)[0], best, 1e-10, "restoredvalidationNLL");
  auto             before = deployed.Arrays();
  nn::NamedTensors output;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    auto              f     = deployed.Predict(pools[k].images);
    output.emplace_back(split + "_code", f[0]);
    output.emplace_back(split + "_features", f[1]);
    output.emplace_back(split + "_logits", f[2]);
    output.emplace_back(split + "_probability", f[3]);
  }
  auto repeat = deployed.Predict(pools[2].images);
  check::Require(ae::Bytes(repeat[3]) == ae::Bytes(ae::Find(output, "test_probability")), "image-onlyheadinferenceBITEXACTrepeat");
  auto         single = deployed.Predict(nn::take_rows(pools[2].images, {0}));
  const double parity = ae::MaxAbsDiff(single[3], repeat[3], 10);
  check::Require(parity < 2e-6, "classifierSingle/batchparity");
  auto after = deployed.Arrays();
  for (std::size_t i = 0; i < before.size(); ++i)
    check::Require(ae::Bytes(before[i].second) == ae::Bytes(after[i].second), "allclassifierinferenceweightsimmutable");
  const auto       probability = repeat[3].to_vector<float>();
  const auto       labels      = pools[2].labels.to_vector<int>();
  std::vector<int> confusion(100, 0), predicted(200);
  for (int b = 0; b < 200; ++b) {
    int top = 0;
    for (int c = 1; c < 10; ++c)
      if (probability[b * 10 + c] > probability[b * 10 + top]) top = c;
    predicted[b] = top;
    ++confusion[labels[b] * 10 + top];
  }
  output.emplace_back("test_predicted", nn::from_vector(predicted, {200}));
  output.emplace_back("test_confusion", nn::from_vector(confusion, {10, 10}));
  nn::save_npz((root / "outputs.npz").string(), output);
  nn::save_npz((root / "model_arrays.npz").string(), after);
  auto          score = ae::TransferScore(repeat[3], pools[2].labels);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"frozen_encoder_parameters\":25120,\"trained_head_parameters\":330,\"epochs\":" << epochs << ",\"head_updates\":" << steps
       << ",\"selected_epoch\":" << selected << ",\"test_nll\":" << score[0] << ",\"test_accuracy\":" << score[1]
       << ",\"balanced_majority_accuracy\":0.1,\"single_batch_max_error\":" << parity
       << ",\"frozen_encoder_max_change\":0,\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html"), examples(root / "examples.tsv");
  html << "<!doctype html><meta charset='utf-8'><title>Frozenencoder transfer classifier</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1100px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid "
          "#bbb}</style><h1>Frozenencoder classifier:"
       << root.filename().string()
       << "</h1><p>784->32ReLUencoder transferred from verifiedTHREEepoch reconstructionAE OR its originalrandom initialweights. "
          "Encoder25120paramsFROZEN;32features->10softmaxHEAD330paramsONLY trained forTHREEepochs96Adamupdates. "
          "Fitting-onlymean/std(feature) normalization,floor1e-6,populationvariance,independently fitted perencoder. Labels NEVERenter "
          "extraction/normalization;fitlabels enter headCE,validationlabels select head,heldouttestlabels score only. Both variants "
          "SAMEhead initialization and batchorder.</p><p>1000fit200validation200test originalMNIST,balancedclasses. Pretrainedencoder's "
          "unsupervised epoch selected by validationRECONSTRUCTIONMSE previously;no classifier-label pretraining. Random control uses "
          "sameAEinitialencoderweights. This is frozen transfer/linear probing,NOTfine-tuning;the decoder is discarded. Pretraining had "
          "extra96unsupervisedupdates,so compute is not matched. No guaranteed advantage from this small model/single "
          "seed.</p><p>Testaccuracy="
       << score[1] << ",NLL=" << score[0] << ",balancedmajoritybaseline.1,selectedhead epoch" << selected
       << ". Encoder weights remain BITEXACT throughout allheadupdates;inference feeds imagesONLY. Perclass confusion(rows truth,columns "
          "prediction):</p><table>";
  for (int t = 0; t < 10; ++t) {
    html << "<tr>";
    for (int p = 0; p < 10; ++p) html << "<td>" << confusion[t * 10 + p] << "</td>";
    html << "</tr>";
  }
  html << "</table><h2>Firstheldoutcase pertruthdigit,unfiltered</h2><table><tr><th>Truth</th><th>Prediction</th><th>Source</th></tr>";
  examples << "true_digit\tpredicted_digit\tsource_id\n";
  const auto source_ids = pools[2].source.to_vector<int>();
  for (int d = 0; d < 10; ++d) {
    const int b = ae::FirstOfDigit(pools[2].labels, d);
    html << "<tr><td>" << d << "</td><td>" << predicted[b] << "</td><td>" << source_ids[b] << "</td></tr>";
    examples << d << '\t' << predicted[b] << '\t' << source_ids[b] << '\n';
  }
  html << "</table><p>All errors retained,no example qualityfilter. Linearhead results do not establish semantic disentanglement or "
          "downstreamtaskgeneralization.</p>";
  check::Require(bool(meta) && bool(html) && bool(examples) && bool(history) && bool(batches), "completeencodertransferartifacts");
  std::cout << root.filename() << ",selected" << selected << ",testacc=" << score[1] << ",NLL=" << score[0]
            << ";ONLYheadupdates/frozenEncoder/fitOnlyscaling/restore/imageOnly/batch/immutable PASS\nArtifacts:"
            << (root / "index.html").string() << std::endl;
  return seconds;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("saliency"), encoder = paths::OutputRoot() / "models/undercomplete_autoencoder",
                          out = paths::OutputRoot() / "models/encoder_pretraining", init;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("encoder=", 0) == 0)
        encoder = a.substr(8);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("init=", 0) == 0)
        init = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N data=PATH encoder=PATH out=PATH init=DIR");
    }
    check::Require(epochs > 0 && epochs <= 100, "THREEdefaultorexplicit1..100epochs");
    auto pools = ae::LoadSaliency(data);
    check::Require(std::filesystem::exists(encoder / "model_arrays.npz") && std::filesystem::exists(encoder / "initial_arrays.npz"),
                   "run undercomplete_autoencoder first (needs model_arrays.npz + initial_arrays.npz in " + encoder.string() + ")");
    auto pretrained = nn::load_npz((encoder / "model_arrays.npz").string()),
         random     = nn::load_npz((encoder / "initial_arrays.npz").string());
    const auto head_init = init.empty() ? init : init / "pretrained" / "initial_arrays.npz";
    double     seconds   = Run(pools, pretrained, out / "pretrained", epochs, head_init);
    seconds += Run(pools, random, out / "random", epochs, init.empty() ? init : init / "random" / "initial_arrays.npz");
    auto a = nn::load_npz((out / "pretrained" / "initial_arrays.npz").string()),
         b = nn::load_npz((out / "random" / "initial_arrays.npz").string());
    for (const auto& name : {"head/linear/W", "head/linear/b"})
      check::Require(ae::Bytes(ae::Find(a, name)) == ae::Bytes(ae::Find(b, name)), "matchedclassifierHEADinitializationEXACT");
    ae::CopyManifests(data, out);
    std::ofstream html(out / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>Pretraining encoders for downstream classifiers</title><h1>Frozenencoder "
            "pretraining comparison</h1><p><a href='pretrained/index.html'>Pretrained3epochAEencoder+3epochhead</a> versus <a "
            "href='random/index.html'>Originalrandomencoder+3epochhead</a>. "
            "Matchedhead330params/init/order,encoder25120paramsFROZEN,fit-onlyfeaturestandardization. Threeepochs96headupdates EACH. "
            "Reconstructionpretraining used extra96unsupervisedupdates;not matched totalcompute. Actual heldoutaccuracy/NLL/confusion and "
            "allerrors reported,no guaranteed pretraining benefit.</p>";
    std::cout << "training time " << seconds << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
