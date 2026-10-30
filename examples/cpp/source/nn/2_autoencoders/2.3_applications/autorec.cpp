// CMD: just run-nn autorec
// Prepare:just autorec-assets. THREEdefault epochs;epochs=N/data=PATH/out=PATH.
//
// Port of tflow's autorec (model: examples/nn/common/autorec.hpp). Preparation (in
// ~/projects/tflow_examples): just autorec-assets; data (ratings.npz, u.data, manifest.json)
// are read from $DATAMUNGE_DATA_DIR/autorec. Outputs (mask_algebra.npz, data.npz,
// initial/first_*/epoch<k>_arrays.npz, history.tsv, batches.tsv, outputs.npz,
// model_arrays.npz, metrics.json, heldout_examples.tsv, index.html) go to
// output/nn/models/autorec. Checkpoints are best.npz instead of TF SaveV2 bundles;
// metrics.json gains "training_seconds". Options: epochs=N data=PATH out=PATH init=PATH
// (start from saved parameters, e.g. tflow's autorec/initial_arrays.npz, to follow its
// trajectory exactly; by default nn's own initialiser is used). Batch order is tflow's.

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "common/autorec.hpp"

namespace {

namespace nn = datamunge::nn;

void Unit(const std::filesystem::path& root) {
  const std::vector<float> pv{.2f, .3f, .4f, .5f, .6f, .7f}, tv{.1f, .9f, .5f, .7f, .2f, .8f}, mv{1, 0, 1, 0, 1, 0};
  std::vector<float>       cv(6);
  for (int i = 0; i < 6; ++i) cv[i] = mv[i] ? tv[i] : 1000;
  auto p = nn::from_vector(pv, {2, 3}).requires_grad_(), t = nn::from_vector(tv, {2, 3}), m = nn::from_vector(mv, {2, 3}),
       changed = nn::from_vector(cv, {2, 3});
  auto loss = ae::ObservedRatingMse(p, t, m), altered = ae::ObservedRatingMse(p, changed, m);
  auto gradient = nn::grad(loss, {p})[0].contiguous();
  check::Near(loss.item<float>(), .06, 2e-8, "observed-onlydenominator3");
  check::Require(ae::Bytes(loss) == ae::Bytes(altered), "changingALLunobservedtargetsleaveslossBITEXACT");
  const auto g = gradient.to_vector<float>();
  for (int i = 0; i < 6; ++i)
    if (!mv[i]) check::Near(g[i], 0, 0, "unobservedoutputgradientEXACTzero");
  nn::save_npz((root / "mask_algebra.npz").string(), {{"prediction", p.detach()},
                                                     {"target", t},
                                                     {"mask", m},
                                                     {"changed_target", changed},
                                                     {"loss", loss.detach()},
                                                     {"changed_loss", altered.detach()},
                                                     {"gradient", gradient}});
}

void Run(const std::filesystem::path& data, const std::filesystem::path& root, int epochs, const std::string& init) {
  auto d = ae::LoadAutoRec(data);
  std::filesystem::create_directories(root);
  Unit(root);
  if (std::filesystem::exists(data / "manifest.json"))
    std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json", std::filesystem::copy_options::overwrite_existing);
  nn::save_npz((root / "data.npz").string(), {{"input", d.input},
                                             {"ratings", d.ratings},
                                             {"train_mask", d.train},
                                             {"validation_mask", d.validation},
                                             {"test_mask", d.test},
                                             {"user_ids", d.users},
                                             {"item_ids", d.items}});
  ae::AutoRecModel model;
  if (!init.empty()) model.parameters.load(init);
  check::Require(model.parameters.count() == 16672 && model.parameters.size() == 4, "256-32sigmoid-256linearAutoRecinventory");
  nn::save_npz((root / "initial_arrays.npz").string(), ae::Arrays(model.parameters));
  nn::save_npz((root / "initial_outputs.npz").string(), {{"prediction", model.Predict(d.input)[1]}});
  nn::BatchOrder order(199, 32, 4621);
  std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
  history.precision(17);
  history << "epoch\ttrain_rmse\tvalidation_raw_rmse\tvalidation_raw_mae\n";
  batches << "epoch\tstep\trow\tuser_id\n";
  const auto users = d.users.to_vector<int>();
  const auto best_path = (root / "best.npz").string();
  double     best      = std::numeric_limits<double>::infinity();
  int        selected = 0, steps = 0;
  const auto started = std::chrono::steady_clock::now();
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    int seen = 0;
    for (;;) {
      auto rows = order.next();
      if (rows.empty()) break;
      auto x = nn::take_rows(d.input, rows), mask = nn::take_rows(d.train, rows);
      if (!steps) {
        nn::save_npz((root / "first_batch.npz").string(), {{"input", x}, {"mask", mask}});
        auto out = model.Predict(x), objective = model.Objective(x, mask);
        nn::save_npz((root / "first_forward.npz").string(), {{"code", out[0]},
                                                            {"prediction", out[1]},
                                                            {"observed_mse", objective[0]},
                                                            {"regularizer", objective[1]},
                                                            {"loss", objective[2]}});
        auto gradients = ae::CheckGradients(model.parameters, model.Gradient(x, mask), "allobserved/L2derivativesfinite",
                                            "allfourAutoRectensorsreceivegradient");
        nn::save_npz((root / "first_gradients.npz").string(), gradients);
      }
      model.Update(x, mask);
      if (!steps) nn::save_npz((root / "first_update_arrays.npz").string(), ae::Arrays(model.parameters));
      for (auto row : rows) batches << epoch << '\t' << steps << '\t' << row << '\t' << users[row] << '\n';
      ++steps;
      seen += int(rows.size());
    }
    check::Require(seen == 199, "everywarmuseronceperepoch");
    auto prediction = model.Predict(d.input)[1];
    auto train = ae::RatingErrors(prediction, d.ratings, d.train), val = ae::RatingErrors(prediction, d.ratings, d.validation);
    history << epoch << '\t' << train[0] << '\t' << val[0] << '\t' << val[1] << '\n';
    nn::save_npz((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(), ae::Arrays(model.parameters));
    if (val[0] < best) {
      best     = val[0];
      selected = epoch;
      model.parameters.save(best_path);  // WITHHELD validation RAW rating RMSE selects model
    }
    std::cout << "epoch" << epoch << ":trainRMSE=" << train[0] << ",validationRMSE=" << val[0] << std::endl;
  }
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ae::AutoRecModel deployed(false);
  deployed.parameters.load(best_path);  // fresh AutoRec model-only restore
  auto before = ae::Snapshot(deployed.parameters);
  auto f = deployed.Predict(d.input), repeat = deployed.Predict(d.input);
  check::Near(ae::RatingErrors(f[1], d.ratings, d.validation)[0], best, 1e-9, "restoredvalidationRMSE");
  check::Require(ae::Bytes(f[1]) == ae::Bytes(repeat[1]), "ratinginferenceBITEXACTrepeat");
  auto         single = deployed.Predict(nn::take_rows(d.input, {0}));
  const double parity = ae::MaxAbsDiff(single[1], f[1], 256);
  check::Require(parity < 2e-6, "singleuser/batchpredictionparity");
  const auto tr = d.train.to_vector<float>(), ratings = d.ratings.to_vector<float>(), test_mask = d.test.to_vector<float>();
  double     global = 0, count = 0, user_sum[199] = {}, user_count[199] = {}, item_sum[256] = {}, item_count[256] = {};
  for (int u = 0; u < 199; ++u)
    for (int i = 0; i < 256; ++i)
      if (tr[u * 256 + i]) {
        const double value = ratings[u * 256 + i];
        global += value;
        ++count;
        user_sum[u] += value;
        ++user_count[u];
        item_sum[i] += value;
        ++item_count[i];
      }
  global /= count;
  std::vector<double> bgv(199 * 256), buv(199 * 256), biv(199 * 256);
  for (int u = 0; u < 199; ++u)
    for (int i = 0; i < 256; ++i) {
      bgv[u * 256 + i] = global;
      buv[u * 256 + i] = (user_sum[u] + 10 * global) / (user_count[u] + 10);
      biv[u * 256 + i] = (item_sum[i] + 10 * global) / (item_count[i] + 10);
    }
  auto baseline_global = nn::from_vector(bgv, {199, 256}), baseline_user = nn::from_vector(buv, {199, 256}),
       baseline_item = nn::from_vector(biv, {199, 256});
  auto errors = ae::RatingErrors(f[1], d.ratings, d.test), clipped = ae::RatingErrors(f[1], d.ratings, d.test, true),
       bg = ae::RatingErrors(baseline_global, d.ratings, d.test), bu = ae::RatingErrors(baseline_user, d.ratings, d.test),
       bi = ae::RatingErrors(baseline_item, d.ratings, d.test);
  check::Require(before == ae::Snapshot(deployed.parameters), "allrecommendationinferenceimmutable");
  nn::save_npz((root / "outputs.npz").string(), {{"code", f[0]},
                                                {"prediction_normalized", f[1]},
                                                {"global_baseline", baseline_global},
                                                {"user_baseline", baseline_user},
                                                {"item_baseline", baseline_item}});
  nn::save_npz((root / "model_arrays.npz").string(), ae::Arrays(deployed.parameters));
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"parameters\":16672,\"tensors\":4,\"epochs\":" << epochs << ",\"updates\":" << steps << ",\"selected_epoch\":" << selected
       << ",\"users\":199,\"items\":256,\"train_ratings\":8079,\"validation_ratings\":898,\"test_ratings\":898,\"kernel_l2_lambda\":0.0001,"
          "\"test_raw_rmse\":"
       << errors[0] << ",\"test_raw_mae\":" << errors[1] << ",\"test_clipped_rmse\":" << clipped[0]
       << ",\"test_clipped_mae\":" << clipped[1] << ",\"fit_global_baseline_rmse\":" << bg[0]
       << ",\"fit_user_shrink10_baseline_rmse\":" << bu[0] << ",\"fit_item_shrink10_baseline_rmse\":" << bi[0]
       << ",\"single_batch_max_error\":" << parity << ",\"training_seconds\":" << seconds << "}\n";
  std::ofstream html(root / "index.html"), examples(root / "heldout_examples.tsv");
  html << "<!doctype html><meta charset='utf-8'><title>User-based AutoRec</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1100px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid "
          "#bbb}</style><h1>Collaborative filtering with U-AutoRec</h1><p><a "
          "href='https://ssanner.github.io/papers/anu/www15.pdf'>AutoRec,2015</a> compact user-based "
          "variant:256ratings->32sigmoid->256LINEAR,16672params. Observedtrainratings/5input,allunknownANDvalidation/testentriesZERO. "
          "Trainingtarget is ONLYsameTRAINratings,loss SUMobserved "
          "squaredresidual/SUMtrainmask+.0001/2*(||Wencoder||^2+||Wdecoder||^2),biases unregularized. Missingratings are NOTzero "
          "negatives. THREEdefault epochs21Adamupdates,32userbatch/final7. Prediction input-only,noheldouttargets or scoringmask.</p><p><a "
          "href='https://grouplens.org/datasets/movielens/100k/'>GroupLensMovieLens100K</a>,research-use dataset:Harper and "
          "Konstan,2015,The MovieLens Datasets:History and Context,DOI10.1145/2827872. Download source+archive/dataSHA256 pinned;assets "
          "stay output-only. Subsetusers1..256with>=10ratings amongitems1..256 gives199users256items9875ratings. Reserveoneobservedrecord "
          "peritem in train by smallest SHA256(autorec-v1:user:item);within eachuser sort "
          "remaininglocations,firstfloor(n/10)validation,nexttest,remaindertrain. 8079train/898validation/898test. Allusers/items "
          "warm;this is transductive ratingcompletion,not time-split/cold-start or paper-scalebenchmark. No demographics used. Source "
          "locations,notratingvalues/timestamps,selectsplit.</p><p>ValidationRAWratingRMSE selects epoch"
       << selected
       << ". All8079fittingobservations remain inferencecontext;withheldratings NEVERfed. Rawlinearoutputs scaled*5back "
          "tostarunits;clip1..5reported SEPARATELY,not optimized or used for checkpointselection. TestRMSE raw="
       << errors[0] << ",clipped=" << clipped[0] << ",rawMAE=" << errors[1]
       << ". FitONLYglobalmean/usermean(shrink10)/itemmean(shrink10) RMSE=" << bg[0] << "/" << bu[0] << "/" << bi[0]
       << ". Short-run failures retained,no recommender superiority/ranking-quality claim. "
          "RMSEdoesnotmeasuretop-Nrecommendations.</p><h2>First20withheldtestentries "
          "inuser/itemorder,unfiltered</h2><table><tr><th>User</th><th>Item</th><th>Heldoutstars</th><th>Rawprediction</th><th>Clipped</"
          "th><th>Fititembaseline</th></tr>";
  examples.precision(17);
  examples << "user_id\titem_id\ttrue_rating\tprediction_raw\tprediction_clipped\tfit_item_baseline\n";
  const auto pred  = f[1].to_vector<float>();
  int        shown = 0;
  for (int u = 0; u < 199 && shown < 20; ++u)
    for (int i = 0; i < 256 && shown < 20; ++i)
      if (test_mask[u * 256 + i]) {
        const double p = double(pred[u * 256 + i]) * 5;
        html << "<tr><td>" << users[u] << "</td><td>" << i + 1 << "</td><td>" << ratings[u * 256 + i] << "</td><td>" << p << "</td><td>"
             << std::clamp(p, 1., 5.) << "</td><td>" << biv[u * 256 + i] << "</td></tr>";
        examples << users[u] << '\t' << i + 1 << '\t' << ratings[u * 256 + i] << '\t' << p << '\t' << std::clamp(p, 1., 5.) << '\t'
                 << biv[u * 256 + i] << '\n';
        ++shown;
      }
  html << "</table><p>No unseenrating labels are assumed negative. Model-only checkpoint restoration does not resume optimizer state. "
          "Section9standaloneAutoRec item remains pending until catalogue order reaches it.</p>";
  check::Require(bool(meta) && bool(html) && bool(examples) && bool(history) && bool(batches), "completeAutoRecartifacts");
  std::cout << "selected" << selected << ",testRMSEraw=" << errors[0] << ",clipped=" << clipped[0] << ",fitglobal/user/item=" << bg[0]
            << "/" << bu[0] << "/" << bi[0]
            << ";observedOnly/noHeldoutInput/warmSplit/maskGradZero/restore/immutable PASS\ntraining time " << seconds
            << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    std::string           init;
    std::filesystem::path data = paths::Dataset("autorec"), out = paths::OutputRoot() / "models/autorec";
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
    check::Require(epochs > 0 && epochs <= 100, "THREEdefaultorexplicit1..100epochs");
    Run(data, out, epochs, init);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
