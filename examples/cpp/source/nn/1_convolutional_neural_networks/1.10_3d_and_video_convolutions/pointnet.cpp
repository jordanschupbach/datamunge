// CMD: just run-nn pointnet
// Both learned T-Nets,orthogonality penalty;default three short training epochs.
// Prepare:just point-cloud-assets. Options:epochs=N,data=PATH.
#include <datamunge/datasets/assets.hpp>
#include "common/point_training.hpp"

int main(int argc, char** argv) {
  using namespace point_training;
  constexpr bool plus = false;
  try {
    nn::set_num_threads(2);
    int                   epochs = 3;
    std::filesystem::path root   = paths::Dataset("point_clouds");
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        root = a.substr(5);
      else
        throw std::invalid_argument("expected epochs=N data=PATH");
    }
    using namespace point_training;
    {
      check::Require(epochs > 0, "positive short point model epochs");
      if (!std::filesystem::exists(root / "point_clouds.npz"))
        datamunge::datasets::prepare_example_data("mesh", (root).string());
      auto              pools      = LoadPoints(root);
      const auto&       fit        = pools[0];
      const auto&       validation = pools[1];
      const auto&       test       = pools[2];
      const std::string name       = plus ? "pointnet_plus_plus" : "pointnet";
      auto directory = paths::OutputDir("models/" + name), checkpoint = directory / "best.npz";
      PointModel model(plus);
      auto       untrained = Predict(model, test);
      double     initial   = sensor::ScoreSensor(Predict(model, fit), fit.labels, 3).nll;
      auto       before    = model.Values();
      std::cout << name << ":" << model.parameters.count() << " parameters/"
                << model.parameters.size() << " learned+" << model.state.size() << " BN tensors;"
                << epochs
                << " epochs,batch4,Adam.001,width/4,256 real points;24fit/6validation/12test\n";
      std::ofstream history(directory / "history.tsv");
      history << "epoch\ttrain_objective\tvalidation_nll\n";
      nn::BatchOrder order(24, 4, 983);
      double         best     = INFINITY;
      int            selected = 0;
      for (int epoch = 1; epoch <= epochs; ++epoch) {
        order.begin_epoch();
        double      sum  = 0;
        std::size_t seen = 0;
        for (;;) {
          auto rows = order.next();
          if (rows.empty())
            break;
          const auto batch = Rows(fit, rows);
          const auto net   = model.Forward(batch.xyz, true, true);
          const auto ce    = nn::sparse_softmax_cross_entropy(net.logits, batch.labels);
          const auto loss =
              plus ? ce
                   : nn::add(ce, point_models::PointTransformPenalty(net.feature_transform, 16));
          const double value = loss.item();
          check::Require(std::isfinite(value) && value >= 0, "finite point objective");
          model.optimizer->minimize(loss);
          sum += rows.size() * value;
          seen += rows.size();
        }
        check::Require(seen == 24, "every fitting object once per epoch");
        double score = sensor::ScoreSensor(Predict(model, validation), validation.labels, 3).nll;
        if (score < best) {
          best     = score;
          selected = epoch;
          model.Save(checkpoint);
        }
        history << epoch << '\t' << sum / seen << '\t' << score << '\n';
        history.flush();
        std::cout << "epoch " << epoch << ":train=" << sum / seen << ",validationNLL=" << score
                  << std::endl;
      }
      auto after = model.Values();
      if (!plus)
        for (const std::string head : {"input_tnet/head/W", "feature_tnet/head/W"}) {
          check::Require(
              after.count(head) &&
                  nn::sum(nn::abs(nn::sub(after.at(head), before.at(head)))).item() > 0,
              "both transform heads learned away from zero");
        }
      const auto diagnostic = Rows(fit, {0, 8, 16, 1});
      const auto net        = model.Forward(diagnostic.xyz, true, false);
      const auto ce         = nn::sparse_softmax_cross_entropy(net.logits, diagnostic.labels);
      const auto objective =
          plus ? ce : nn::add(ce, point_models::PointTransformPenalty(net.feature_transform, 16));
      auto gradients = nn::grad(objective, model.parameters.parameters());
      int  nonzero   = 0;
      for (const auto& g : gradients) {
        double norm = 0;
        for (float v : g.to_vector<float>()) {
          check::Require(std::isfinite(v), "every gradient finite");
          norm += double(v) * v;
        }
        nonzero += norm > 0;
      }
      std::cout << "actual finite gradient tensors=" << gradients.size() << ",nonzero=" << nonzero
                << " (pre-BN biases can be zero)\n";
      model.Load(checkpoint);
      check::Near(
          sensor::ScoreSensor(Predict(model, validation), validation.labels, 3).nll,
          best,
          1e-6,
          "selected validation restore");
      PointModel deployed(plus, false);
      deployed.Load(checkpoint);
      auto immutable = deployed.Values();
      auto logits = Predict(deployed, test), reference = Predict(model, test),
           repeat = Predict(deployed, test);
      check::Near(replay::MaxAbsDiff(logits, reference), 0, 1e-6, "fresh classifier restore");
      check::Near(replay::MaxAbsDiff(logits, repeat), 0, 0, "deterministic inference");
      auto single = deployed.Predict(Rows(test, {0}));
      check::Near(
          replay::MaxAbsDiff(single, nn::slice(logits, {0, 0}, {1, 3})),
          0,
          2e-6,
          "single/batch moving BN inference");
      auto reversed   = test;
      reversed.xyz    = nn::reverse(test.xyz, {1});
      auto   permuted = Predict(deployed, reversed);
      double change   = replay::MaxAbsDiff(logits, permuted);
      if (!plus)
        check::Near(change, 0, 2e-6, "trained PointNet permutation invariance");
      auto altered = Rows(test, {0});
      altered.xyz  = nn::zeros({1, 256, 3});
      auto blank   = deployed.Predict(altered);
      check::Require(
          nn::sum(nn::abs(nn::sub(blank, single))).item() > 1e-7, "prediction depends on XYZ");
      check::Require(
          replay::SameBits(immutable, deployed.Values()),
          "all learned/BN tensors immutable at inference");
      deployed.Save(directory / "model_arrays.npz");
      nn::save_npz(
          (directory / "test_outputs.npz").string(),
          {{"xyz", test.xyz},
           {"labels", test.labels},
           {"source_id", test.source},
           {"logits", logits},
           {"untrained", untrained},
           {"reversed", permuted},
           {"fit_xyz", fit.xyz},
           {"fit_labels", fit.labels},
           {"fit_source_id", fit.source}});
      std::filesystem::copy_file(
          root / "manifest.json",
          directory / "data_manifest.json",
          std::filesystem::copy_options::overwrite_existing);
      Metrics("trained " + name, test, logits);
      Metrics("untrained " + name, test, untrained);
      std::cout
          << "selected=" << selected << ",fit NLL " << initial << " -> "
          << sensor::ScoreSensor(Predict(model, fit), fit.labels, 3).nll
          << ",reverse maxlogit change=" << change
          << ";freshrestore/XYZ-only API/sensitivity/single-batch/determinism/immutable state "
             "passed\n"
          << "Artifacts:" << directory
          << "\nTiny three-class subset,short unpretrained run;not original40-class benchmark. "
          << "PointNet++ first-seed FPS/first-K ball grouping remains order-dependent.\n";
    }
    // Independent forward parity: use TensorFlow's trained arrays and saved
    // observed XYZ, without depending on either framework's initialization.
    const auto tf = replay::TfModels(plus ? "pointnet_plus_plus" : "pointnet");
    if (std::filesystem::exists(tf / "model_arrays.npz") &&
        std::filesystem::exists(tf / "test_outputs.npz")) {
      PointModel recorded(plus, false);
      recorded.Load(tf / "model_arrays.npz");
      const auto arrays = replay::ToMap(nn::load_npz((tf / "test_outputs.npz").string()));
      PointData  data{arrays.at("xyz"), arrays.at("labels"), arrays.at("source_id")};
      const auto error = replay::MaxAbsDiff(Predict(recorded, data), arrays.at("logits"));
      check::Near(error, 0, 1e-5, "TensorFlow trained-weight XYZ inference parity");
      std::cout << "TF trained-weight inference max difference=" << error << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
