// CMD: just run-nn nice_density
// NICE exact likelihood/inverse/Jacobians on a reproducible analytic banana density.
//
// Port of tflow's nice_density (common/ml/nice_density.h, common/nn/nice_density.h
// -> examples/nn/common/{flow_density,nice_density}.hpp). Four alternating
// additive couplings (each 1 -> 16 tanh -> 1 shift) and a learned positive global
// diagonal scale; 198 parameters / 17 tensors; unit coupling determinants, so
// logdet = sum(log_scale). Data: SYNTHETIC banana generated in-process,
// bit-identical to tflow's data.npz. Three epochs of Adam .003, batch 32 in
// tflow's BatchOrder(512, 32, 9323), fixed final weights.
//
// Checks (as tflow): census; bit-exact restore; actual input Jacobian
// determinants positive and equal to the analytic logdet (forward, and inverse on
// all 1024 prior draws); both roundtrips < 5e-6; latent-only repeat sampling
// bit-exact; change-of-variables normalization identity for every sample;
// single/batch parity; immutable inference. Added: data and TF replay checks
// (see flow_density.hpp). Deviations: final.npz instead of a TF checkpoint, no
// independent_audit.json.
// Options: epochs=3 out=DIR init=auto|tf|nn|PATH tf=DIR threads=4 verbose=0.
// Output: output/nn/models/nice_density.
#include "common/nice_density.hpp"

using namespace flowd;

int main(int argc, char** argv) {
  try {
    const auto o = ParseOptions(argc, argv, "nice_density");
    Spec       s;
    s.name = "nice_density", s.tag = "NICE", s.title = "NICE density";
    s.parameters = 198, s.tensors = 17, s.batch_seed = 9323, s.global_logdet = true;
    s.protocol = BananaProtocol();
    s.make     = [](bool training) { return std::make_unique<NiceDensity>(training); };
    s.html =
        "<h1>NICE: exact invertible density</h1><p><a "
        "href='https://arxiv.org/abs/1410.8516'>NICE</a>: four alternating "
        "additive couplings, each 1 -> 16 tanh -> 1 shift, then a learned positive global diagonal "
        "scale exp(log_s); 198 "
        "parameters / 17 tensors. Coupling Jacobian determinant 1, total logdet = sum(log_s), "
        "standard Gaussian 2-D base, "
        "NLL = log 2pi + .5 sum(z^2) - sum(log_s). Exact inverse: unscale, then reverse couplings "
        "subtracting the shift.</p><p>"
        "SYNTHETIC analytic banana u ~ N(0, I), x0 = u0, x1 = u1 + .5 (u0^2 - 1); 512 fit / 128 "
        "validation / 128 test, three "
        "epochs (48 Adam .003 updates), fixed final. Additive NICE with only a global determinant "
        "cannot create multiple "
        "density modes; this banana has one curved mode.</p>";
    {
      const auto     started = std::chrono::steady_clock::now();
      const fs::path root    = s.variant.empty() ? o.out : o.out / s.variant;
      const fs::path tf      = s.variant.empty() ? o.tf : o.tf / s.variant;
      fs::create_directories(root);
      if (s.before)
        s.before(root);
      const Data data = MakeData(s.kind);
      nn::save_npz((root / "data.npz").string(), replay::ToNamed(data.arrays));
      {
        std::ofstream protocol(root / "source_protocol.json");
        protocol << s.protocol;
        check::Require(bool(protocol), "source protocol written");
      }
      if (fs::exists(tf / "data.npz")) {
        const auto theirs = replay::ToMap(nn::load_npz((tf / "data.npz").string()));
        for (const auto& [k, v] : data.arrays)
          check::Require(
              theirs.count(k) && Bits(v) == Bits(theirs.at(k)),
              "data array " + k + " bit-identical to TF's data.npz");
        std::cout << "data: all " << data.arrays.size()
                  << " arrays bit-identical to TF's data.npz\n";
      }
      const Tensor& fit   = data.pools[0];
      auto          model = s.make(true);
      check::Require(
          model->store.count() == s.parameters && std::int64_t(model->store.size()) == s.tensors,
          std::to_string(s.parameters) + " " + s.tag + " parameters / " +
              std::to_string(s.tensors) + " tensors");
      model->Prepare(fit, root);
      const fs::path init = InitFile(o.init, tf);
      if (!init.empty()) {
        replay::LoadInto(model->store, init);
        std::cout << "replaying TF's initial arrays " << init.string() << "\n";
        if (s.replayed)
          s.replayed(*model, data);
      } else {
        std::cout << "weights from nn's generator (statistical parity with TF only)\n";
      }
      nn::save_npz((root / "initial_arrays.npz").string(), replay::ToNamed(model->Arrays()));
      nn::save_npz(
          (root / "initial_outputs.npz").string(), replay::ToNamed(model->Evaluate(data.pools[2])));
      nn::BatchOrder order(512, 32, s.batch_seed);
      std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
      history.precision(17);
      history
          << (s.global_logdet ? "epoch\tvalidation_nll\tvalidation_forward_logdet\n"
                              : "epoch\tvalidation_nll\tvalidation_mean_forward_logdet\n");
      batches << "epoch\tstep\trow\n";
      int    steps = 0;
      Tensor first;
      for (int epoch = 1; epoch <= o.epochs; ++epoch) {
        order.begin_epoch();
        for (;;) {
          const auto rows = order.next();
          if (rows.empty())
            break;
          const Tensor batch = nn::take_rows(fit, rows);
          if (!steps) {
            first = batch;
            nn::save_npz((root / "first_batch.npz").string(), {{"points", batch}});
            nn::save_npz(
                (root / "first_forward.npz").string(), replay::ToNamed(model->Evaluate(batch)));
            nn::save_npz(
                (root / "initial_gradients.npz").string(), replay::ToNamed(model->Gradient(batch)));
          }
          model->Update(batch);
          if (!steps)
            nn::save_npz((root / "first_after.npz").string(), replay::ToNamed(model->Arrays()));
          for (auto row : rows)
            batches << epoch << '\t' << steps << '\t' << row << '\n';
          ++steps;
        }
        const auto   output = model->Evaluate(data.pools[1]);
        const double loss = output.at("loss").item<float>(), ld = Mean(output.at("forward_logdet"));
        history << epoch << '\t' << loss << '\t' << ld << '\n';
        nn::save_npz(
            (root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string(),
            replay::ToNamed(model->Arrays()));
        nn::save_npz(
            (root / ("epoch" + std::to_string(epoch) + "_validation.npz")).string(),
            replay::ToNamed(output));
        std::cout << "epoch" << epoch << ":validationNLL=" << loss << ",forwardlogdet=" << ld
                  << std::endl;
      }
      const double train_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      model->store.save((root / "final.npz").string());
      nn::save_npz(
          (root / "final_gradients.npz").string(), replay::ToNamed(model->Gradient(first)));
      auto deployed = s.make(false);
      deployed->store.load((root / "final.npz").string(), true);
      check::Require(
          SameBits(model->Arrays(), deployed->Arrays()),
          "all " + s.tag + " weights BITEXACT restore");
      if (s.after)
        s.after(*deployed, data, root);
      const auto before = deployed->Arrays();
      const auto output = deployed->Evaluate(data.pools[2]);
      nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(output));
      nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(deployed->Arrays()));

      // Exact (or, for the CNF, numerical) inverse and analytic logdet against the actual Jacobian.
      const auto test  = data.pools[2].to_vector<float>();
      const auto rec   = output.at("reconstruction").to_vector<float>(),
                 j0    = output.at("jacobian_row0").to_vector<float>(),
                 j1    = output.at("jacobian_row1").to_vector<float>(),
                 fld   = output.at("forward_logdet").to_vector<float>();
      double roundtrip = 0, det_error = 0;
      for (int b = 0; b < 128; ++b) {
        for (int c = 0; c < 2; ++c)
          roundtrip = std::max(roundtrip, std::abs(double(rec[b * 2 + c]) - test[b * 2 + c]));
        const double determinant =
            double(j0[b * 2]) * j1[b * 2 + 1] - double(j0[b * 2 + 1]) * j1[b * 2];
        check::Require(determinant > 0, "orientation preserving " + s.tag + " Jacobian");
        det_error = std::max(det_error, std::abs(std::log(determinant) - At(fld, b)));
      }
      check::Require(
          roundtrip < 5e-6 && det_error < 5e-6,
          std::string(s.numerical ? "numerical" : "exact") +
              " inverse and analytic logdet within F32");
      const auto samples = deployed->Sample(data.prior);
      check::Require(
          SameBits(samples, deployed->Sample(data.prior)),
          "latent ONLY repeat " + s.tag + " samples and Jacobians BITEXACT");
      const auto pv      = data.prior.to_vector<float>(),
                 pr      = samples.at("prior_roundtrip").to_vector<float>(),
                 i0      = samples.at("inverse_jacobian_row0").to_vector<float>(),
                 i1      = samples.at("inverse_jacobian_row1").to_vector<float>(),
                 ild     = samples.at("inverse_logdet").to_vector<float>();
      double prior_error = 0;
      for (int b = 0; b < 1024; ++b) {
        for (int c = 0; c < 2; ++c)
          prior_error = std::max(prior_error, std::abs(double(pr[b * 2 + c]) - pv[b * 2 + c]));
        const double determinant =
            double(i0[b * 2]) * i1[b * 2 + 1] - double(i0[b * 2 + 1]) * i1[b * 2];
        check::Require(determinant > 0, "inverse orientation preserving");
        check::Near(
            std::log(determinant),
            At(ild, b),
            5e-6,
            "analytic INVERSE logdet matches actual Jacobian");
      }
      check::Require(prior_error < 5e-6, "forward(inverse(prior)) roundtrip");
      const auto sample_density = deployed->Evaluate(samples.at("samples"));
      const auto sdl            = sample_density.at("log_probability").to_vector<float>();
      for (int b = 0; b < 1024; ++b) {
        const double base_log = -std::log(2 * M_PI) - .5 * (std::pow(double(pv[b * 2]), 2) +
                                                            std::pow(double(pv[b * 2 + 1]), 2));
        check::Near(
            sdl[b] + At(ild, b),
            base_log,
            2e-5,
            "normalization CHANGE-OF-VARIABLES identity for all ancestral samples");
      }
      nn::save_npz((root / "samples.npz").string(), replay::ToNamed(samples));
      nn::save_npz((root / "sample_density.npz").string(), replay::ToNamed(sample_density));

      // Density grid and plot.
      std::vector<float>  grid(128 * 128 * 2);
      std::vector<double> truth(128 * 128);
      for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x) {
          const int j = y * 128 + x;
          grid[j * 2] = float((x + .5) / 16 - 4), grid[j * 2 + 1] = float(4 - (y + .5) / 16);
          truth[j] = TrueLog(grid[j * 2], grid[j * 2 + 1], s.kind);
        }
      const Tensor grid_t  = nn::from_vector(grid, {128 * 128, 2});
      const auto   density = deployed->Evaluate(grid_t);
      nn::save_npz(
          (root / "grid.npz").string(),
          {{"points", grid_t},
           {"model_log_probability", density.at("log_probability")},
           {"true_log_probability", nn::from_vector(truth, {128 * 128})}});
      Plot(
          root / "density.png",
          density.at("log_probability"),
          truth,
          data.pools[2],
          samples.at("samples"),
          output.at("latent"),
          data.prior,
          s.density_scale);

      // Fit-only Gaussian baselines and the analytic truth.
      const auto fv      = fit.to_vector<float>();
      double     mean[2] = {0, 0}, cov[2][2] = {{0, 0}, {0, 0}};
      for (int b = 0; b < 512; ++b)
        for (int c = 0; c < 2; ++c)
          mean[c] += fv[b * 2 + c] / 512.;
      for (int b = 0; b < 512; ++b)
        for (int c = 0; c < 2; ++c)
          for (int d = 0; d < 2; ++d)
            cov[c][d] +=
                (double(fv[b * 2 + c]) - mean[c]) * (double(fv[b * 2 + d]) - mean[d]) / 512.;
      const double det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
      check::Require(det > 0, "fitting MLE Gaussian positive covariance");
      double              full_nll = 0, diag_nll = 0, true_nll = 0, identity_nll = 0;
      std::vector<double> true_logp(128);
      for (int b = 0; b < 128; ++b) {
        const double x = test[b * 2], y = test[b * 2 + 1], a = x - mean[0], c = y - mean[1];
        true_logp[b] = TrueLog(x, y, s.kind);
        true_nll -= true_logp[b] / 128;
        identity_nll += (std::log(2 * M_PI) + .5 * (x * x + y * y)) / 128;
        full_nll += (std::log(2 * M_PI) + .5 * std::log(det) +
                     .5 * (a * a * cov[1][1] - 2 * a * c * cov[0][1] + c * c * cov[0][0]) / det) /
                    128;
        diag_nll += (std::log(2 * M_PI) + .5 * std::log(cov[0][0] * cov[1][1]) +
                     .5 * (a * a / cov[0][0] + c * c / cov[1][1])) /
                    128;
      }
      nn::save_npz(
          (root / "baseline.npz").string(),
          {{"fit_mean", nn::from_vector(std::vector<double>{mean[0], mean[1]}, {2})},
           {"fit_covariance",
            nn::from_vector(
                std::vector<double>{cov[0][0], cov[0][1], cov[1][0], cov[1][1]}, {2, 2})},
           {"true_test_log_probability", nn::from_vector(true_logp, {128})}});
      const auto single = deployed->Evaluate(nn::take_rows(data.pools[2], {0}));
      const auto sl     = single.at("latent").to_vector<float>(),
                 ol     = output.at("latent").to_vector<float>();
      for (int c = 0; c < 2; ++c)
        check::Near(sl[c], ol[c], 2e-6, "single/batch " + s.tag + " parity");
      check::Require(
          SameBits(before, deployed->Arrays()), "density/Jacobian/inverse/sampling ALL immutable");
      double min_ld = INFINITY, max_ld = -INFINITY;
      if (!s.global_logdet) {
        const auto inv_ld = output.at("inverse_logdet").to_vector<float>();
        for (int b = 0; b < 128; ++b) {
          min_ld = std::min(min_ld, double(fld[b]));
          max_ld = std::max(max_ld, double(fld[b]));
          check::Near(
              fld[b] + inv_ld[b],
              0,
              5e-6,
              "forward and reconstruction inverse logdets cancel PER POINT");
        }
        check::Require(max_ld - min_ld > 1e-5, "spatially varying non-volume-preserving Jacobian");
      }

      // flow_density_2d's sample diagnostics.
      [[maybe_unused]] double      diagnostic = 0;
      std::string diagnostics_json;
      if (s.kind >= 0) {
        const auto         sv      = samples.at("samples").to_vector<float>();
        double             support = 0, curvature = 0;
        std::array<int, 8> mode_counts{};
        for (int b = 0; b < 1024; ++b) {
          const double x = sv[b * 2], y = sv[b * 2 + 1];
          support += CheckerSupport(x, y) / 1024.;
          curvature += y * (x * x - 1) / 1024.;
          double nearest = INFINITY;
          int    mode    = -1;
          for (int j = 0; j < 8; ++j) {
            const double d =
                std::hypot(x - 2 * std::cos(2 * M_PI * j / 8), y - 2 * std::sin(2 * M_PI * j / 8));
            if (d < nearest)
              nearest = d, mode = j;
          }
          if (nearest <= .45)
            ++mode_counts[mode];
        }
        int covered = 0, total_near = 0;
        for (int count : mode_counts)
          covered += count > 0, total_near += count;
        nn::save_npz(
            (root / "sample_diagnostics.npz").string(),
            {{"mode_counts_within_three_sigma",
              nn::from_vector(
                  std::vector<std::int32_t>(mode_counts.begin(), mode_counts.end()), {8})}});
        std::ofstream diagnostics(root / "sample_diagnostics.json");
        diagnostics.precision(17);
        diagnostics << "{\"checkerboard_support_fraction\":" << support
                    << ",\"eight_gaussian_fraction_within_three_sigma_of_centers\":"
                    << total_near / 1024.
                    << ",\"eight_gaussian_modes_covered_at_three_sigma\":" << covered
                    << ",\"banana_curvature_moment\":" << curvature << "}\n";
        check::Require(bool(diagnostics), "sample diagnostics written");
        diagnostic = s.kind == 0 ? curvature : s.kind == 1 ? total_near / 1024. : support;
      }

      const double  nll = output.at("loss").item<float>();
      std::ofstream meta(root / "metrics.json");
      meta.precision(17);
      meta << "{\"parameters\":" << s.parameters << ",\"parameter_tensors\":" << s.tensors
           << ",\"coupling_layers\":4,\"hidden\":16,\"base\":\"standardGaussian2D\",\"epochs\":"
           << o.epochs << ",\"optimizer_updates\":" << steps
           << ",\"selected_epoch_fixed_final\":" << o.epochs << ",\"test_nll\":" << nll
           << ",\"true_test_nll\":" << true_nll << ",\"fit_full_gaussian_nll\":" << full_nll
           << ",\"fit_diagonal_gaussian_nll\":" << diag_nll
           << ",\"identity_gaussian_nll\":" << identity_nll;
      if (s.global_logdet)
        meta << ",\"forward_logdet\":" << fld[0];
      else
        meta << ",\"forward_logdet_min\":" << min_ld << ",\"forward_logdet_max\":" << max_ld
             << ",\"mean_forward_logdet\":" << Mean(output.at("forward_logdet"));
      meta << ",\"forward_inverse_max_error\":" << roundtrip
           << ",\"inverse_forward_max_error\":" << prior_error
           << ",\"actual_vs_analytic_logdet_max_error\":" << det_error
           << ",\"ancestral_samples\":1024,\"train_seconds\":" << train_seconds << "}\n";
      std::ofstream html(root / "index.html");
      html << "<!doctype html><meta charset='utf-8'><title>" << s.title
           << "</title><style>body{font:16px "
              "system-ui;margin:30px;max-width:1100px}img{max-width:100%}</style>"
           << s.html
           << "<img src='density.png'><p>Panels: top left analytic true PDF, top right learned PDF "
              "(fixed density scale 0.."
           << s.density_scale
           << "); bottom left held-out orange and generated blue, bottom right transformed "
              "held-out purple and prior gray. Axes [-4, 4], "
              "outside points hidden only for the plot. See metrics.json, history.tsv and "
              "source_protocol.json. Short training can "
              "remain Gaussian-like; no fit-quality guarantee.</p>";
      check::Require(
          bool(meta) && bool(html) && bool(history) && bool(batches),
          "complete " + s.tag + " records");
      std::cout << (s.kind >= 0 ? "dataset=" + std::to_string(s.kind) + "," : std::string())
                << "testNLL=" << nll << ",true=" << true_nll << ",fitGaussian=" << full_nll
                << ",inverse/Jacobianerrors=" << roundtrip << '/' << det_error << ";"
                << (s.numerical ? "numerical" : "exact")
                << "inverse/logdet/normalization/restore/1024samples/immutable PASS\n";

      // ---- TF parity ----
      const auto close = [](double d, double tol) { return d >= 0 && d <= tol; };
      for (const auto& f : s.extra_files)
        Compare(tf, root, f, o.verbose);
      if (!init.empty() && init == tf / "initial_arrays.npz") {
        std::cout << "replay vs TF's recorded run (" << tf.string() << "):\n";
        double first_diff = 0;
        for (auto f :
             {"initial_arrays.npz",
              "initial_outputs.npz",
              "first_batch.npz",
              "first_forward.npz",
              "initial_gradients.npz"})
          first_diff = std::max(first_diff, Compare(tf, root, f, o.verbose));
        check::Require(
            close(first_diff, 2e-5), "replayed initial outputs/first forward/gradients match TF");
        check::Require(
            close(Compare(tf, root, "first_after.npz", o.verbose), 1e-4),
            "first Adam update matches TF");
        if (o.epochs == 3) {
          const auto ours_h = replay::History(root / "history.tsv"),
                     tf_h   = replay::History(tf / "history.tsv");
          double hd         = 0;
          for (std::size_t i = 0; i < std::min(ours_h.size(), tf_h.size()); ++i)
            hd = std::max(hd, std::abs(ours_h[i] - tf_h[i]));
          std::printf("  vs TF %-30s max |ours - TF| validation NLL = %.3g\n", "history.tsv", hd);
          for (auto f :
               {"epoch3_arrays.npz",
                "model_arrays.npz",
                "outputs.npz",
                "final_gradients.npz",
                "samples.npz",
                "sample_density.npz",
                "grid.npz",
                "baseline.npz"})
            Compare(tf, root, f, o.verbose);
          check::Require(hd < 2e-3, "replayed validation history follows TF's");
        }
      }
      if (fs::exists(tf / "model_arrays.npz") && fs::exists(tf / "outputs.npz")) {
        auto fresh = s.make(false);
        replay::LoadInto(fresh->store, tf / "model_arrays.npz");
        std::cout << "from TF's final weights:\n";
        const auto   out = fresh->Evaluate(data.pools[2]);
        const double d   = Compare(tf, "outputs.npz", out, o.verbose);
        const double s   = Compare(tf, "samples.npz", fresh->Sample(data.prior), o.verbose);
        check::Require(
            close(d, 1e-4) && close(s, 1e-4),
            "TF's final weights reproduce TF's test outputs and samples");
        std::printf(
            "  test NLL %.8f (TF %.8f)\n",
            out.at("loss").item<float>(),
            replay::Get(nn::load_npz((tf / "outputs.npz").string()), "loss").item<float>());
      }
      std::cout << "training " << train_seconds << " s, wall "
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << " s\nArtifacts:" << (root / "index.html").string() << std::endl;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
