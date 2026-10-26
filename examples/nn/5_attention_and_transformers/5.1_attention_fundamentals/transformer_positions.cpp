// CMD: just run-nn transformer_positions
// Five position mechanisms: absolute tables, rotary queries/keys, ALiBi and T5
// relative bias.
//
// Port of tflow's transformer_positions. On x [2, 5, 8]: a sinusoidal table
// added to x; a learned table [9, 8] gathered at positions {0, 2, 2, 5, 7} and
// added; rotary embedding at positions 3..7. On q, k [3, 5, 8], v [3, 5, 4]:
// attention with a T5 relative-position bias from a learned table [16 buckets,
// 3 heads] (bidirectional, max distance 32, queries offset by 3), plus its
// unidirectional form and the causal ALiBi bias for 3 heads. One objective
// sums all four probed outputs; its gradient with respect to x and both tables
// is checked against float32 central differences at EVERY coordinate (200,
// max error < .006). Also saved: the full T5 bucket grid for distances
// -256..256, ALiBi slopes for 1, 2, 3, 4, 6, 8 heads (authors' ordering for
// non-powers of two), rotary applied to a cached suffix and rotary dot
// products at offsets 0 and 7. Seven invalid inputs are rejected.
// Checks added here: the cached-suffix rotary equals the matching rows of the
// full rotary, the rotary dot products do not depend on the shared offset,
// and the library's offset-0 relative buckets and power-of-two ALiBi slopes
// agree with tflow's.
//
// Tables: init=auto (default) replays the learned and relative tables TF saved
// (tflow's output/models/transformer_positions/positions.npz) when present,
// init=tf requires them, init=nn draws them from nn's generator with TF's
// distributions (N(0, .02) and N(0, .2)). Inputs come from TF's std::mt19937
// stream, so with TF's tables every array is compared with TF's (to 1e-5).
//
// TF -> datamunge: SinusoidalTable/ApplyRotary/AlibiSlopes/AlibiBias/
// RelativePositionBucket/RelativePositionBias are in common/positional.hpp
// (tflow's API and guards over nn::sinusoidal_encoding / nn::apply_rotary;
// tflow's ALiBi slope ordering and a caller-owned relative table with a query
// offset, which the library does not provide), LearnedEncoding is
// nn::learned_encoding, attention is attn::ScaledDotProductAttention.
//
// Options: out=DIR (default output/nn/transformer_positions), tf=DIR,
// init=auto|tf|nn, verbose=1.

#include <cstdio>
#include <iostream>
#include <string>

#include "common/attention.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/positional.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

int main(int argc, char** argv) {
  try {
    const auto opt = attn::ParseOptions(argc, argv, "transformer_positions", paths::OutputDir("transformer_positions"));
    attn::TfRandom     random(45801);
    nn::ParameterStore store(45803);
    Tensor             x = random({2, 5, 8});
    const Tensor       probe = random({2, 5, 8});
    const auto         tf_file = opt.tf / "positions.npz";
    std::string        source  = "nn";
    if (opt.init != "nn" && std::filesystem::exists(tf_file)) {
      const auto ref = nn::load_npz(tf_file.string());
      store.get("absolute/pos", {9, 8}, attn::FixedInit(attn::Named(ref, "learned_table")));
      store.get("relative", {16, 3}, attn::FixedInit(attn::Named(ref, "relative_table")));
      source = "TF";
    } else if (opt.init == "tf") {
      throw std::runtime_error("init=tf: " + tf_file.string() + " not found");
    }
    const Tensor sin      = positional::SinusoidalTable(5, 8);
    Tensor       table    = nn::learned_encoding(store, "absolute", 9, 8, .02);
    Tensor       reltable = store.get("relative", {16, 3}, nn::init::normal(.2));
    const Tensor indices  = attn::Ints({0, 2, 2, 5, 7});
    const Tensor qv = random({3, 5, 8}), kv = random({3, 5, 8}), vv = random({3, 5, 4}), ap = random({3, 5, 4});
    x.requires_grad_();

    struct Outputs {
      Tensor absolute, learned, rotary, relative, causal;
      nn::Attended attention;
      Tensor objective;
    };
    auto forward = [&] {
      Outputs o;
      o.absolute  = x + sin;
      o.learned   = x + nn::gather(table, indices, 0);
      o.rotary    = positional::ApplyRotary(x, 3);
      o.relative  = positional::RelativePositionBias(reltable, 5, 5, true, 16, 32, 3);
      o.causal    = positional::RelativePositionBias(reltable, 5, 5, false, 16, 32, 3);
      o.attention = attn::ScaledDotProductAttention(qv, kv, vv, 8, o.relative);
      o.objective = nn::sum(o.absolute * probe) + nn::sum(o.learned * probe) + nn::sum(o.rotary * probe) +
                    nn::sum(o.attention.output * ap);
      return o;
    };
    const Outputs o     = forward();
    const Tensor  alibi = positional::AlibiBias(3, 5);
    auto          grads = nn::grad(o.objective, {x, table, reltable});

    const Tensor cached = positional::ApplyRotary(nn::slice(x.detach(), {0, 2, 0}, {2, 3, 8}), 5);
    auto         dots   = [&](int offset) {
      return nn::matmul(positional::ApplyRotary(qv, offset), positional::ApplyRotary(kv, offset), false, true);
    };
    const Tensor dot0 = dots(0), dot7 = dots(7);

    nn::NamedTensors saved{{"probe", probe},
                           {"indices", indices},
                           {"q", qv},
                           {"k", kv},
                           {"v", vv},
                           {"attention_probe", ap},
                           {"x", x.detach().clone()},
                           {"learned_table", table.detach().clone()},
                           {"relative_table", reltable.detach().clone()},
                           {"sinusoidal", sin},
                           {"absolute", o.absolute.detach()},
                           {"learned", o.learned.detach()},
                           {"rotary", o.rotary.detach()},
                           {"alibi", alibi},
                           {"relative", o.relative.detach()},
                           {"relative_causal", o.causal.detach()},
                           {"weights", o.attention.weights.detach()},
                           {"output", o.attention.output.detach()},
                           {"objective", o.objective.detach()},
                           {"x_gradient", grads[0]},
                           {"learned_gradient", grads[1]},
                           {"relative_gradient", grads[2]},
                           {"rotary_cached", cached},
                           {"rotary_dot0", dot0},
                           {"rotary_dot7", dot7}};
    std::vector<int> buckets(2 * 513);
    for (int r = -256; r <= 256; ++r)
      for (int bidir = 0; bidir < 2; ++bidir)
        buckets[bidir * 513 + r + 256] = positional::RelativePositionBucket(r, bidir, 16, 32);
    saved.push_back({"bucket_grid", nn::from_vector(buckets, {2, 513})});
    for (int heads : {1, 2, 3, 4, 6, 8}) {
      const auto slopes = positional::AlibiSlopes(heads);
      saved.push_back({"slopes_" + std::to_string(heads), nn::from_vector(slopes, {heads})});
      if ((heads & (heads - 1)) == 0) {  // powers of two: the library's slopes agree
        const auto lib = nn::alibi_slopes(heads);
        for (int h = 0; h < heads; ++h) check::Near(lib[h], slopes[h], 1e-7, "library ALiBi slopes (power of two)");
      }
    }

    // Properties: a cached suffix rotated from its absolute offset equals the
    // full rotary rows; q.k after rotating both by the same offset is unchanged.
    check::Near(attn::MaxAbsDiff(cached, nn::slice(o.rotary.detach(), {0, 2, 0}, {2, 3, 8})), 0, 1e-6,
                "rotary of a cached suffix equals the full rotary rows");
    check::Near(attn::MaxAbsDiff(dot0, dot7), 0, 1e-5, "rotary dot products depend only on relative position");
    // The library's bottom-right-aligned buckets agree with tflow's at query offset 0.
    check::Near(attn::MaxAbsDiff(nn::relative_position_buckets(5, 5, true, 16, 32),
                                 positional::RelativePositionBuckets(5, 5, true, 16, 32)),
                0, 0, "library relative buckets (bidirectional)");
    check::Near(attn::MaxAbsDiff(nn::relative_position_buckets(5, 5, false, 16, 32),
                                 positional::RelativePositionBuckets(5, 5, false, 16, 32)),
                0, 0, "library relative buckets (unidirectional)");

    double error = 0;
    int    count = 0;
    {
      nn::NoGradGuard no_grad;
      auto            f         = [&] { return forward().objective.item<float>(); };
      Tensor*         leaves[3] = {&x, &table, &reltable};
      for (int leaf = 0; leaf < 3; ++leaf)
        error = std::max(error, attn::CentralDifferenceError(*leaves[leaf], grads[leaf],
                                                              static_cast<int>(leaves[leaf]->numel()), f, &count));
    }
    check::Require(error < .006, "all position native central differences");
    saved.push_back({"native_fd", nn::from_vector(std::vector<double>{double(count), error}, {2})});
    nn::save_npz((opt.out / "positions.npz").string(), saved);

    int rejected = 0;
    auto reject  = [&](auto&& f) {
      try {
        f();
      } catch (const std::invalid_argument&) {
        ++rejected;
      }
    };
    for (int d : {0, 3}) reject([&] { positional::SinusoidalTable(5, d); });
    reject([&] { positional::ApplyRotary(nn::slice(x.detach(), {0, 0, 0}, {2, 5, 7})); });
    reject([&] { positional::ApplyRotary(x.detach(), -1); });
    reject([&] { positional::AlibiSlopes(0); });
    reject([&] { positional::RelativePositionBucket(1, true, 5); });
    reject([&] { positional::RelativePositionBucket(1, true, 16, 4); });
    check::Require(rejected == 7, "position guards");
    std::printf("Five position mechanisms (tables from %s): %d native FD coordinates, max error %.3g; seven invalid "
                "inputs rejected PASS\n",
                source.c_str(), count, error);
    if (source == "TF")
      attn::CheckAgainstTf(opt.tf, "positions.npz", saved,
                           {"x", "probe", "q", "k", "v", "attention_probe", "learned_table", "relative_table"}, 1e-5,
                           opt.verbose);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
