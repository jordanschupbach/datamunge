// CMD: just run-nn embedding
// Embedding lookup.
//
// Port of tflow's embedding. nn::embedding (datamunge/nn/layers.hpp) gathers
// rows of a trainable table "<name>/table" by integer id. The program shows
// that the lookup is a one-hot matrix product, that its gradient is sparse
// (only the rows that were looked up receive a gradient, and a repeated id
// accumulates), how padding ids are masked out of a mean over a sequence
// (nn::masked_mean), and finally trains embeddings on a synthetic corpus with
// a skip-gram objective so that words that share contexts end up close to
// each other.
//
// TF -> datamunge: no graph/session; AddSymbolicGradients is nn::grad, ids
// are int64 tensors, and SoftmaxCrossEntropyWithLogits on one-hot labels is
// nn::sparse_softmax_cross_entropy (the same loss). The corpus and the
// training batches use the same std::mt19937_64 stream as TF; the tables are
// initialised by datamunge's generator, so losses and cosines match TF
// statistically.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

Tensor Ids(const std::vector<std::vector<std::int64_t>>& rows) {
  std::vector<std::int64_t> flat;
  for (const auto& r : rows) flat.insert(flat.end(), r.begin(), r.end());
  return nn::from_vector(flat, {static_cast<std::int64_t>(rows.size()), static_cast<std::int64_t>(rows[0].size())});
}

// ---------------------------------------------------------------------------
// 1. Lookup as a one-hot product, and the sparse gradient
// ---------------------------------------------------------------------------

void LookupAndGradient() {
  std::printf("== Embedding table [vocab 10, dim 4], ids [2, 3]\n");
  nn::ParameterStore p(1);
  const Tensor       idv = Ids({{3, 7, 3}, {0, 9, 7}});
  Tensor             rv  = nn::empty({2, 3, 4});
  for (int i = 0; i < 24; ++i) rv.data_ptr<float>()[i] = 0.5f * (i % 5) - 1;
  const Tensor e     = nn::embedding(p, "words", idv, 10, 4, 1.0);
  const Tensor table = p.at("words/table");
  // The same lookup as a one-hot matmul: [2*3, 10] x [10, 4].
  const Tensor e2   = nn::reshape(nn::matmul(nn::one_hot(nn::reshape(idv, {-1}), 10), table), {2, 3, 4});
  const Tensor loss = nn::sum(e * rv);  // dL/de = r
  const Tensor g    = nn::grad(loss, {table})[0];
  const float  d    = nn::max(nn::abs(e - e2)).item<float>();
  std::printf("  output shape %s; max |gather - onehot matmul| = %.1e\n", nn::shape_str(e.shape()).c_str(), d);
  std::printf("  ids: [3 7 3] [0 9 7]. Gradient dL/dtable, row by row (L = sum(e * r)):\n");
  double err = 0;
  for (std::int64_t v = 0; v < 10; ++v) {
    std::printf("    row %lld:", static_cast<long long>(v));
    for (std::int64_t k = 0; k < 4; ++k) std::printf(" %6.2f", g.at({v, k}));
    // Expected: the sum of r over the positions holding id v.
    std::printf("   expected:");
    for (std::int64_t k = 0; k < 4; ++k) {
      float acc = 0;
      for (std::int64_t i = 0; i < 2; ++i)
        for (std::int64_t j = 0; j < 3; ++j)
          if (idv.at({i, j}) == v) acc += static_cast<float>(rv.at({i, j, k}));
      std::printf(" %6.2f", acc);
      err = std::max(err, std::abs(acc - g.at({v, k})));
    }
    std::printf("%s\n", v == 3 || v == 7 ? "   (used twice: the two r rows add up)" : "");
  }
  check::Require(e.shape() == nn::Shape{2, 3, 4}, "embedding output shape");
  check::Require(d == 0, "gather and one-hot matmul differ");
  check::Require(err < 1e-6, "embedding gradient is not the scattered sum of r");
}

// ---------------------------------------------------------------------------
// 2. Padding and masked pooling
// ---------------------------------------------------------------------------

void Masking() {
  std::printf("\n== Variable-length sequences padded with id 0, pooled by a masked mean\n");
  nn::ParameterStore p(2);
  const Tensor       idv    = Ids({{4, 2, 0, 0}, {1, 5, 9, 3}});
  const Tensor       e      = nn::embedding(p, "words", idv, 10, 3, 1.0);
  const Tensor       mask   = nn::not_equal(idv, 0.0);
  const Tensor       pooled = nn::masked_mean(e, mask);
  const Tensor       naive  = nn::mean(e, {1});
  double             err    = 0;
  for (std::int64_t i = 0; i < 2; ++i) {
    std::printf("  sequence %lld (%d real tokens): masked mean", static_cast<long long>(i), i == 0 ? 2 : 4);
    for (std::int64_t k = 0; k < 3; ++k) std::printf(" %7.3f", pooled.at({i, k}));
    std::printf("   hand");
    for (std::int64_t k = 0; k < 3; ++k) {
      float acc = 0;
      int   n   = 0;
      for (std::int64_t j = 0; j < 4; ++j)
        if (idv.at({i, j}) != 0) acc += static_cast<float>(e.at({i, j, k})), ++n;
      std::printf(" %7.3f", acc / n);
      err = std::max(err, std::abs(acc / n - pooled.at({i, k})));
    }
    std::printf("   unmasked mean");
    for (std::int64_t k = 0; k < 3; ++k) std::printf(" %7.3f", naive.at({i, k}));
    std::printf("\n");
  }
  std::printf("  (the unmasked mean of sequence 0 is diluted by two PAD vectors, which are not even zero)\n");
  check::Require(err < 1e-6, "masked mean disagrees with the hand computation");
  check::Require(nn::max(nn::abs(nn::select(pooled, 0, 1) - nn::select(naive, 0, 1))).item() < 1e-6,
                 "an unpadded sequence must pool the same with or without the mask");
}

// ---------------------------------------------------------------------------
// 3. Learning embeddings with a skip-gram objective
// ---------------------------------------------------------------------------

void SkipGram() {
  std::printf("\n== Skip-gram on a synthetic corpus: 12 words in 3 topics, sentences drawn within a topic\n");
  const int       vocab = 12, dim = 8, window = 2;
  std::mt19937_64 rng(3);
  // Corpus: 400 sentences of 8 words, each from one topic (words 0-3, 4-7, 8-11).
  std::vector<std::pair<int, int>> pairs;
  for (int s = 0; s < 400; ++s) {
    const int        topic = static_cast<int>(rng() % 3);
    std::vector<int> sentence(8);
    for (int& w : sentence) w = 4 * topic + static_cast<int>(rng() % 4);
    for (int i = 0; i < 8; ++i)
      for (int j = std::max(0, i - window); j <= std::min(7, i + window); ++j)
        if (i != j) pairs.push_back({sentence[i], sentence[j]});
  }
  std::printf("  %zu (target, context) pairs\n", pairs.size());
  nn::ParameterStore p(4);
  auto               loss_of = [&](const Tensor& target, const Tensor& context) {
    const Tensor e         = nn::embedding(p, "in", target, vocab, dim, 0.5);                        // [B, dim]
    const Tensor out_table = p.get("out/table", {vocab, dim}, nn::init::uniform(0.5));               // [vocab, dim]
    const Tensor logits    = nn::matmul(e, out_table, false, true);                                  // [B, vocab]
    return nn::sparse_softmax_cross_entropy(logits, context);
  };
  const int                 B = 256;
  std::vector<std::int64_t> tv(B), cv(B);
  {
    nn::NoGradGuard g;
    loss_of(nn::from_vector(tv), nn::from_vector(cv));  // create both tables before the optimizer
  }
  nn::Optimizer adam(p, 0.05, {nn::OptimizerKind::Adam});
  double        last = 0;
  for (int step = 1; step <= 300; ++step) {
    for (int i = 0; i < B; ++i) {
      const auto& pr = pairs[rng() % pairs.size()];
      tv[i] = pr.first, cv[i] = pr.second;
    }
    const Tensor loss = loss_of(nn::from_vector(tv), nn::from_vector(cv));
    adam.minimize(loss);
    last = loss.item();
    if (step == 1 || step % 100 == 0)
      std::printf("  step %3d  loss %.3f  (log 4 = %.3f is the floor: 4 equally likely contexts)\n", step, last,
                  std::log(4.0));
  }
  const Tensor T      = p.at("in/table");
  auto         cosine = [&](std::int64_t a, std::int64_t b) {
    double dot = 0, na = 0, nb = 0;
    for (std::int64_t k = 0; k < dim; ++k) {
      const double ta = T.at({a, k}), tb = T.at({b, k});
      dot += ta * tb, na += ta * ta, nb += tb * tb;
    }
    return dot / std::sqrt(na * nb);
  };
  std::printf("  cosine similarity between word embeddings (rows / columns: words 0-11; topics are 0-3, 4-7, 8-11):\n       ");
  for (int b = 0; b < vocab; ++b) std::printf("%6d", b);
  std::printf("\n");
  double within = 0, between = 0;
  int    nw = 0, nb = 0;
  for (int a = 0; a < vocab; ++a) {
    std::printf("    %2d ", a);
    for (int b = 0; b < vocab; ++b) {
      const double c = cosine(a, b);
      std::printf("%6.2f", c);
      if (a != b) (a / 4 == b / 4 ? within : between) += c, (a / 4 == b / 4 ? nw : nb)++;
    }
    std::printf("\n");
  }
  std::printf("  mean cosine within a topic %.2f, between topics %.2f\n", within / nw, between / nb);
  // TF: loss 2.555 -> 1.399 (floor log 4 = 1.386); cosine within 0.93, between -0.44.
  check::Require(last < std::log(4.0) + 0.06, "skip-gram loss should approach log 4");
  check::Require(within / nw > 0.8, "words of a topic should be close");
  check::Require(between / nb < -0.2, "words of different topics should be apart");
}

}  // namespace

int main() {
  try {
    LookupAndGradient();
    Masking();
    SkipGram();
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
