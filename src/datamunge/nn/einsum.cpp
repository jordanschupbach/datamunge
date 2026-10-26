// einsum lowered to diagonal / sum / transpose / reshape / mul / matmul.

#include "datamunge/nn/einsum.hpp"

#include <algorithm>
#include <cctype>
#include <array>
#include <stdexcept>

#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

namespace {

using I64    = std::int64_t;
using Labels = std::vector<int>;

// Label sets are tiny and immutable during contraction. Keep them sorted and
// contiguous instead of allocating a tree node for each axis.
bool contains(const Labels& labels, int label) {
  return std::binary_search(labels.begin(), labels.end(), label);
}
void unique_labels(Labels& labels) {
  std::sort(labels.begin(), labels.end());
  labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
}

constexpr int kEllipsis = 1000;  // label of the first broadcast axis; letters are their char codes

[[noreturn]] void fail(const std::string& equation, const std::string& why) {
  throw std::invalid_argument("einsum \"" + equation + "\": " + why);
}

/// A term's letters, with -1 marking "...".
std::vector<int> parse_term(const std::string& term, const std::string& equation) {
  std::vector<int> out;
  bool             seen_ellipsis = false;
  for (std::size_t i = 0; i < term.size(); ++i) {
    const char c = term[i];
    if (c == '.') {
      if (term.compare(i, 3, "...") != 0 || seen_ellipsis) fail(equation, "malformed ellipsis");
      seen_ellipsis = true;
      out.push_back(-1);
      i += 2;
    } else if (std::isalpha(static_cast<unsigned char>(c))) {
      out.push_back(static_cast<unsigned char>(c));
    } else {
      fail(equation, std::string("unexpected character '") + c + "'");
    }
  }
  return out;
}

/// Expands "..." in a term to `count` broadcast labels, right-aligned to `total`.
Labels expand(const std::vector<int>& term, I64 count, I64 total) {
  Labels out;
  for (int t : term) {
    if (t != -1) {
      out.push_back(t);
      continue;
    }
    for (I64 i = 0; i < count; ++i) out.push_back(kEllipsis + static_cast<int>(total - count + i));
  }
  return out;
}

I64 index_of(const Labels& labels, int label) {
  return static_cast<I64>(std::find(labels.begin(), labels.end(), label) - labels.begin());
}

bool is_identity(const Axes& perm) {
  for (std::size_t i = 0; i < perm.size(); ++i)
    if (perm[i] != static_cast<I64>(i)) return false;
  return true;
}

Tensor permute(const Tensor& x, const Axes& perm) { return is_identity(perm) ? x : transpose(x, perm); }

/// Takes diagonals for repeated labels and sums labels not in `keep`.
Tensor reduce_operand(Tensor x, Labels& labels, const Labels& keep, const std::string& equation) {
  // Diagonals: move each repeated pair to the end and take its diagonal.
  while (true) {
    I64 first = -1, second = -1;
    for (std::size_t i = 0; i < labels.size() && first < 0; ++i)
      for (std::size_t j = i + 1; j < labels.size(); ++j)
        if (labels[i] == labels[j]) {
          first = static_cast<I64>(i), second = static_cast<I64>(j);
          break;
        }
    if (first < 0) break;
    if (x.size(first) != x.size(second)) fail(equation, "repeated index with different sizes");
    Axes   perm;
    Labels rest;
    for (I64 i = 0; i < static_cast<I64>(labels.size()); ++i)
      if (i != first && i != second) perm.push_back(i), rest.push_back(labels[static_cast<std::size_t>(i)]);
    perm.push_back(first);
    perm.push_back(second);
    const int label = labels[static_cast<std::size_t>(first)];
    x               = matrix_diag_part(permute(x, perm));
    rest.push_back(label);
    labels = rest;
  }
  Axes   axes;
  Labels rest;
  for (std::size_t i = 0; i < labels.size(); ++i)
    if (contains(keep, labels[i]))
      rest.push_back(labels[i]);
    else
      axes.push_back(static_cast<I64>(i));
  if (!axes.empty()) x = sum(x, axes);
  labels = rest;
  return x;
}

/// Contracts two reduced operands, keeping the labels in `keep`.
Tensor contract(Tensor a, Labels la, Tensor b, Labels lb, const Labels& keep, Labels& out_labels,
                const std::string& equation) {
  Labels batch, aonly, bonly, con;
  for (int l : la) {
    const bool in_b = std::count(lb.begin(), lb.end(), l) > 0;
    if (in_b)
      (contains(keep, l) ? batch : con).push_back(l);
    else
      aonly.push_back(l);
  }
  for (int l : lb)
    if (!std::count(la.begin(), la.end(), l)) bonly.push_back(l);
  // (Labels only in one operand are always kept: reduce_operand summed the others.)
  // Size checks; a contracted axis of size 1 on one side sums the other side first.
  for (int l : con) {
    const I64 ia = index_of(la, l), ib = index_of(lb, l);
    const I64 sa = a.size(ia), sb = b.size(ib);
    if (sa == sb) continue;
    if (sa == 1)
      b = sum(b, {ib}, true);
    else if (sb == 1)
      a = sum(a, {ia}, true);
    else
      fail(equation, "contracted index sizes differ");
  }
  for (int l : batch) {
    const I64 sa = a.size(index_of(la, l)), sb = b.size(index_of(lb, l));
    if (sa != sb && sa != 1 && sb != 1) fail(equation, "index sizes differ and do not broadcast");
  }
  const auto positions = [](const Labels& from, std::initializer_list<const Labels*> groups) {
    Axes p;
    for (const Labels* g : groups)
      for (int l : *g) p.push_back(index_of(from, l));
    return p;
  };
  const auto sizes = [](const Tensor& t, const Labels& from, const Labels& group) {
    Shape s;
    for (int l : group) s.push_back(t.size(index_of(from, l)));
    return s;
  };
  const auto prod = [](const Shape& s) { return shape_numel(s); };
  const Shape bsa = sizes(a, la, batch), bsb = sizes(b, lb, batch);
  const Shape as = sizes(a, la, aonly), bs = sizes(b, lb, bonly);
  const I64   M = prod(as), N = prod(bs), K = prod(sizes(a, la, con));
  const auto  with = [](Shape s, std::initializer_list<I64> tail) {
    s.insert(s.end(), tail);
    return s;
  };
  out_labels = batch;
  out_labels.insert(out_labels.end(), aonly.begin(), aonly.end());
  out_labels.insert(out_labels.end(), bonly.begin(), bonly.end());

  Tensor result;
  if (con.empty()) {
    // Outer / elementwise product: broadcast multiply.
    const Tensor ap = reshape(permute(a, positions(la, {&batch, &aonly})), with(bsa, {M, 1}));
    const Tensor bp = reshape(permute(b, positions(lb, {&batch, &bonly})), with(bsb, {1, N}));
    result          = mul(ap, bp);
  } else if (M == 1 && N == 1) {
    // Batched dot products: multiply and sum.
    const Tensor ap = reshape(permute(a, positions(la, {&batch, &aonly, &con})), with(bsa, {K}));
    const Tensor bp = reshape(permute(b, positions(lb, {&batch, &bonly, &con})), with(bsb, {K}));
    result          = sum(mul(ap, bp), {-1});
  } else {
    const Axes pa1 = positions(la, {&batch, &aonly, &con}), pa2 = positions(la, {&batch, &con, &aonly});
    const Axes pb1 = positions(lb, {&batch, &con, &bonly}), pb2 = positions(lb, {&batch, &bonly, &con});
    const bool ta  = !is_identity(pa1) && is_identity(pa2);
    const bool tb  = !is_identity(pb1) && is_identity(pb2);
    const Tensor ap = ta ? reshape(a, with(bsa, {K, M})) : reshape(permute(a, pa1), with(bsa, {M, K}));
    const Tensor bp = tb ? reshape(b, with(bsb, {N, K})) : reshape(permute(b, pb1), with(bsb, {K, N}));
    result          = matmul(ap, bp, ta, tb);
  }
  Shape final_shape(result.shape().begin(), result.shape().begin() + static_cast<I64>(batch.size()));
  final_shape.insert(final_shape.end(), as.begin(), as.end());
  final_shape.insert(final_shape.end(), bs.begin(), bs.end());
  return reshape(result, final_shape);
}

}  // namespace

Tensor einsum(const std::string& equation_in, const std::vector<Tensor>& operands) {
  std::string equation;
  for (char c : equation_in)
    if (c != ' ') equation += c;
  if (operands.empty()) fail(equation, "no operands");
  const auto  arrow    = equation.find("->");
  const bool  explicit_out = arrow != std::string::npos;
  std::string lhs      = explicit_out ? equation.substr(0, arrow) : equation;
  std::vector<std::vector<int>> terms;
  {
    std::size_t start = 0;
    while (true) {
      const auto comma = lhs.find(',', start);
      terms.push_back(parse_term(lhs.substr(start, comma == std::string::npos ? std::string::npos : comma - start), equation));
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
  }
  if (terms.size() != operands.size())
    fail(equation, std::to_string(terms.size()) + " terms for " + std::to_string(operands.size()) + " operands");
  // Ellipsis widths.
  std::vector<I64> widths;
  I64              total = 0;
  for (std::size_t i = 0; i < terms.size(); ++i) {
    const I64  letters  = static_cast<I64>(std::count_if(terms[i].begin(), terms[i].end(), [](int t) { return t != -1; }));
    const bool has_ell  = std::count(terms[i].begin(), terms[i].end(), -1) > 0;
    const I64  width    = operands[i].dim() - letters;
    if (width < 0 || (!has_ell && width != 0))
      fail(equation, "term " + std::to_string(i) + " does not match operand of shape " + shape_str(operands[i].shape()));
    widths.push_back(width);
    total = std::max(total, width);
  }
  std::vector<Labels> labels;
  for (std::size_t i = 0; i < terms.size(); ++i) labels.push_back(expand(terms[i], widths[i], total));
  Labels out;
  if (explicit_out) {
    const auto out_term = parse_term(equation.substr(arrow + 2), equation);
    out                 = expand(out_term, total, total);
    Labels seen = out;
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) fail(equation, "repeated output index");
    for (int l : out) {
      bool found = false;
      for (const auto& ls : labels) found = found || std::count(ls.begin(), ls.end(), l) > 0;
      if (!found) fail(equation, "output index not in any operand");
    }
  } else {
    std::array<std::size_t, 256> counts{};
    for (const auto& ls : labels)
      for (int l : ls)
        if (l < kEllipsis) ++counts[static_cast<std::size_t>(l)];
    for (I64 i = 0; i < total; ++i) out.push_back(kEllipsis + static_cast<int>(i));
    for (std::size_t l = 0; l < counts.size(); ++l)
      if (counts[l] == 1) out.push_back(static_cast<int>(l));
  }
  Labels out_set = out;
  unique_labels(out_set);
  // Labels needed after operand i: the output plus every other operand.
  const auto needed_except = [&](std::size_t skip, std::size_t from) {
    Labels keep = out_set;
    for (std::size_t j = from; j < labels.size(); ++j)
      if (j != skip) keep.insert(keep.end(), labels[j].begin(), labels[j].end());
    unique_labels(keep);
    return keep;
  };
  Labels lx = labels[0];
  Tensor x  = reduce_operand(operands[0], lx, needed_except(0, 0), equation);
  for (std::size_t i = 1; i < operands.size(); ++i) {
    Labels       ly = labels[i];
    Tensor       y  = reduce_operand(operands[i], ly, needed_except(i, 0), equation);
    const auto   keep = needed_except(i, i + 1);  // output + operands after i
    Labels       lr;
    x  = contract(x, lx, y, ly, keep, lr, equation);
    lx = lr;
  }
  // Remaining labels are exactly the output's, in some order.
  {
    Axes   drop;
    Labels rest;
    for (std::size_t i = 0; i < lx.size(); ++i)
      if (contains(out_set, lx[i]))
        rest.push_back(lx[i]);
      else
        drop.push_back(static_cast<I64>(i));
    if (!drop.empty()) x = sum(x, drop), lx = rest;
  }
  Axes perm;
  for (int l : out) perm.push_back(index_of(lx, l));
  return permute(x, perm);
}

Tensor einsum(const std::string& equation, const Tensor& a) { return einsum(equation, std::vector<Tensor>{a}); }

Tensor einsum(const std::string& equation, const Tensor& a, const Tensor& b) {
  return einsum(equation, std::vector<Tensor>{a, b});
}

}  // namespace datamunge::nn
