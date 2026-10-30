/// @file recurrent_variants.cpp
/// @brief linear_scan(), minGRU, minLSTM and indrnn_step().

#include "datamunge/nn/recurrent_variants.hpp"

#include "datamunge/nn/ops.hpp"

#include <stdexcept>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief A dense map of a [B, T, in] sequence or a [B, in] step, with parameters "<name>/W" and "<name>/b".
/// @param store  Parameter store.
/// @param name  Parameter prefix.
/// @param x  Input [B, T, in] or [B, in].
/// @param units  Output size.
/// @param bias  Initial bias value.
/// @return x W + b with the leading axes of x.
Tensor affine(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units, double bias) {
    const std::int64_t in = x.size(x.dim() - 1);
    const Tensor W = store.get(name + "/W", {in, units}, init::glorot_uniform());
    const Tensor b = store.get(name + "/b", {units}, init::constant(bias));
    if (x.dim() == 2) return matmul(x, W) + b;
    const std::int64_t B = x.size(0), T = x.size(1);
    return reshape(matmul(reshape(x, {B * T, in}), W) + b, {B, T, units});
}

/// @brief Throws unless x is [B, T, in].
/// @param x  The tensor to check.
/// @param who  Function name for the message.
void check_sequence(const Tensor& x, const char* who) {
    if (x.dim() != 3) throw std::invalid_argument(std::string(who) + ": x must be [B, T, in]");
}

}  // namespace

Tensor linear_scan(const Tensor& a, const Tensor& b, const Tensor& h0) {
    if (a.dim() != 3 || a.shape() != b.shape()) throw std::invalid_argument("linear_scan: a and b must both be [B, T, H]");
    const std::int64_t B = a.size(0), T = a.size(1), H = a.size(2);
    Tensor A = a, Bv = b;
    if (h0.defined()) {
        if (h0.dim() != 2 || h0.size(0) != B || h0.size(1) != H) throw std::invalid_argument("linear_scan: h0 must be [B, H]");
        // Fold h0 into the first input: b_0 <- a_0 h0 + b_0.
        const Tensor first = slice(a, {0, 0, 0}, {B, 1, H}) * reshape(h0, {B, 1, H}) + slice(b, {0, 0, 0}, {B, 1, H});
        Bv = T > 1 ? concat({first, slice(b, {0, 1, 0}, {B, T - 1, H})}, 1) : first;
    }
    for (std::int64_t d = 1; d < T; d *= 2) {
        // Combine each prefix with the one d steps earlier; the first d entries are already complete.
        const Tensor A_prev = concat({ones({B, d, H}, A.dtype()), slice(A, {0, 0, 0}, {B, T - d, H})}, 1);
        const Tensor B_prev = concat({zeros({B, d, H}, Bv.dtype()), slice(Bv, {0, 0, 0}, {B, T - d, H})}, 1);
        Bv = A * B_prev + Bv;
        A  = A * A_prev;
    }
    return Bv;
}

Tensor mingru(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units, const Tensor& h0,
              double update_bias) {
    check_sequence(x, "mingru");
    const Tensor z = sigmoid(affine(store, name + "/z", x, units, update_bias));
    const Tensor h = affine(store, name + "/h", x, units, 0.0);
    return linear_scan(1.0 - z, z * h, h0);
}

Tensor mingru_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h, double update_bias) {
    const std::int64_t units = h.size(h.dim() - 1);
    const Tensor z = sigmoid(affine(store, name + "/z", x, units, update_bias));
    return (1.0 - z) * h + z * affine(store, name + "/h", x, units, 0.0);
}

Tensor minlstm(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units, const Tensor& h0,
               double forget_bias) {
    check_sequence(x, "minlstm");
    const Tensor f = sigmoid(affine(store, name + "/f", x, units, forget_bias));
    const Tensor i = sigmoid(affine(store, name + "/i", x, units, 0.0));
    const Tensor s = f + i;
    return linear_scan(f / s, i / s * affine(store, name + "/h", x, units, 0.0), h0);
}

Tensor minlstm_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h, double forget_bias) {
    const std::int64_t units = h.size(h.dim() - 1);
    const Tensor f = sigmoid(affine(store, name + "/f", x, units, forget_bias));
    const Tensor i = sigmoid(affine(store, name + "/i", x, units, 0.0));
    const Tensor s = f + i;
    return f / s * h + i / s * affine(store, name + "/h", x, units, 0.0);
}

Tensor indrnn_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h, Activation activation) {
    const std::int64_t units = h.size(h.dim() - 1);
    const Tensor u = store.get(name + "/u", {units}, init::uniform(0.0, 1.0));
    return activate(affine(store, name, x, units, 0.0) + u * h, activation);
}

}  // namespace datamunge::nn
