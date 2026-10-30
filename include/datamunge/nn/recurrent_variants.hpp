#pragma once

/// @file recurrent_variants.hpp
/// @brief A parallel scan for diagonal linear recurrences, the minimal gated cells minGRU and minLSTM (in parallel and
///        step form), and the independently recurrent IndRNN cell.
///
/// layers.hpp provides the classical cells rnn_step(), lstm_step() and gru_step(), whose gates depend on the previous
/// hidden state and so must be evaluated one step at a time. The cells here remove that dependence from the gates.
/// minGRU and minLSTM (Feng et al. 2024) compute their gates and candidate states from the input alone, so the
/// hidden state obeys the diagonal linear recurrence \f$h_t=a_t\odot h_{t-1}+b_t\f$. linear_scan() solves that recurrence for all t
/// at once in \f$\lceil\log_2T\rceil\f$ rounds of elementwise operations (a Hillis–Steele scan; Blelloch 1990), which is
/// differentiable. IndRNN (Li et al. 2018) keeps a nonlinearity but replaces the recurrent matrix by a vector, so each
/// unit only sees its own past and its gradient through time is a product of scalars.

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/params.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cstdint>
#include <string>

namespace datamunge::nn {

/// @brief All states of the diagonal linear recurrence \f$h_t=a_t\odot h_{t-1}+b_t\f$, t = 0, ..., T - 1, by a parallel scan.
///
/// The pairs \f$(a_t,b_t)\f$ compose associatively, \f$(a,b)\circ(a',b')=(a\,a',\ a\,b'+b)\f$, so \f$\lceil\log_2T\rceil\f$ rounds of shifted elementwise products
/// give every prefix (Hillis and Steele 1986). This takes \f$O(T\log T)\f$ work and \f$O(\log T)\f$ sequential depth, against \f$O(T)\f$ of both for
/// a loop. Differentiable with respect to a, b and h0.
///
/// @param a  Decay factors [B, T, H].
/// @param b  Inputs [B, T, H].
/// @param h0  Initial state [B, H]; an undefined tensor means zeros.
/// @return The states [B, T, H]; entry t is \f$h_t\f$.
/// @throws std::invalid_argument  If a and b differ in shape, are not rank 3, or h0 does not match.
[[nodiscard]] Tensor linear_scan(const Tensor& a, const Tensor& b, const Tensor& h0 = Tensor());

/// @brief minGRU over a whole sequence (Feng et al. 2024): \f$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t\f$ with \f$z_t=\sigma(x_tW_z+b_z)\f$, \f$\tilde h_t=x_tW_h+b_h\f$.
///
/// The gates do not depend on \f$h_{t-1}\f$, so the sequence is solved with linear_scan(). Parameters are
/// "<name>/z/W", "<name>/z/b" (initialised to `update_bias`) and "<name>/h/W", "<name>/h/b". A negative update_bias
/// makes \f$z\f$ small at the start, so the state is retained for longer.
///
/// @param store  Parameter store.
/// @param name  Parameter prefix.
/// @param x  Inputs [B, T, in].
/// @param units  Hidden size.
/// @param h0  Initial state [B, units]; an undefined tensor means zeros.
/// @param update_bias  Initial bias of the update gate.
/// @return All hidden states [B, T, units].
/// @throws std::invalid_argument  If x is not rank 3.
[[nodiscard]] Tensor mingru(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
                            const Tensor& h0 = Tensor(), double update_bias = 0.0);

/// @brief One minGRU step with the parameters of mingru(): returns \f$h'=(1-z)\odot h+z\odot\tilde h\f$ for x [B, in] and h [B, units].
/// @param store  Parameter store.
/// @param name  Parameter prefix (shared with mingru()).
/// @param x  Input at one time step [B, in].
/// @param h  Previous state [B, units].
/// @param update_bias  Initial bias of the update gate (used only when the parameters are created).
/// @return The next state [B, units].
[[nodiscard]] Tensor mingru_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h,
                                 double update_bias = 0.0);

/// @brief minLSTM over a whole sequence (Feng et al. 2024): \f$h_t=f'_t\odot h_{t-1}+i'_t\odot\tilde h_t\f$ with normalised gates \f$f'=f/(f+i)\f$, \f$i'=i/(f+i)\f$.
///
/// \f$f_t=\sigma(x_tW_f+b_f)\f$, \f$i_t=\sigma(x_tW_i+b_i)\f$ and \f$\tilde h_t=x_tW_h+b_h\f$; normalising makes \f$f'+i'=1\f$, so the state stays a convex
/// combination of its past and the candidates. Parameters "<name>/f/..." (bias initialised to `forget_bias`),
/// "<name>/i/..." and "<name>/h/...". Solved with linear_scan().
///
/// @param store  Parameter store.
/// @param name  Parameter prefix.
/// @param x  Inputs [B, T, in].
/// @param units  Hidden size.
/// @param h0  Initial state [B, units]; an undefined tensor means zeros.
/// @param forget_bias  Initial bias of the forget gate.
/// @return All hidden states [B, T, units].
/// @throws std::invalid_argument  If x is not rank 3.
[[nodiscard]] Tensor minlstm(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
                             const Tensor& h0 = Tensor(), double forget_bias = 0.0);

/// @brief One minLSTM step with the parameters of minlstm().
/// @param store  Parameter store.
/// @param name  Parameter prefix (shared with minlstm()).
/// @param x  Input at one time step [B, in].
/// @param h  Previous state [B, units].
/// @param forget_bias  Initial bias of the forget gate (used only when the parameters are created).
/// @return The next state [B, units].
[[nodiscard]] Tensor minlstm_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h,
                                  double forget_bias = 0.0);

/// @brief One IndRNN step (Li et al. 2018): \f$h'=\phi(xW+u\odot h+b)\f$ with a recurrent weight vector u instead of a matrix.
///
/// Parameters "<name>/W" [in, units] (Glorot uniform), "<name>/u" [units] (uniform on [0, 1]) and "<name>/b" [units].
/// With ReLU, \f$\partial h_T/\partial h_0\f$ is \f$\prod_t u\,\phi'(\cdot)\f$ elementwise, so keeping \f$|u|\le1\f$ controls explosion while \f$u\f$ near 1 keeps
/// memory.
///
/// @param store  Parameter store.
/// @param name  Parameter prefix.
/// @param x  Input [B, in].
/// @param h  Previous state [B, units].
/// @param activation  Nonlinearity \f$\phi\f$.
/// @return The next state [B, units].
[[nodiscard]] Tensor indrnn_step(ParameterStore& store, const std::string& name, const Tensor& x, const Tensor& h,
                                 Activation activation = Activation::Relu);

}  // namespace datamunge::nn
