#pragma once
/// @file recurrent_pixels.hpp
/// @brief Explicit convolutional Row-LSTM step and compact raster-causal PixelRNN.
#include "datamunge/nn/autoregressive_pixels.hpp"

namespace datamunge::nn::models {
/// @brief Own seven live boundaries of one convolutional Row-LSTM transition.
///
/// Flat Tensor handles retain input/state/kernel graphs and storage beyond the
/// parent lifetime. Four gate maps have width4; gates has width16. No optimizer,
/// parameter store, model object, sampling stream or fitted initial state is
/// retained. Gate order is input, forget, output, candidate; forget adds one.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto out=n::models::PixelRowLstmStep(n::zeros({2,1,5,16}),n::zeros({2,1,5,4}),
///                                     n::ones({2,1,5,4}),n::zeros({1,3,4,16}));
/// // out.cell=sigmoid(1), out.hidden=.5*tanh(sigmoid(1)), at every coordinate.
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . Compact width4 Row-LSTM teaching variant.
/// @see PixelRowLstmStep(), PixelRnnOutput, PixelRnn()
struct PixelRowLstmOutput {
  Tensor gates;       ///< Owning live input_gates+conv(previous_hidden,kernel)[N,1,W,16].
  Tensor input_gate;  ///< Owning live sigmoid(gates first4)[N,1,W,4].
  Tensor forget_gate; ///< Owning live sigmoid(gates channels4..7+1)[N,1,W,4].
  Tensor output_gate; ///< Owning live sigmoid(gates channels8..11)[N,1,W,4].
  Tensor candidate;   ///< Owning live tanh(gates channels12..15)[N,1,W,4].
  Tensor cell;        ///< Owning live forget_gate*previous_cell+input_gate*candidate[N,1,W,4].
  Tensor hidden;      ///< Owning live output_gate*tanh(cell)[N,1,W,4].
};

/// @brief Advance all columns of one Row-LSTM row from explicit previous-row states.
///
/// Matching finite Float32/64 CPU input_gates[N,1,W,16], previous hidden/cell
/// [N,1,W,4] and recurrent_kernel[1,3,4,16], positive N/W, arbitrary strides.
/// Add a SAME stride1 convolution of previous_hidden to input_gates, without
/// a recurrent bias. Split into four contiguous width4 groups: input sigmoid,
/// forget sigmoid with fixed+1 bias, output sigmoid, candidate tanh. Then
/// cell=f*previous_cell+i*candidate and hidden=o*tanh(cell). All current columns
/// advance in parallel; the recurrent kernel couples neighboring columns only
/// in the previous row. Causality depends on the supplied input features/states;
/// this primitive alone imposes no image mask or zero initialization.
///
/// Owning flat outputs retain applicable first/higher input/state/kernel paths.
/// No mutation, parameter creation, fitting, optimizer, sampling, RNG or hidden
/// no-grad. Independent calls share no mutable state. Work/storage O(NW) with
/// fixed width4 and kernel1x3. Invalid geometry/dtypes/values reject before math;
/// nonfinite summed gates are also rejected. Caller owns the state sequence.
/// @param input_gates Finite matching CPU[N,1,W,16] input projection including its bias.
/// @param previous_hidden Finite matching CPU[N,1,W,4] previous-row hidden state.
/// @param previous_cell Finite matching CPU[N,1,W,4] previous-row cell state.
/// @param recurrent_kernel Finite matching live CPU[1,3,4,16] unbiased row kernel.
/// @return Owning seven-field PixelRowLstmOutput with actual gate and state boundaries.
/// @throws std::invalid_argument For incompatible geometry/dtype/values or nonfinite summed gates.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto t=n::DType::Float64;
/// auto out=n::models::PixelRowLstmStep(n::zeros({1,1,3,16},t),n::zeros({1,1,3,4},t),
///                                     n::full({1,1,3,4},2,t),n::zeros({1,3,4,16},t));
/// // Forget gate=.73105858; cell=1.46211716; hidden=.5*tanh(cell).
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . Source teaching cell uses i/f/o/
/// candidate ordering, width4, an unbiased1x3 recurrent kernel and forget bias+1.
/// @see PixelRowLstmOutput, PixelRnn(), lstm_step()
inline PixelRowLstmOutput PixelRowLstmStep(
    const Tensor& input_gates,const Tensor& previous_hidden,const Tensor& previous_cell,
    const Tensor& recurrent_kernel) {
  const auto dtype=input_gates.dtype();
  autoregressive_pixel_detail::Image(input_gates,16,dtype);
  autoregressive_pixel_detail::Image(previous_hidden,4,dtype);
  autoregressive_pixel_detail::Image(previous_cell,4,dtype);
  gan_detail::Finite(recurrent_kernel);
  if(input_gates.size(1)!=1 || previous_hidden.shape()!=previous_cell.shape() ||
     previous_hidden.size(0)!=input_gates.size(0) || previous_hidden.size(1)!=1 ||
     previous_hidden.size(2)!=input_gates.size(2) ||
     recurrent_kernel.dtype()!=dtype || recurrent_kernel.shape()!=Shape({1,3,4,16}))
    throw std::invalid_argument("PixelRowLstmStep requires aligned single-row states and a1x3 recurrent kernel");
  PixelRowLstmOutput out;
  out.gates=input_gates+conv2d(previous_hidden,recurrent_kernel);
  gan_detail::Finite(out.gates);
  out.input_gate=sigmoid(slice_axis(out.gates,3,0,4));
  out.forget_gate=sigmoid(slice_axis(out.gates,3,4,8)+1.0);
  out.output_gate=sigmoid(slice_axis(out.gates,3,8,12));
  out.candidate=tanh(slice_axis(out.gates,3,12,16));
  out.cell=out.forget_gate*previous_cell+out.input_gate*out.candidate;
  out.hidden=out.output_gate*tanh(out.cell);
  return out;
}

/// @brief Own the zero-state paths, gate projections and full Row-LSTM image sequence.
///
/// Seven flat Tensor handles retain applicable graphs/storage after parent or
/// store collection. initial_hidden and initial_cell share the same first-row
/// input feature times zero, preserving the source's connected zero path and
/// signed-zero arithmetic. They are not learned states. The remaining fields
/// concatenate real row outputs along the image-height axis. No optimizer,
/// model object, moving statistics or sampling stream is retained.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8303);
/// auto out=n::models::PixelRnn(p,n::zeros({2,8,8,1}));
/// // logits[2,8,8,1], hidden/cell[2,8,8,4], input/row gates[2,8,8,16].
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . One-layer width4 teaching variant.
/// @see PixelRnn(), PixelRowLstmOutput
struct PixelRnnOutput {
  Tensor initial_hidden; ///< Owning live first-row input feature times zero[N,1,W,4].
  Tensor initial_cell;   ///< Owning live same zero tensor as initial_hidden[N,1,W,4].
  Tensor input_gates;    ///< Owning live biased mask-A3x3 input projection[N,H,W,16].
  Tensor row_gates;      ///< Owning live concatenated input+recurrent preactivations[N,H,W,16].
  Tensor hidden;         ///< Owning live concatenated output-gated hidden states[N,H,W,4].
  Tensor cell;           ///< Owning live concatenated recurrent cell states[N,H,W,4].
  Tensor logits;         ///< Owning live biased1x1 hidden4-to1 Bernoulli logits[N,H,W,1].
};

/// @brief Evaluate a compact strictly raster-causal image density with one Row-LSTM layer.
///
/// Finite signed matching Float32/64 CPU image[N,H,W,1], positive rectangles,
/// arbitrary strides. A biased3x3 SAME mask-A input convolution1->16 excludes
/// current/future raw pixels. A live unbiased1x3 recurrent kernel4->16 couples
/// the previous hidden row's neighboring columns. Every image row advances by
/// PixelRowLstmStep, with i/f/o/candidate ordering and fixed forget bias+1.
/// Initial hidden/cell are the first input row's first4 features times zero,
/// retaining the exact source connected zero path. Concatenate all H hidden/
/// cell rows, then biased1x1 hidden4->1 logits. Columns advance in parallel;
/// there is no same-row hidden recurrence. The compact receptive field does
/// not guarantee access to every earlier pixel. No batch coupling, class
/// conditioning, normalization, dropout or pooling.
///
/// Missing W use normal(0,.1), biases zero, source seed8303. Create leaves in
/// source order input/W,input/b,row/W,output/W,output/b:357 values in5 leaves,
/// 277 active and80 masked input-kernel coordinates. Masked partials are exact
/// zero; there are no learned initial states or recurrent biases. Validate
/// supplied image before leaves; stored-leaf/math errors may follow creation.
/// Only missing leaves initialize; no existing-leaf writes, later RNG, fitting,
/// optimizer, sampling or hidden no-grad. Seven owning fields retain applicable
/// first/higher image/weight paths through all rows. Work/storage O(NHW), fixed
/// width4, graph depth O(H); synchronize shared mutable parameter stores.
/// @param parameters Caller-owned matching floating store, source seed8303.
/// @param image Finite signed matching CPU[N,H,W,1] image/canvas; source experiment uses8x8.
/// @return Owning seven-field PixelRnnOutput with actual zero, gate and sequence boundaries.
/// @throws std::invalid_argument For incompatible geometry/dtype/products/values or stored leaves.
/// @throws std::logic_error For unsupported storage or primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(8303);
/// auto out=n::models::PixelRnn(p,n::zeros({2,8,8,1}));
/// // p.count()==357,p.size()==5; pass out.logits and binary targets to BernoulliPixels.
/// @endcode
/// @par References
/// - van den Oord, Kalchbrenner and Kavukcuoglu(2016), Pixel Recurrent Neural
/// Networks, https://arxiv.org/abs/1601.06759 . One width4 Row-LSTM and a binary
/// head are a CPU teaching variant, not the paper's12-layer RGB model.
/// @see PixelRnnOutput, PixelRowLstmStep(), CausalPixelMask(), BernoulliPixels()
inline PixelRnnOutput PixelRnn(ParameterStore& parameters,const Tensor& image) {
  autoregressive_pixel_detail::Image(image,1,parameters.dtype());
  if(image.numel()>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("PixelRnn gate activation product overflows");
  const auto input_w=parameters.get("input/W",{3,3,1,16},init::normal(.1));
  const auto input_b=parameters.get("input/b",{16},init::zeros());
  const auto row_w=parameters.get("row/W",{1,3,4,16},init::normal(.1));
  const auto output_w=parameters.get("output/W",{1,1,4,1},init::normal(.1));
  const auto output_b=parameters.get("output/b",{1},init::zeros());
  PixelRnnOutput out;
  out.input_gates=bias_add(conv2d(image,input_w*CausalPixelMask(1,16,false,parameters.dtype())),input_b);
  out.initial_hidden=slice_axis(slice_axis(out.input_gates,1,0,1),3,0,4)*0.0;
  out.initial_cell=out.initial_hidden;
  Tensor hidden=out.initial_hidden,cell=out.initial_cell;
  std::vector<Tensor> states,cells,gates;
  for(std::int64_t y=0;y<image.size(1);++y) {
    auto row=PixelRowLstmStep(slice_axis(out.input_gates,1,y,y+1),hidden,cell,row_w);
    hidden=row.hidden;cell=row.cell;
    states.push_back(hidden);cells.push_back(cell);gates.push_back(row.gates);
  }
  out.hidden=concat(states,1);out.cell=concat(cells,1);out.row_gates=concat(gates,1);
  out.logits=bias_add(conv2d(out.hidden,output_w),output_b);
  gan_detail::Finite(out.logits);
  return out;
}
} // namespace datamunge::nn::models
