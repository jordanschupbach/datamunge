#pragma once
/// @file maf_density.hpp
/// @brief Strict degree-masked Gaussian conditioners and compact masked autoregressive density flows.
#include "datamunge/nn/affine_flows.hpp"
#include "datamunge/nn/made_density.hpp"

namespace datamunge::nn::models {
/// @brief Own the compact two-coordinate MAF degree metadata and structural masks.
///
/// Five owning detached fields survive parent collection. Integer degrees are
/// Int32; masks use the requested Float32/64 dtype. No learned leaves, optimizer,
/// order state or RNG is retained. These masks also serve the paired IAF model.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto r=dm::nn::models::MafMasks(1);
/// // input degrees[2,1], hidden degrees eight0s/eight1s; heads degrees[2,1,2,1].
/// @endcode
/// @par References
/// - Papamakarios, Pavlakou and Murray(2017), Masked Autoregressive Flow for
/// Density Estimation, https://arxiv.org/abs/1705.07057 .
/// @see MafMasks(), MafConditioner(), AutoregressiveDegreeMask()
struct MafMaskInventory {
  Tensor input_degrees;  ///< Owning detached Int32[2] degrees[1,2] for even blocks,[2,1] for odd.
  Tensor hidden_degrees; ///< Owning detached Int32[16] degree floor(j/8),j=0..15.
  Tensor output_degrees; ///< Owning detached Int32[4] input degrees repeated for mean and log-scale heads.
  Tensor input_mask;     ///< Owning detached matching floating[2,16], incoming degree<=hidden degree.
  Tensor output_mask;    ///< Owning detached matching floating[16,4], strict hidden degree<head degree.
};

/// @brief Build the exact alternating strict degree masks of a compact MAF conditioner.
///
/// Source block0..3: input degrees[1,2] for even blocks,[2,1] for odd; sixteen
/// hidden degrees floor(j/8), and output degrees repeating the input pair for
/// mu0,mu1,raw_s0,raw_s1. Input mask uses<=, output uses strict<, so no head
/// sees its own or a later input. Int32 degrees and detached Float32(default)/64
/// masks, five flat owning fields,472 metadata values across all four blocks.
/// Each block stores40 forbidden kernel coordinates. Fixed small work/storage;
/// no parameter store, graph, RNG, mutation or shared order state.
/// @param block Source index0..3, alternating the dependency order.
/// @param dtype Float32(default) or Float64 structural mask dtype.
/// @return Owning detached MafMaskInventory with degrees[2],[16],[4] and masks[2,16],[16,4].
/// @throws std::invalid_argument For invalid block or nonfloating mask dtype.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;auto r=n::models::MafMasks(0,n::DType::Float64);
/// // Eight hidden units see input0; output0 sees neither input, output1 may see input0.
/// @endcode
/// @par References
/// - Papamakarios et al.(2017), Masked Autoregressive Flow for Density Estimation,
/// https://arxiv.org/abs/1705.07057 ; Germain et al.(2015), MADE,
/// https://arxiv.org/abs/1502.03509 . These dimensions/degrees are the source teaching adaptation.
/// @see MafMaskInventory, MafConditioner(), AutoregressiveDegreeMask()
inline MafMaskInventory MafMasks(int block,DType dtype=DType::Float32) {
  if(block<0 || block>3 || (dtype!=DType::Float32 && dtype!=DType::Float64))
    throw std::invalid_argument("MafMasks needs block0..3 and a floating mask dtype");
  std::vector<std::int32_t> input(2),hidden(16),output(4);
  for(int c=0;c<2;++c)input[c]=block%2?2-c:c+1;
  for(int j=0;j<16;++j)hidden[j]=j/8;
  for(int c=0;c<4;++c)output[c]=input[c%2];
  MafMaskInventory out;
  out.input_degrees=from_vector(input,{2});out.hidden_degrees=from_vector(hidden,{16});
  out.output_degrees=from_vector(output,{4});
  out.input_mask=AutoregressiveDegreeMask(out.input_degrees,out.hidden_degrees,false,dtype);
  out.output_mask=AutoregressiveDegreeMask(out.hidden_degrees,out.output_degrees,true,dtype);
  return out;
}

/// @brief Own live mean and bounded log-scale fields of a strict Gaussian conditioner.
///
/// Two flat Tensor handles retain applicable input/parameter first/higher graphs
/// and storage after parent/store collection. scale means log scale, not exp(scale).
/// Structural masks remain inside the graph; no optimizer or fitting state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9603);
/// auto c=n::models::MafConditioner(p,n::ones({3,2}),0);
/// // c.mean[3,2], c.scale[3,2]; scale=.8*tanh(raw), own/future input partials zero.
/// @endcode
/// @par References
/// - Papamakarios et al.(2017), MAF, https://arxiv.org/abs/1705.07057 .
/// @see MafConditioner(), MafDensity(), MafMasks()
struct MafConditionerOutput {
  Tensor mean;  ///< Owning live matching floating CPU[N,2] Gaussian means in raw coordinate order.
  Tensor scale; ///< Owning live matching floating CPU[N,2] bounded log scales .8*tanh(raw), not exponentials.
};

/// @brief Apply a source strict degree-masked Gaussian conditioner to two coordinates.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides, block0..3.
/// Prefix autoregressive<block>/hidden is2->16 tanh; missing W normal(.3),
/// b=.05. /heads is16->4 linear, W normal(.01), b=0. Create hidden W/b then
/// heads W/b before applying MafMasks and MaskedDensityDense. Heads0,1 give
/// means; heads2,3 give s=.8*tanh(raw), with .8 rounded to computation dtype.
/// Four leaves/116 stored values per block include40 masked kernel entries;
/// their exact zero partials prevent optimizer writes from zero initial moments.
/// Every mean/log-scale head excludes its own and later input in dependency
/// order, even though all heads evaluate in parallel. Only missing leaves
/// initialize; existing leaves never change. No sampling, fitting, optimizer,
/// RNG after warmup or hidden no-grad. Owning fields retain applicable first/
/// higher paths after collection. O(96N) work plus activations, fixed small
/// masks; synchronize shared stores. Inputs validate before creation; stored-
/// leaf/math failures may follow partial creation. Nonfinite results reject.
/// @param parameters Caller-owned matching floating store, source MAF seed9603 or IAF seed9703.
/// @param input Finite matching Float32/64 CPU[N,2] observed conditioning coordinates.
/// @param block Source index0..3, alternating raw input degree order.
/// @return Owning live MafConditionerOutput with mean/scale[N,2] in raw coordinate order.
/// @throws std::invalid_argument For invalid inputs/block/store, unsafe activations or nonfinite values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9603,n::DType::Float64);
/// auto x=n::ones({2,2},n::DType::Float64).requires_grad_();
/// auto c=n::models::MafConditioner(p,x,1);
/// // First generated coordinate1 has constant heads; coordinate0 may depend on coordinate1.
/// @endcode
/// @par References
/// - Papamakarios et al.(2017), Masked Autoregressive Flow for Density Estimation,
/// https://arxiv.org/abs/1705.07057 . Tiny2-D teaching adaptation with bounded log scales.
/// @see MafConditionerOutput, MafMasks(), MafDensity(), MaskedDensityDense()
inline MafConditionerOutput MafConditioner(ParameterStore& parameters,const Tensor& input,int block) {
  flow_detail::Matrix(input,2);
  if(block<0 || block>3 || input.dtype()!=parameters.dtype() ||
     input.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("MafConditioner needs block0..3 and matching safe floating matrices");
  const auto name="autoregressive"+std::to_string(block);auto masks=MafMasks(block,input.dtype());
  parameters.get(name+"/hidden/W",{2,16},init::normal(.3));
  parameters.get(name+"/hidden/b",{16},init::constant(.05));
  parameters.get(name+"/heads/W",{16,4},init::normal(.01));
  parameters.get(name+"/heads/b",{4},init::zeros());
  auto hidden=tanh(MaskedDensityDense(parameters,name+"/hidden",input,masks.input_mask));
  auto heads=MaskedDensityDense(parameters,name+"/heads",hidden,masks.output_mask);
  auto mean=slice_axis(heads,1,0,2),scale=.8*tanh(slice_axis(heads,1,2,4));
  gan_detail::Finite(scale);return {mean,scale};
}

/// @brief Own all compact MAF conditioner inputs, autoregressive stages and density intermediates.
///
/// Twenty-three flat Tensor handles retain applicable first/higher graphs and
/// storage after parent/store collection. No vector/nested record ownership,
/// fitting workflow, optimizer or sampler. Scale fields mean bounded log scales;
/// each stage's conditioner sees that stage's recorded input in parallel.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9603);
/// auto out=n::models::MafDensity(p,n::zeros({2,2}));
/// // latent/unscaled/conditioner_input0..3/coupling0..3/scale/shift[2,2]; logdets[2].
/// @endcode
/// @par References
/// - Papamakarios et al.(2017), MAF, https://arxiv.org/abs/1705.07057 .
/// @see MafDensity(), MafConditioner(), MafInverse()
struct MafDensityOutput {
  Tensor conditioner_input0; ///< Owning live matching floating[N,2] stage0 observed conditioner input.
  Tensor coupling0; ///< Owning live matching floating[N,2] stage0 autoregressive stage value.
  Tensor coupling_logdet0; ///< Owning live matching floating[N] stage0 negative sum of the two bounded log scales.
  Tensor coupling_scale0; ///< Owning live matching floating[N,2] stage0 bounded log scales.
  Tensor coupling_shift0; ///< Owning live matching floating[N,2] stage0 Gaussian conditional means.
  Tensor conditioner_input1; ///< Owning live matching floating[N,2] stage1 observed conditioner input.
  Tensor coupling1; ///< Owning live matching floating[N,2] stage1 autoregressive stage value.
  Tensor coupling_logdet1; ///< Owning live matching floating[N] stage1 negative sum of the two bounded log scales.
  Tensor coupling_scale1; ///< Owning live matching floating[N,2] stage1 bounded log scales.
  Tensor coupling_shift1; ///< Owning live matching floating[N,2] stage1 Gaussian conditional means.
  Tensor conditioner_input2; ///< Owning live matching floating[N,2] stage2 observed conditioner input.
  Tensor coupling2; ///< Owning live matching floating[N,2] stage2 autoregressive stage value.
  Tensor coupling_logdet2; ///< Owning live matching floating[N] stage2 negative sum of the two bounded log scales.
  Tensor coupling_scale2; ///< Owning live matching floating[N,2] stage2 bounded log scales.
  Tensor coupling_shift2; ///< Owning live matching floating[N,2] stage2 Gaussian conditional means.
  Tensor conditioner_input3; ///< Owning live matching floating[N,2] stage3 observed conditioner input.
  Tensor coupling3; ///< Owning live matching floating[N,2] stage3 autoregressive stage value.
  Tensor coupling_logdet3; ///< Owning live matching floating[N] stage3 negative sum of the two bounded log scales.
  Tensor coupling_scale3; ///< Owning live matching floating[N,2] stage3 bounded log scales.
  Tensor coupling_shift3; ///< Owning live matching floating[N,2] stage3 Gaussian conditional means.
  Tensor unscaled; ///< Live floating[N,2] alias of coupling3 before final global scaling.
  Tensor latent;   ///< Owning live matching floating[N,2] globally scaled Gaussian coordinates.
  Tensor logdet;   ///< Owning live matching floating[N] total forward log absolute determinant, nats.
};

/// @brief Map observations through four parallel masked Gaussian autoregressions and global scale.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. In blocks
/// 0..3 evaluate MafConditioner on each current input, transform both coordinates
/// to z=(x-mu)*exp(-s) in parallel, and add -sum(s) per row in source order.
/// Apply FlowDiagonalScale last. Four2->16 tanh->4 conditioners plus global
/// log_scale[2] give466 stored values/17 leaves, including160 masked entries.
/// Warm forward with source seed9603 to preserve initializers/order. Only missing
/// leaves initialize; warmed inference is immutable and consumes no RNG. All
/// stage inputs, values, means, log scales and determinants remain live; own/
/// future conditioner partials stay zero. No fitting, sampling, optimizer,
/// automatic order selection or hidden no-grad. Owning fields retain applicable
/// first/higher paths after collection. O(384N) work plus activations; synchronize
/// shared stores. Validate supplied inputs before leaves; stored-leaf/math
/// failures may follow partial creation. Both global scale directions must
/// remain positive finite. Reject nonfinite results; finite roundtrip error remains.
/// @param parameters Caller-owned matching floating store, source initialization seed9603.
/// @param input Finite matching Float32/64 CPU[N,2] observations, positive rows.
/// @return Owning live MafDensityOutput with23 flat source diagnostic fields.
/// @throws std::invalid_argument For invalid inputs/store/scale or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9603);
/// auto path=n::models::MafDensity(p,n::zeros({3,2}));
/// auto density=n::models::NormalFlowDensity(path.latent,path.logdet,true);
/// // p.count()==466,p.size()==17; density.loss is mean source-compatible NLL.
/// @endcode
/// @par References
/// - Papamakarios, Pavlakou and Murray(2017), Masked Autoregressive Flow for
/// Density Estimation, https://arxiv.org/abs/1705.07057 . Four tiny2-D masked
/// conditioners and .8*tanh bounding follow the migrated teaching example.
/// @see MafDensityOutput, MafConditioner(), MafInverse(), FlowDiagonalScale()
inline MafDensityOutput MafDensity(ParameterStore& parameters,const Tensor& input) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype())throw std::invalid_argument("MAF input/store dtypes must match");
  Tensor h=input,ld=reshape(slice_axis(input,1,0,1),{-1})*0.;
  MafDensityOutput out;
  out.conditioner_input0=h;auto c0=MafConditioner(parameters,h,0);
  h=(h-c0.mean)*exp(-c0.scale);auto ld0=-sum(c0.scale,{1});ld=ld+ld0;
  gan_detail::Finite(h);out.coupling0=h;out.coupling_logdet0=ld0;
  out.coupling_scale0=c0.scale;out.coupling_shift0=c0.mean;
  out.conditioner_input1=h;auto c1=MafConditioner(parameters,h,1);
  h=(h-c1.mean)*exp(-c1.scale);auto ld1=-sum(c1.scale,{1});ld=ld+ld1;
  gan_detail::Finite(h);out.coupling1=h;out.coupling_logdet1=ld1;
  out.coupling_scale1=c1.scale;out.coupling_shift1=c1.mean;
  out.conditioner_input2=h;auto c2=MafConditioner(parameters,h,2);
  h=(h-c2.mean)*exp(-c2.scale);auto ld2=-sum(c2.scale,{1});ld=ld+ld2;
  gan_detail::Finite(h);out.coupling2=h;out.coupling_logdet2=ld2;
  out.coupling_scale2=c2.scale;out.coupling_shift2=c2.mean;
  out.conditioner_input3=h;auto c3=MafConditioner(parameters,h,3);
  h=(h-c3.mean)*exp(-c3.scale);auto ld3=-sum(c3.scale,{1});ld=ld+ld3;
  gan_detail::Finite(h);out.coupling3=h;out.coupling_logdet3=ld3;
  out.coupling_scale3=c3.scale;out.coupling_shift3=c3.mean;
  auto scaled=FlowDiagonalScale(parameters,h,ld);
  out.unscaled=h;out.latent=scaled.value;out.logdet=scaled.logdet;return out;
}

/// @brief Invert compact MAF by recovering coordinates in strict dependency order.
///
/// Finite matching Float32/64 CPU latent[N,2],N>0, arbitrary strides. Unscale
/// globally, then reverse blocks3..0. For each block start recovered=input*0;
/// recover coordinates0,1 for even blocks or1,0 for odd, recomputing the strict
/// conditioner on the recovered prefix before each coordinate. Set
/// x_c=z_c*exp(s_c)+mu_c and add the used s_c to the incoming determinant in
/// dependency order, preserving source floating accumulation. Own/future
/// conditioner inputs cannot affect that head. This sequential inverse uses
/// two conditioner passes per block; forward density uses one. Its logdet is
/// computed at its own recovered points, with no cached forward state. A
/// forward-warmed store reuses all466 values without writes/RNG; missing leaves
/// initialize global scale then reversed blocks, so warm MafDensity to preserve
/// source seed9603 ordering. No sampling stream, fitting, optimizer or hidden
/// no-grad. Owns live value/logdet after collection; applicable first/higher
/// paths remain live. O(768N) work plus activations; synchronize shared stores.
/// Inputs validate before leaves; stored-leaf/math failures may follow creation.
/// Both global scale directions must remain positive finite, nonfinite results
/// reject, finite roundtrip error remains.
/// @param parameters Caller-owned matching floating store, preferably forward-warmed.
/// @param latent Finite matching Float32/64 CPU[N,2] caller-supplied Gaussian coordinates.
/// @return Owning live FlowScaleOutput with inverse value[N,2] and inverse-path logdet[N].
/// @throws std::invalid_argument For invalid inputs/store/scale or nonfinite stored/computed values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9603);
/// auto x=n::ones({2,2});auto path=n::models::MafDensity(p,x);
/// auto rec=n::models::MafInverse(p,path.latent);
/// // rec.value approximates x; rec.logdet+path.logdet approximates zero per row.
/// @endcode
/// @par References
/// - Papamakarios et al.(2017), Masked Autoregressive Flow for Density Estimation,
/// https://arxiv.org/abs/1705.07057 . Parallel likelihood and sequential generation.
/// @see MafDensity(), MafConditioner(), MafMasks(), FlowScaleOutput
inline FlowScaleOutput MafInverse(ParameterStore& parameters,const Tensor& latent) {
  flow_detail::Matrix(latent,2);
  if(latent.dtype()!=parameters.dtype() || latent.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("MAF inverse needs matching safe floating points");
  auto zero=reshape(slice_axis(latent,1,0,1),{-1})*0.;
  auto out=FlowDiagonalScale(parameters,latent,zero,true);
  for(int i=3;i>=0;--i){
    auto recovered=out.value*0.;
    for(int ordinal=0;ordinal<2;++ordinal){
      const int c=i%2?1-ordinal:ordinal;auto cond=MafConditioner(parameters,recovered,i);
      auto s=slice_axis(cond.scale,1,c,c+1),mean=slice_axis(cond.mean,1,c,c+1);
      auto value=slice_axis(out.value,1,c,c+1)*exp(s)+mean;
      auto other=slice_axis(recovered,1,1-c,2-c);
      recovered=c?concat({other,value},1):concat({value,other},1);
      out.logdet=out.logdet+reshape(s,{-1});
    }
    gan_detail::Finite(recovered);out.value=recovered;
  }
  gan_detail::Finite(out.logdet);return out;
}
} // namespace datamunge::nn::models
