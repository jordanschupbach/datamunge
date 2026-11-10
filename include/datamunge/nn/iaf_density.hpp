#pragma once
/// @file iaf_density.hpp
/// @brief Compact affine IAF with sequential density recovery and parallel inverse sampling maps.
#include "datamunge/nn/maf_density.hpp"
#include <array>

namespace datamunge::nn::models {
/// @brief Own all compact IAF recovered inputs, autoregressive stages and density intermediates.
///
/// Twenty-three flat Tensor handles retain applicable first/higher graphs and
/// storage after parent/store collection. Stage indices are source block names,
/// applied3,2,1,0 in density order. Each conditioner_input is the recovered
/// coordinate batch, also exposed as coupling, on which the final heads are
/// re-evaluated for that stage's density determinant. Scale fields are bounded
/// log scales, not exponentials. No fitting workflow, optimizer or sampler.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9703);
/// auto out=n::models::IafDensity(p,n::zeros({2,2}));
/// // unscaled aliases coupling0; every recovered stage is[N,2], logdet[N].
/// @endcode
/// @par References
/// - Kingma et al.(2016), Improving Variational Inference with Inverse
/// Autoregressive Flow, https://arxiv.org/abs/1606.04934 . This standalone
/// exact-density example uses affine exp(.8*tanh(raw)) scaling, not sigmoid gating.
/// @see IafDensity(), IafInverse(), MafConditioner()
struct IafDensityOutput {
  Tensor conditioner_input0; ///< Owning live matching floating[N,2] source block0 fully recovered input for the final conditioner re-evaluation.
  Tensor coupling0; ///< Owning live matching floating[N,2] source block0 recovered autoregressive stage value, alias of conditioner_input.
  Tensor coupling_logdet0; ///< Owning live matching floating[N] source block0 negative sum of the final two bounded log scales.
  Tensor coupling_scale0; ///< Owning live matching floating[N,2] source block0 bounded log scales re-evaluated on the recovered input.
  Tensor coupling_shift0; ///< Owning live matching floating[N,2] source block0 Gaussian means re-evaluated on the recovered input.
  Tensor conditioner_input1; ///< Owning live matching floating[N,2] source block1 fully recovered input for the final conditioner re-evaluation.
  Tensor coupling1; ///< Owning live matching floating[N,2] source block1 recovered autoregressive stage value, alias of conditioner_input.
  Tensor coupling_logdet1; ///< Owning live matching floating[N] source block1 negative sum of the final two bounded log scales.
  Tensor coupling_scale1; ///< Owning live matching floating[N,2] source block1 bounded log scales re-evaluated on the recovered input.
  Tensor coupling_shift1; ///< Owning live matching floating[N,2] source block1 Gaussian means re-evaluated on the recovered input.
  Tensor conditioner_input2; ///< Owning live matching floating[N,2] source block2 fully recovered input for the final conditioner re-evaluation.
  Tensor coupling2; ///< Owning live matching floating[N,2] source block2 recovered autoregressive stage value, alias of conditioner_input.
  Tensor coupling_logdet2; ///< Owning live matching floating[N] source block2 negative sum of the final two bounded log scales.
  Tensor coupling_scale2; ///< Owning live matching floating[N,2] source block2 bounded log scales re-evaluated on the recovered input.
  Tensor coupling_shift2; ///< Owning live matching floating[N,2] source block2 Gaussian means re-evaluated on the recovered input.
  Tensor conditioner_input3; ///< Owning live matching floating[N,2] source block3 fully recovered input for the final conditioner re-evaluation.
  Tensor coupling3; ///< Owning live matching floating[N,2] source block3 recovered autoregressive stage value, alias of conditioner_input.
  Tensor coupling_logdet3; ///< Owning live matching floating[N] source block3 negative sum of the final two bounded log scales.
  Tensor coupling_scale3; ///< Owning live matching floating[N,2] source block3 bounded log scales re-evaluated on the recovered input.
  Tensor coupling_shift3; ///< Owning live matching floating[N,2] source block3 Gaussian means re-evaluated on the recovered input.
  Tensor unscaled; ///< Live floating[N,2] alias of coupling0, the last density stage before scaling.
  Tensor latent;   ///< Owning live matching floating[N,2] globally scaled Gaussian coordinates.
  Tensor logdet;   ///< Owning live matching floating[N] full forward density log absolute determinant, nats.
};

/// @brief Recover Gaussian density coordinates sequentially through the source affine IAF.
///
/// Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. Reverse
/// source blocks3..0. For each block start recovered=input*0 and recover
/// coordinates0,1(even) or1,0(odd), calling MafConditioner on the recovered
/// prefix before setting u_c=(observed_c-mu_c)*exp(-s_c). Re-evaluate heads
/// once more on the fully recovered input, add -sum(s) per row, and expose
/// those final means/log scales and live recovered inputs. This third call
/// preserves the source's determinant graph and rounding. Finish with
/// FlowDiagonalScale. Each strict conditioner excludes its own and future
/// inputs; all original first/higher parameter/input paths remain live.
/// Four2->16 tanh->4 conditioners plus log_scale[2] give466 stored values in
///17 leaves, including160 masked kernel entries. Source seed9703 initializes
/// blocks3,2,1,0 then global scale; only missing leaves initialize. Warmed
/// inference changes no leaves and consumes no RNG. Source scale is .8*tanh
/// rounded to computation dtype. This compact affine exact-density model is
/// distinct from the paper's sigmoid-gated variational posterior; no VAE/ELBO,
/// fitting, sampling stream, optimizer or hidden no-grad is implied. Owning
/// flat fields retain graphs/storage after collection. O(1152N) work plus
/// activations, three conditioner passes per block; synchronize shared stores.
/// Inputs validate before leaves; stored-leaf/math failures may follow partial
/// creation. Both global scale directions must remain positive finite;
/// nonfinite intermediate/results reject, floating roundtrip errors remain.
/// @param parameters Caller-owned matching floating store, source seed9703.
/// @param input Finite matching Float32/64 CPU[N,2] observed coordinates, positive rows.
/// @return Owning live IafDensityOutput with23 flat recovered stage and density fields.
/// @throws std::invalid_argument For invalid inputs/store/scale, unsafe activations or nonfinite values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9703);
/// auto path=n::models::IafDensity(p,n::ones({3,2}));
/// auto density=n::models::NormalFlowDensity(path.latent,path.logdet,true);
/// //466 stored values/17 leaves; loss is mean source-compatible NLL in nats/row.
/// @endcode
/// @par References
/// - Kingma et al.(2016), Improving Variational Inference with Inverse
/// Autoregressive Flow, https://arxiv.org/abs/1606.04934 . The migrated
///2-D standalone source uses bounded affine scales and sequential exact likelihood.
/// @see IafDensityOutput, IafInverse(), MafConditioner(), FlowDiagonalScale()
inline IafDensityOutput IafDensity(ParameterStore& parameters,const Tensor& input) {
  flow_detail::Matrix(input,2);
  if(input.dtype()!=parameters.dtype() || input.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("IAF density needs matching safe floating matrices");
  Tensor h=input,ld=reshape(slice_axis(input,1,0,1),{-1})*0.;
  std::array<Tensor,4> values,scales,shifts,determinants;
  for(int i=3;i>=0;--i){
    auto recovered=h*0.;
    for(int ordinal=0;ordinal<2;++ordinal){
      const int c=i%2?1-ordinal:ordinal;auto cond=MafConditioner(parameters,recovered,i);
      auto mean=slice_axis(cond.mean,1,c,c+1),scale=slice_axis(cond.scale,1,c,c+1);
      auto value=(slice_axis(h,1,c,c+1)-mean)*exp(-scale);
      auto other=slice_axis(recovered,1,1-c,2-c);
      recovered=c?concat({other,value},1):concat({value,other},1);
    }
    gan_detail::Finite(recovered);auto cond=MafConditioner(parameters,recovered,i);
    auto contribution=-sum(cond.scale,{1});ld=ld+contribution;
    values[i]=recovered;scales[i]=cond.scale;shifts[i]=cond.mean;determinants[i]=contribution;h=recovered;
  }
  auto scaled=FlowDiagonalScale(parameters,h,ld);
  return {values[0],values[0],determinants[0],scales[0],shifts[0],
          values[1],values[1],determinants[1],scales[1],shifts[1],
          values[2],values[2],determinants[2],scales[2],shifts[2],
          values[3],values[3],determinants[3],scales[3],shifts[3],
          h,scaled.value,scaled.logdet};
}

/// @brief Map explicit Gaussian coordinates to IAF observations with parallel autoregressive stages.
///
/// Finite matching Float32/64 CPU latent[N,2],N>0, arbitrary strides. Unscale
/// globally, then source blocks0..3. One MafConditioner call per block evaluates
/// all heads on the known current input; set h=h*exp(s)+mu in parallel and add
/// sum(s) per row. Each block's inverse determinant uses its own supplied input,
/// independent of cached forward points. Four conditioner calls total, versus
/// twelve in IafDensity's sequential recovery and final head re-evaluation.
/// A forward-warmed store reuses466 values/17 leaves without writes or RNG;
/// missing leaves initialize global scale then blocks0..3, so warm IafDensity
/// for source seed9703 ordering. Owns live value/logdet after collection;
/// applicable first/higher input/parameter paths remain live. No sampling stream,
/// fitting, optimizer, hidden no-grad or VAE/ELBO workflow. O(384N) work plus
/// activations; synchronize shared stores. Validate inputs before leaves;
/// stored-leaf/math failures may follow partial creation. Both global scale
/// directions must remain positive finite. Nonfinite results reject without
/// clipping; finite floating roundtrip errors remain.
/// @param parameters Caller-owned matching floating store, preferably forward-warmed.
/// @param latent Finite matching Float32/64 CPU[N,2] caller-supplied Gaussian coordinates.
/// @return Owning live FlowScaleOutput with observation value[N,2] and inverse-path logdet[N].
/// @throws std::invalid_argument For invalid inputs/store/scale, unsafe activations or nonfinite values.
/// @throws std::logic_error For unsupported primitive execution.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;n::ParameterStore p(9703);
/// auto x=n::ones({2,2});auto path=n::models::IafDensity(p,x);
/// auto rec=n::models::IafInverse(p,path.latent);
/// // rec.value approximates x; rec.logdet+path.logdet approximates zero per row.
/// @endcode
/// @par References
/// - Kingma et al.(2016), Improving Variational Inference with Inverse
/// Autoregressive Flow, https://arxiv.org/abs/1606.04934 . Parallel generation
/// with source .8*tanh affine log scales, distinct from sigmoid gating.
/// @see IafDensity(), IafDensityOutput, MafConditioner(), FlowScaleOutput
inline FlowScaleOutput IafInverse(ParameterStore& parameters,const Tensor& latent) {
  flow_detail::Matrix(latent,2);
  if(latent.dtype()!=parameters.dtype() || latent.size(0)>std::numeric_limits<std::int64_t>::max()/16)
    throw std::invalid_argument("IAF inverse needs matching safe floating matrices");
  auto zero=reshape(slice_axis(latent,1,0,1),{-1})*0.;
  auto out=FlowDiagonalScale(parameters,latent,zero,true);
  for(int i=0;i<4;++i){auto cond=MafConditioner(parameters,out.value,i);
    out.value=out.value*exp(cond.scale)+cond.mean;out.logdet=out.logdet+sum(cond.scale,{1});
    gan_detail::Finite(out.value);}
  gan_detail::Finite(out.logdet);return out;
}
} // namespace datamunge::nn::models
