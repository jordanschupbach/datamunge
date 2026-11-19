#pragma once

/// @file mnist_vision_transformer.hpp
/// @brief Two-block Vision Transformer for single-channel MNIST images.

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/attention_variants.hpp>
#include <datamunge/nn/completion_graphs.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::nn {

/// @brief Class scores and intermediate tensors from the MNIST Vision Transformer.
/// @par Example
/// @code
/// const dm::nn::MnistVisionTransformerOutput empty{};
/// @endcode
struct MnistVisionTransformerOutput {
  Tensor logits; ///< Unnormalized class scores [batch,10].
  Tensor patches; ///< Raw 7-by-7 patches [batch,16,49] in raster order.
  Tensor embedded; ///< Special and patch tokens with positional embeddings [batch,tokens,24].
  std::vector<Tensor> attention; ///< Two attention probability arrays, or FAVOR+ query normalizers for Performer.
};

/// @brief DeiT student heads and shared encoder diagnostics.
struct MnistDeitOutput {
  MnistVisionTransformerOutput classification; ///< Class head and encoder tensors.
  Tensor distillation; ///< Distillation head scores [batch,10].
  Tensor logits; ///< Mean of class and distillation scores [batch,10].
};

/// @brief Attend to a packed sequence of causal keys at a fixed spacing.
/// @param q,hk,hv Matching CPU floating tensors [batch,heads,tokens,depth].
/// @param slots Number of candidate keys per query, including the query itself.
/// @param spacing Positive gap between consecutive keys.
/// @return Context and weights [batch,heads,tokens,slots], newest key first.
/// @throws std::invalid_argument Invalid tensor geometry, spacing or slot count.
inline Attended causal_factor_attention(const Tensor& q,const Tensor& hk,const Tensor& hv,
    std::int64_t slots,std::int64_t spacing) {
  if (!q.defined() || !hk.defined() || !hv.defined() || q.dim()!=4 || q.numel()==0 ||
      q.shape()!=hk.shape() || q.shape()!=hv.shape() || spacing<1 ||
      slots<1 || slots>1+(q.size(2)-1)/spacing) {
    throw std::invalid_argument("causal_factor_attention: invalid geometry");
  }
  const auto tokens=q.size(2);
  std::vector<std::int64_t> indices;
  std::vector<float> bias;
  for (std::int64_t query=0;query<tokens;++query) {
    for (std::int64_t slot=0;slot<slots;++slot) {
      const auto key=query-slot*spacing;
      indices.push_back(std::max<std::int64_t>(key,0));
      bias.push_back(key<0 ? -1e9f : 0.f);
    }
  }
  const auto keys=from_vector(indices,{tokens,slots});
  const auto scores=sum(expand_dims(q,3)*gather(hk,keys,2),{-1})/std::sqrt(double(q.size(3)));
  const auto weights=softmax(where(equal(from_vector(bias,{tokens,slots}),0),scores,-std::numeric_limits<double>::infinity()),-1);
  return {sum(expand_dims(weights,4)*gather(hv,keys,2),{3}),weights};
}

namespace detail {

/// @brief Select the global-token local-window layout.
enum class MnistWindowAttention {
  None, ///< Leave the backbone attention configuration unchanged.
  Longformer, ///< Use separate projections for the global query.
  BigBird ///< Add seeded random connections to local queries.
};

/// @brief Evaluate local windows and a global token with a packed diagnostic.
/// @param q,k,v Local-path query, key and value heads.
/// @param qg,kg,vg Heads for the separate global query path.
/// @param big_bird Add two random nonlocal connections per patch.
/// @param seed Seed used for the random connections.
/// @return Token contexts and packed global and local probabilities.
inline Attended mnist_global_windows(const Tensor& q,const Tensor& k,const Tensor& v,
    const Tensor& qg,const Tensor& kg,const Tensor& vg,bool big_bird,unsigned seed) {
  constexpr int tokens=17,radius=2;
  const int slots=big_bird ? 8 : 6;
  std::vector<int> keys,keep;
  std::mt19937 random(seed);
  for (int query=1;query<tokens;++query) {
    keys.push_back(0);
    keep.push_back(1);
    for (int offset=-radius;offset<=radius;++offset) {
      const int key=query+offset;
      const bool valid=key>=1 && key<tokens;
      keys.push_back(valid ? key : 0);
      keep.push_back(valid);
    }
    if (big_bird) {
      const int left=std::max(0,query-radius-1);
      const int right=std::min(tokens,query+radius+1);
      const int count=left+tokens-right;
      const int first=random()%count;
      const int other=random()%(count-1);
      const auto position=[=](int i) { return i<left ? 1+i : right+i-left; };
      keys.push_back(position(first));
      keys.push_back(position(other+(other>=first)));
      keep.insert(keep.end(),2,1);
    }
  }
  const auto indices=from_vector(keys,{tokens-1,slots});
  const auto valid=equal(from_vector(keep,{tokens-1,slots}),1);
  const auto local_q=slice_axis(q,2,1,tokens);
  const auto score=sum(expand_dims(local_q,3)*gather(k,indices,2),{-1})/std::sqrt(8.);
  const auto local_weights=softmax(where(valid,score,-std::numeric_limits<double>::infinity()),-1);
  const auto local=sum(expand_dims(local_weights,4)*gather(v,indices,2),{3});
  const auto global=scaled_dot_product_attention(slice_axis(qg,2,0,1),kg,vg,{.need_weights=true});
  const auto packed=concat({squeeze(global.weights,{2}),local_weights.reshape({q.size(0),q.size(1),-1})},2);
  return {concat({global.output,local},2),packed};
}

/// @brief Apply single-round LSH attention with sorted four-token chunks.
/// @param qk Shared query and key heads [batch,3,17,8].
/// @param value Value heads with the same geometry.
/// @param rotation Fixed Gaussian hash projection [8,4].
/// @return Restored contexts and sorted position/probability traces.
inline Attended mnist_lsh_attention(const Tensor& qk,const Tensor& value,const Tensor& rotation) {
  constexpr int heads=3,tokens=17,padded=20,depth=8,groups=5,chunk=4;
  const auto batch=qk.size(0), rows=batch*heads;
  const auto project=matmul(qk.detach(),rotation);
  const auto hash=argmax(concat({project,-project},3),3).to_vector<std::int64_t>();
  std::vector<int> order,sorted,gather_rows,inverse(rows*padded);
  for (std::int64_t row=0;row<rows;++row) {
    std::array<int,padded> ids;
    std::iota(ids.begin(),ids.end(),0);
    const auto bucket=[&](int i) { return i<tokens ? int(hash[row*tokens+i]) : 8; };
    std::stable_sort(ids.begin(),ids.end(),[&](int a,int b) { return bucket(a)<bucket(b); });
    for (int i=0;i<padded;++i) {
      order.push_back(ids[i]);
      sorted.push_back(bucket(ids[i]));
      gather_rows.push_back(row*padded+ids[i]);
      inverse[row*padded+ids[i]]=row*padded+i;
    }
  }
  const auto permutation=from_vector(gather_rows);
  const auto pack=[&](const Tensor& input) {
    const auto extended=pad(input,{{0,0},{0,0},{0,padded-tokens},{0,0}});
    return gather(extended.reshape({rows*padded,depth}),permutation).reshape({rows,groups,chunk,depth});
  };
  const auto previous=from_vector(std::vector<int>{0,0,1,2,3});
  const auto adjacent=[&](const Tensor& x) { return concat({gather(x,previous,1),x},2); };
  const auto query=pack(qk);
  const auto norm=sqrt(maximum(mean(square(query),{-1},true),scalar(1e-12,qk.dtype())));
  const auto key=adjacent(query/norm);
  const auto ids=from_vector(order,{rows,groups,chunk});
  const auto buckets=from_vector(sorted,{rows,groups,chunk});
  const auto same=equal(expand_dims(buckets,3),expand_dims(adjacent(buckets),2));
  const auto self=equal(expand_dims(ids,3),expand_dims(adjacent(ids),2));
  const auto valid=greater_equal(arange(0,groups).reshape({1,groups,1,1}),1);
  const auto current=greater_equal(arange(0,2*chunk).reshape({1,1,1,2*chunk}),chunk);
  const auto allowed=logical_and(same,logical_or(valid,current));
  const auto scores=matmul(query,key,false,true)/std::sqrt(8.);
  const auto bias=where(allowed,where(self,full_like(scores,-1e5),0.0),-1e9);
  const auto weights=softmax(scores+bias,-1);
  const auto mixed=matmul(weights,adjacent(pack(value))).reshape({rows*padded,depth});
  const auto restored=gather(mixed,from_vector(inverse)).reshape({batch,heads,padded,depth});
  const auto trace=concat({ids.reshape({batch,heads,padded,1}).to(qk.dtype()),
                           weights.reshape({batch,heads,padded,2*chunk})},3);
  return {slice_axis(restored,2,0,tokens),trace};
}

/// @brief Evaluate the shared two-block backbone with an optional distillation token.
/// @param parameters Model parameters.
/// @param image Input digit images.
/// @param distill Whether to add a distillation token and prediction head.
/// @param compression_rank Positive compressed sequence length, or zero for full attention.
/// @param causal_sparse Place the class token last and use alternating causal factors.
/// @param random_features Optional fixed orthogonal Performer projections.
/// @param windows Optional bidirectional global-token and local-window layout.
/// @param rotations Optional fixed Reformer hash projections.
/// @return Class head, encoder diagnostics, and optional distillation head.
inline std::pair<MnistVisionTransformerOutput,Tensor> mnist_vit_heads(
    ParameterStore& parameters,const Tensor& image,bool distill,int compression_rank=0,bool causal_sparse=false,
    const std::array<Tensor,2>* random_features=nullptr,
    MnistWindowAttention windows=MnistWindowAttention::None,
    const std::array<Tensor,2>* rotations=nullptr) {
  if (!image.defined() || image.dim() != 4 || image.size(0) == 0 ||
      image.size(1) != 28 || image.size(2) != 28 || image.size(3) != 1 ||
      image.device().backend != compute::Backend::CPU || !is_floating(image.dtype()) ||
      image.dtype() != parameters.dtype())
    throw std::invalid_argument("mnist_vision_transformer: expected CPU [batch,28,28,1]");
  const auto batch = image.size(0);
  MnistVisionTransformerOutput result;
  result.patches = image.reshape({batch,4,7,4,7}).transpose({0,1,3,2,4}).reshape({batch,16,49});
  const auto tokens = dense(parameters,"patch",result.patches,24);
  const auto cls = parameters.get("cls",{1,1,24},init::normal(.02));
  std::vector<Tensor> prefix{broadcast_to(cls,{batch,1,24})};
  if (distill) {
    const auto token=parameters.get("distillation_token",{1,1,24},init::normal(.02));
    prefix.push_back(broadcast_to(token,{batch,1,24}));
  }
  if (causal_sparse) prefix.insert(prefix.begin(),tokens);
  else prefix.push_back(tokens);
  const int length=17+distill;
  const auto position=parameters.get("position",{1,length,24},init::normal(.02));
  result.embedded=concat(prefix,1)+position;
  auto state = result.embedded;
  for (int block = 0; block < 2; ++block) {
    const auto name = "block" + std::to_string(block);
    const auto normalized = layer_norm(parameters,name+"/ln1",state);
    const auto matrix = [&](const char* key) {
      return parameters.get(name+"/attention/"+key,{24,24},init::uniform(std::sqrt(3./24.)));
    };
    const auto heads = [&](const char* key) {
      return matmul(normalized,matrix(key)).reshape({batch,length,3,8}).transpose({0,2,1,3});
    };
    const auto q = heads(rotations ? "Wqk" : "Wq");
    auto k = rotations ? q : heads("Wk");
    auto v = heads("Wv");
    if (compression_rank>0) {
      const auto compress=[&](const Tensor& states,const char* key) {
        const auto projection=parameters.get(name+"/attention/"+key,
            {compression_rank,length},init::uniform(std::sqrt(3./length)));
        return matmul(projection,states);
      };
      k=compress(k,"E");
      v=compress(v,"F");
    }
    const auto attention=[&]() -> Attended {
      if (rotations) return mnist_lsh_attention(q,v,rotations->at(block).detach());
      if (random_features) {
        const auto favor=models::performer::Context(q,k,v,random_features->at(block).detach(),8);
        return {favor.context,favor.denominator};
      }
      if (causal_sparse) return causal_factor_attention(q,k,v,4+block,block ? 4 : 1);
      if (windows==MnistWindowAttention::Longformer) {
        const auto qg=heads("Wq_global"),kg=heads("Wk_global"),vg=heads("Wv_global");
        return mnist_global_windows(q,k,v,qg,kg,vg,false,0);
      }
      if (windows==MnistWindowAttention::BigBird) {
        return mnist_global_windows(q,k,v,q,k,v,true,41901+block);
      }
      return scaled_dot_product_attention(q,k,v,{.need_weights=true});
    }();
    result.attention.push_back(attention.weights);
    const auto merged = attention.output.transpose({0,2,1,3}).reshape({batch,length,24});
    const auto residual = state + matmul(merged,matrix("Wo"));
    const auto expanded = dense(parameters,name+"/up",layer_norm(parameters,name+"/ln2",residual),
                                48,Activation::GeluTanh);
    state = residual + dense(parameters,name+"/down",expanded,24);
  }
  result.logits = dense(parameters,"head",layer_norm(parameters,"final_norm",select(state,1,causal_sparse ? length-1 : 0)),10);
  Tensor secondary;
  if (distill) {
    secondary=dense(parameters,"distillation_head",
                    layer_norm(parameters,"final_norm",select(state,1,1)),10);
  }
  return {std::move(result),secondary};
}

} // namespace detail

/// @brief Classify MNIST images with a class token and two pre-normalized transformer blocks.
/// @param parameters Trainable CPU parameter store.
/// @param image CPU floating tensor [batch,28,28,1] with the same dtype as the store.
/// @return Class logits and diagnostic patch, embedding and attention tensors.
/// @throws std::invalid_argument If the input geometry, dtype or backend is incompatible.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto output = dm::nn::mnist_vision_transformer(model,dm::nn::zeros({2,28,28,1}));
/// @endcode
/// @par References
/// - Dosovitskiy et al. (2021), An Image is Worth 16x16 Words, https://arxiv.org/abs/2010.11929.
/// @see scaled_dot_product_attention()
inline MnistVisionTransformerOutput mnist_vision_transformer(ParameterStore& parameters,
                                                             const Tensor& image) {
  return detail::mnist_vit_heads(parameters,image,false).first;
}

/// @brief Classify MNIST digits with learned key and value sequence projections.
/// @param parameters Trainable CPU parameter store.
/// @param image Float32 or Float64 images [batch,28,28,1] matching the store.
/// @param rank Number of compressed slots, from 1 through 17.
/// @return Logits and attention over compressed slots [batch,3,17,rank].
/// @throws std::invalid_argument Invalid rank or incompatible images.
inline MnistVisionTransformerOutput mnist_linformer(ParameterStore& parameters,
    const Tensor& image,int rank=5) {
  if (rank<1 || rank>17) throw std::invalid_argument("mnist_linformer: rank must be in [1,17]");
  return detail::mnist_vit_heads(parameters,image,false,rank).first;
}

/// @brief Classify digits with local then strided causal attention factors.
/// @param parameters Trainable CPU parameter store.
/// @param image CPU floating images [batch,28,28,1] matching the store.
/// @return Class logits and packed attention traces for both factors.
/// @throws std::invalid_argument Incompatible image shape, dtype or backend.
inline MnistVisionTransformerOutput mnist_sparse_transformer(ParameterStore& parameters,
    const Tensor& image) {
  return detail::mnist_vit_heads(parameters,image,false,0,true).first;
}

/// @brief Create two fixed banks of orthogonal features for the MNIST Performer.
/// @param dtype Model floating-point dtype.
/// @return Two independent [64,8] projections, one per block.
inline std::array<Tensor,2> mnist_performer_features(DType dtype=DType::Float32) {
  Generator first(41801),second(41802);
  return {orthogonal_gaussian_projection(64,8,first,dtype),
          orthogonal_gaussian_projection(64,8,second,dtype)};
}

/// @brief Classify digits with bidirectional FAVOR+ attention.
/// @param parameters Trainable CPU parameter store.
/// @param image CPU floating images [batch,28,28,1] matching the store.
/// @param features Fixed [64,8] projections from mnist_performer_features().
/// @return Class logits and per-block query normalizers [batch,3,17,1].
/// @throws std::invalid_argument Incompatible feature or image geometry.
inline MnistVisionTransformerOutput mnist_performer(ParameterStore& parameters,
    const Tensor& image,const std::array<Tensor,2>& features) {
  for (const auto& projection:features) {
    if (!projection.defined() || projection.shape()!=Shape{64,8} ||
        projection.dtype()!=parameters.dtype() || projection.device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("mnist_performer: expected CPU [64,8] feature projections");
    }
  }
  return detail::mnist_vit_heads(parameters,image,false,0,false,&features).first;
}

/// @brief Fit or evaluate the MNIST backbone with global and local Longformer attention.
/// @param parameters Trainable CPU parameter store.
/// @param image CPU floating images [batch,28,28,1] matching the store.
/// @return Logits and two packed attention traces, each [batch,3,113].
inline MnistVisionTransformerOutput mnist_longformer(ParameterStore& parameters,const Tensor& image) {
  return detail::mnist_vit_heads(parameters,image,false,0,false,nullptr,
                                detail::MnistWindowAttention::Longformer).first;
}

/// @brief Evaluate the MNIST backbone with BigBird global, window and random attention.
/// @param parameters Trainable CPU parameter store.
/// @param image CPU floating images [batch,28,28,1] matching the store.
/// @return Logits and two packed attention traces, each [batch,3,145].
inline MnistVisionTransformerOutput mnist_big_bird(ParameterStore& parameters,const Tensor& image) {
  return detail::mnist_vit_heads(parameters,image,false,0,false,nullptr,
                                detail::MnistWindowAttention::BigBird).first;
}

/// @brief Classify MNIST digits with shared query/key LSH attention.
/// @param parameters Trainable CPU parameter store.
/// @param image CPU floating images [batch,28,28,1] matching the store.
/// @param rotations Fixed [8,4] Gaussian projections, one per block.
/// @return Logits and sorted traces [batch,3,20,9].
/// @throws std::invalid_argument Invalid images or projections.
inline MnistVisionTransformerOutput mnist_reformer(ParameterStore& parameters,const Tensor& image,
    const std::array<Tensor,2>& rotations) {
  for (const auto& projection:rotations) {
    if (!projection.defined() || projection.shape()!=Shape{8,4} ||
        projection.dtype()!=parameters.dtype() || projection.device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("mnist_reformer: expected CPU [8,4] rotations");
    }
  }
  return detail::mnist_vit_heads(parameters,image,false,0,false,nullptr,
                                detail::MnistWindowAttention::None,&rotations).first;
}

/// @brief Evaluate a DeiT student with class and distillation tokens.
/// @param parameters Shared trainable parameter store.
/// @param image CPU floating images [batch,28,28,1] with the store dtype.
/// @return Separate training scores and their mean for inference.
/// @throws std::invalid_argument Incompatible image shape, dtype or backend.
inline MnistDeitOutput mnist_deit(ParameterStore& parameters,const Tensor& image) {
  auto [classification,distillation]=detail::mnist_vit_heads(parameters,image,true);
  const auto logits=.5*(classification.logits+distillation);
  return {std::move(classification),distillation,logits};
}

} // namespace datamunge::nn
