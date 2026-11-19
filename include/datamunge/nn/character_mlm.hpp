/// @file
/// @brief Bidirectional character masked-language modeling.
#pragma once

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/layers.hpp>

#include <array>
#include <cmath>
#include <tuple>
#include <numeric>
#include <random>

namespace datamunge::nn {

/// @brief Select ten positions per sequence using BERT's 80/10/10 corruption.
///
/// @param gold Integral [N,64] character IDs in [0,65).
/// @param rng Caller-owned random stream.
/// @return Corrupted IDs, Float32 selected-position mask, and Int32 operation
/// codes (0 unselected, 1 MASK, 2 random replacement, 3 unchanged).
/// @throws std::invalid_argument For invalid shape or token IDs.
/// @par Example
/// @code
/// std::mt19937 rng(7);
/// const auto batch = dm::nn::corrupt_mlm_characters(dm::nn::zeros({2,64},dm::nn::DType::Int64),rng);
/// @endcode
/// @par References
/// - Devlin et al. (2019), https://aclanthology.org/N19-1423/
/// @see character_mlm()
inline std::array<Tensor,3> corrupt_mlm_characters(const Tensor &gold, std::mt19937 &rng) {
  if(gold.dim()!=2 || gold.shape()[0]<1 || gold.shape()[1]!=64 || !is_integral(gold.dtype()) ||
     min(gold).item()<0 || max(gold).item()>=65)
    throw std::invalid_argument("corrupt_mlm_characters: expected [N,64] character IDs");
  auto input=gold.to_vector<int>();
  std::vector<float> selected(input.size());
  std::vector<int> operations(input.size());
  for(std::size_t row=0;row<input.size();row+=64) {
    std::array<int,64> order;
    std::iota(order.begin(),order.end(),0);
    std::shuffle(order.begin(),order.end(),rng);
    for(int i=0;i<10;++i) {
      const auto t=row+order[i];
      const auto draw=rng()%10;
      selected[t]=1.f;
      operations[t]=draw<8?1:draw==8?2:3;
      if(draw<8) input[t]=65;
      else if(draw==8) input[t]=rng()%65;
    }
  }
  return {from_vector(input,gold.shape()),from_vector(selected,gold.shape()),
          from_vector(operations,gold.shape())};
}

/// @brief Choose a parameter-efficient character encoder update.
enum class CharacterBertAdaptation {
  None,    ///< Ordinary encoder with no additional parameters.
  Lora,    ///< Low-rank attention query and value corrections.
  Adapter  ///< Residual bottleneck after each encoder block.
};

/// @brief Encode character tokens with optional padding-aware bidirectional attention.
///
/// The encoder width is 24 with three bidirectional attention heads and
/// a 48-unit feed-forward layer per block.
///
/// @param model Owns the encoder tensors.
/// @param tokens Integral [N,64] input IDs in [0,66).
/// @param layers Number of blocks, one or two, default two.
/// @param factorized Use eight-dimensional embeddings and share block parameters when true.
/// @param lengths Optional integral [N] valid lengths in [1,64].
/// @param method Optional attention or residual adaptation, default None.
/// @param adaptation Independent store for adaptation parameters when selected.
/// @return Token table, hidden states [N,64,24], and attention per block.
/// @throws std::invalid_argument For invalid input or block count.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto path = dm::nn::character_bert_encoder(model,dm::nn::zeros({2,64},dm::nn::DType::Int64));
/// @endcode
/// @par References
/// - Devlin et al. (2019), https://aclanthology.org/N19-1423/
/// @see corrupt_mlm_characters()
inline std::tuple<Tensor,Tensor,std::vector<Tensor>> character_bert_encoder(
    ParameterStore &model,const Tensor &tokens,int layers=2,bool factorized=false,
    const Tensor &lengths={}, CharacterBertAdaptation method=CharacterBertAdaptation::None,
    ParameterStore *adaptation=nullptr) {
  if(tokens.dim()!=2 || tokens.shape()[0]<1 || tokens.shape()[1]!=64 ||
     !is_integral(tokens.dtype()) || min(tokens).item()<0 || max(tokens).item()>=66 ||
     layers<1 || layers>2 || (lengths.defined() &&
     (lengths.shape()!=Shape{tokens.shape()[0]} || !is_integral(lengths.dtype()) ||
      min(lengths).item()<1 || max(lengths).item()>64)))
    throw std::invalid_argument("character_bert_encoder: invalid input geometry or block count");
  if (method!=CharacterBertAdaptation::None &&
      (!adaptation || adaptation==&model || adaptation->dtype()!=model.dtype() ||
       (method!=CharacterBertAdaptation::Lora && method!=CharacterBertAdaptation::Adapter)))
    throw std::invalid_argument("character_bert_encoder: independent adaptation store required");
  const auto batch=tokens.shape()[0];
  const int width=factorized?8:24;
  const auto table=model.get("word/W",{66,width},init::normal(.02));
  const auto pos=model.get("position/W",{64,width},init::normal(.02));
  auto x=layer_norm(model,"embedding_norm",gather(table,tokens)+pos);
  if(factorized)
    x=dense(model,"embedding_projection",x,24);
  AttentionOptions options{.need_weights=true};
  if(lengths.defined())
    options.mask=less(arange(0,64,1,DType::Int64),lengths.reshape({batch,1,1,1}));
  std::vector<Tensor> weights;
  for(int layer=0;layer<layers;++layer) {
    const std::string name=factorized?"shared_block":"block"+std::to_string(layer);
    const auto matrix=[&](const std::string &key) {
      return model.get(name+"/attention/"+key,{24,24},init::uniform(std::sqrt(3./24.)));
    };
    const auto heads=[&](const std::string &key){
      auto projected=matmul(x,matrix(key));
      if(method==CharacterBertAdaptation::Lora && (key=="Wq" || key=="Wv")) {
        const auto prefix=name+"/attention/"+key;
        const auto a=adaptation->get(prefix+"/A",{24,4},init::normal(.02));
        const auto b=adaptation->get(prefix+"/B",{4,24},init::zeros());
        projected=projected+matmul(matmul(x,a),b);
      }
      return projected.reshape({batch,64,3,8}).transpose({0,2,1,3});
    };
    const auto a=scaled_dot_product_attention(heads("Wq"),heads("Wk"),heads("Wv"),options);
    weights.push_back(a.weights);
    const auto merged=a.output.transpose({0,2,1,3}).reshape({batch,64,24});
    const auto context=layer_norm(model,name+"/attention_norm",x+matmul(merged,matrix("Wo")));
    const auto expanded=gelu(dense(model,name+"/expand",context,48),true);
    x=layer_norm(model,name+"/output_norm",context+dense(model,name+"/contract",expanded,24));
    if(method==CharacterBertAdaptation::Adapter) {
      const auto down=gelu(dense(*adaptation,name+"/adapter/down",x,4),true);
      const auto prefix=name+"/adapter/up";
      const auto w=adaptation->get(prefix+"/W",{4,24},init::zeros());
      const auto b=adaptation->get(prefix+"/b",{24},init::zeros());
      x=x+matmul(down,w)+b;
    }
  }
  return {table,x,weights};
}

/// @brief Encode a masked character sequence and score it with tied token embeddings.
///
/// The encoder width is 24 with three bidirectional attention heads and
/// a 48-unit feed-forward layer per block. The head shares the first 65
/// embedding rows with the input table. MASK is input-only.
///
/// @param model Owns the encoder and tied-head tensors.
/// @param tokens Integral [N,64] input IDs in [0,66).
/// @param layers Number of blocks, one or two, default two.
/// @param factorized Use eight-dimensional embeddings and share block parameters when true.
/// @return Logits [N,64,65], hidden [N,64,24], and one [N,3,64,64] attention tensor per block.
/// @throws std::invalid_argument For invalid input or block count.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto path = dm::nn::character_mlm(model,dm::nn::zeros({2,64},dm::nn::DType::Int64));
/// @endcode
/// @par References
/// - Devlin et al. (2019), https://aclanthology.org/N19-1423/
/// @see corrupt_mlm_characters()
inline std::tuple<Tensor,Tensor,std::vector<Tensor>> character_mlm(
    ParameterStore &model,const Tensor &tokens,int layers=2,bool factorized=false) {
  const auto [table,x,weights]=character_bert_encoder(model,tokens,layers,factorized);
  const auto width=table.shape()[1];
  const auto out=layer_norm(model,"mlm/norm",gelu(dense(model,"mlm/transform",x,width),true));
  const auto bias=model.get("mlm/bias",{65},init::zeros());
  return {matmul(out,slice_axis(table,0,0,65).transpose())+bias,x,weights};
}


/// @brief Compute the character DistilBERT penalty against a fixed teacher.
///
/// The penalty is four times the mean KL divergence at temperature two
/// plus one quarter of the mean cosine distance between hidden rows.
/// Add it to the student's supervised masked-language loss.
///
/// @param student_logits Student class scores [N,T,C].
/// @param teacher_logits Floating teacher scores with the same shape.
/// @param student_hidden Student hidden rows [N,T,D].
/// @param teacher_hidden Floating teacher rows with the same shape.
/// @return Scalar loss with gradients only through the student.
/// @throws std::invalid_argument For nonmatching, empty, or nonfloating inputs.
/// @par Example
/// @code
/// const auto scores = dm::nn::zeros({1,2,3});
/// const auto hidden = dm::nn::ones({1,2,4});
/// const auto penalty = dm::nn::character_distillation_loss(scores,scores,hidden,hidden);
/// @endcode
/// @par References
/// - Sanh et al. (2019), https://arxiv.org/abs/1910.01108
/// @see character_mlm()
inline Tensor character_distillation_loss(const Tensor &student_logits, const Tensor &teacher_logits,
    const Tensor &student_hidden, const Tensor &teacher_hidden) {
  if (student_logits.dim()!=3 || student_hidden.dim()!=3 ||
      student_logits.shape()!=teacher_logits.shape() ||
      student_hidden.shape()!=teacher_hidden.shape() ||
      student_logits.shape()[0]!=student_hidden.shape()[0] ||
      student_logits.shape()[1]!=student_hidden.shape()[1] ||
      student_logits.numel()==0 || student_hidden.numel()==0 ||
      !is_floating(student_logits.dtype()) || !is_floating(teacher_logits.dtype()) ||
      !is_floating(student_hidden.dtype()) || !is_floating(teacher_hidden.dtype()))
    throw std::invalid_argument("character_distillation_loss: incompatible teacher and student");
  const auto teacher_log=log_softmax(teacher_logits.detach()/2.,2);
  const auto student_log=log_softmax(student_logits/2.,2);
  const auto kl=mean(sum(exp(teacher_log)*(teacher_log-student_log),{2}));
  const auto unit=[](const Tensor &x){return x*rsqrt(sum(square(x),{2},true)+1e-8);};
  const auto cosine=mean(sum(unit(student_hidden)*unit(teacher_hidden.detach()),{2}));
  return 4.*kl+.25*(1.-cosine);
}


/// @brief Mean-pool valid character states and normalize for cosine retrieval.
///
/// @param model Owns the shared character BERT encoder.
/// @param tokens Integral character IDs [N,64].
/// @param lengths Integral valid counts [N] in [1,64].
/// @return Differentiable embeddings [N,24] independent of padded tokens.
/// @throws std::invalid_argument For incompatible inputs.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto z = dm::nn::character_sentence_embedding(model,
///     dm::nn::zeros({2,64},dm::nn::DType::Int64), dm::nn::ones({2},dm::nn::DType::Int64));
/// @endcode
/// @par References
/// - Reimers and Gurevych (2019), https://aclanthology.org/D19-1410/
/// @see character_bert_encoder()
inline Tensor character_sentence_embedding(ParameterStore &model,const Tensor &tokens,
                                            const Tensor &lengths) {
  const auto [table,hidden,attention]=character_bert_encoder(model,tokens,2,false,lengths);
  const auto valid=less(arange(0,64,1,DType::Int64),lengths.reshape({-1,1}));
  const auto pooled=sum(where(expand_dims(valid,2),hidden,0.),{1}) /
                    lengths.to(hidden.dtype()).reshape({-1,1});
  return pooled/sqrt(sum(square(pooled),{1},true)+1e-12);
}


/// @brief Select the supervised prediction head on the character encoder.
enum class CharacterBertTask {
  Classification, ///< Mean-pooled binary classification.
  BioTagging,     ///< Five BIO tags at every position.
  ExtractiveQa    ///< Start and end scores over valid context positions.
};

/// @brief Apply a supervised head to padding-aware character BERT states.
///
/// @param model Owns both the pretrained encoder and task head parameters.
/// @param tokens Integral [N,64] character IDs.
/// @param lengths Integral [N] valid lengths.
/// @param task The task to evaluate, with QA context starting at position 16.
/// @param method Optional encoder adaptation, default None.
/// @param adaptation Independent store for the adaptation and task head.
/// @return Classification [N,2], tagging [N,64,5], or QA [N,2,64] logits.
/// @throws std::invalid_argument For invalid encoder inputs, task or empty QA context.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore model(7);
/// auto z=n::character_bert_task_logits(model,n::zeros({1,64},n::DType::Int64),
///     n::full({1},24,n::DType::Int64),n::CharacterBertTask::ExtractiveQa);
/// @endcode
/// @see character_bert_encoder()
inline Tensor character_bert_task_logits(ParameterStore &model,const Tensor &tokens,
    const Tensor &lengths,CharacterBertTask task,
    CharacterBertAdaptation method=CharacterBertAdaptation::None, ParameterStore *adaptation=nullptr) {
  const auto [table,hidden,attention]=character_bert_encoder(model,tokens,2,false,lengths,method,adaptation);
  auto &trainable=method==CharacterBertAdaptation::None?model:*adaptation;
  if(task==CharacterBertTask::BioTagging)
    return dense(trainable,"ner",hidden,5);
  const auto positions=arange(0,64,1,DType::Int64);
  const auto valid=less(positions,lengths.reshape({-1,1}));
  if(task==CharacterBertTask::Classification) {
    const auto mean=sum(where(expand_dims(valid,2),hidden,0.),{1}) /
        lengths.to(hidden.dtype()).reshape({-1,1});
    return dense(trainable,"classification",mean,2);
  }
  if(task!=CharacterBertTask::ExtractiveQa || min(lengths).item()<=16)
    throw std::invalid_argument("character_bert_task_logits: invalid task or QA span");
  const auto context=logical_and(valid,greater_equal(positions,16.));
  return where(expand_dims(context,2),dense(trainable,"qa",hidden,2),-1e9).transpose({0,2,1});
}

}  // namespace datamunge::nn
