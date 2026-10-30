#pragma once
/// @file guided_backprop.hpp
/// @brief Explicit positive-positive guided visualization gate for a frozen classifier.
#include "datamunge/nn/saliency_models.hpp"
#include <atomic>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Apply the guided-ReLU positive-activation and positive-adjoint gate.
///
/// Tensor-only visualization rule. It discards negative incoming evidence and
/// is not a linear VJP,so it is NOT ordinary differentiation of forward ReLU.
///
/// @param activation Forward pre-ReLU tensor.
/// @param incoming Matching upstream adjoint.
/// @return incoming*1[activation>0]*1[incoming>0],zero at either boundary.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto g = m::GuidedReluBackward(n::ones({2}), n::ones({2}));
/// @endcode
/// @par References
/// - Springenberg,Dosovitskiy,Brox and Riedmiller(2014/2015),Striving for Simplicity:The All Convolutional Net, https://arxiv.org/abs/1412.6806. Guided backprop retains only positive incoming adjoints at positive forward activations. This modified visualization rule is NOT the derivative of the unchanged forward classifier.
/// @see GuidedRelu(),SaliencyCnn()
inline Tensor GuidedReluBackward(const Tensor &activation, const Tensor &incoming) {
  return nn::mul(nn::mul(incoming, nn::cast(nn::greater(activation, 0.0), incoming.dtype())),
                 nn::cast(nn::greater(incoming, 0.0), incoming.dtype()));
}

/// @brief Inspect the atomic count of guided-ReLU applications in this process.
///
/// Each GuidedRelu call increments once. Diagnostic metadata only,no tensor
/// or weight mutation. Concurrent increments are atomic;counts are process-global.
///
/// @return Reference to process-local atomic counter,initially0.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto count = m::GuidedReluCount().load();
/// @endcode
/// @par References
/// - Springenberg,Dosovitskiy,Brox and Riedmiller(2014/2015),Striving for Simplicity:The All Convolutional Net, https://arxiv.org/abs/1412.6806. Guided backprop retains only positive incoming adjoints at positive forward activations. This modified visualization rule is NOT the derivative of the unchanged forward classifier.
/// @see GuidedRelu(),SaliencyCnn()
inline std::atomic<long> &GuidedReluCount() {
  /// @brief Process-local diagnostic call counter;atomic metadata only.
  static std::atomic<long> count{0};
  return count;
}

/// @brief Access the immutable custom guided-ReLU forward/backward function.
///
/// Static initialization is thread-safe. Pure tensor forward/backward callbacks
/// retain caller storage through the autograd graph;no fitted state.
///
/// @return Shared process-local function with ordinary ReLU values and guided adjoints.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::GuidedReluFunction()(n::ones({2}));
/// @endcode
/// @par References
/// - Springenberg,Dosovitskiy,Brox and Riedmiller(2014/2015),Striving for Simplicity:The All Convolutional Net, https://arxiv.org/abs/1412.6806. Guided backprop retains only positive incoming adjoints at positive forward activations. This modified visualization rule is NOT the derivative of the unchanged forward classifier.
/// @see GuidedRelu(),SaliencyCnn()
inline const nn::Function &GuidedReluFunction() {
  /// @brief Immutable ordinary-forward/guided-backward custom function.
  static const nn::Function f(
      "TflowGuidedRelu",
      [](const std::vector<Tensor> &in) {
        return std::vector<Tensor>{nn::relu(in[0])};
      },
      [](const std::vector<Tensor> &in, const std::vector<Tensor> &,
         const std::vector<Tensor> &g) {
        return std::vector<Tensor>{GuidedReluBackward(in[0], g[0])};
      });
  return f;
}

/// @brief Apply ordinary ReLU values with the explicitly modified guided backward rule.
///
/// Increments the process diagnostic counter;learned weights/input unchanged.
/// Incoming negative adjoints are suppressed even at positive x. The backward
/// rule is nonlinear in incoming adjoints and is not a true forward derivative.
///
/// @param x Floating pre-activation tensor.
/// @return ReLU(x)with guided positive-positive incoming gate on reverse traversal.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::GuidedRelu(n::ones({2}));
/// @endcode
/// @par References
/// - Springenberg,Dosovitskiy,Brox and Riedmiller(2014/2015),Striving for Simplicity:The All Convolutional Net, https://arxiv.org/abs/1412.6806. Guided backprop retains only positive incoming adjoints at positive forward activations. This modified visualization rule is NOT the derivative of the unchanged forward classifier.
/// @see GuidedRelu(),SaliencyCnn()
inline Tensor GuidedRelu(const Tensor &x) {
  ++GuidedReluCount();
  return GuidedReluFunction()(x);
}

/// @brief Evaluate the shared digit classifier with all three guided-ReLU gates.
///
/// Restore the exact same ordinary classifier weights;do not fit with this
/// modified visualization rule. Conv1,conv2 and hidden all replace ReLU.
/// No optimizer update;diagnostic guided call count increases by3.
///
/// @param p Mutable learned store,matching the ordinary shared eight-leaf classifier.
/// @param image Float32[N,28,28,1]original normalized grayscale input.
/// @return Same forward endpoints/logits as SaliencyCnn,with modified guided adjoints.
/// @throws std::invalid_argument For invalid input/stored parameter geometry or dtype.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(1103);
/// const auto f = m::GuidedSaliencyCnn(p, n::zeros({1, 28, 28, 1}));
/// @endcode
/// @par References
/// - Springenberg,Dosovitskiy,Brox and Riedmiller(2014/2015),Striving for Simplicity:The All Convolutional Net, https://arxiv.org/abs/1412.6806. Guided backprop retains only positive incoming adjoints at positive forward activations. This modified visualization rule is NOT the derivative of the unchanged forward classifier.
/// @see GuidedRelu(),SaliencyCnn()
inline SaliencyCnnOutput GuidedSaliencyCnn(nn::ParameterStore &p, const Tensor &image) {
  return SaliencyCnn(p, image, GuidedRelu);
}

} // namespace datamunge::nn::models
