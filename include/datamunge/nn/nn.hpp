#pragma once
#include "datamunge/nn/array_io.hpp"

/// @file nn.hpp
/// @brief Umbrella header for `datamunge::nn`: tensors, autograd, ops,
///        convolution, attention, linear algebra, FFT and audio, CTC and CRF, boxes and NMS,
///        layers, losses, optimizers, training utilities, image ops and
///        augmentation, and data I/O.

#include "datamunge/nn/activations.hpp"
#include "datamunge/nn/adaptive_variants.hpp"
#include "datamunge/nn/advanced_variants.hpp"
#include "datamunge/nn/attention.hpp"
#include "datamunge/nn/attention_rollout.hpp"
#include "datamunge/nn/linear_counterfactual.hpp"
#include "datamunge/nn/group_fairness.hpp"
#include "datamunge/nn/kernel_shap.hpp"
#include "datamunge/nn/local_linear_surrogate.hpp"
#include "datamunge/nn/tcav.hpp"
#include "datamunge/nn/ood_scores.hpp"
#include "datamunge/nn/federated_average.hpp"
#include "datamunge/nn/randomized_smoothing.hpp"
#include "datamunge/nn/ridge_influence.hpp"
#include "datamunge/nn/recommendation_metrics.hpp"
#include "datamunge/nn/factorization_machine.hpp"
#include "datamunge/nn/explicit_factorization.hpp"
#include "datamunge/nn/implicit_feedback.hpp"
#include "datamunge/nn/retention.hpp"
#include "datamunge/nn/hyena.hpp"
#include "datamunge/nn/s4.hpp"
#include "datamunge/nn/selective_scan.hpp"
#include "datamunge/nn/rwkv.hpp"
#include "datamunge/nn/causal_relative_attention.hpp"
#include "datamunge/nn/attention_variants.hpp"
#include "datamunge/nn/audio.hpp"
#include "datamunge/nn/beat_tracking.hpp"
#include "datamunge/nn/conv_tasnet.hpp"
#include "datamunge/nn/deepspeech_ctc.hpp"
#include "datamunge/nn/listen_attend_spell.hpp"
#include "datamunge/nn/tacotron2.hpp"
#include "datamunge/nn/tabular_image_fusion.hpp"
#include "datamunge/nn/audio_visual_speech.hpp"
#include "datamunge/nn/digit_backbone.hpp"
#include "datamunge/nn/continual_learning.hpp"
#include "datamunge/nn/progressive_column.hpp"
#include "datamunge/nn/expected_improvement.hpp"
#include "datamunge/nn/halving_search.hpp"
#include "datamunge/nn/population_training.hpp"
#include "datamunge/nn/neat.hpp"
#include "datamunge/nn/darts_regressor.hpp"
#include "datamunge/nn/enas_controller.hpp"
#include "datamunge/nn/sinusoid_hypernetwork.hpp"
#include "datamunge/nn/learned_optimizer.hpp"
#include "datamunge/nn/episodic_classifier.hpp"
#include "datamunge/nn/stacked_vqa.hpp"
#include "datamunge/nn/image_captioner.hpp"
#include "datamunge/nn/rnn_transducer.hpp"
#include "datamunge/nn/audio_tagger.hpp"
#include "datamunge/nn/mfcc_summary.hpp"
#include "datamunge/nn/xvector.hpp"
#include "datamunge/nn/wav2vec2.hpp"
#include "datamunge/nn/augment.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/autoencoder_models.hpp"
#include "datamunge/nn/competitive_autoencoders.hpp"
#include "datamunge/nn/masked_autoencoder.hpp"
#include "datamunge/nn/patch_restoration.hpp"
#include "datamunge/nn/encoder_pretraining.hpp"
#include "datamunge/nn/latent_arithmetic.hpp"
#include "datamunge/nn/latent_visualization.hpp"
#include "datamunge/nn/learned_image_codec.hpp"
#include "datamunge/nn/reconstruction_anomaly.hpp"
#include "datamunge/nn/gan_models.hpp"
#include "datamunge/nn/auxiliary_classifier_gan.hpp"
#include "datamunge/nn/gan_data.hpp"
#include "datamunge/nn/bidirectional_gan.hpp"
#include "datamunge/nn/dense_gan.hpp"
#include "datamunge/nn/conditional_gan.hpp"
#include "datamunge/nn/info_gan.hpp"
#include "datamunge/nn/spectral_gan.hpp"
#include "datamunge/nn/self_attention_gan.hpp"
#include "datamunge/nn/big_gan.hpp"
#include "datamunge/nn/dcgan.hpp"
#include "datamunge/nn/midpoint_uniform.hpp"
#include "datamunge/nn/wasserstein_gan.hpp"
#include "datamunge/nn/progressive_gan.hpp"
#include "datamunge/nn/quantized_sobel.hpp"
#include "datamunge/nn/pix2pix.hpp"
#include "datamunge/nn/cycle_domains.hpp"
#include "datamunge/nn/cycle_gan.hpp"
#include "datamunge/nn/modulated_conv.hpp"
#include "datamunge/nn/spatial_instance_norm.hpp"
#include "datamunge/nn/style_gan.hpp"
#include "datamunge/nn/sequence_autoencoders.hpp"
#include "datamunge/nn/activation_sparse_autoencoder.hpp"
#include "datamunge/nn/layered_autoencoders.hpp"
#include "datamunge/nn/tied_autoencoder.hpp"
#include "datamunge/nn/factor_lattice.hpp"
#include "datamunge/nn/adversarial_autoencoder.hpp"
#include "datamunge/nn/batch_statistics.hpp"
#include "datamunge/nn/boxes.hpp"
#include "datamunge/nn/build_info.hpp"
#include "datamunge/nn/checkpoint.hpp"
#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/conv_analysis.hpp"
#include "datamunge/nn/convert.hpp"
#include "datamunge/nn/datasets.hpp"
#include "datamunge/nn/derivatives.hpp"
#include "datamunge/nn/differential_privacy.hpp"
#include "datamunge/nn/dropout_variants.hpp"
#include "datamunge/nn/dtype.hpp"
#include "datamunge/nn/einsum.hpp"
#include "datamunge/nn/fft.hpp"
#include "datamunge/nn/gradcheck.hpp"
#include "datamunge/nn/gradient_tricks.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/image_explainers.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/frozen_graph.hpp"
#include "datamunge/nn/saved_model.hpp"
#include "datamunge/nn/saved_model_http.hpp"
#include "datamunge/nn/graph_session.hpp"
#include "datamunge/nn/lite_model.hpp"
#include "datamunge/nn/low_rank_dense.hpp"
#include "datamunge/nn/hashed_dense.hpp"
#include "datamunge/nn/channel_pruning.hpp"
#include "datamunge/nn/magnitude_pruning.hpp"
#include "datamunge/nn/int8_dense.hpp"
#include "datamunge/nn/cuda_half_matmul.hpp"
#include "datamunge/nn/distillation.hpp"
#include "datamunge/nn/inference_benchmark.hpp"
#include "datamunge/nn/kernel.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/recurrent_forecast.hpp"
#include "datamunge/nn/forecast_evaluation.hpp"
#include "datamunge/nn/forecast_baselines.hpp"
#include "datamunge/nn/forecast_losses.hpp"
#include "datamunge/nn/dlinear.hpp"
#include "datamunge/nn/change_point.hpp"
#include "datamunge/nn/series_anomaly.hpp"
#include "datamunge/nn/graph_forecast.hpp"
#include "datamunge/nn/differentiable_kalman.hpp"
#include "datamunge/nn/nbeats.hpp"
#include "datamunge/nn/nhits.hpp"
#include "datamunge/nn/patchtst.hpp"
#include "datamunge/nn/seq2seq_forecast.hpp"
#include "datamunge/nn/deepar.hpp"
#include "datamunge/nn/irregular_ode.hpp"
#include "datamunge/nn/autoformer_forecast.hpp"
#include "datamunge/nn/informer_forecast.hpp"
#include "datamunge/nn/series_classification.hpp"
#include "datamunge/nn/probabilistic_regression.hpp"
#include "datamunge/nn/langevin.hpp"
#include "datamunge/nn/weight_gaussian.hpp"
#include "datamunge/nn/beam_search.hpp"
#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/memory.hpp"
#include "datamunge/nn/model_plot.hpp"
#include "datamunge/nn/metrics.hpp"
#include "datamunge/nn/calibration.hpp"
#include "datamunge/nn/mixed_precision.hpp"
#include "datamunge/nn/normalization_variants.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/nn/optimizer_analysis.hpp"
#include "datamunge/nn/parallel.hpp"
#include "datamunge/nn/params.hpp"
#include "datamunge/nn/pooling_variants.hpp"
#include "datamunge/nn/positional_variants.hpp"
#include "datamunge/nn/profiler.hpp"
#include "datamunge/nn/random.hpp"
#include "datamunge/nn/recurrent_variants.hpp"
#include "datamunge/nn/recurrent_components.hpp"
#include "datamunge/nn/recurrent_data.hpp"
#include "datamunge/nn/reservoir.hpp"
#include "datamunge/nn/spiking.hpp"
#include "datamunge/nn/neural_tangent_kernel.hpp"
#include "datamunge/nn/poincare.hpp"
#include "datamunge/nn/spline_edges.hpp"
#include "datamunge/nn/soft_circle_raster.hpp"
#include "datamunge/nn/cellular_automata.hpp"
#include "datamunge/nn/program_induction.hpp"
#include "datamunge/nn/volume_rendering.hpp"
#include "datamunge/nn/equivariant.hpp"
#include "datamunge/nn/schedules.hpp"
#include "datamunge/nn/sequence.hpp"
#include "datamunge/nn/tape.hpp"
#include "datamunge/nn/tensor.hpp"
#include "datamunge/nn/training.hpp"
#include "datamunge/nn/variance_scaling.hpp"
#include "datamunge/nn/views.hpp"
#include "datamunge/nn/weight_reparam.hpp"
#include "datamunge/nn/word_embeddings.hpp"

// Data preprocessing and portable record formats.
#include "datamunge/nn/binary.hpp"
#include "datamunge/nn/cifar.hpp"
#include "datamunge/nn/csv.hpp"
#include "datamunge/nn/tokenization.hpp"
#include "datamunge/nn/tfrecord.hpp"
#include "datamunge/nn/artifacts.hpp"
#include "datamunge/nn/linear.hpp"
#include "datamunge/nn/gaussian_conditioning.hpp"
#include "datamunge/nn/polynomial.hpp"
#include "datamunge/nn/validation.hpp"
#include "datamunge/nn/glm.hpp"
#include "datamunge/nn/gaussian_mixture.hpp"
#include "datamunge/nn/naive_bayes.hpp"
#include "datamunge/nn/classical_inference.hpp"
#include <datamunge/nn/classical_components.hpp>

#include "datamunge/nn/convolution_components.hpp"
#include "datamunge/nn/vision_plots.hpp"

#include "datamunge/nn/cnn_models.hpp"
#include "datamunge/nn/bar_images.hpp"
#include "datamunge/nn/small_image_models.hpp"
#include "datamunge/nn/zplus_relevance.hpp"

#include "datamunge/nn/classic_cnn_models.hpp"

#include "datamunge/nn/classic_cnn_components.hpp"
#include "datamunge/nn/alexnet_googlenet.hpp"

#include "datamunge/nn/vision_architecture_components.hpp"
#include "datamunge/nn/zfnet_xception.hpp"

#include "datamunge/nn/inception_models.hpp"

#include "datamunge/nn/resnet_models.hpp"

#include "datamunge/nn/preactivation_densenet_models.hpp"

#include "datamunge/nn/highway_models.hpp"

#include "datamunge/nn/dual_path_models.hpp"

#include "datamunge/nn/grouped_wide_residual_models.hpp"

#include "datamunge/nn/pyramidal_multiscale_models.hpp"

#include "datamunge/nn/stochastic_residual_models.hpp"

#include "datamunge/nn/reversible_residual_models.hpp"

#include "datamunge/nn/mobilenet_models.hpp"

#include "datamunge/nn/squeezenet_models.hpp"

#include "datamunge/nn/shufflenet_models.hpp"

#include "datamunge/nn/ghostnet_models.hpp"

#include "datamunge/nn/regnet_models.hpp"

#include "datamunge/nn/condensenet_models.hpp"

#include "datamunge/nn/mbconv_models.hpp"

#include "datamunge/nn/deformable_models.hpp"

#include "datamunge/nn/attention_cnn_models.hpp"

#include "datamunge/nn/convnext_models.hpp"

#include "datamunge/nn/repvgg_models.hpp"

#include "datamunge/nn/nonlocal_models.hpp"

#include "datamunge/nn/octave_models.hpp"

#include "datamunge/nn/spatial_transformer_models.hpp"

#include "datamunge/nn/capsule_models.hpp"

#include "datamunge/nn/detection_components.hpp"

#include "datamunge/nn/detection_evaluation.hpp"

#include "datamunge/nn/detection_scene_models.hpp"

#include "datamunge/nn/detection_inference.hpp"

#include "datamunge/nn/two_stage_detection_models.hpp"

#include "datamunge/nn/feature_pyramid_models.hpp"

#include "datamunge/nn/ssd_models.hpp"

#include "datamunge/nn/retina_detection_models.hpp"

#include "datamunge/nn/fcos_detection_models.hpp"

#include "datamunge/nn/center_detection_models.hpp"

#include "datamunge/nn/bifpn_models.hpp"
#include "datamunge/nn/efficientdet_models.hpp"

#include "datamunge/nn/tensor_assignment.hpp"

#include "datamunge/nn/detr_models.hpp"

#include "datamunge/nn/detection_augmentation.hpp"

#include "datamunge/nn/yolo_v1_models.hpp"

#include "datamunge/nn/word_tree_models.hpp"

#include "datamunge/nn/yolo_v2_models.hpp"

#include "datamunge/nn/yolo_v3_models.hpp"

#include "datamunge/nn/detection_mosaic.hpp"

#include "datamunge/nn/yolo_v5_models.hpp"

#include "datamunge/nn/segmentation_losses.hpp"

#include "datamunge/nn/segmentation_data.hpp"

#include "datamunge/nn/unet_models.hpp"

#include "datamunge/nn/fcn_models.hpp"

#include "datamunge/nn/segnet_models.hpp"

#include "datamunge/nn/unet_variant_models.hpp"

#include "datamunge/nn/deeplab_models.hpp"

#include "datamunge/nn/dense_crf_models.hpp"

#include "datamunge/nn/deeplab_v3_models.hpp"

#include "datamunge/nn/pyramid_pool_models.hpp"

#include "datamunge/nn/hrnet_models.hpp"

#include "datamunge/nn/instance_scene_models.hpp"

#include "datamunge/nn/mask_rcnn_models.hpp"

#include "datamunge/nn/yolact_models.hpp"

#include "datamunge/nn/panoptic_fpn_models.hpp"

#include "datamunge/nn/panoptic_models.hpp"

#include "datamunge/nn/panoptic_scene_models.hpp"

#include "datamunge/nn/volumetric_models.hpp"
#include "datamunge/nn/volumetric_data.hpp"

#include "datamunge/nn/siamese_models.hpp"

#include "datamunge/nn/metric_learning_data.hpp"

#include "datamunge/nn/facenet_models.hpp"

#include "datamunge/nn/metric_learning.hpp"

#include "datamunge/nn/angular_margin_models.hpp"

#include "datamunge/nn/bilinear_models.hpp"

#include "datamunge/nn/flower_data.hpp"

#include "datamunge/nn/retrieval_models.hpp"

#include "datamunge/nn/retrieval_data.hpp"

#include "datamunge/nn/retrieval_index.hpp"

#include "datamunge/nn/multilabel_data.hpp"

#include "datamunge/nn/multilabel_metrics.hpp"

#include "datamunge/nn/multilabel_models.hpp"

#include "datamunge/nn/inpainting_models.hpp"

#include "datamunge/nn/inpainting_data.hpp"

#include "datamunge/nn/colorization_data.hpp"
#include "datamunge/nn/colorization_models.hpp"

#include "datamunge/nn/dncnn_models.hpp"
#include "datamunge/nn/denoising_data.hpp"

#include "datamunge/nn/crnn_models.hpp"

#include "datamunge/nn/ocr_data.hpp"

#include "datamunge/nn/flow_components.hpp"
#include "datamunge/nn/flownet_models.hpp"
#include "datamunge/nn/flow_losses.hpp"

#include "datamunge/nn/flow_data.hpp"

#include "datamunge/nn/flow_visualization.hpp"

#include "datamunge/nn/pwcnet_models.hpp"

#include "datamunge/nn/depth_models.hpp"

#include "datamunge/nn/depth_data.hpp"

#include "datamunge/nn/super_resolution_models.hpp"

#include "datamunge/nn/super_resolution_data.hpp"

#include "datamunge/nn/perceptual_models.hpp"

#include "datamunge/nn/style_losses.hpp"

#include "datamunge/nn/johnson_models.hpp"

#include "datamunge/nn/johnson_losses.hpp"

#include "datamunge/nn/partial_unet_models.hpp"

#include "datamunge/nn/partial_inpainting_losses.hpp"

#include "datamunge/nn/partial_inpainting_data.hpp"

#include "datamunge/nn/hourglass_models.hpp"

#include "datamunge/nn/pose_data.hpp"

#include "datamunge/nn/pose_visualization.hpp"

#include "datamunge/nn/openpose_models.hpp"

#include "datamunge/nn/paf_data.hpp"

#include "datamunge/nn/paf_decode.hpp"

#include "datamunge/nn/paf_evaluation.hpp"

#include "datamunge/nn/vqa_models.hpp"

#include "datamunge/nn/vqa_data.hpp"

#include "datamunge/nn/vqa_visualization.hpp"

#include "datamunge/nn/sequence_classification.hpp"

#include "datamunge/nn/time_series_models.hpp"
#include "datamunge/nn/time_series_data.hpp"

#include "datamunge/nn/tcn_models.hpp"

#include "datamunge/nn/sensor_models.hpp"

#include "datamunge/nn/sensor_data.hpp"

#include "datamunge/nn/sensor_evaluation.hpp"

#include "datamunge/nn/kim_cnn_models.hpp"

#include "datamunge/nn/sentence_data.hpp"

#include "datamunge/nn/character_cnn_models.hpp"

#include "datamunge/nn/character_data.hpp"

#include "datamunge/nn/wavenet_stack_models.hpp"
#include "datamunge/nn/hifigan.hpp"
#include "datamunge/nn/waveglow.hpp"
#include "datamunge/nn/wavenet_vocoder.hpp"
#include "datamunge/nn/speech_denoising.hpp"
#include "datamunge/nn/wavenet_stack_data.hpp"

#include "datamunge/nn/point_geometry.hpp"

#include "datamunge/nn/c3d_models.hpp"
#include "datamunge/nn/video_data.hpp"

#include "datamunge/nn/factorized_video_models.hpp"

#include "datamunge/nn/i3d_models.hpp"

#include "datamunge/nn/medical_volume_models.hpp"
#include "datamunge/nn/medical_volume_data.hpp"

#include "datamunge/nn/point_models.hpp"

#include "datamunge/nn/point_data.hpp"

#include "datamunge/nn/slowfast_models.hpp"

#include "datamunge/nn/voxnet_models.hpp"

#include "datamunge/nn/two_stream_models.hpp"

#include "datamunge/nn/saliency_models.hpp"

#include "datamunge/nn/guided_backprop.hpp"

#include "datamunge/nn/grad_cam.hpp"

#include "datamunge/nn/integrated_gradients.hpp"

#include "datamunge/nn/smoothgrad.hpp"

#include "datamunge/nn/occlusion.hpp"

#include "datamunge/nn/dream.hpp"

#include "datamunge/nn/feature_visualization.hpp"

#include "datamunge/nn/visual_features.hpp"

#include <datamunge/nn/mobile_classifier.hpp>

#include "datamunge/nn/gaussian_autoencoder.hpp"
#include "datamunge/nn/vector_quantized_autoencoder.hpp"
#include "datamunge/nn/vector_quantized_autoencoder_2.hpp"
#include "datamunge/nn/wasserstein_autoencoder.hpp"
#include "datamunge/nn/vae_gan.hpp"
#include "datamunge/nn/latent_factor_diagnostics.hpp"

#include "datamunge/nn/conditional_autoencoder.hpp"

#include "datamunge/nn/convolutional_variational_autoencoder.hpp"

#include "datamunge/nn/total_correlation_autoencoder.hpp"

#include "datamunge/nn/flow_variational_autoencoder.hpp"

#include "datamunge/nn/gumbel_categorical_autoencoder.hpp"

#include "datamunge/nn/hierarchical_variational_autoencoder.hpp"

#include "datamunge/nn/importance_weighted_autoencoder.hpp"

#include "datamunge/nn/kernel_discrepancy.hpp"
#include "datamunge/nn/info_variational_autoencoder.hpp"

#include "datamunge/nn/ladder_variational_autoencoder.hpp"
#include "datamunge/nn/sequence_variational_autoencoder.hpp"
#include "datamunge/nn/posterior_collapse.hpp"
#include "datamunge/nn/sequence_vae_data.hpp"

#include "datamunge/nn/uniform_categorical.hpp"
#include "datamunge/nn/star_domains.hpp"
#include "datamunge/nn/star_gan.hpp"

#include "datamunge/nn/relativistic_gan.hpp"
#include "datamunge/nn/adversarial_super_resolution.hpp"
#include "datamunge/nn/single_image_gan.hpp"
#include "datamunge/nn/generative_metrics.hpp"
#include "datamunge/nn/autoregressive_pixels.hpp"
#include "datamunge/nn/gated_pixels.hpp"
#include "datamunge/nn/recurrent_pixels.hpp"
#include "datamunge/nn/made_density.hpp"

#include "datamunge/nn/logistic_pixels.hpp"

#include "datamunge/nn/wavenet_generation.hpp"
#include "datamunge/nn/normalizing_flows.hpp"
#include "datamunge/nn/nice_density.hpp"
#include "datamunge/nn/affine_flows.hpp"
#include "datamunge/nn/glow_density.hpp"
#include "datamunge/nn/maf_density.hpp"
#include "datamunge/nn/iaf_density.hpp"
#include "datamunge/nn/spline_density.hpp"
#include "datamunge/nn/planar_radial_density.hpp"
#include "datamunge/nn/cnf_density.hpp"
#include "datamunge/nn/flow_study.hpp"

#include "datamunge/nn/diffusion.hpp"
#include "datamunge/nn/diffusion_components.hpp"
#include <datamunge/nn/rbf.hpp>
#include <datamunge/nn/self_organizing_map.hpp>

#include "datamunge/nn/recurrent_extensions.hpp"
#include "datamunge/nn/memory_networks.hpp"

#include "datamunge/nn/autorec.hpp"

#include "datamunge/nn/energy_components.hpp"
#include "datamunge/nn/dnc.hpp"
#include "datamunge/nn/attention_components.hpp"

#include <datamunge/nn/attention_diagnostics.hpp>

#include <datamunge/nn/deep_q.hpp>

#include <datamunge/nn/actor_critic.hpp>

#include <datamunge/nn/continuous_control.hpp>

#include <datamunge/nn/rssm.hpp>
#include <datamunge/nn/model_based_control.hpp>

#include <datamunge/nn/pixel_q.hpp>

#include <datamunge/nn/autoregressive_components.hpp>

#include <datamunge/nn/decoder_cache.hpp>

#include <datamunge/nn/token_distribution.hpp>

#include <datamunge/nn/sequence_language_graphs.hpp>

#include <datamunge/nn/safetensors.hpp>
#include <datamunge/nn/coo.hpp>
#include <datamunge/nn/graph_convolution.hpp>
#include <datamunge/nn/graph_pooling.hpp>
#include <datamunge/nn/graph_walks.hpp>
#include <datamunge/nn/knowledge_graph.hpp>
#include <datamunge/nn/temporal_graph.hpp>
#include <datamunge/nn/molecular_graph.hpp>
#include <datamunge/nn/contrastive_models.hpp>
#include <datamunge/nn/image_pretext.hpp>
#include <datamunge/nn/extended_pointer.hpp>
#include <datamunge/nn/handwriting_ctc.hpp>
#include <datamunge/nn/pen_synthesis.hpp>
#include <datamunge/nn/character_mlm.hpp>
#include <datamunge/nn/grouped_query_attention.hpp>
#include <datamunge/nn/chunked_attention.hpp>
#include <datamunge/nn/switch_transformer.hpp>
#include <datamunge/nn/transformer_xl.hpp>
#include <datamunge/nn/mnist_vision_transformer.hpp>
#include <datamunge/nn/mnist_dino.hpp>
#include <datamunge/nn/heat_surrogate.hpp>
#include <datamunge/nn/deep_ritz.hpp>
#include <datamunge/nn/deeponet.hpp>
#include <datamunge/nn/fourier_operator.hpp>
#include <datamunge/nn/inverse_heat.hpp>
#include <datamunge/nn/scalar_neural_ode.hpp>
#include <datamunge/nn/hamiltonian_network.hpp>
#include <datamunge/nn/lagrangian_network.hpp>
#include <datamunge/nn/neural_sde.hpp>
#include <datamunge/nn/latent_ode.hpp>
#include <datamunge/nn/particle_network.hpp>
#include <datamunge/nn/pinn_equations.hpp>
#include <datamunge/nn/sindy_autoencoder.hpp>
#include <datamunge/nn/instruction_tuning.hpp>
#include <datamunge/nn/embedded_clustering.hpp>
#include <datamunge/nn/word_prediction.hpp>
#include <datamunge/nn/representation_evaluation.hpp>

#include <datamunge/nn/completion_graphs.hpp>
#include <datamunge/nn/published_bert.hpp>

#include "datamunge/nn/neural_process.hpp"

#include "datamunge/nn/cross_network.hpp"

#include "datamunge/nn/neural_additive.hpp"

#include "datamunge/nn/multi_gate_experts.hpp"

#include "datamunge/nn/neural_oblivious.hpp"

#include "datamunge/nn/survival_losses.hpp"

#include "datamunge/nn/ft_transformer.hpp"

#include "datamunge/nn/ranking_losses.hpp"

#include "datamunge/nn/tabnet.hpp"

#include "datamunge/nn/feature_preprocessing.hpp"
