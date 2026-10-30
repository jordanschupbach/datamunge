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
#include "datamunge/nn/attention_variants.hpp"
#include "datamunge/nn/audio.hpp"
#include "datamunge/nn/augment.hpp"
#include "datamunge/nn/autograd.hpp"
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
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/kernel.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/memory.hpp"
#include "datamunge/nn/model_plot.hpp"
#include "datamunge/nn/metrics.hpp"
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
#include "datamunge/nn/polynomial.hpp"
#include "datamunge/nn/validation.hpp"
#include "datamunge/nn/glm.hpp"
#include "datamunge/nn/gaussian_mixture.hpp"
#include "datamunge/nn/naive_bayes.hpp"
#include "datamunge/nn/classical_inference.hpp"

#include "datamunge/nn/convolution_components.hpp"
#include "datamunge/nn/vision_plots.hpp"

#include "datamunge/nn/cnn_models.hpp"

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
