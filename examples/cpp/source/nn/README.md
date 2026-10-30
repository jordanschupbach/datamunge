# CNN and autoencoder examples in C++

Experiment orchestration lives in each `.cpp` example. Dataset setup, epoch and
batch loops, optimizer updates, validation, checkpoint selection and result
reporting are visible in that source file. Shared `common/` headers provide
model components, numerical operations, data readers and small reusable helpers;
they no longer own end-to-end `Run`/`Train` harnesses. MNIST examples show the
workflow directly in `main`; examples with multiple experiment variants may
keep their workflow in a function defined in the same file, taking plain
parameters (a depth, a version) and defining the model inside. A generic harness
parameterized by model callbacks does not count, even when pasted into the file:
the R and Python examples must be able to mirror the loop line by line over the
SWIG-bound components (`examples/bindings/nn_components.i`). `just nn-audit`
reports every example's status (see `docs/nn-example-audit.md`).


This is the canonical home for section 1 (129 convolutional / vision examples)
and section 2 (36 autoencoder examples). Their numbered subdirectories and
executable names are preserved. The remaining sections stay in `examples/nn`.
Shared model, data, training and numerical-check headers currently remain in
`examples/nn/common`, which both CMake projects include.

```sh
just build-nn-sections          # build these 165 targets only
just run-nn convolution_im2col  # data-free numerical / derivative fixtures
just run-nn cnn_mnist           # quick training; downloads MNIST when missing
just run-nn undercomplete_autoencoder epochs=1
just run-nn-full cnn_mnist      # explicit full-training opt-in
```

These targets are also part of the normal `examples/cpp` CMake project:

```sh
cmake -S examples/cpp -B build/cpp-examples -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp-examples --target convolution_im2col -j4
./build/cpp-examples/convolution_im2col
```

Set `DATAMUNGE_BUILD_NN_EXAMPLES=OFF` at CMake configuration time to build only
the original general-purpose C++ examples. The legacy `examples/nn` CMake project
and `just run-nn` / `run-all-nn` commands still discover the full catalogue across
both locations. The cached runner checks include dependencies, checkpoint
prerequisites and a 180-second default deadline.

Python replicas call these exact programs' entry functions through the optional
SWIG extension. See [the Python workflow](../../../python/nn/README.md). Quick-mode
training is bounded and does not claim full convergence; asset paths and full
training behavior are unchanged from [the NN workflow](../../../nn/README.md).

## Catalogue

### 1.10_3d_and_video_convolutions

- [c3d](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/c3d.cpp)
- [factorized_video](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/factorized_video.cpp)
- [i3d](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/i3d.cpp)
- [medical_volume_classification](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/medical_volume_classification.cpp)
- [point_geometry](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_geometry.cpp)
- [point_models_checks](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_models_checks.cpp)
- [pointnet](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/pointnet.cpp)
- [pointnet_plus_plus](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/pointnet_plus_plus.cpp)
- [slowfast](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/slowfast.cpp)
- [two_stream](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/two_stream.cpp)
- [voxnet](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/voxnet.cpp)

### 1.11_interpretability

- [deepdream](1_convolutional_neural_networks/1.11_interpretability/deepdream.cpp)
- [feature_visualization](1_convolutional_neural_networks/1.11_interpretability/feature_visualization.cpp)
- [filter_activations](1_convolutional_neural_networks/1.11_interpretability/filter_activations.cpp)
- [grad_cam](1_convolutional_neural_networks/1.11_interpretability/grad_cam.cpp)
- [guided_backprop](1_convolutional_neural_networks/1.11_interpretability/guided_backprop.cpp)
- [integrated_gradients](1_convolutional_neural_networks/1.11_interpretability/integrated_gradients.cpp)
- [occlusion_sensitivity](1_convolutional_neural_networks/1.11_interpretability/occlusion_sensitivity.cpp)
- [smoothgrad](1_convolutional_neural_networks/1.11_interpretability/smoothgrad.cpp)
- [vanilla_saliency](1_convolutional_neural_networks/1.11_interpretability/vanilla_saliency.cpp)

### 1.1_fundamentals

- [batch_norm_ablation](1_convolutional_neural_networks/1.1_fundamentals/batch_norm_ablation.cpp)
- [cnn_cifar10](1_convolutional_neural_networks/1.1_fundamentals/cnn_cifar10.cpp)
- [cnn_mnist](1_convolutional_neural_networks/1.1_fundamentals/cnn_mnist.cpp)
- [convolution_im2col](1_convolutional_neural_networks/1.1_fundamentals/convolution_im2col.cpp)
- [depthwise_separable](1_convolutional_neural_networks/1.1_fundamentals/depthwise_separable.cpp)
- [dilated_convolutions](1_convolutional_neural_networks/1.1_fundamentals/dilated_convolutions.cpp)
- [global_pooling_vs_flatten](1_convolutional_neural_networks/1.1_fundamentals/global_pooling_vs_flatten.cpp)
- [lenet5](1_convolutional_neural_networks/1.1_fundamentals/lenet5.cpp)
- [receptive_fields](1_convolutional_neural_networks/1.1_fundamentals/receptive_fields.cpp)
- [transposed_convolutions](1_convolutional_neural_networks/1.1_fundamentals/transposed_convolutions.cpp)

### 1.2_classic_architectures

- [alexnet](1_convolutional_neural_networks/1.2_classic_architectures/alexnet.cpp)
- [googlenet](1_convolutional_neural_networks/1.2_classic_architectures/googlenet.cpp)
- [inception_v2_v3](1_convolutional_neural_networks/1.2_classic_architectures/inception_v2_v3.cpp)
- [inception_v4_resnet](1_convolutional_neural_networks/1.2_classic_architectures/inception_v4_resnet.cpp)
- [network_in_network](1_convolutional_neural_networks/1.2_classic_architectures/network_in_network.cpp)
- [vgg](1_convolutional_neural_networks/1.2_classic_architectures/vgg.cpp)
- [xception](1_convolutional_neural_networks/1.2_classic_architectures/xception.cpp)
- [zfnet](1_convolutional_neural_networks/1.2_classic_architectures/zfnet.cpp)

### 1.3_residual_and_dense_families

- [densenet](1_convolutional_neural_networks/1.3_residual_and_dense_families/densenet.cpp)
- [dpn](1_convolutional_neural_networks/1.3_residual_and_dense_families/dpn.cpp)
- [highway_networks](1_convolutional_neural_networks/1.3_residual_and_dense_families/highway_networks.cpp)
- [preactivation_resnet](1_convolutional_neural_networks/1.3_residual_and_dense_families/preactivation_resnet.cpp)
- [pyramidnet](1_convolutional_neural_networks/1.3_residual_and_dense_families/pyramidnet.cpp)
- [res2net](1_convolutional_neural_networks/1.3_residual_and_dense_families/res2net.cpp)
- [resnet_basic](1_convolutional_neural_networks/1.3_residual_and_dense_families/resnet_basic.cpp)
- [resnet_bottleneck](1_convolutional_neural_networks/1.3_residual_and_dense_families/resnet_bottleneck.cpp)
- [resnext](1_convolutional_neural_networks/1.3_residual_and_dense_families/resnext.cpp)
- [revnet](1_convolutional_neural_networks/1.3_residual_and_dense_families/revnet.cpp)
- [stochastic_depth](1_convolutional_neural_networks/1.3_residual_and_dense_families/stochastic_depth.cpp)
- [wide_resnet](1_convolutional_neural_networks/1.3_residual_and_dense_families/wide_resnet.cpp)

### 1.4_efficient_and_mobile_architectures

- [condensenet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/condensenet.cpp)
- [efficientnet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/efficientnet.cpp)
- [efficientnet_v2](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/efficientnet_v2.cpp)
- [ghostnet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/ghostnet.cpp)
- [mnasnet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mnasnet.cpp)
- [mobilenet_v1](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v1.cpp)
- [mobilenet_v2](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v2.cpp)
- [mobilenet_v3](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v3.cpp)
- [regnet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/regnet.cpp)
- [shufflenet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/shufflenet.cpp)
- [squeezenet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.cpp)

### 1.5_attention_and_modern_convnets

- [capsule_networks](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/capsule_networks.cpp)
- [cbam](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/cbam.cpp)
- [convnext](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/convnext.cpp)
- [convnext_v2](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/convnext_v2.cpp)
- [deformable_conv](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/deformable_conv.cpp)
- [eca_net](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/eca_net.cpp)
- [non_local](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/non_local.cpp)
- [octconv](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/octconv.cpp)
- [repvgg](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/repvgg.cpp)
- [senet](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/senet.cpp)
- [sknet](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/sknet.cpp)
- [spatial_transformer](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/spatial_transformer.cpp)

### 1.6_object_detection

- [centernet](1_convolutional_neural_networks/1.6_object_detection/centernet.cpp)
- [detection_baselines](1_convolutional_neural_networks/1.6_object_detection/detection_baselines.cpp)
- [detection_evaluation](1_convolutional_neural_networks/1.6_object_detection/detection_evaluation.cpp)
- [detr](1_convolutional_neural_networks/1.6_object_detection/detr.cpp)
- [efficientdet](1_convolutional_neural_networks/1.6_object_detection/efficientdet.cpp)
- [fast_rcnn](1_convolutional_neural_networks/1.6_object_detection/fast_rcnn.cpp)
- [faster_rcnn](1_convolutional_neural_networks/1.6_object_detection/faster_rcnn.cpp)
- [fcos](1_convolutional_neural_networks/1.6_object_detection/fcos.cpp)
- [fpn](1_convolutional_neural_networks/1.6_object_detection/fpn.cpp)
- [nms_matching](1_convolutional_neural_networks/1.6_object_detection/nms_matching.cpp)
- [rcnn](1_convolutional_neural_networks/1.6_object_detection/rcnn.cpp)
- [retinanet](1_convolutional_neural_networks/1.6_object_detection/retinanet.cpp)
- [ssd](1_convolutional_neural_networks/1.6_object_detection/ssd.cpp)
- [yolo_v1](1_convolutional_neural_networks/1.6_object_detection/yolo_v1.cpp)
- [yolo_v2](1_convolutional_neural_networks/1.6_object_detection/yolo_v2.cpp)
- [yolo_v3](1_convolutional_neural_networks/1.6_object_detection/yolo_v3.cpp)
- [yolo_v5_style](1_convolutional_neural_networks/1.6_object_detection/yolo_v5_style.cpp)

### 1.7_segmentation

- [deeplab_v1_v2](1_convolutional_neural_networks/1.7_segmentation/deeplab_v1_v2.cpp)
- [deeplab_v3_plus](1_convolutional_neural_networks/1.7_segmentation/deeplab_v3_plus.cpp)
- [fcn](1_convolutional_neural_networks/1.7_segmentation/fcn.cpp)
- [hrnet](1_convolutional_neural_networks/1.7_segmentation/hrnet.cpp)
- [mask_rcnn](1_convolutional_neural_networks/1.7_segmentation/mask_rcnn.cpp)
- [panoptic_fpn](1_convolutional_neural_networks/1.7_segmentation/panoptic_fpn.cpp)
- [pspnet](1_convolutional_neural_networks/1.7_segmentation/pspnet.cpp)
- [segmentation_losses](1_convolutional_neural_networks/1.7_segmentation/segmentation_losses.cpp)
- [segnet](1_convolutional_neural_networks/1.7_segmentation/segnet.cpp)
- [unet](1_convolutional_neural_networks/1.7_segmentation/unet.cpp)
- [unet_variants](1_convolutional_neural_networks/1.7_segmentation/unet_variants.cpp)
- [volumetric_segmentation](1_convolutional_neural_networks/1.7_segmentation/volumetric_segmentation.cpp)
- [yolact](1_convolutional_neural_networks/1.7_segmentation/yolact.cpp)

### 1.8_other_vision_tasks

- [angular_margin](1_convolutional_neural_networks/1.8_other_vision_tasks/angular_margin.cpp)
- [bilinear_cnn](1_convolutional_neural_networks/1.8_other_vision_tasks/bilinear_cnn.cpp)
- [context_encoder](1_convolutional_neural_networks/1.8_other_vision_tasks/context_encoder.cpp)
- [crnn_ctc](1_convolutional_neural_networks/1.8_other_vision_tasks/crnn_ctc.cpp)
- [dncnn](1_convolutional_neural_networks/1.8_other_vision_tasks/dncnn.cpp)
- [facenet_triplet](1_convolutional_neural_networks/1.8_other_vision_tasks/facenet_triplet.cpp)
- [flownets](1_convolutional_neural_networks/1.8_other_vision_tasks/flownets.cpp)
- [gatys_style_transfer](1_convolutional_neural_networks/1.8_other_vision_tasks/gatys_style_transfer.cpp)
- [image_colorization](1_convolutional_neural_networks/1.8_other_vision_tasks/image_colorization.cpp)
- [image_retrieval](1_convolutional_neural_networks/1.8_other_vision_tasks/image_retrieval.cpp)
- [johnson_fast_style](1_convolutional_neural_networks/1.8_other_vision_tasks/johnson_fast_style.cpp)
- [monocular_depth](1_convolutional_neural_networks/1.8_other_vision_tasks/monocular_depth.cpp)
- [multilabel_classification](1_convolutional_neural_networks/1.8_other_vision_tasks/multilabel_classification.cpp)
- [openpose_pafs](1_convolutional_neural_networks/1.8_other_vision_tasks/openpose_pafs.cpp)
- [partial_inpainting](1_convolutional_neural_networks/1.8_other_vision_tasks/partial_inpainting.cpp)
- [pwcnet](1_convolutional_neural_networks/1.8_other_vision_tasks/pwcnet.cpp)
- [siamese](1_convolutional_neural_networks/1.8_other_vision_tasks/siamese.cpp)
- [stacked_hourglass](1_convolutional_neural_networks/1.8_other_vision_tasks/stacked_hourglass.cpp)
- [super_resolution](1_convolutional_neural_networks/1.8_other_vision_tasks/super_resolution.cpp)
- [vqa_baseline](1_convolutional_neural_networks/1.8_other_vision_tasks/vqa_baseline.cpp)

### 1.9_1d_convolutions

- [character_cnn](1_convolutional_neural_networks/1.9_1d_convolutions/character_cnn.cpp)
- [ecg_sensor_cnn](1_convolutional_neural_networks/1.9_1d_convolutions/ecg_sensor_cnn.cpp)
- [kim_sentence_cnn](1_convolutional_neural_networks/1.9_1d_convolutions/kim_sentence_cnn.cpp)
- [tcn](1_convolutional_neural_networks/1.9_1d_convolutions/tcn.cpp)
- [time_series_cnn](1_convolutional_neural_networks/1.9_1d_convolutions/time_series_cnn.cpp)
- [wavenet_stack](1_convolutional_neural_networks/1.9_1d_convolutions/wavenet_stack.cpp)

### 2.1_deterministic_autoencoders

- [contractive_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/contractive_autoencoder.cpp)
- [convolutional_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/convolutional_autoencoder.cpp)
- [denoising_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/denoising_autoencoder.cpp)
- [k_sparse_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/k_sparse_autoencoder.cpp)
- [masked_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/masked_autoencoder.cpp)
- [recurrent_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/recurrent_autoencoder.cpp)
- [sparse_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/sparse_autoencoder.cpp)
- [stacked_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/stacked_autoencoder.cpp)
- [tied_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/tied_autoencoder.cpp)
- [undercomplete_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/undercomplete_autoencoder.cpp)
- [winner_take_all_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/winner_take_all_autoencoder.cpp)

### 2.2_variational_autoencoders

- [adversarial_autoencoder](2_autoencoders/2.2_variational_autoencoders/adversarial_autoencoder.cpp)
- [beta_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/beta_variational_autoencoder.cpp)
- [conditional_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/conditional_variational_autoencoder.cpp)
- [convolutional_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/convolutional_variational_autoencoder.cpp)
- [factor_tc_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/factor_tc_variational_autoencoder.cpp)
- [flow_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/flow_variational_autoencoder.cpp)
- [gumbel_categorical_autoencoder](2_autoencoders/2.2_variational_autoencoders/gumbel_categorical_autoencoder.cpp)
- [hierarchical_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/hierarchical_variational_autoencoder.cpp)
- [importance_weighted_autoencoder](2_autoencoders/2.2_variational_autoencoders/importance_weighted_autoencoder.cpp)
- [info_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/info_variational_autoencoder.cpp)
- [ladder_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/ladder_variational_autoencoder.cpp)
- [posterior_collapse](2_autoencoders/2.2_variational_autoencoders/posterior_collapse.cpp)
- [sequence_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/sequence_variational_autoencoder.cpp)
- [vae_gan](2_autoencoders/2.2_variational_autoencoders/vae_gan.cpp)
- [variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/variational_autoencoder.cpp)
- [vector_quantized_autoencoder](2_autoencoders/2.2_variational_autoencoders/vector_quantized_autoencoder.cpp)
- [vector_quantized_autoencoder_2](2_autoencoders/2.2_variational_autoencoders/vector_quantized_autoencoder_2.cpp)
- [wasserstein_autoencoder](2_autoencoders/2.2_variational_autoencoders/wasserstein_autoencoder.cpp)

### 2.3_applications

- [autoencoder_restoration](2_autoencoders/2.3_applications/autoencoder_restoration.cpp)
- [autorec](2_autoencoders/2.3_applications/autorec.cpp)
- [encoder_pretraining](2_autoencoders/2.3_applications/encoder_pretraining.cpp)
- [latent_arithmetic](2_autoencoders/2.3_applications/latent_arithmetic.cpp)
- [latent_visualization](2_autoencoders/2.3_applications/latent_visualization.cpp)
- [learned_image_codec](2_autoencoders/2.3_applications/learned_image_codec.cpp)
- [reconstruction_anomaly](2_autoencoders/2.3_applications/reconstruction_anomaly.cpp)

## Validation

All 165 migrated C++ quick examples passed after rebuilding against the current
core. Representative convolution, point-model and autoencoder targets also
built and ran through the standard `examples/cpp` CMake project. All moved source
files are byte-identical to their original versions. See the
[Python validation record](../../../python/nn/README.md#validation) for cross-language
checkpoint/fixture parity and packaging checks.
