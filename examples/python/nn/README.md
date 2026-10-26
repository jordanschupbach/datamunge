# Neural network examples in Python

The migration to actual language implementations is **in progress** across all
414 canonical NN C++ examples. Run `just nn-language-status` for the current
counts and see [the ledger](../../nn/language_status.json) for every example.
Only scripts marked `native` count as Python implementations. Unported entries
are rejected by the normal language runner.

```sh
just python-example cnn_mnist epochs=5
just python-example tensor_basics
just python-example autodiff epochs=5
just python-example linear_regression epochs=5
just python-example polynomial_regression
just python-example principal_components
just python-example logistic_regression epochs=5
just python-example softmax_mnist epochs=5
just python-example mlp_mnist epochs=5
just python-example sessions
just python-example learning_rate_schedules epochs=5
just python-example gradient_clipping_accumulation epochs=5
just python-example adaptive_optimizers epochs=5
just python-example advanced_optimizers epochs=5
just python-example sgd_momentum epochs=5
just python-example training_loop epochs=5
just python-example early_stopping_checkpoints epochs=5
just python-example dense_layer
just python-example activations
just python-example embedding epochs=5
just python-example pooling
just python-example dropout epochs=5
just python-example normalization_layers
just python-example batch_norm epochs=5
just python-example conv_layers
just python-example weight_spectral_norm
just python-example variables_initializers
just python-example convolution_im2col
just python-example undercomplete_autoencoder epochs=5
```

These scripts compose the functional `datamunge::nn` SWIG API directly: models,
training, evaluation and numerical checks live in Python. Import it with
`from pydatamunge import nn`. The binding exposes tensors, parameters, layers,
autograd, optimizers, batching, I/O, datasets, linear algebra and random streams.
The new `python-example` recipe builds the functional API without the compiled
experiment bridge. Classification examples print the final confusion matrix.

Keep development runs small by passing `epochs=5`; `--quick` is the default and
limits dataset sizes. An explicit epoch count is honored. No-data demonstrations
and direct solvers perform no training. The migration ledger records source
hashes and validation commands; changing source or the binding invalidates its
old validation evidence. See [migration notes](../../../docs/nn-language-migration.md).

Handwritten ports carry a `# Native` header. Reference regeneration preserves
those files. `nn_examples.available()` lists native Python ports;
`nn_examples.available(native_only=False)` lists the entire canonical inventory.
`nn_examples.run()` runs native scripts only. `run_reference()` is a separately
named legacy tool for compiled experiments, requiring its optional extension.

The sections below describe the original reference bridge and its commands.


All 165 examples in sections 1 and 2 have matching Python entry points with the
same numbered layout as [`examples/cpp/nn`](../../cpp/nn/README.md).

These are **thin Python wrappers over the canonical C++ experiments**, not
independent Python implementations of their models or training loops. An
optional SWIG extension calls the C++ entry functions in the Python process.
It preserves each architecture, numerical/gradient fixtures, training routine,
command-line options, datasets, artifacts and failure status. It never shells
out to a C++ executable. The native demo API is separate from the stable tensor
and model API in `pydatamunge`.

```sh
just build-nn-python
just run-nn-python convolution_im2col
just run-nn-python cnn_mnist
just run-nn-python undercomplete_autoencoder epochs=1
just run-all-nn-python --with-library --jobs 3
just run-nn-python-full cnn_mnist
```

The first build compiles the 165 experiments and can take several minutes.
Subsequent builds reuse unchanged objects. Normal Python builds do not compile
the example extension until opted in with `just build-nn-python`; after that,
`just build-python` keeps it enabled for this checkout. Explicitly set
`DATAMUNGE_BUILD_NN_EXAMPLES=0` to request a core-only build. Outside Nix:

```sh
python3 scripts/prebuild_bindings.py python
DATAMUNGE_BUILD_NN_EXAMPLES=1 CMAKE_BUILD_PARALLEL_LEVEL=4 python -m pip install .
```

This optional build needs SWIG and a C++23 compiler. Ordinary core-only source
package installation remains independent of SWIG. Explicit
`DATAMUNGE_USE_SYSTEM_LIB=1` linking is supported for the example extension too.
The source archive includes the canonical experiments, shared headers, SWIG
interface and generator needed to rebuild it.

Run a file directly with the installed binding:

```sh
build/venv/pydatamunge-run/bin/python -I \
  examples/python/nn/1_convolutional_neural_networks/1.1_fundamentals/convolution_im2col.py
```

Or call an experiment from Python:

```python
from pydatamunge.nn_examples import available, run

print(available())
status = run("undercomplete_autoencoder", ["epochs=1"])
if status:
    raise RuntimeError("the experiment's own checks failed")
```

Quick mode is the default. `--full` on a direct entry point, or
`DATAMUNGE_EXAMPLES_FULL=1`, opts into the original training schedule. Dataset
resolution is unchanged: `DATAMUNGE_DATA_DIR` or the existing
`~/projects/tflow_examples/output/datasets` cache. See the corresponding C++
source's opening comments for required assets and supported `key=value` options.
Missing assets fail normally; examples do not substitute toy data.

Python output defaults to `output/nn/python/quick` or `output/nn/python/full`,
separate from the C++ outputs. `DATAMUNGE_OUTPUT_DIR` overrides this. The runner
uses an isolated interpreter so inherited `PYTHONPATH` cannot shadow the installed
binding, keeps separate caches and logs under `build/nn-python`, includes dataset/output
overrides in cache keys, runs checkpoint producers
before consumers and stops failed or timed-out runs without caching a pass.
For direct scripts that consume a checkpoint (such as `receptive_fields`), run
the producer first; the runner handles this automatically.

Python files and `src/pydatamunge/_nn_catalog.py` are generated deterministically
by `scripts/prebuild_nn_examples.py`; edit the C++ experiments or generator and
run `just prebuild-nn-python`. The generated C++ adapters live only under `build`.
They isolate each program's private symbols and preserve the implicit return
behavior of `main`. The remaining sections and other language replicas are
reserved for later.

## Catalogue

### 1.10_3d_and_video_convolutions

- [c3d](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/c3d.py)
- [factorized_video](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/factorized_video.py)
- [i3d](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/i3d.py)
- [medical_volume_classification](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/medical_volume_classification.py)
- [point_geometry](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_geometry.py)
- [point_models_checks](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_models_checks.py)
- [pointnet](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/pointnet.py)
- [pointnet_plus_plus](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/pointnet_plus_plus.py)
- [slowfast](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/slowfast.py)
- [two_stream](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/two_stream.py)
- [voxnet](1_convolutional_neural_networks/1.10_3d_and_video_convolutions/voxnet.py)

### 1.11_interpretability

- [deepdream](1_convolutional_neural_networks/1.11_interpretability/deepdream.py)
- [feature_visualization](1_convolutional_neural_networks/1.11_interpretability/feature_visualization.py)
- [filter_activations](1_convolutional_neural_networks/1.11_interpretability/filter_activations.py)
- [grad_cam](1_convolutional_neural_networks/1.11_interpretability/grad_cam.py)
- [guided_backprop](1_convolutional_neural_networks/1.11_interpretability/guided_backprop.py)
- [integrated_gradients](1_convolutional_neural_networks/1.11_interpretability/integrated_gradients.py)
- [occlusion_sensitivity](1_convolutional_neural_networks/1.11_interpretability/occlusion_sensitivity.py)
- [smoothgrad](1_convolutional_neural_networks/1.11_interpretability/smoothgrad.py)
- [vanilla_saliency](1_convolutional_neural_networks/1.11_interpretability/vanilla_saliency.py)

### 1.1_fundamentals

- [batch_norm_ablation](1_convolutional_neural_networks/1.1_fundamentals/batch_norm_ablation.py)
- [cnn_cifar10](1_convolutional_neural_networks/1.1_fundamentals/cnn_cifar10.py)
- [cnn_mnist](1_convolutional_neural_networks/1.1_fundamentals/cnn_mnist.py)
- [convolution_im2col](1_convolutional_neural_networks/1.1_fundamentals/convolution_im2col.py)
- [depthwise_separable](1_convolutional_neural_networks/1.1_fundamentals/depthwise_separable.py)
- [dilated_convolutions](1_convolutional_neural_networks/1.1_fundamentals/dilated_convolutions.py)
- [global_pooling_vs_flatten](1_convolutional_neural_networks/1.1_fundamentals/global_pooling_vs_flatten.py)
- [lenet5](1_convolutional_neural_networks/1.1_fundamentals/lenet5.py)
- [receptive_fields](1_convolutional_neural_networks/1.1_fundamentals/receptive_fields.py)
- [transposed_convolutions](1_convolutional_neural_networks/1.1_fundamentals/transposed_convolutions.py)

### 1.2_classic_architectures

- [alexnet](1_convolutional_neural_networks/1.2_classic_architectures/alexnet.py)
- [googlenet](1_convolutional_neural_networks/1.2_classic_architectures/googlenet.py)
- [inception_v2_v3](1_convolutional_neural_networks/1.2_classic_architectures/inception_v2_v3.py)
- [inception_v4_resnet](1_convolutional_neural_networks/1.2_classic_architectures/inception_v4_resnet.py)
- [network_in_network](1_convolutional_neural_networks/1.2_classic_architectures/network_in_network.py)
- [vgg](1_convolutional_neural_networks/1.2_classic_architectures/vgg.py)
- [xception](1_convolutional_neural_networks/1.2_classic_architectures/xception.py)
- [zfnet](1_convolutional_neural_networks/1.2_classic_architectures/zfnet.py)

### 1.3_residual_and_dense_families

- [densenet](1_convolutional_neural_networks/1.3_residual_and_dense_families/densenet.py)
- [dpn](1_convolutional_neural_networks/1.3_residual_and_dense_families/dpn.py)
- [highway_networks](1_convolutional_neural_networks/1.3_residual_and_dense_families/highway_networks.py)
- [preactivation_resnet](1_convolutional_neural_networks/1.3_residual_and_dense_families/preactivation_resnet.py)
- [pyramidnet](1_convolutional_neural_networks/1.3_residual_and_dense_families/pyramidnet.py)
- [res2net](1_convolutional_neural_networks/1.3_residual_and_dense_families/res2net.py)
- [resnet_basic](1_convolutional_neural_networks/1.3_residual_and_dense_families/resnet_basic.py)
- [resnet_bottleneck](1_convolutional_neural_networks/1.3_residual_and_dense_families/resnet_bottleneck.py)
- [resnext](1_convolutional_neural_networks/1.3_residual_and_dense_families/resnext.py)
- [revnet](1_convolutional_neural_networks/1.3_residual_and_dense_families/revnet.py)
- [stochastic_depth](1_convolutional_neural_networks/1.3_residual_and_dense_families/stochastic_depth.py)
- [wide_resnet](1_convolutional_neural_networks/1.3_residual_and_dense_families/wide_resnet.py)

### 1.4_efficient_and_mobile_architectures

- [condensenet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/condensenet.py)
- [efficientnet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/efficientnet.py)
- [efficientnet_v2](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/efficientnet_v2.py)
- [ghostnet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/ghostnet.py)
- [mnasnet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mnasnet.py)
- [mobilenet_v1](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v1.py)
- [mobilenet_v2](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v2.py)
- [mobilenet_v3](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v3.py)
- [regnet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/regnet.py)
- [shufflenet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/shufflenet.py)
- [squeezenet](1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.py)

### 1.5_attention_and_modern_convnets

- [capsule_networks](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/capsule_networks.py)
- [cbam](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/cbam.py)
- [convnext](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/convnext.py)
- [convnext_v2](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/convnext_v2.py)
- [deformable_conv](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/deformable_conv.py)
- [eca_net](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/eca_net.py)
- [non_local](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/non_local.py)
- [octconv](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/octconv.py)
- [repvgg](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/repvgg.py)
- [senet](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/senet.py)
- [sknet](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/sknet.py)
- [spatial_transformer](1_convolutional_neural_networks/1.5_attention_and_modern_convnets/spatial_transformer.py)

### 1.6_object_detection

- [centernet](1_convolutional_neural_networks/1.6_object_detection/centernet.py)
- [detection_baselines](1_convolutional_neural_networks/1.6_object_detection/detection_baselines.py)
- [detection_evaluation](1_convolutional_neural_networks/1.6_object_detection/detection_evaluation.py)
- [detr](1_convolutional_neural_networks/1.6_object_detection/detr.py)
- [efficientdet](1_convolutional_neural_networks/1.6_object_detection/efficientdet.py)
- [fast_rcnn](1_convolutional_neural_networks/1.6_object_detection/fast_rcnn.py)
- [faster_rcnn](1_convolutional_neural_networks/1.6_object_detection/faster_rcnn.py)
- [fcos](1_convolutional_neural_networks/1.6_object_detection/fcos.py)
- [fpn](1_convolutional_neural_networks/1.6_object_detection/fpn.py)
- [nms_matching](1_convolutional_neural_networks/1.6_object_detection/nms_matching.py)
- [rcnn](1_convolutional_neural_networks/1.6_object_detection/rcnn.py)
- [retinanet](1_convolutional_neural_networks/1.6_object_detection/retinanet.py)
- [ssd](1_convolutional_neural_networks/1.6_object_detection/ssd.py)
- [yolo_v1](1_convolutional_neural_networks/1.6_object_detection/yolo_v1.py)
- [yolo_v2](1_convolutional_neural_networks/1.6_object_detection/yolo_v2.py)
- [yolo_v3](1_convolutional_neural_networks/1.6_object_detection/yolo_v3.py)
- [yolo_v5_style](1_convolutional_neural_networks/1.6_object_detection/yolo_v5_style.py)

### 1.7_segmentation

- [deeplab_v1_v2](1_convolutional_neural_networks/1.7_segmentation/deeplab_v1_v2.py)
- [deeplab_v3_plus](1_convolutional_neural_networks/1.7_segmentation/deeplab_v3_plus.py)
- [fcn](1_convolutional_neural_networks/1.7_segmentation/fcn.py)
- [hrnet](1_convolutional_neural_networks/1.7_segmentation/hrnet.py)
- [mask_rcnn](1_convolutional_neural_networks/1.7_segmentation/mask_rcnn.py)
- [panoptic_fpn](1_convolutional_neural_networks/1.7_segmentation/panoptic_fpn.py)
- [pspnet](1_convolutional_neural_networks/1.7_segmentation/pspnet.py)
- [segmentation_losses](1_convolutional_neural_networks/1.7_segmentation/segmentation_losses.py)
- [segnet](1_convolutional_neural_networks/1.7_segmentation/segnet.py)
- [unet](1_convolutional_neural_networks/1.7_segmentation/unet.py)
- [unet_variants](1_convolutional_neural_networks/1.7_segmentation/unet_variants.py)
- [volumetric_segmentation](1_convolutional_neural_networks/1.7_segmentation/volumetric_segmentation.py)
- [yolact](1_convolutional_neural_networks/1.7_segmentation/yolact.py)

### 1.8_other_vision_tasks

- [angular_margin](1_convolutional_neural_networks/1.8_other_vision_tasks/angular_margin.py)
- [bilinear_cnn](1_convolutional_neural_networks/1.8_other_vision_tasks/bilinear_cnn.py)
- [context_encoder](1_convolutional_neural_networks/1.8_other_vision_tasks/context_encoder.py)
- [crnn_ctc](1_convolutional_neural_networks/1.8_other_vision_tasks/crnn_ctc.py)
- [dncnn](1_convolutional_neural_networks/1.8_other_vision_tasks/dncnn.py)
- [facenet_triplet](1_convolutional_neural_networks/1.8_other_vision_tasks/facenet_triplet.py)
- [flownets](1_convolutional_neural_networks/1.8_other_vision_tasks/flownets.py)
- [gatys_style_transfer](1_convolutional_neural_networks/1.8_other_vision_tasks/gatys_style_transfer.py)
- [image_colorization](1_convolutional_neural_networks/1.8_other_vision_tasks/image_colorization.py)
- [image_retrieval](1_convolutional_neural_networks/1.8_other_vision_tasks/image_retrieval.py)
- [johnson_fast_style](1_convolutional_neural_networks/1.8_other_vision_tasks/johnson_fast_style.py)
- [monocular_depth](1_convolutional_neural_networks/1.8_other_vision_tasks/monocular_depth.py)
- [multilabel_classification](1_convolutional_neural_networks/1.8_other_vision_tasks/multilabel_classification.py)
- [openpose_pafs](1_convolutional_neural_networks/1.8_other_vision_tasks/openpose_pafs.py)
- [partial_inpainting](1_convolutional_neural_networks/1.8_other_vision_tasks/partial_inpainting.py)
- [pwcnet](1_convolutional_neural_networks/1.8_other_vision_tasks/pwcnet.py)
- [siamese](1_convolutional_neural_networks/1.8_other_vision_tasks/siamese.py)
- [stacked_hourglass](1_convolutional_neural_networks/1.8_other_vision_tasks/stacked_hourglass.py)
- [super_resolution](1_convolutional_neural_networks/1.8_other_vision_tasks/super_resolution.py)
- [vqa_baseline](1_convolutional_neural_networks/1.8_other_vision_tasks/vqa_baseline.py)

### 1.9_1d_convolutions

- [character_cnn](1_convolutional_neural_networks/1.9_1d_convolutions/character_cnn.py)
- [ecg_sensor_cnn](1_convolutional_neural_networks/1.9_1d_convolutions/ecg_sensor_cnn.py)
- [kim_sentence_cnn](1_convolutional_neural_networks/1.9_1d_convolutions/kim_sentence_cnn.py)
- [tcn](1_convolutional_neural_networks/1.9_1d_convolutions/tcn.py)
- [time_series_cnn](1_convolutional_neural_networks/1.9_1d_convolutions/time_series_cnn.py)
- [wavenet_stack](1_convolutional_neural_networks/1.9_1d_convolutions/wavenet_stack.py)

### 2.1_deterministic_autoencoders

- [contractive_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/contractive_autoencoder.py)
- [convolutional_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/convolutional_autoencoder.py)
- [denoising_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/denoising_autoencoder.py)
- [k_sparse_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/k_sparse_autoencoder.py)
- [masked_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/masked_autoencoder.py)
- [recurrent_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/recurrent_autoencoder.py)
- [sparse_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/sparse_autoencoder.py)
- [stacked_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/stacked_autoencoder.py)
- [tied_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/tied_autoencoder.py)
- [undercomplete_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/undercomplete_autoencoder.py)
- [winner_take_all_autoencoder](2_autoencoders/2.1_deterministic_autoencoders/winner_take_all_autoencoder.py)

### 2.2_variational_autoencoders

- [adversarial_autoencoder](2_autoencoders/2.2_variational_autoencoders/adversarial_autoencoder.py)
- [beta_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/beta_variational_autoencoder.py)
- [conditional_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/conditional_variational_autoencoder.py)
- [convolutional_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/convolutional_variational_autoencoder.py)
- [factor_tc_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/factor_tc_variational_autoencoder.py)
- [flow_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/flow_variational_autoencoder.py)
- [gumbel_categorical_autoencoder](2_autoencoders/2.2_variational_autoencoders/gumbel_categorical_autoencoder.py)
- [hierarchical_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/hierarchical_variational_autoencoder.py)
- [importance_weighted_autoencoder](2_autoencoders/2.2_variational_autoencoders/importance_weighted_autoencoder.py)
- [info_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/info_variational_autoencoder.py)
- [ladder_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/ladder_variational_autoencoder.py)
- [posterior_collapse](2_autoencoders/2.2_variational_autoencoders/posterior_collapse.py)
- [sequence_variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/sequence_variational_autoencoder.py)
- [vae_gan](2_autoencoders/2.2_variational_autoencoders/vae_gan.py)
- [variational_autoencoder](2_autoencoders/2.2_variational_autoencoders/variational_autoencoder.py)
- [vector_quantized_autoencoder](2_autoencoders/2.2_variational_autoencoders/vector_quantized_autoencoder.py)
- [vector_quantized_autoencoder_2](2_autoencoders/2.2_variational_autoencoders/vector_quantized_autoencoder_2.py)
- [wasserstein_autoencoder](2_autoencoders/2.2_variational_autoencoders/wasserstein_autoencoder.py)

### 2.3_applications

- [autoencoder_restoration](2_autoencoders/2.3_applications/autoencoder_restoration.py)
- [autorec](2_autoencoders/2.3_applications/autorec.py)
- [encoder_pretraining](2_autoencoders/2.3_applications/encoder_pretraining.py)
- [latent_arithmetic](2_autoencoders/2.3_applications/latent_arithmetic.py)
- [latent_visualization](2_autoencoders/2.3_applications/latent_visualization.py)
- [learned_image_codec](2_autoencoders/2.3_applications/learned_image_codec.py)
- [reconstruction_anomaly](2_autoencoders/2.3_applications/reconstruction_anomaly.py)

## Validation

This migration was validated on macOS arm64 with Python 3.13.11 and SWIG 4.6.0:

- All 165 C++ quick runs and all 165 Python/SWIG quick runs passed their own checks.
- All 35 Python tests passed, including catalogue parity, C++ entry-point
  semantics, object lifetimes, gradients, checkpoint paths and stale-PYTHONPATH isolation.
- Ten checkpoint/fixture archives from CNN, LeNet, ResNet, PointNet,
  deterministic-autoencoder and VAE runs contained 309 byte-identical arrays.
- Every moved C++ file is byte-identical to its pre-migration source.
- The source archive contains all 165 experiments and 271 core translation units;
  regeneration from an extracted archive reproduces all 167 native adapter/SWIG
  wrapper sources exactly. Compiled objects are excluded from the archive.
- A repeated native build invoked neither compiler nor linker. The optional
  extension reuses core objects and compiles changed sources in parallel.

Detailed local records are in `build/nn-python/migration-sources.json`,
`cpp-validation.json`, `python-validation.json` and `artifact-parity.json`.
Validation outputs were isolated under `output/nn/migration-cpp/quick` and
`output/nn/migration-python/quick`. Normal runs keep the default locations above.
Linux CI now builds the optional extension from its source archive and exercises
its data-free fixtures. That CI job and Windows builds were not run locally.
Full training was not run; it remains an explicit opt-in using the same C++ code.

R exposes the genuine R ports through R-only build and run commands; see
[the R examples](../../r/nn/README.md). Regeneration updates both catalogues together.

The native MNIST classifier now contains dataset loading, batching, fitting,
validation and checkpoint selection directly in its `main` function, with a
local evaluation helper. Shared language utilities contain only path handling,
option parsing and tensor snapshots; there is no shared MNIST experiment runner.
The other two native examples already keep their own workflows in their scripts.
