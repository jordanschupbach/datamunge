# A namespace matching datamunge::nn. Keep model composition and loops in R.
nn <- new.env(parent = emptyenv())
.local_nn_enum <- function(prefix) {
  symbols <- ls(environment(.local_nn_enum), all.names = TRUE)
  selected <- symbols[startsWith(symbols, paste0(prefix, "_"))]
  selected <- selected[endsWith(selected, "_get")]
  values <- lapply(selected, function(name) get(name, environment(.local_nn_enum))())
  names(values) <- base::sub("_get$", "", substring(selected, nchar(prefix) + 2L))
  values
}
.local_nn_populate <- function() {
  namespace <- environment(.local_nn_populate)
  for (name in ls(namespace, all.names = TRUE)) {
    value <- get(name, namespace)
    if (is.function(value) && !startsWith(name, ".")) nn[[name]] <- value
  }
  nn$Tensor <- CoreTensor
  for (kind in c("DType", "Activation", "Padding", "OptimizerKind", "Reduction", "AttentionImpl", "RotaryLayout",
                "PadMode", "ResizeMethod", "DepthToSpaceMode", "GridSampleMode", "GridSamplePadding", "WavFormat"))
    nn[[kind]] <- .local_nn_enum(kind)
  initializer_names <- ls(namespace, pattern = "^init_")
  nn$init <- setNames(lapply(initializer_names, get, envir = namespace), substring(initializer_names, 6L))
  augment_names <- ls(namespace, pattern = "^augment_")
  nn$augment <- setNames(lapply(augment_names, get, envir = namespace), substring(augment_names, 9L))
  schedule_names <- ls(namespace, pattern = "^schedule_")
  nn$schedule <- setNames(lapply(schedule_names, get, envir = namespace), substring(schedule_names, 10L))
  nn$no_grad <- function(code) {
    previous <- is_grad_enabled()
    on.exit(set_grad_enabled(previous), add = TRUE)
    set_grad_enabled(FALSE)
    code()
  }
}
# Shared example components (examples/nn/common/*.hpp, bound by
# examples/bindings/nn_components.i) regrouped by C++ namespace: C++
# image_training::ImageBatch is nn_components$image_training$ImageBatch. The R
# examples run the same loops as the C++ examples over these pieces.
nn_components <- new.env(parent = emptyenv())
.local_nn_component_namespaces <- c("example_runtime", "paths", "dataset_assets", "check", "image_training", "resnet", "resnet_demo",
                                    "architecture", "mobile_checks", "nn_components", "grouped", "lrn", "inception",
                                    "normalized_conv", "deconvnet", "alexnet", "googlenet", "inception23", "inception4",
                                    "nin", "vgg", "xception", "zfnet", "preact", "densenet", "dpn", "pyramidnet",
                                    "res2net", "resnext", "revnet", "stochastic_depth", "wide_resnet", "mobilenet",
                                    "mbconv", "condensenet", "ghostnet", "regnet", "shufflenet", "squeezenet", "convnext",
                                    "deformable", "non_local", "octconv", "repvgg", "attention_cnn", "mnist",
                                    "cnn_baselines", "alexnet_checks", "xception_checks", "densenet_checks", "ghostnet_checks", "mobilenet_v2_checks", "mnasnet_checks", "convnext_checks", "efficientnet_checks", "efficientnet_v2_checks", "preactivation_resnet_checks", "pyramidnet_checks", "res2net_checks", "resnext_checks", "revnet_checks", "cbam_checks", "non_local_checks", "octconv_checks", "repvgg_checks", "sknet_checks", "ae", "memory_components",
                                    "seg", "fcn", "unet", "segnet", "hrnet", "pspnet", "deeplab", "sentence", "replay", "series", "sensor", "charcnn", "tcn", "wavestack", "sal", "dream", "featurevis", "cam", "occlusion", "ig", "smooth", "guided", "detector", "fpn", "retinanet", "fcos", "ssd", "centernet", "detr", "hungarian", "efficientdet", "bifpn",
                                    "yolo", "wordtree", "maskrcnn", "instance", "panoptic", "yolact", "detection_evaluation", "detection", "highway", "stn", "capsnet", "im2col", "receptive")
.local_nn_populate_components <- function() {
  namespace <- environment(.local_nn_populate_components)
  symbols <- ls(namespace, all.names = TRUE)
  for (prefix in .local_nn_component_namespaces) {
    marker <- paste0(prefix, "__")
    group <- new.env(parent = emptyenv())
    selected <- symbols[startsWith(symbols, marker)]
    # Drop generated accessors and overload variants; the dispatching function keeps the plain name.
    selected <- selected[!grepl("__SWIG_[0-9]+$|_(get|set)$", selected)]
    for (name in selected) {
      value <- get(name, namespace)
      if (!is.function(value)) next
      # Flush R's buffered output before a C++ call so std::cout text stays in order.
      wrapped <- eval(bquote(function(...) { flush(stdout()); .(value)(...) }))
      group[[substring(name, nchar(marker) + 1L)]] <- wrapped
    }
    # SWIG exposes enum constants as zero-argument getters in R. Preserve
    # their enum class so they can be assigned to bound configuration fields.
    for (name in symbols[startsWith(symbols, marker) & endsWith(symbols, "_get")]) {
      value <- get(name, namespace)
      if (is.function(value) && identical(names(formals(value)), ".copy") &&
          base::any(grepl("enumFromInteger", deparse(body(value)), fixed = TRUE))) {
        key <- base::sub("_get$", "", substring(name, nchar(marker) + 1L))
        group[[key]] <- value()
      }
    }
    nn_components[[prefix]] <- group
  }
}
# Called after all generated proxy functions have been defined.

.local_nn_tensor_list <- function(vector) {
  lapply(seq_len(TensorVector_size(vector)) - 1L, function(i) {
    value <- TensorVector___getitem__(vector, i)
    attr(value, "nn_parent") <- vector
    value
  })
}
.local_nn_tensor_vector <- function(values) {
  if (!is.list(values)) return(values)
  vector <- TensorVector()
  for (value in values) vector$push_back(value)
  vector
}
.local_nn_named_list <- function(vector) {
  result <- list()
  for (i in seq_len(NamedTensors_size(vector)) - 1L) {
    pair <- NamedTensors___getitem__(vector, i)
    attr(pair, "nn_parent") <- vector
    tensor <- pair$second
    attr(tensor, "nn_parent") <- pair
    result[[pair$first]] <- tensor
  }
  result
}
.local_nn_named_tensors <- function(arrays) {
  if (!is.list(arrays)) return(arrays)
  vector <- NamedTensors()
  if (!length(arrays)) return(vector)
  if (is.null(names(arrays)) || base::any(!nzchar(names(arrays))) || anyDuplicated(names(arrays)))
    stop("arrays must have unique nonempty names")
  for (name in names(arrays)) vector$push_back(NamedTensor(name, arrays[[name]]))
  vector
}
.local_nn_load_npz <- function(path) .local_nn_named_list(load_npz(path))
.local_nn_save_npz <- function(path, arrays) save_npz(path, .local_nn_named_tensors(arrays))
.local_nn_grad <- function(output, inputs, options = GradOptions(), grad_outputs = NULL) {
  if (is.list(output) || !is.null(grad_outputs)) {
    if (!is.list(output)) output <- list(output)
    if (is.null(grad_outputs)) grad_outputs <- list()
    return(.local_nn_tensor_list(grad(.local_nn_tensor_vector(output),
      .local_nn_tensor_vector(inputs), .local_nn_tensor_vector(grad_outputs), options)))
  }
  .local_nn_tensor_list(grad(output, .local_nn_tensor_vector(inputs), options))
}
.local_nn_image_size <- function(size) {
  if (!is.numeric(size)) return(size) # Already a bound C++ ImageSize.
  if (length(size) != 2L || base::any(!is.finite(size) | size != base::floor(size)))
    stop("image size must contain two finite integer dimensions")
  result <- ImageSize()
  result$height <- size[1L]
  result$width <- size[2L]
  result
}
.onLoad <- function(libname, pkgname) {
  .local_nn_populate()
  .local_nn_populate_components()
  nn$load_npz <- .local_nn_load_npz
  nn$save_npz <- .local_nn_save_npz
  nn$grad <- .local_nn_grad
  nn$tensor_list <- .local_nn_tensor_list
  nn$tensor_vector <- .local_nn_tensor_vector
  nn$named_tensors <- .local_nn_named_tensors
  nn$named_list <- .local_nn_named_list
  nn$resize_bilinear <- function(images, size, align_corners = FALSE, half_pixel_centers = FALSE)
    resize_bilinear(images, .local_nn_image_size(size), align_corners, half_pixel_centers)
  nn$resize_nearest <- function(images, size, align_corners = FALSE, half_pixel_centers = FALSE)
    resize_nearest(images, .local_nn_image_size(size), align_corners, half_pixel_centers)
}
