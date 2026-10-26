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
.local_nn_load_npz <- function(path) {
  vector <- load_npz(path)
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
.local_nn_save_npz <- function(path, arrays) {
  vector <- NamedTensors()
  if (is.null(names(arrays)) || base::any(!nzchar(names(arrays)))) stop("arrays must be named")
  for (name in names(arrays)) vector$push_back(NamedTensor(name, arrays[[name]]))
  save_npz(path, vector)
}
.local_nn_grad <- function(output, inputs, options = GradOptions(), grad_outputs = NULL) {
  if (is.list(output) || !is.null(grad_outputs)) {
    if (!is.list(output)) output <- list(output)
    if (is.null(grad_outputs)) grad_outputs <- list()
    return(.local_nn_tensor_list(grad(.local_nn_tensor_vector(output),
      .local_nn_tensor_vector(inputs), .local_nn_tensor_vector(grad_outputs), options)))
  }
  .local_nn_tensor_list(grad(output, .local_nn_tensor_vector(inputs), options))
}
.onLoad <- function(libname, pkgname) {
  .local_nn_populate()
  nn$load_npz <- .local_nn_load_npz
  nn$save_npz <- .local_nn_save_npz
  nn$grad <- .local_nn_grad
  nn$tensor_list <- .local_nn_tensor_list
  nn$tensor_vector <- .local_nn_tensor_vector
}
