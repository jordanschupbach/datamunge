# Named-array helpers for replaying recorded runs (examples/nn/common/tf_replay.hpp).
if (!exists("nn_require", inherits = TRUE)) nn_helper("checks")
replay_get <- function(arrays, key) {
    if (is.null(arrays[[key]])) stop("missing array ", key, call. = FALSE)
    arrays[[key]]
}
replay_rows <- function(arrays, rows) lapply(arrays, function(v) nn$take_rows(v, rows))
replay_split <- function(arrays, split, keys) setNames(lapply(keys, function(k) replay_get(arrays, paste0(split, "_", k))), keys)
replay_max_abs_diff <- function(a, b) {
    x <- a$values(); y <- b$values()
    nn_require(length(x) == length(y), "compared tensors have the same size")
    if (length(x)) max(abs(x - y)) else 0
}
replay_same_bits <- function(a, b) {
    if (is.list(a)) return(setequal(names(a), names(b)) && all(vapply(names(a), function(k) replay_same_bits(a[[k]], b[[k]]), logical(1))))
    a$dtype() == b$dtype() && identical(as.numeric(a$shape()), as.numeric(b$shape())) && identical(a$values(), b$values())
}
replay_arrays <- function(store) nn_arrays(store)
replay_named <- function(store, gradients) {
    values <- nn$tensor_list(store$parameters())
    setNames(lapply(seq_along(gradients), function(i) {
        g <- gradients[[i]]
        if (g$defined()) g$detach() else nn$zeros_like(values[[i]])$detach()
    }), store$names())
}
replay_load_into <- function(store, path) {
    loaded <- nn$load_npz(path)
    nn_require(length(loaded) == store$size(), paste("array file covers exactly the model's parameters:", path))
    for (name in names(loaded)) {
        nn_require(store$contains(name), paste("array", name, "is a model parameter"))
        store$set(name, nn$cast(loaded[[name]], store$at(name)$dtype()))
    }
}
replay_max_diff <- function(a, b) {
    if (is.character(b)) b <- nn$load_npz(b)
    result <- 0
    for (k in names(a)) {
        nn_require(!is.null(b[[k]]), paste("reference has array", k))
        result <- max(result, replay_max_abs_diff(a[[k]], b[[k]]))
    }
    result
}
replay_history <- function(path, column = 1L) {
    lines <- readLines(path)[-1L]
    lines <- lines[nzchar(lines)]
    vapply(strsplit(lines, "\t", fixed = TRUE), function(cells) as.numeric(cells[[column + 1L]]), numeric(1))
}
replay_adam <- function() { config <- nn$OptimizerConfig(); config$kind <- nn$OptimizerKind$Adam; config }
replay_copy_assets <- function(from, to, files) for (f in files)
    if (file.exists(file.path(from, f))) file.copy(file.path(from, f), file.path(to, f), overwrite = TRUE)
