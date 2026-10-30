# Named-tensor marshalling for bound C++ memory models; no fitting workflow.
memory_records <- function(batch) nn$named_tensors(batch)
memory_rows <- function(batch, indices) lapply(batch, function(tensor) nn$take_rows(tensor, indices))
memory_evaluate <- function(model, batch) nn$named_list(model$Evaluate(memory_records(batch)))
memory_same <- function(first, second) {
    if (is.list(first)) return(setequal(names(first), names(second)) &&
        all(vapply(names(first), function(key) memory_same(first[[key]], second[[key]]), logical(1))))
    nn$equal_bits(first, second)
}
memory_save <- function(path, batch) {
    if (is.list(batch)) batch <- batch[base::order(names(batch))]
    nn$save_npz(path, batch)
}
