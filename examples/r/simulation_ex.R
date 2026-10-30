# Native CPU pendulum batch with explicit end-of-episode handling.
library(datamunger)
task <- TaskBatch_create("InvertedPendulum", 4L, 2L, 42L, 100L)
returns <- numeric(4)
active <- rep(1L, 4)
for (tick in seq_len(100)) {
  if (!any(active)) break
  state <- task$output()$observations
  actions <- vapply(seq_len(4), function(i) {
    row <- state[(i - 1L) * 4L + seq_len(4)]
    max(-3, min(3, .08*row[1] + .15*row[3] - 1.5*row[2] - .3*row[4]))
  }, numeric(1))
  result <- task$step(actions, active)
  returns <- returns + result$rewards * active
  active <- active * as.integer(!(result$terminated | result$truncated))
}
stopifnot(all(is.finite(returns)), all(returns > 0))
print(returns)
