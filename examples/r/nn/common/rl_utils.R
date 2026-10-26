# Episode statistics, argmax helpers and observation printing (examples/nn/common/rl/utils.hpp).
# The episode runners (RunEpisodes / RunRandom) are experiment loops and live in each example.

nn_helper("rl_env", environment())

# Episode statistics: returns (float32) and lengths, as list(returns = , lengths = ).
rl_utils_episode_stats <- function() list(returns = numeric(), lengths = integer())

# std::accumulate over floats with a float accumulator, then / size.
rl_utils_float_mean <- function(values) {
    if (length(values) == 0L) return(0)
    total <- 0
    for (value in values) total <- rl_env_f32(total + value)
    rl_env_f32(total / length(values))
}
rl_utils_mean_return <- function(stats) rl_utils_float_mean(stats$returns)
rl_utils_mean_length <- function(stats) rl_utils_float_mean(stats$lengths)
rl_utils_min_return <- function(stats) min(stats$returns)
rl_utils_max_return <- function(stats) max(stats$returns)

rl_utils_print_stats <- function(label, stats) {
    cat(sprintf("  %-32s return mean %8.2f  min %8.2f  max %8.2f   length mean %6.1f\n", label,
                rl_utils_mean_return(stats), rl_utils_min_return(stats), rl_utils_max_return(stats),
                rl_utils_mean_length(stats)))
}

# 0-based index of the first maximum.
rl_utils_argmax <- function(values) which.max(values) - 1L

# 0-based argmax with ties broken uniformly at random (rng: nn$Mt19937_64, rng() % ties).
rl_utils_argmax_random_tie <- function(q, n, rng) {
    q <- q[seq_len(n)]
    ties <- which(q == max(q)) - 1L
    ties[as.integer(rng$modulo(length(ties))) + 1L]
}

rl_utils_format_obs <- function(obs, max_elems = 8L) {
    shown <- obs[seq_len(min(length(obs), max_elems))]
    text <- paste(sprintf("%.3f", shown), collapse = " ")
    if (length(obs) > max_elems) text <- paste0(text, " ...")
    paste0("[", text, "]")
}
