# Gymnasium-style environment interface shared by the RL examples (examples/nn/common/rl/env.hpp).
#
# reset() returns the first observation, step() a step result list(obs, reward, terminated,
# truncated): `terminated` means the MDP reached a terminal state, `truncated` that a time
# limit cut the episode short (bootstrapping is still valid).
#
# Environments are R environments holding their state and method closures:
#   env$name(), env$obs_size(), env$action_space(), env$reset(seed = NULL), env$step(action),
#   env$render(), env$action_names(), env$seed(seed), env$max_steps().
# Actions are rl_env_action() lists: discrete actions are R *integers* (2L), continuous ones
# numeric vectors; rl_env_as_action() applies the C++ implicit conversions.
#
# The C++ environments compute in float32 and draw from std::mt19937_64. The ports reproduce
# both: every environment RNG is an nn$Mt19937_64 (same engine, same libc++ distributions) and
# the physics rounds each operation to float32 with rl_env_f32(). The canonical build (clang,
# arm64, -ffp-contract=on) fuses `a + b * c` written in one expression into one fma;
# rl_env_fma() reproduces that, so trajectories match the C++ bit for bit.

# x rounded to the nearest float32 (vectorised).
rl_env_f32 <- function(x) readBin(writeBin(as.double(x), raw(), size = 4L), "double", size = 4L, n = length(x))

# float32 a * b + c with one rounding: clang's contraction of `c + a * b` for floats. For float32
# operands a * b is exact in double, so only the sum is rounded.
rl_env_fma <- function(a, b, c) rl_env_f32(a * b + c)

# std::round: halves away from zero (R's round() rounds halves to even).
rl_env_cpp_round <- function(x) as.integer(sign(x) * floor(abs(x) + 0.5))

# std::to_string(float): printf("%f").
rl_env_cpp_float <- function(x) sprintf("%f", x)

# The C++ calls the platform's float libm (cosf, sinf, atan2f, expf), which is not always
# correctly rounded. nn$cos/nn$sin/nn$atan2/nn$exp on a small float32 tensor call exactly those
# functions, so the ports use them for scalars too.
rl_env_cosf <- function(x) nn$cos(nn$from_vector(x, 1L))$item()
rl_env_sinf <- function(x) nn$sin(nn$from_vector(x, 1L))$item()
rl_env_atan2f <- function(y, x) nn$atan2(nn$from_vector(y, 1L), nn$from_vector(x, 1L))$item()
rl_env_expf <- function(x) nn$exp(nn$from_vector(x, 1L))$item()
rl_env_sqrtf <- function(x) rl_env_f32(sqrt(x))

# Where one C++ function takes both the sine and the cosine of the same float, clang on Darwin
# merges them into one __sincosf_stret call. Its results are the correctly rounded values in all
# but ~0.03% of arguments (a 1e7-sample scan); no binding exposes the platform's float sincos,
# so this pair (cos, sin) uses the correctly rounded values: the one known 1-ulp deviation.
rl_env_sincosf <- function(x) rl_env_f32(c(cos(x), sin(x)))

rl_env_action <- function(discrete = -1L, continuous = numeric()) {
    structure(list(discrete = as.integer(discrete), continuous = as.double(continuous)), class = "rl_action")
}

# The C++ implicit conversions: integer -> discrete, double vector -> continuous,
# a hud decision -> its action.
rl_env_as_action <- function(value) {
    if (inherits(value, "rl_action")) return(value)
    if (inherits(value, "rl_decision")) return(value$action)
    if (is.integer(value) && length(value) == 1L) return(rl_env_action(value))
    if (is.double(value)) return(rl_env_action(continuous = rl_env_f32(value)))
    stop("an action is an integer (discrete), a double vector (continuous) or an rl_action")
}

rl_env_discrete_space <- function(n) {
    structure(list(kind = "discrete", n = as.integer(n), dim = 0L, low = numeric(), high = numeric()),
              class = "rl_action_space")
}

rl_env_continuous_space <- function(low, high) {
    structure(list(kind = "continuous", n = 0L, dim = length(low), low = rl_env_f32(low), high = rl_env_f32(high)),
              class = "rl_action_space")
}

# A uniformly random action; rng is an nn$Mt19937_64 (rng() % n, uniform_real<float>).
rl_env_space_sample <- function(space, rng) {
    if (space$kind == "discrete") return(rl_env_action(as.integer(rng$modulo(space$n))))
    values <- numeric(space$dim)
    for (i in seq_len(space$dim)) values[i] <- rng$uniform32(space$low[i], space$high[i])
    rl_env_action(continuous = values)
}

rl_env_space_string <- function(space) {
    if (space$kind == "discrete") return(sprintf("Discrete(%d)", space$n))
    bounds <- paste(paste0(rl_env_cpp_float(space$low), "..", rl_env_cpp_float(space$high)), collapse = ", ")
    sprintf("Box(%d) [%s]", space$dim, bounds)
}

rl_env_step_result <- function(obs = numeric(), reward = 0, terminated = FALSE, truncated = FALSE) {
    list(obs = obs, reward = reward, terminated = terminated, truncated = truncated)
}

rl_env_done <- function(result) result$terminated || result$truncated

# rl::EnvBase: an mt19937_64 RNG, a step counter and time-limit truncation. Subclasses add
# name/obs_size/action_space/step and reset_impl (the C++ protected Reset()).
rl_env_base <- function(max_steps = 0L) {
    self <- new.env()
    self$max_steps_ <- as.integer(max_steps)
    self$steps_ <- 0L
    self$rng_ <- nn$Mt19937_64(0)
    self$seed <- function(seed) self$rng_ <- nn$Mt19937_64(seed)
    self$max_steps <- function() self$max_steps_
    self$steps <- function() self$steps_
    self$rng <- function() self$rng_
    # Start a new episode; with a seed the environment's RNG is reseeded first.
    self$reset <- function(seed = NULL) {
        if (!is.null(seed)) self$seed(seed)
        self$reset_impl()
    }
    self$render <- function() ""
    self$action_names <- function() character()
    self$begin_episode <- function() self$steps_ <- 0L
    # TRUE if the time limit is reached after this step.
    self$count_step <- function() {
        if (self$max_steps_ <= 0L) return(FALSE)
        self$steps_ <- self$steps_ + 1L
        self$steps_ >= self$max_steps_
    }
    self$uniform <- function(lo, hi) self$rng_$uniform32(lo, hi)
    # EnvBase builds a fresh std::normal_distribution per draw: no cached second sample.
    self$normal <- function(mean, stddev) {
        self$rng_$reset_normal()
        self$rng_$normal32(mean, stddev)
    }
    self
}

# Outcome of taking action a in state s, for planning methods.
rl_env_transition <- function(prob, next_state, reward, terminal) {
    list(prob = prob, next_state = next_state, reward = reward, terminal = terminal)
}

# rl::TabularEnv: states are indices, observations are the index as a double. Subclasses add
# num_states(), state() and transitions(s, a).
rl_env_tabular <- function(max_steps = 0L) {
    self <- rl_env_base(max_steps)
    self$obs_size <- function() 1L
    self$state_obs <- function() as.double(self$state())
    self
}
