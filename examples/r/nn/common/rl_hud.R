# The agent's explained move and the overlay state (examples/nn/common/rl/hud.hpp).
#
# A decision is one move as the agent explains it: the action, an estimate per candidate
# action, the candidate taken and the return it expects. Drawing the overlay onto video frames
# is out of scope (as in the C++ port); the bookkeeping is kept so policies written for the
# recorders work unchanged.

nn_helper("rl_env", environment())

rl_hud_decision <- function(action = rl_env_action()) {
    structure(list(action = rl_env_as_action(action), label = "", kind = "", labels = character(),
                   values = numeric(), chosen = -1L, value = NaN, value_name = "expected return",
                   lo = NaN, hi = NaN, decimals = -1L, note = ""), class = "rl_decision")
}

# Greedy over Q(s, .): the argmax, the whole row, and max_a Q as the expected return.
rl_hud_greedy_q <- function(q, kind = "Q(s,a)") {
    chosen <- which.max(q) - 1L
    d <- rl_hud_decision(rl_env_action(chosen))
    d$values <- q
    d$chosen <- chosen
    d$kind <- kind
    d$value <- q[chosen + 1L]
    d
}

# The most probable action of a stochastic policy, with the probabilities and V(s).
rl_hud_greedy_pi <- function(probs, v = NaN, kind = "pi(a|s)") {
    chosen <- which.max(probs) - 1L
    d <- rl_hud_decision(rl_env_action(chosen))
    d$values <- probs
    d$chosen <- chosen
    d$kind <- kind
    d$lo <- 0
    d$hi <- 1
    d$decimals <- 2L
    d$value <- v
    d$value_name <- "V(s)"
    d
}

# An action with one estimate attached (a torque and its critic value, a plan's return).
rl_hud_explain <- function(action, value, value_name = "expected return", note = "") {
    d <- rl_hud_decision(action)
    d$value <- value
    d$value_name <- value_name
    d$note <- note
    d
}

# An action with a remark for the overlay: what a hand-written rule looked at.
rl_hud_note <- function(action, note) {
    d <- rl_hud_decision(action)
    d$note <- note
    d
}

# A number with as many decimals as its size warrants.
rl_hud_num <- function(v, decimals = -1L) {
    if (is.nan(v)) return("-")
    if (decimals < 0L) decimals <- if (abs(v) >= 100) 0L else if (abs(v) >= 10) 1L else 2L
    sprintf("%.*f", decimals, v)
}

rl_hud_signed <- function(v) if (is.nan(v)) "-" else sprintf("%+.2f", v)

# Everything the overlay shows: the episode so far and the current decision.
rl_hud_hud <- function() {
    self <- new.env()
    self$title <- ""
    self$reset <- function(space, action_names) {
        self$step <- 0L
        self$reward <- NaN
        self$ret <- 0
        self$done <- FALSE
        self$truncated <- FALSE
        self$has_decision <- FALSE
        self$decision <- rl_hud_decision()
        self$space <- space
        self$names <- action_names
    }
    # The move taken; the step that follows reports its reward through observe().
    self$choose <- function(decision) {
        self$decision <- decision
        self$has_decision <- TRUE
        self$complete()
    }
    self$observe <- function(result) {
        self$step <- self$step + 1L
        self$reward <- result$reward
        self$ret <- rl_env_f32(self$ret + result$reward)
        self$done <- rl_env_done(result)
        self$truncated <- result$truncated
    }
    self$name_of <- function(i, prefix) {
        if (i >= 0L && i < length(self$names)) self$names[[i + 1L]] else paste0(prefix, i)
    }
    # Fills the action label and the candidate labels the policy left empty.
    self$complete <- function() {
        d <- self$decision
        if (!nzchar(d$label)) {
            if (self$space$kind == "discrete") {
                d$label <- self$name_of(d$action$discrete, "action ")
            } else {
                parts <- character()
                for (i in seq_along(d$action$continuous)) {
                    prefix <- if (i <= length(self$names)) paste0(self$names[[i]], " ") else ""
                    parts <- c(parts, paste0(prefix, rl_hud_signed(d$action$continuous[i])))
                }
                d$label <- paste(parts, collapse = "  ")
            }
        }
        if (length(d$labels) != length(d$values)) {
            per_action <- self$space$kind == "discrete" && length(d$values) == self$space$n
            indices <- seq_along(d$values) - 1L
            d$labels <- vapply(indices, function(i) if (per_action) self$name_of(i, "action ") else as.character(i), "")
        }
        self$decision <- d
    }
    self$reset(rl_env_discrete_space(0L), character())
    self
}
