# Plays one recorded episode without a video (examples/nn/common/rl/playback.hpp).
#
# tflow films episodes with vsim's renderer and ffmpeg; both are out of scope.
# rl_playback_play_episode runs exactly the episode the recorder would film (same reset seed,
# the same decide/hud bookkeeping, one env step per iteration) and prints the same summary
# line. It is a playback utility, not an experiment: examples call it for one episode they
# already ran explicitly.

nn_helper("rl_env", environment())
nn_helper("rl_hud", environment())

# The policy's decision for obs: a decision passes through, a bare action is wrapped.
rl_playback_decide <- function(policy, obs) {
    choice <- policy(obs)
    if (inherits(choice, "rl_decision")) choice else rl_hud_decision(choice)
}

# One episode of policy from env$reset(seed); after_step(env, hud) runs after the reset and
# after every step. Returns list(steps, ret, hud).
rl_playback_play_episode <- function(env, policy, name, seed, after_step = NULL) {
    hud <- rl_hud_hud()
    hud$reset(env$action_space(), env$action_names())
    steps <- 0L
    ret <- 0
    obs <- env$reset(seed)
    if (!is.null(after_step)) after_step(env, hud)
    repeat {
        hud$choose(rl_playback_decide(policy, obs))
        result <- env$step(hud$decision$action)
        hud$observe(result)
        ret <- rl_env_f32(ret + result$reward)
        steps <- steps + 1L
        obs <- result$obs
        if (!is.null(after_step)) after_step(env, hud)
        if (rl_env_done(result)) break
    }
    cat(sprintf("  played %s (video omitted): %d steps, return %.1f\n", name, steps, ret))
    list(steps = steps, ret = ret, hud = hud)
}
