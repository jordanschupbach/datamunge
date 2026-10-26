"""Plays one recorded episode without a video (examples/nn/common/rl/playback.hpp).

tflow films episodes with vsim's renderer and ffmpeg; both are out of scope. play_episode
runs exactly the episode the recorder would film (same reset seed, the same Decide/Hud
bookkeeping, one env step per iteration) and prints the same summary line. It is a playback
utility, not an experiment: examples call it for one episode they already ran explicitly.
"""
from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
rl_hud = helper("rl_hud")
f32 = rl_env.f32


def decide(policy, obs):
    """The policy's decision for obs: a Decision passes through, a bare action is wrapped."""
    choice = policy(obs)
    return choice if isinstance(choice, rl_hud.Decision) else rl_hud.Decision(choice)


class Playback:
    def __init__(self):
        self.steps = 0
        self.ret = 0.0
        self.hud = rl_hud.Hud()


def play_episode(env, policy, name, seed, after_step=None):
    """One episode of policy from env.reset(seed); after_step(env, hud) runs after the reset and every step."""
    pb = Playback()
    pb.hud.reset(env.action_space(), env.action_names())
    obs = env.reset(seed)
    if after_step is not None:
        after_step(env, pb.hud)
    while True:
        pb.hud.choose(decide(policy, obs))
        result = env.step(pb.hud.decision.action)
        pb.hud.observe(result)
        pb.ret = f32(pb.ret + result.reward)
        pb.steps += 1
        obs = result.obs
        if after_step is not None:
            after_step(env, pb.hud)
        if result.done():
            break
    print("  played %s (video omitted): %d steps, return %.1f" % (name, pb.steps, pb.ret))
    return pb
