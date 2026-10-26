"""The agent's explained move and the overlay state (examples/nn/common/rl/hud.hpp).

A Decision is one move as the agent explains it: the action, an estimate per candidate
action, the candidate taken and the return it expects. Drawing the overlay onto video frames
is out of scope (as in the C++ port); the bookkeeping is kept so policies written for the
recorders work unchanged.
"""
import math

from pydatamunge.nn_workflows import helper

rl_env = helper("rl_env")
Action, ActionSpace, as_action = rl_env.Action, rl_env.ActionSpace, rl_env.as_action

NAN = float("nan")


class Decision:
    def __init__(self, action=None):
        self.action = Action() if action is None else as_action(action)
        self.label = ""          # the move in words; from the action space when empty
        self.kind = ""           # what `values` are: "Q(s,a)", "pi(a|s)", ...
        self.labels = []         # one per candidate
        self.values = []         # the agent's estimate per candidate action
        self.chosen = -1
        self.value = NAN         # the return the agent expects from here
        self.value_name = "expected return"
        self.lo = NAN
        self.hi = NAN
        self.decimals = -1
        self.note = ""


def greedy_q(q, kind="Q(s,a)"):
    d = Decision()
    d.values = list(q)
    d.chosen = max(range(len(q)), key=lambda i: (q[i], -i))  # first maximum
    d.action = Action(d.chosen)
    d.kind = kind
    d.value = q[d.chosen]
    return d


def greedy_pi(probs, v=NAN, kind="pi(a|s)"):
    d = Decision()
    d.values = list(probs)
    d.chosen = max(range(len(probs)), key=lambda i: (probs[i], -i))
    d.action = Action(d.chosen)
    d.kind = kind
    d.lo, d.hi = 0.0, 1.0
    d.decimals = 2
    d.value = v
    d.value_name = "V(s)"
    return d


def explain(action, value, value_name="expected return", note=""):
    d = Decision(action)
    d.value = value
    d.value_name = value_name
    d.note = note
    return d


def note(action, text):
    d = Decision(action)
    d.note = text
    return d


def hud_num(v, decimals=-1):
    if math.isnan(v):
        return "-"
    if decimals < 0:
        decimals = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2
    return "%.*f" % (decimals, v)


def hud_signed(v):
    return "-" if math.isnan(v) else "%+.2f" % v


class Hud:
    """Everything the overlay shows: the episode so far and the current decision."""

    def __init__(self):
        self.title = ""
        self.reset(ActionSpace(), [])

    def reset(self, space, action_names):
        self.step = 0
        self.reward = NAN
        self.ret = 0.0
        self.done = self.truncated = False
        self.has_decision = False
        self.decision = Decision()
        self.space = space
        self.names = list(action_names)

    def choose(self, decision):
        self.decision = decision
        self.has_decision = True
        self.complete()

    def observe(self, result):
        self.step += 1
        self.reward = result.reward
        self.ret = rl_env.f32(self.ret + result.reward)
        self.done = result.done()
        self.truncated = result.truncated

    def name_of(self, i, prefix):
        return self.names[i] if 0 <= i < len(self.names) else prefix + str(i)

    def complete(self):
        d = self.decision
        if not d.label:
            if self.space.kind == ActionSpace.DISCRETE:
                d.label = self.name_of(d.action.discrete, "action ")
            else:
                parts = []
                for i, value in enumerate(d.action.continuous):
                    parts.append((self.names[i] + " " if i < len(self.names) else "") + hud_signed(value))
                d.label = "  ".join(parts)
        if len(d.labels) != len(d.values):
            per_action = self.space.kind == ActionSpace.DISCRETE and len(d.values) == self.space.n
            d.labels = [self.name_of(i, "action ") if per_action else str(i) for i in range(len(d.values))]
