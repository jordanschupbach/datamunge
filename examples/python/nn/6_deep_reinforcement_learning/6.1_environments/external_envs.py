# Native Python NN example. Binding to external environments: ALE (Atari 2600) and vsim.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/6_deep_reinforcement_learning/6.1_environments/external_envs.py
"""Binding to external environments: ALE (Atari 2600) and vsim.

Port of tflow's external_envs. tflow reaches two real simulators through
their C++ APIs:
  - the Arcade Learning Environment, wrapped as rl::AtariEnv with DQN-style
    preprocessing (common/rl/ale_env.hpp);
  - vsim, the game engine whose scene files and Jolt-based PhysicsWorld the
    articulated tasks in tflow's common/rl/vsim_env.h are built on.
Atari ROMs are not redistributable; the free Tetris homebrew ships with ALE
and other ROMs can be dropped into data/roms/.

Deviations: ALE and vsim are not datamunge dependencies (the plan keeps
emulators, vsim and video recording out of scope). The Atari demo runs only
when the example is compiled with ALE on the include path and linked with
libale (DATAMUNGE_EXAMPLES_HAVE_ALE); `just run-nn` builds without it and
then only reports where the ROMs are. The vsim demo (a ball knocking over a
crate stack, stepped at 120 Hz) and both videos are omitted.
"""
import os

from pydatamunge.nn_workflows import helper, options

rl_ale_env = helper("rl_ale_env")


def atari_demo():
    print("#### ALE")
    # rl_ale_env has no emulator (HAVE_ALE is False), as in the C++ built without ALE.
    print("  ALE is not built in (see the header): the Atari demo is skipped.")
    try:
        print("  tetris ROM: " + rl_ale_env.find_rom("tetris"))
    except RuntimeError as error:
        print("  " + str(error))


def vsim_demo():
    print("\n#### vsim")
    print("  vsim is not available to the datamunge ports (out of scope): the box-stack scene demo is skipped.")


def main(arguments=None):
    options(arguments, dict(full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    atari_demo()
    vsim_demo()
    print("\n  no videos are written by the port")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
