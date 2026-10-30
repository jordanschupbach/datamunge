#!/usr/bin/env python3
"""Run the datamunge::nn example ports, skipping ones that already passed.

Each example gets a key: the SHA-256 of its .cpp plus every repository header it
includes, followed transitively (examples/nn/common/..., include/datamunge/...).
The built libdatamunge's hash is recorded with each pass; with --with-library an
example also reruns when the library changed since it passed. An example is skipped when its key
equals the key of its last successful run; failures always rerun.

The cache lives next to the binaries (build/nn-examples/run-cache.tsv), so
deleting the build directory resets it.

  run_nn_examples.py                  run new or changed examples
  run_nn_examples.py --force          run everything
  run_nn_examples.py --with-library   also rerun after library changes
  run_nn_examples.py NAME...          only these examples (still cached)
  run_nn_examples.py --mark [NAME...] record NAMEs (default: all) as passed at
                                      their current key without running them
                                      (e.g. after a full run made elsewhere)
  run_nn_examples.py --list           show each example's state and exit
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "nn"
CPP_EXAMPLES = ROOT / "examples" / "cpp" / "source" / "nn"
INCLUDE_DIRS = [CPP_EXAMPLES, EXAMPLES, ROOT / "include"]
BIN_DIR = ROOT / "build" / "nn-examples" / "bin"
CACHE = ROOT / "build" / "nn-examples" / "run-cache.tsv"
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)

# Checkpoint consumers need their producers, including when selected alone.
PREREQUISITES = {
    name: ("vanilla_saliency",)
    for name in ("deepdream", "feature_visualization", "filter_activations", "grad_cam",
                 "guided_backprop", "integrated_gradients", "occlusion_sensitivity", "smoothgrad")
}
PREREQUISITES["receptive_fields"] = ("cnn_mnist",)
OUTPUT_BASE = ROOT / "output" / "nn"
OUTPUT_ROOT = Path(os.environ.get("DATAMUNGE_OUTPUT_DIR") or OUTPUT_BASE / "quick")
REQUIRED_OUTPUTS = {
    "cnn_mnist": (OUTPUT_ROOT / "models" / "cnn_mnist" / "best.npz",),
    "vanilla_saliency": tuple(OUTPUT_ROOT / "models" / "vanilla_saliency" / name
                              for name in ("best.npz", "model_arrays.npz", "initial_arrays.npz")),
}


def execution_order(names):
    """Include prerequisite examples once, before each checkpoint consumer."""
    ordered, visiting, seen = [], set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError(f"cyclic example prerequisites: {name}")
        if name in seen:
            return
        visiting.add(name)
        for prerequisite in PREREQUISITES.get(name, ()):
            visit(prerequisite)
        visiting.remove(name)
        seen.add(name)
        ordered.append(name)

    for name in names:
        visit(name)
    return ordered


def examples():
    """name -> source path for every example (.cpp under a numbered section)."""
    found = {}
    for path in sorted([*EXAMPLES.glob("[0-9]*/**/*.cpp"), *CPP_EXAMPLES.glob("[0-9]*/**/*.cpp")]):
        if path.stem in found:
            raise ValueError(f"duplicate NN example: {path.stem}")
        found[path.stem] = path
    return found


def resolve(name, including_dir):
    for base in [including_dir, *INCLUDE_DIRS]:
        candidate = base / name
        if candidate.is_file():
            return candidate.resolve()
    return None  # a system or third-party header


def dependencies(source):
    """The source and every repository file it includes, transitively."""
    seen, stack = set(), [source.resolve()]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        text = path.read_text(errors="replace")
        for name in INCLUDE_RE.findall(text):
            dep = resolve(name, path.parent)
            if dep is not None and ROOT in dep.parents:
                stack.append(dep)
    return sorted(seen)


def file_digest(path, cache={}):
    if path not in cache:
        cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cache[path]


def library_file():
    for pattern in ("_deps/datamunge-build/libdatamunge*", "**/libdatamunge.*"):
        hits = sorted(p for p in (ROOT / "build" / "nn-examples").glob(pattern) if p.is_file())
        if hits:
            return hits[0]
    return None


def source_key(source):
    h = hashlib.sha256()
    for dep in dependencies(source):
        h.update(str(dep.relative_to(ROOT)).encode())
        h.update(file_digest(dep).encode())
    return h.hexdigest()


def library_key():
    lib = library_file()
    return file_digest(lib) if lib is not None else "-"


def load_cache(path=None):
    """name -> (source key, library key at that pass)."""
    cache = {}
    path = path or CACHE
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                cache[parts[0]] = (parts[1], parts[2])
    return cache


def save_cache(cache, path=None):
    path = path or CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(f"{name}\t{src}\t{lib}\n" for name, (src, lib) in sorted(cache.items())))
    tmp.replace(path)


def run_example(binary, arguments=(), timeout=180, env=None, log_path=None):
    """Bound execution and terminate subprocesses too when the deadline expires."""
    if not binary.is_file():
        print(f"missing binary {binary}; build the examples first")
        return False
    with (log_path.open("w") if log_path else contextlib.nullcontext(None)) as output:
        process = subprocess.Popen([str(binary), *arguments], cwd=ROOT, env=env, start_new_session=True,
                                   stdout=output, stderr=subprocess.STDOUT if output else None)
        try:
            return process.wait(timeout=timeout) == 0
        except subprocess.TimeoutExpired:
            print(f"TIME LIMIT: {binary.name} exceeded {timeout:g} seconds; stopping its process group", flush=True)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="only these examples")
    parser.add_argument("--force", action="store_true", help="ignore the cache and run everything selected")
    parser.add_argument("--with-library", action="store_true", help="include the built library in each key")
    parser.add_argument("--mark", action="store_true", help="record NAMEs (default: all) as passed without running them")
    parser.add_argument("--list", action="store_true", help="print each example's state and exit")
    parser.add_argument("--full", action="store_true", default=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1",
                        help="opt into the original full training schedules and convergence checks")
    parser.add_argument("--timeout", type=float, help="per-example deadline in seconds (quick default: 180; full: unlimited)")
    parser.add_argument("--example-args", nargs=argparse.REMAINDER, help="arguments for one selected example; customized runs are not cached")
    parser.add_argument("--jobs", type=int, default=1, help="concurrent examples (default: 1; dependencies still run first)")
    parser.add_argument("--build-only", action="store_true", help="build the selected examples and their prerequisites, then exit")
    args = parser.parse_args()
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be positive and finite")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.example_args and len(args.names) != 1:
        parser.error("--example-args requires exactly one example")
    timeout = args.timeout if args.timeout is not None else (None if args.full else 180)
    profile = "full" if args.full else "quick"
    cache_path = CACHE.with_name(f"{CACHE.stem}-{profile}{CACHE.suffix}")
    output_root = Path(os.environ.get("DATAMUNGE_OUTPUT_DIR") or OUTPUT_BASE / profile)
    required_outputs = REQUIRED_OUTPUTS if output_root == OUTPUT_ROOT else {
        name: tuple(output_root / p.relative_to(OUTPUT_ROOT) for p in paths)
        for name, paths in REQUIRED_OUTPUTS.items()
    }
    env = dict(os.environ, DATAMUNGE_EXAMPLES_FULL="1" if args.full else "0",
               DATAMUNGE_OUTPUT_DIR=str(output_root))
    env.setdefault("DATAMUNGE_NUM_THREADS", "2")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "2")

    all_examples = examples()
    unknown = [n for n in args.names if n not in all_examples]
    if unknown:
        sys.exit("run_nn_examples: unknown example(s): " + " ".join(unknown))
    selected = execution_order(args.names or list(all_examples))
    if args.build_only:
        return subprocess.run(["cmake", "--build", str(BIN_DIR.parent), "--parallel",
                               os.environ.get("DATAMUNGE_EXAMPLE_BUILD_JOBS", "4"),
                               "--target", *selected], cwd=ROOT).returncode
    cache = load_cache(cache_path)
    lib = library_key()
    if args.with_library and lib == "-":
        sys.exit("run_nn_examples: --with-library needs the built library; build the examples first")
    keys = {name: (source_key(all_examples[name]), lib) for name in selected}

    def up_to_date(name):
        if name not in cache:
            return False
        if any(not path.is_file() for path in required_outputs.get(name, ())):
            return False
        src, lib_then = cache[name]
        return src == keys[name][0] and (not args.with_library or lib_then == keys[name][1])

    if args.mark:
        for name in selected:
            cache[name] = keys[name]
        save_cache(cache, cache_path)
        print(f"marked {len(selected)} example(s) as passed")
        return 0

    stale = []
    for name in selected:
        if (args.force or not up_to_date(name)
                or any(dep in stale for dep in PREREQUISITES.get(name, ()))):
            stale.append(name)
    if args.list:
        for name in selected:
            state = "up to date" if name not in stale else ("changed" if name in cache else "never passed")
            print(f"{name:40s} {state}")
        return 0

    print(f"nn examples ({profile}): {len(stale)} to run, {len(selected) - len(stale)} unchanged since they last passed")
    passed, failed = [], []

    def execute(name):
        binary = BIN_DIR / name
        print(f"\n========== {name} ==========", flush=True)
        start = time.monotonic()
        arguments = (args.example_args or ()) if name in args.names else ()
        log_path = None
        if args.jobs > 1:
            log_path = CACHE.parent / "run-logs" / profile / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
        ok = run_example(binary, arguments, timeout, env, log_path)
        elapsed = time.monotonic() - start
        return ok, elapsed, arguments, log_path

    def record(name, result):
        ok, elapsed, arguments, log_path = result
        if ok:
            passed.append(name)
            if not arguments:
                cache[name] = keys[name]
        else:
            failed.append(name)
            cache.pop(name, None)
        save_cache(cache, cache_path)  # interrupted runs keep their progress
        print(f"---------- {name}: {'passed' if ok else 'FAILED'} in {elapsed:.1f} s", flush=True)
        if log_path:
            print(f"  log: {log_path}", flush=True)

    pending, completed = list(stale), set(selected) - set(stale)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        running = {}
        while pending or running:
            for name in list(pending):
                prerequisites = PREREQUISITES.get(name, ())
                if any(dep in failed for dep in prerequisites):
                    print(f"not running {name}: checkpoint prerequisite failed", flush=True)
                    record(name, (False, 0., (), None))
                    pending.remove(name)
                    completed.add(name)
                elif len(running) < args.jobs and all(dep in completed for dep in prerequisites):
                    running[pool.submit(execute, name)] = name
                    pending.remove(name)
            if running:
                done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    name = running.pop(future)
                    record(name, future.result())
                    completed.add(name)

    print(f"\n########## nn examples: {len(passed)} passed, {len(failed)} failed, "
          f"{len(selected) - len(stale)} skipped (unchanged) ##########")
    if failed:
        print("FAILED: " + " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
