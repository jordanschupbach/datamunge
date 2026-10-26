#!/usr/bin/env python3
"""Validate native Python/R NN ports against their C++ reference and record evidence.

For each NAME this builds and runs the C++ example, then the native Python and
R scripts, with the development budget (epochs=5 when the example takes an
epoch option). Logs go to build/nn-language-validation/NAME.{cpp,python,r}.log
and passing runs are recorded in examples/nn/language_status.json. Safe to run
concurrently: ledger updates and C++ builds are serialized with file locks.

    python3 scripts/validate_nn_ports.py NAME... [--classify NAME=yes|no] [--timeout S]
"""
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nn_migration_status  # noqa: E402

LOGS = ROOT / "build/nn-language-validation"
LOCKS = ROOT / "build/nn-language-validation/locks"
NIX = ["nix", "develop", "--accept-flake-config"]
R_EXPR = "a<-commandArgs(TRUE);quit(status=datamungerNN::nn_example_run(a[[1L]],a[-1L]))"


@contextlib.contextmanager
def locked(name):
    LOCKS.mkdir(parents=True, exist_ok=True)
    with open(LOCKS / name, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def execute(command, log, timeout, env=None):
    started = time.monotonic()
    with open(log, "w") as output:
        output.write("$ " + " ".join(command) + "\n")
        output.flush()
        try:
            process = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                     timeout=timeout, env=env, start_new_session=True)
            status = process.returncode
        except subprocess.TimeoutExpired:
            output.write(f"\nTIMEOUT after {timeout} s\n")
            status = "timeout"
    return status, round(time.monotonic() - started, 3)


def wants_epochs(cpp_source):
    return "epochs" in (ROOT / cpp_source).read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--classify", action="append", default=[], metavar="NAME=yes|no")
    parser.add_argument("--timeout", type=int, default=900, help="per-language seconds (C++ uses min(180, this))")
    parser.add_argument("--languages", default="cpp,python,r")
    args = parser.parse_args()
    classify = {}
    for item in args.classify:
        name, _, value = item.partition("=")
        if value not in ("yes", "no"):
            parser.error("--classify takes NAME=yes|no")
        classify[name] = value == "yes"
    languages = args.languages.split(",")
    LOGS.mkdir(parents=True, exist_ok=True)
    with locked("ledger"):
        report = nn_migration_status.refresh(write_catalog=False)
    failures = 0
    for name in args.names:
        entry = report["examples"].get(name)
        if entry is None:
            print(f"{name}: unknown example")
            failures += 1
            continue
        extra = ["epochs=5"] if wants_epochs(entry["cpp"]) else []
        results = {}
        if "cpp" in languages:
            with locked("cpp-build"):
                status, _ = execute(NIX + [".#cpp-tools", "--command", "python3", "scripts/run_nn_examples.py",
                                           "--build-only", name], LOGS / f"{name}.cpp-build.log", 1800)
            if status != 0:
                results["cpp"] = (status, 0, None)
            else:
                env = dict(os.environ, DATAMUNGE_OUTPUT_DIR=str(ROOT / "output/nn/language-migration/cpp/quick"),
                           DATAMUNGE_EXAMPLES_FULL="0")
                command = [str(ROOT / "build/nn-examples/bin" / name)] + extra
                results["cpp"] = execute(command, LOGS / f"{name}.cpp.log", min(180, args.timeout), env) + (command,)
        if "python" in languages:
            env = dict(os.environ, DATAMUNGE_OUTPUT_DIR=str(ROOT / "output/nn/language-migration/python/quick"))
            command = ["build/venv/pydatamunge-run/bin/python", "-I", "scripts/python_example.py", name] + extra + ["--quick"]
            results["python"] = execute(NIX + [".#python", "--command"] + command, LOGS / f"{name}.python.log",
                                        args.timeout, env) + (command,)
        if "r" in languages:
            env = dict(os.environ, DATAMUNGE_OUTPUT_DIR=str(ROOT / "output/nn/language-migration/r/quick"),
                       R_LIBS_USER=str(ROOT / "build/r/library"))
            command = ["Rscript", "--vanilla", "-e", R_EXPR, name] + extra + ["--quick"]
            results["r"] = execute(NIX + [".#r", "--command"] + command, LOGS / f"{name}.r.log",
                                   args.timeout, env) + (command,)
        with locked("ledger"):
            report = nn_migration_status.refresh(write_catalog=False)
            record = report["examples"][name]
            if name in classify:
                record["classification"] = classify[name]
            summary = []
            for language, (status, seconds, command) in results.items():
                log = LOGS / f"{name}.{language}.log"
                text = log.read_text(errors="replace") if log.exists() else ""
                confusion = "onfusion" in text
                ok = status == 0 and (not record.get("classification") or confusion)
                if language != "cpp" and record[language]["implementation"] != "native":
                    ok = False
                summary.append(f"{language}={'PASS' if ok else 'FAIL'}({status},{seconds}s)")
                failures += not ok
                evidence = dict(status="passed", command=command, log=log.relative_to(ROOT).as_posix(),
                                seconds=seconds, epochs=5 if extra else None, confusion_matrix=confusion)
                if language == "cpp":
                    record["cpp_reference_validation"] = evidence if ok else None
                else:
                    record[language]["validation"] = evidence if ok else None
            nn_migration_status.STATUS.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: " + " ".join(summary), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
