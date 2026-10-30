#!/usr/bin/env python3
"""The C++ compiler for the docs build: report blocks against a precompiled dm.hpp.

mep compiles each C++ block of a report as `$CXX BLOCK.cpp -o BIN FLAGS... LIBS...`,
and nearly all of that time goes on parsing <dm/dm.hpp>. The docs build makes
this program mep's $CXX (an older mep, which always runs `g++`, finds it first
on its PATH as `g++` instead; see reports.toolchain), with DM_PCH_FILE (clang's
precompiled dm.hpp), DM_PCH_FLAGS (the compile flags it was built with, one per
line) and DM_PCH_REAL_GXX (the compiler it stands in for, one argument per line).

A block whose first #include is <dm/dm.hpp> and whose compile flags are the
precompiled header's is compiled by clang++ with that header: the same
libstdc++ the library was built with, about six times faster. Anything else,
or a block clang++ cannot compile, goes to the real compiler unchanged, so a
block's errors are always that compiler's.
"""
import os
import re
import subprocess
import sys

FIRST_INCLUDE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.M)


def compile_flags(argv):
    """The arguments that decide how a header is compiled (include paths, macros, -std, -O, -f, -m)."""
    out = []
    it = iter(argv)
    for a in it:
        if a in ("-I", "-D", "-U", "-isystem", "-include"):
            out += [a, next(it, "")]
        elif a.startswith(("-I", "-D", "-U", "-std=", "-O", "-f", "-m", "-isystem")):
            out.append(a)
    return out


def sources(argv):
    return [a for a in argv if not a.startswith("-") and a.endswith((".cpp", ".cc", ".cxx", ".C"))]


def use_pch(argv, pch, pch_flags):
    if not pch or not os.path.exists(pch):
        return False
    srcs = sources(argv)
    if len(srcs) != 1 or compile_flags(argv) != pch_flags:
        return False
    try:
        with open(srcs[0], encoding="utf-8", errors="replace") as f:
            m = FIRST_INCLUDE.search(f.read(64 * 1024))
    except OSError:
        return False
    return bool(m and m.group(1) == "dm/dm.hpp")


def main(argv):
    real = [w for w in (os.environ.get("DM_PCH_REAL_GXX") or "g++").split("\n") if w]
    pch = os.environ.get("DM_PCH_FILE", "")
    pch_flags = [f for f in os.environ.get("DM_PCH_FLAGS", "").split("\n") if f]
    if use_pch(argv, pch, pch_flags):
        clang = os.environ.get("DM_PCH_CLANG") or "clang++"
        r = subprocess.run([clang, "-include-pch", pch, *argv], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return 0
    os.execvp(real[0], [*real, *argv])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
