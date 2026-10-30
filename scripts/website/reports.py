"""The mepml reports: run with mep, exported by mep, fitted into the website.

With `run`, each report's code blocks are run first by mep itself
(`mep-mepml run`, the editor's C-c C-c without a window), which writes their
results and figures back into the report. A report is rerun only when it
changed since its last run: each run leaves a JSON record of how every block
went in the status directory (kept between builds), which the report's page
(a warning when a block failed) and the status page show.

Then mep's HTML export of each report (`mep-mepml convert X.mepml X.html`) is
cut down to its body: mep's own contents list gives way to the site's, code
blocks are highlighted again with their datamunge names linked to the API
reference, and the pictures it names are copied next to the page.
"""
import glob
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

BODY = re.compile(r"<body[^>]*>(.*)</body>", re.S)
CODE = re.compile(r'<pre><code class="language-([\w+#-]+)">(.*?)</code></pre>', re.S)
CODE_PLAIN = re.compile(r"<pre><code>(.*?)</code></pre>", re.S)
TITLE = re.compile(r'<h1 class="title">(.*?)</h1>', re.S)
SUBTITLE = re.compile(r'<p class="subtitle">(.*?)</p>', re.S)
TOC = re.compile(r'<nav[^>]*class="toc"[^>]*>.*?</nav>', re.S)
SRC = re.compile(r'(<(?:img|video|source)\b[^>]*?\s(?:src|poster)=")([^"]+)(")')
IMG = re.compile(r'<img\b[^>]*>')
ABSTRACT = re.compile(r'<section[^>]*class="abstract"[^>]*>(.*?)</section>', re.S)
CPP_BLOCK = re.compile(r'^```(?:\{\s*(?:cpp|c\+\+)(?=[\s,}])[^\n]*\}|cpp|c\+\+)\s*\n(.*?)^```\s*$', re.M | re.S)
LIBRARY_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](?:dm/dm\.hpp|datamunge/[^>"]+)[>"]', re.M)


class ToolchainError(RuntimeError):
    """Report setup failed before any report results or figures were rewritten."""


def validate_cpp_toolchain(paths, env=None):
    """Check package discovery and compile/link/run a public-header canary once.

    Only reports with a C++ datamunge include need this check. The caller's CXX,
    package overrides and the reports' common flags are used, independently of
    whether optional PCH acceleration is enabled. Fail before launching workers
    so one broken environment cannot overwrite hundreds of report statuses.
    """
    library_paths = []
    for path in paths:
        with open(path, encoding="utf-8") as source:
            if any(LIBRARY_INCLUDE.search(block) for block in CPP_BLOCK.findall(source.read())):
                library_paths.append(path)
    if not library_paths:
        return
    env = mep_env(env)

    def checked(command, stage):
        """Run a bounded setup probe, retaining native diagnostics on failure."""
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ToolchainError(f"Docs C++ toolchain: {stage}: {error}") from error
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise ToolchainError(
                f"Docs C++ toolchain: {stage} failed (exit {result.returncode}).\n"
                f"Command: {shlex.join(command)}\n{detail[-12000:]}\n"
                "Use `nix develop .#cpp` for matching headers and library. Check any "
                "DATAMUNGE_MEPML_PKG_CONFIG_PATH / DOCS_MEPML_PKG_CONFIG_PATH override.")
        return result.stdout.strip()

    package_flags = shlex.split(checked(
        ["pkg-config", "--cflags", "--libs", "datamunge"], "pkg-config datamunge discovery"))
    libdir = checked(["pkg-config", "--variable=libdir", "datamunge"], "library directory discovery")
    flags_text = common_cpp_flags(library_paths)
    try:
        flags = shell_words(flags_text, env) if flags_text else [*package_flags, "-Wl,-rpath," + libdir]
    except RuntimeError as error:
        raise ToolchainError(f"Docs C++ toolchain: report compiler flags: {error}") from error
    compiler = shlex.split(env.get("CXX") or "g++")
    with tempfile.TemporaryDirectory(prefix="docs-cxx-check-") as temporary:
        source = os.path.join(temporary, "check.cpp")
        executable = os.path.join(temporary, "check")
        with open(source, "w") as output:
            output.write("#include <dm/dm.hpp>\nint main() { "
                         "return dm::nn::scalar(1).item<float>() == 1 ? 0 : 1; }\n")
        checked([*compiler, "-std=c++23", source, "-o", executable, *flags],
                "compiling/linking <dm/dm.hpp>")
        checked([executable], "executing the linked datamunge canary")
    print("Docs C++ toolchain: datamunge headers, linking and execution OK", flush=True)


class Report:
    def __init__(self, path, rel):
        self.path = path            # the .mepml
        self.rel = rel              # its path under docs/mepml/examples
        self.url = "reports/" + os.path.splitext(rel)[0] + ".html"
        self.title = os.path.splitext(os.path.basename(rel))[0].replace("_", " ")
        self.subtitle = ""
        self.summary = ""
        self.body = ""
        self.status = None

    @property
    def group(self):
        return os.path.dirname(self.rel)

    @property
    def broken(self):
        return bool(self.status and (self.status.get("failed") or self.status.get("error")))


def shorten(text, n):
    """`text` cut to about `n` characters at a space, never inside a \\( ... \\) formula."""
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    if cut.count("\\(") > cut.count("\\)"):
        cut = cut[:cut.rindex("\\(")].rstrip()
    return cut + "…"


def mep_env(extra=None):
    """The environment mep runs in: the caller's without LD_LIBRARY_PATH (plus `extra`).

    mep is a Nix-built program that finds its libraries itself; a library path
    from another environment (a shell's alsa-lib built against a newer glibc)
    stops it from starting at all."""
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(extra or {})
    return env


def default_jobs():
    """How many reports to run at once: one per core, as many as memory allows
    (a block's compile takes up to about 1.5 GB)."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        with open("/proc/meminfo") as f:
            avail_kb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:"))
        cores = min(cores, max(1, avail_kb // (1536 * 1024)))
    except (OSError, StopIteration, ValueError):
        pass
    return max(1, cores)


# -- the precompiled dm.hpp ---------------------------------------------------------
# Almost all of a block's time is its compiler parsing <dm/dm.hpp>. With clang++ on
# PATH, the report blocks are compiled against a precompiled dm.hpp instead
# (pch_cxx.py, standing in for g++ on mep's PATH); see pch_cxx.py for which blocks.

def shell_words(text, env):
    """`text` split into words by sh, as mep splits a block's flags= (expanding $(...))."""
    r = subprocess.run(["sh", "-c", 'printf "%s\\0" ' + text], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"cannot expand {text!r}")
    return [w for w in r.stdout.split("\0") if w]


def common_cpp_flags(paths):
    """The flags="..." most of the reports' C++ blocks use, or None."""
    counts = defaultdict(int)
    head = re.compile(r"^```\{\s*cpp\b([^\n]*)\}\s*$", re.M)
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for m in head.finditer(f.read()):
                fm = re.search(r'\bflags="([^"]*)"', m.group(1))
                if fm:
                    counts[fm.group(1)] += 1
    return max(counts, key=counts.get) if counts else None


def mep_honors_cxx(mep_mepml, work_dir, env):
    """Whether this mep compiles C++ blocks with $CXX (newer meps do; older ones always run g++).

    Found by running a one-block probe report with CXX set to a script that leaves a
    mark, once per mep binary (the answer is kept in `work_dir`)."""
    exe = shutil.which(mep_mepml, path=env.get("PATH")) or mep_mepml
    key = os.path.realpath(exe)
    cache = os.path.join(work_dir, "mep-cxx.json")
    try:
        with open(cache) as f:
            known = json.load(f)
    except (OSError, ValueError):
        known = {}
    if key in known:
        return known[key]
    probe = tempfile.mkdtemp(prefix="cxx-probe-", dir=work_dir)
    try:
        mark, cxx, doc = (os.path.join(probe, n) for n in ("used", "cxx", "probe.mepml"))
        with open(cxx, "w") as f:
            f.write(f'#!/bin/sh\ntouch "{mark}"\nexec g++ "$@"\n')
        os.chmod(cxx, 0o755)
        with open(doc, "w") as f:
            f.write("```{cpp}\nint main() { return 0; }\n```\n")
        try:
            subprocess.run([exe, "run", "--quiet", doc], env={**env, "CXX": cxx}, capture_output=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            pass
        honors = os.path.exists(mark)
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    known[key] = honors
    tmp = f"{cache}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(known, f)
    os.replace(tmp, cache)
    return honors


def toolchain(work_dir, reports_paths, mep_mepml="mep-mepml"):
    """The environment that makes mep compile report blocks with the precompiled
    dm.hpp ({} when it cannot: no clang++, no dm.hpp, the header does not compile).

    The header is precompiled once, under `work_dir`, and again only when clang,
    the flags or a header it includes changed. mep is pointed at the launcher
    (pch_cxx.py) with $CXX; a mep too old to read $CXX gets it as the `g++` on
    its PATH instead."""
    import hashlib
    import shlex
    import pch_cxx

    env = mep_env()
    clang = shutil.which("clang++", path=env.get("PATH"))
    # The compiler the launcher stands in for: the caller's $CXX (`g++` in a Nix shell), else g++.
    real = shlex.split(env.get("CXX") or "g++")
    real[0] = shutil.which(real[0], path=env.get("PATH")) if real else None
    real = real if real and real[0] else None
    flags_text = common_cpp_flags(reports_paths)
    if not clang or not real or not flags_text:
        print("precompiled dm.hpp: off (" + ("no clang++ on PATH" if not clang else "no $CXX or g++ on PATH" if not real
                                             else "no C++ blocks with flags=") + ")", flush=True)
        return {}
    try:
        flags = pch_cxx.compile_flags(shell_words(flags_text, env))
    except RuntimeError as e:
        print(f"precompiled dm.hpp: off (the blocks' flags: {e})", flush=True)
        return {}
    incs = [f[2:] for f in flags if f.startswith("-I") and len(f) > 2] + \
           [flags[i + 1] for i, f in enumerate(flags[:-1]) if f in ("-I", "-isystem")]
    header = next((os.path.join(d, "dm", "dm.hpp") for d in incs if os.path.isfile(os.path.join(d, "dm", "dm.hpp"))),
                  None)
    if not header:
        print("precompiled dm.hpp: off (dm/dm.hpp is not under the blocks' include paths)", flush=True)
        return {}
    version = subprocess.run([clang, "--version"], capture_output=True, text=True).stdout
    key = hashlib.sha256("\0".join([os.path.realpath(clang), version, header, *flags]).encode()).hexdigest()[:16]
    os.makedirs(work_dir, exist_ok=True)
    pch = os.path.join(work_dir, f"dm-{key}.pch")
    dep = pch + ".d"
    if not pch_fresh(pch, dep):
        print(f"precompiling {header} with {os.path.basename(clang)}", flush=True)
        tmp = f"{pch}.{os.getpid()}.tmp"
        r = subprocess.run([clang, *flags, "-fpch-instantiate-templates", "-x", "c++-header", header,
                            "-o", tmp, "-MD", "-MF", dep], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print("precompiled dm.hpp: off (clang++ cannot compile it):\n" + r.stderr[-2000:], flush=True)
            return {}
        os.replace(tmp, pch)
        for old in glob.glob(os.path.join(work_dir, "dm-*.pch")):
            if old != pch:
                for stale_file in (old, old + ".d"):
                    if os.path.exists(stale_file):
                        os.remove(stale_file)
    launch = f'#!/bin/sh\nexec "{sys.executable}" "{os.path.abspath(pch_cxx.__file__)}" "$@"\n'
    out = {"DM_PCH_FILE": pch, "DM_PCH_FLAGS": "\n".join(flags), "DM_PCH_REAL_GXX": "\n".join(real),
           "DM_PCH_CLANG": clang}
    if mep_honors_cxx(mep_mepml, work_dir, env):
        launcher = os.path.join(work_dir, "dm-pch-cxx")
        write_script(launcher, launch)
        out["CXX"] = launcher
        how = "$CXX"
    else:
        bindir = os.path.join(work_dir, "bin")
        os.makedirs(bindir, exist_ok=True)
        write_script(os.path.join(bindir, "g++"), launch)
        out["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        how = "the g++ on mep's PATH (this mep ignores $CXX; `just update-mep`)"
    print(f"report blocks that include <dm/dm.hpp> compile with {os.path.basename(clang)} and a precompiled "
          f"dm.hpp ({os.path.basename(pch)}), via {how}", flush=True)
    return out


def write_script(path, text):
    """Write an executable script, atomically (several builds may share it)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.chmod(tmp, 0o755)
    os.replace(tmp, path)


def pch_fresh(pch, dep):
    """Whether `pch` is newer than every file its depfile lists."""
    if not (os.path.exists(pch) and os.path.exists(dep)):
        return False
    with open(dep) as f:
        text = f.read().replace("\\\n", " ")
    files = text.split(":", 1)[1].split() if ":" in text else []
    t = os.path.getmtime(pch)
    try:
        return all(os.path.getmtime(p) <= t for p in files)
    except OSError:
        return False


def report_rel(path, examples_root):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(examples_root)).replace(os.sep, "/")


def is_stale(path, status_path, force=False):
    """Whether the report has to run: it changed since its last run, or that run failed."""
    if force or not os.path.exists(status_path) or os.path.getmtime(path) > os.path.getmtime(status_path):
        return True
    # A report that failed (or that mep could not run) is rerun until it passes:
    # the fix may be in the library, not in the report.
    try:
        with open(status_path) as f:
            last = (json.load(f).get("files") or [{}])[0]
        return bool(last.get("failed") or last.get("error"))
    except (OSError, ValueError):
        return True


def run_report(mep_mepml, path, status_path, timeout=0, env=None):
    """Run the report's code blocks with mep, recording how they went in `status_path`.

    Returns (seconds, mep's output)."""
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    cmd = [mep_mepml, "run", "--quiet", "--report", status_path, path]
    if timeout:
        cmd += ["--timeout", str(timeout)]
    t0 = time.time()
    r = subprocess.run(cmd, env=mep_env(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode not in (0, 1) or not os.path.exists(status_path):
        # mep could not run it at all (missing, too old, crashed).
        with open(status_path, "w") as f:
            json.dump({"files": [{"file": os.path.abspath(path), "failed": 0, "skipped": 0, "blocks": [],
                                  "error": f"mep-mepml run exited with status {r.returncode}"}]}, f)
    return time.time() - t0, r.stdout


def prerequisites(path):
    """Read native-report prerequisites declared as relative `//? Requires:` paths."""
    with open(path, encoding="utf-8") as source:
        names = re.findall(r"^//\? Requires:\s*(.+?)\s*$", source.read(), re.M)
    result = []
    for name in names:
        required = os.path.abspath(os.path.join(os.path.dirname(path), name))
        if not os.path.isfile(required) or not required.endswith(".mepml"):
            raise ToolchainError(f"Missing report prerequisite {name!r} required by {path}")
        result.append(required)
    return result


def with_prerequisites(jobs_list, status_path):
    """Include prerequisite reports once, validating missing paths and cycles before execution."""
    ordered, active, seen = [], set(), set()
    supplied = {os.path.abspath(path): status for path, status in jobs_list}

    def visit(path):
        path = os.path.abspath(path)
        if path in active:
            raise ToolchainError(f"Cyclic report prerequisites involving {path}")
        if path in seen:
            return
        active.add(path)
        for required in prerequisites(path):
            visit(required)
        active.remove(path)
        seen.add(path)
        ordered.append((path, supplied.get(path) or status_path(path)))

    for path, _ in jobs_list:
        visit(path)
    return ordered


def run_reports(mep_mepml, jobs_list, timeout=0, jobs=1, env=None):
    """Run each (path, status_path) in `jobs_list`, `jobs` reports at a time."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def last_seconds(job):
        try:
            with open(job[1]) as f:
                return sum(b.get("seconds") or 0 for b in (json.load(f).get("files") or [{}])[0].get("blocks", []))
        except (OSError, ValueError):
            return float("inf")  # never ran: could be long

    # The slowest reports first (by their last run), so none of them starts last.
    jobs_list = sorted(jobs_list, key=last_seconds, reverse=True)
    pending = {os.path.abspath(path): (path, status) for path, status in jobs_list}
    dependencies = {path: set(prerequisites(path)) for path in pending}
    required = set().union(*dependencies.values()) if dependencies else set()
    missing = required - pending.keys()
    if missing:
        raise ToolchainError(f"Report prerequisites were not scheduled: {', '.join(sorted(missing))}")
    t0 = time.time()
    n = len(jobs_list)
    print(f"running {n} report(s) with mep, {jobs} at a time", flush=True)
    completed, failed, count = set(), set(), 0
    while pending:
        ready = [path for path in pending if dependencies[path] <= completed]
        if not ready:
            raise ToolchainError("Cyclic report prerequisites")
        runnable = []
        for path in ready:
            original, status = pending[path]
            if dependencies[path] & failed:
                os.makedirs(os.path.dirname(status), exist_ok=True)
                with open(status, "w") as output:
                    json.dump({"files": [{"file": path, "failed": 0, "skipped": 0, "blocks": [],
                                          "error": "A required report failed; dependent execution was skipped"}]}, output)
                failed.add(path)
                count += 1
            else:
                runnable.append((original, status))
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futs = {pool.submit(run_report, mep_mepml, p, s, timeout, env): (p, s) for p, s in runnable}
            for fut in as_completed(futs):
                secs, out = fut.result()
                path, status = futs[fut]
                count += 1
                print(f"[{count}/{n}] {os.path.relpath(path)} ({secs:.0f} s)", flush=True)
                if out.strip():
                    print(out.rstrip(), flush=True)
                last = load_status(status) or {}
                if last.get("failed") or last.get("error"):
                    failed.add(os.path.abspath(path))
        completed.update(ready)
        for path in ready:
            del pending[path]
    print(f"ran {n} report(s) in {time.time() - t0:.0f} s", flush=True)


def load_status(status_path):
    """The status of the report's last run (a `mep-mepml run --report` file entry:
    blocks, failed, skipped, error, plus `when`), or None when it never ran."""
    if not os.path.exists(status_path):
        return None
    with open(status_path) as f:
        files = json.load(f).get("files") or [{}]
    st = files[0]
    st.setdefault("blocks", [])
    st.setdefault("failed", 0)
    st.setdefault("skipped", 0)
    st["when"] = os.path.getmtime(status_path)
    return st


def export(mep_mepml, report, out_root, highlight):
    """Export `report` with mep and fill in its title, summary and body.

    The pictures it names are copied under `out_root` beside its page;
    `highlight(code, lang)` highlights its code blocks."""
    page_dir = os.path.join(out_root, os.path.dirname(report.url))
    src_dir = os.path.dirname(os.path.abspath(report.path))
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.html")
        subprocess.run([mep_mepml, "convert", "--no-notice", report.path, out], check=True,
                       stdout=subprocess.DEVNULL, env=mep_env())
        with open(out, encoding="utf-8") as f:
            page = f.read()
        m = BODY.search(page)
        body = m.group(1) if m else page
        tm = TITLE.search(body)
        if tm:
            report.title = html.unescape(re.sub(r"<[^>]+>", "", tm.group(1))).strip()
        sm = SUBTITLE.search(body)
        if sm:
            report.subtitle = html.unescape(re.sub(r"<[^>]+>", "", sm.group(1))).strip()
        am = ABSTRACT.search(body)
        if am:
            text = html.unescape(re.sub(r"<[^>]+>", " ", am.group(1)))
            text = re.sub(r"\s+", " ", text.replace("Abstract", "", 1)).strip()
            text = re.sub(r"\s+([,.;:)])", r"\1", text)  # where a citation or note was
            report.summary = shorten(text, 260)
        body = TOC.sub("", body)

        def code(match):
            lang, text = match.group(1), html.unescape(match.group(2))
            return f'<pre class="hl"><code>{highlight(text, lang)}</code></pre>'

        body = CODE.sub(code, body)
        body = CODE_PLAIN.sub(lambda mm: f'<pre class="hl"><code>{mm.group(1)}</code></pre>', body)

        # Pictures: copied beside the page at the same relative path.
        model_diagrams = set()

        def picture(match):
            src = html.unescape(match.group(2))
            if re.match(r"^[a-z]+:|^/|^#", src):
                return match.group(0)
            for base in (src_dir, tmp):
                full = os.path.normpath(os.path.join(base, src))
                if os.path.isfile(full):
                    break
            else:
                return match.group(0)
            rel = os.path.normpath(src)
            if rel.startswith(".."):
                rel = os.path.join("assets", os.path.basename(src))
            dest = os.path.join(page_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(full):
                shutil.copyfile(full, dest)
            if full.lower().endswith(".svg"):
                with open(full, "rb") as image:
                    if b'data-model-layout="left-to-right"' in image.read(4096):
                        model_diagrams.add(rel.replace(os.sep, "/"))
            return match.group(1) + html.escape(rel.replace(os.sep, "/")) + match.group(3)

        body = SRC.sub(picture, body)

        def model_picture(match):
            """Keep wide network diagrams readable without changing ordinary report images."""
            source = SRC.search(match.group(0))
            if not source or html.unescape(source.group(2)) not in model_diagrams:
                return match.group(0)
            return ('<span class="model-diagram" tabindex="0" role="region" '
                    'aria-label="Network diagram; scroll to explore">'
                    + match.group(0) + '</span>')

        body = IMG.sub(model_picture, body)
        report.body = body.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the docs report C++ toolchain")
    parser.add_argument("--check-toolchain", action="store_true", required=True)
    parser.add_argument("reports", nargs="+")
    arguments = parser.parse_args()
    try:
        validate_cpp_toolchain(arguments.reports)
    except ToolchainError as error:
        parser.exit(1, str(error) + "\n")
