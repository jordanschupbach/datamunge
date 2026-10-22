#!/usr/bin/env python3
"""Convert the literate Org reports in examples/org/ to mepml (examples/mepml/).

mep's own importer (`mep-mepml convert IN.org OUT.mepml`) does the prose; this
script then rewrites the code so each report runs as mepml C++ blocks:

  - The shell block that wrote a program out with a heredoc, compiled it and
    ran it becomes one runnable `{cpp, results="output", flags="..."}` block
    holding that program, its `<<noweb>>` chunks expanded (mepml has no
    noweb). The flags are the compile line's -std/-O plus datamunge's
    pkg-config flags and an rpath (libdatamunge's install name is
    @rpath-relative on macOS).
  - A report with no shell block has its self-contained `main()` blocks made
    runnable the same way.
  - Every other C++ block is a fragment shown for the narrative:
    `{cpp, eval="no"}`.
  - mepml runs a block in the document's directory, so the programs write
    their figures into `build/` beside it (the document's export directory)
    instead of the repository's `build/org/`, creating it first.
  - A runnable block that saves exactly one figure, which the report shows
    with an `\\image(...)` line, gets `file="figures/<name>"`: mep copies the
    picture the program saved there and shows it in the block's results, so
    the separate `\\image` line goes (its `\\alttext` moves under the block).

It also keeps what the importer drops: `#+begin_export latex` lines around a
block become `\\raw(tex pdf, ...)`, `toc:t` becomes `\\toc`, org verbatim
markup in the title/subtitle is unwrapped, and links to sibling `X.org`
reports point at `X.html` (what the exporters write side by side).

With --run, each runnable block is compiled and run (in OUT_DIR, as mep
would) and its output written under it as `// result_begin:` ... results,
its file= figure copied into place and named there as mep does.

usage: scripts/org_to_mepml.py [--run] [-j N] [ORG_DIR [OUT_DIR]]
"""
import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC_BEGIN = re.compile(r"^\s*#\+begin_src\s+(\S+)(.*)$", re.I)
SRC_END = re.compile(r"^\s*#\+end_src\s*$", re.I)
NAME = re.compile(r"^\s*#\+name:\s*(\S+)\s*$", re.I)
EXPORT_BEGIN = re.compile(r"^\s*#\+begin_export\s+latex\s*$", re.I)
EXPORT_END = re.compile(r"^\s*#\+end_export\s*$", re.I)
NOWEB = re.compile(r"^(.*?)<<([^<>\s]+)>>(.*)$")
HEREDOC = re.compile(r"""^\s*cat\s*>\s*\S+\s*<<\s*'?(\w+)'?\s*$""")
MAIN = re.compile(r"^\s*int\s+main\s*\([^)]*\)\s*\{")

PKG = "$(pkg-config --cflags --libs datamunge) -Wl,-rpath,$(pkg-config --variable=libdir datamunge)"
OUT_DIR_IN_CODE = "build"
FIGURE_DIR = "figures"
SAVED_FIGURE = re.compile(r'"(%s/[^"]+\.(?:svg|png|jpe?g|gif|pdf))"' % OUT_DIR_IN_CODE)


def header_arg(args, key):
    m = re.search(r":%s\s+(\S+)" % re.escape(key), args)
    return m.group(1) if m else None


def parse_org(lines):
    """Source blocks in order, as dicts, plus the file-level shell :exports and toc flag."""
    shell_exports = None
    toc = False
    blocks = []
    pending_name = None
    pending_export = []
    i = 0
    while i < len(lines):
        line = lines[i]
        low = line.lower()
        if low.startswith("#+property:") and "header-args:shell" in low:
            shell_exports = header_arg(line, "exports") or shell_exports
        if low.startswith("#+options:") and re.search(r"\btoc:t\b", low):
            toc = True
        m = NAME.match(line)
        if m:
            pending_name = m.group(1)
            i += 1
            continue
        if EXPORT_BEGIN.match(line):
            j = i + 1
            body = []
            while j < len(lines) and not EXPORT_END.match(lines[j]):
                body.append(lines[j])
                j += 1
            text = "\n".join(body).strip()
            # Attach to the previous block if it ended right above, else to the next.
            if blocks and blocks[-1]["end"] == i - 1:
                blocks[-1]["after"].append(text)
            else:
                pending_export.append(text)
            i = j + 1
            continue
        m = SRC_BEGIN.match(line)
        if m:
            j = i + 1
            body = []
            while j < len(lines) and not SRC_END.match(lines[j]):
                body.append(lines[j])
                j += 1
            blocks.append({
                "lang": m.group(1),
                "args": m.group(2),
                "name": pending_name,
                "body": body,
                "end": j,
                "before": pending_export,
                "after": [],
            })
            pending_name = None
            pending_export = []
            i = j + 1
            continue
        if line.strip():
            pending_name = None
            pending_export = []
        i += 1
    return blocks, shell_exports, toc


def expand(body, named, seen=()):
    out = []
    for line in body:
        m = NOWEB.match(line)
        if m and m.group(2) in named and m.group(2) not in seen:
            prefix, ref, suffix = m.groups()
            chunk = expand(named[m.group(2)], named, seen + (ref,))
            for k, c in enumerate(chunk):
                out.append(prefix + c + (suffix if k == len(chunk) - 1 else ""))
        else:
            out.append(line)
    return out


def relocate(text):
    """Repository-root build/org/ paths -> the document's own build/."""
    return text.replace("../../build/org/", OUT_DIR_IN_CODE + "/").replace("build/org/", OUT_DIR_IN_CODE + "/")


def ensure_out_dir(program, where):
    """Have a program that writes into build/ create it first."""
    if not any('"' + OUT_DIR_IN_CODE + "/" in l for l in program):
        return program
    out = list(program)
    k = next((i for i, l in enumerate(out) if MAIN.match(l) and l.rstrip().endswith("{")), None)
    if k is None:
        raise SystemExit(f"{where}: writes into {OUT_DIR_IN_CODE}/ but has no `int main(...) {{` line")
    indent = re.match(r"\s*", out[k + 1] if k + 1 < len(out) and out[k + 1].strip() else "    ").group(0) or "    "
    out.insert(k + 1, f'{indent}std::filesystem::create_directories("{OUT_DIR_IN_CODE}");')
    if not any(re.match(r"\s*#\s*include\s*<filesystem>", l) for l in out):
        first = next((i for i, l in enumerate(out) if l.lstrip().startswith("#include")), 0)
        out.insert(first, "#include <filesystem>")
    return out


def program_from_shell(body, named, where):
    """(program lines, compiler flags) from a heredoc-compile-run shell block."""
    body = expand(body, named)
    prog, std, opt = None, "-std=c++23", ""
    i = 0
    while i < len(body):
        m = HEREDOC.match(body[i])
        if m and prog is None:
            j = i + 1
            while j < len(body) and body[j].strip() != m.group(1):
                j += 1
            prog = body[i + 1:j]
            i = j + 1
            continue
        s = re.search(r"-std=\S+", body[i])
        if s:
            std = s.group(0)
        o = re.search(r"\s(-O\d)\b", body[i])
        if o:
            opt = " " + o.group(1)
        i += 1
    if prog is None:
        raise SystemExit(f"{where}: shell block without a heredoc program")
    while prog and not prog[-1].strip():
        prog.pop()
    return prog, std + opt


def cpp_header(*, run, exports=None, flags=None, file=None):
    if not run:
        return '```{cpp, eval="no"}'
    opts = ['results="output"', f'exports="{exports or "both"}"']
    if file:
        opts.append(f'file="{file}"')
    opts.append(f'flags="{flags}"')
    return "```{cpp, " + ", ".join(opts) + "}"


def unwrap_verbatim(s):
    return re.sub(r"(?<!\w)[=~]([^=~\s](?:[^=~]*[^=~\s])?)[=~](?!\w)", r"\1", s)


def claim_figure(prog, shown, after_line, skip):
    """For a program saving exactly one figure the report shows (below the
    block): ((saved path, file= path), its alt text line), marking the
    report's own \\image line (and alt text) as carried by the block."""
    saved = {m for l in prog for m in SAVED_FIGURE.findall(l)}
    if len(saved) != 1:
        return None, None
    src = saved.pop()
    if src not in shown or shown[src][0] <= after_line:
        return None, None
    n, alt = shown.pop(src)
    skip.add(n)
    if alt:
        skip.add(n + 1)
    return (src, FIGURE_DIR + "/" + os.path.basename(src)), alt


def import_org(org_path):
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "out.mepml")
        subprocess.run(["mep-mepml", "convert", org_path, raw_path], check=True)
        with open(raw_path) as f:
            return f.read().splitlines()


def convert(org_path):
    """The .mepml lines, and the runnable blocks as
    (index of closing fence, program, flags, (saved figure, file= figure) or None)."""
    with open(org_path) as f:
        org = f.read().splitlines()
    blocks, shell_exports, toc = parse_org(org)
    named = {}
    for b in blocks:
        if b["name"]:
            named.setdefault(b["name"], []).extend(b["body"])
    has_shell = any(b["lang"].lower() in ("shell", "sh", "bash") for b in blocks)
    mep = import_org(org_path)
    where = os.path.basename(org_path)

    # The figures the report shows: \image line index and the alt text under it.
    shown = {}
    for n, l in enumerate(mep):
        m = re.match(r"^\\image\((.*)\)$", relocate(l))
        if m and m.group(1) not in shown:
            alt = mep[n + 1] if n + 1 < len(mep) and mep[n + 1].startswith("\\alttext(") else None
            shown[m.group(1)] = (n, alt)
    skip = set()  # importer lines now carried by a block's file=

    out = []
    runs = []
    k = 0  # index into blocks
    i = 0
    in_header = True
    abstract_open = False
    toc_done = not toc
    while i < len(mep):
        line = mep[i]
        if in_header and line.startswith("//?"):
            key, _, val = line.partition(":")
            if key in ("//? Title", "//? Subtitle"):
                line = key + ": " + unwrap_verbatim(val.strip()).replace("examples/org/", "examples/mepml/")
            out.append(line)
            i += 1
            continue
        in_header = False
        if line.startswith("\\abstract("):
            abstract_open = True
        if abstract_open and line == ")":
            abstract_open = False
            out.append(line)
            if not toc_done:
                out += ["", "\\toc"]
                toc_done = True
            i += 1
            continue
        if not toc_done and line.startswith(">") and not abstract_open:
            out += ["\\toc", ""]
            toc_done = True
        if line.startswith("```") and line != "```":
            # A source block: find its closing fence.
            j = i + 1
            while j < len(mep) and mep[j] != "```":
                j += 1
            if k >= len(blocks):
                raise SystemExit(f"{where}: more code blocks in mepml than org")
            b = blocks[k]
            k += 1
            lang = b["lang"].lower()
            fig, alt = None, None
            if lang in ("shell", "sh", "bash"):
                prog, flags = program_from_shell(b["body"], named, where)
                prog = ensure_out_dir([relocate(l) for l in prog], where)
                exports = header_arg(b["args"], "exports") or shell_exports
                fig, alt = claim_figure(prog, shown, j, skip)
                head = cpp_header(run=True, exports=exports, flags=flags + " " + PKG, file=fig and fig[1])
                runs.append((len(out) + 1 + len(prog) + len(b["before"]), prog, flags + " " + PKG, fig))
            elif lang in ("c++", "cpp") and not has_shell and any(MAIN.match(l) or "int main(" in l for l in b["body"]):
                prog = ensure_out_dir([relocate(l) for l in b["body"]], where)
                flags = "-std=c++23 -O2"
                fig, alt = claim_figure(prog, shown, j, skip)
                head = cpp_header(run=True, exports="both", flags=flags + " " + PKG, file=fig and fig[1])
                runs.append((len(out) + 1 + len(prog) + len(b["before"]), prog, flags + " " + PKG, fig))
            elif lang in ("c++", "cpp"):
                prog = [relocate(l) for l in b["body"]]
                head = cpp_header(run=False)
            else:
                raise SystemExit(f"{where}: unexpected source language {b['lang']}")
            for t in b["before"]:
                out.append("\\raw(tex pdf, %s)" % t)
            out.append(head)
            out += prog
            out.append("```")
            if alt:
                out.append(alt)
            for t in b["after"]:
                out.append("\\raw(tex pdf, %s)" % t)
            i = j + 1
            continue
        if line.startswith("```"):
            # An example block: copy verbatim (bar the relocated paths).
            j = i + 1
            while j < len(mep) and mep[j] != "```":
                j += 1
            out += [relocate(l) for l in mep[i:j + 1]]
            i = j + 1
            continue
        if i in skip:
            i += 1
            continue
        # Links to sibling reports: their exports sit side by side as .html.
        line = re.sub(r"\|(\./)?([\w.-]+)\.org\]", r"|\1\2.html]", line)
        out.append(relocate(line))
        i += 1
    if k != len(blocks):
        raise SystemExit(f"{where}: {len(blocks)} org source blocks but {k} in mepml")
    return out, runs


def format_results(output):
    """mep's own layout of a block's results (mepml_doc.cpp, FormatResults)."""
    lines = ["// result_begin:"]
    o = output.rstrip("\r\n")
    if o:
        for ln in o.split("\n"):
            ln = ln.rstrip("\r")
            lines.append("//" if not ln else "// " + ln)
    lines.append("// result_end")
    return lines


def run_block(prog, flags, cwd):
    """Compile and run one block as mep does; its stdout, or a failure message."""
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "block.cpp")
        exe = os.path.join(tmp, "block")
        with open(src, "w") as f:
            f.write("\n".join(prog) + "\n")
        cc = subprocess.run(f'c++ {src} -o {exe} {flags}', shell=True, capture_output=True, text=True)
        if cc.returncode != 0:
            return None, cc.stderr
        r = subprocess.run([exe], cwd=cwd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return None, r.stdout + r.stderr
        return r.stdout, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true", help="compile and run each runnable block, writing its results")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4, help="parallel compiles with --run")
    ap.add_argument("org_dir", nargs="?")
    ap.add_argument("out_dir", nargs="?")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    org_dir = a.org_dir or os.path.join(root, "examples", "org")
    out_dir = a.out_dir or os.path.join(root, "examples", "mepml")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(org_dir) if n.endswith(".org"))

    docs = {}
    for n in names:
        docs[n] = convert(os.path.join(org_dir, n))

    failures = []
    if a.run:
        jobs = {}
        with concurrent.futures.ThreadPoolExecutor(a.j) as pool:
            for n, (_, runs) in docs.items():
                for idx, (at, prog, flags, fig) in enumerate(runs):
                    jobs[pool.submit(run_block, prog, flags, out_dir)] = (n, idx)
            results = {}
            for fut in concurrent.futures.as_completed(jobs):
                results[jobs[fut]] = fut.result()
        for n, (lines, runs) in docs.items():
            # Insert from the bottom so earlier fence indices stay valid.
            for idx in reversed(range(len(runs))):
                at = runs[idx][0]
                output, err = results[(n, idx)]
                if output is None:
                    failures.append((n, err))
                    continue
                fig = runs[idx][3]
                if fig:
                    # What mep does after the run: the saved picture copied to file=.
                    src, dst = (os.path.join(out_dir, p) for p in fig)
                    if not os.path.exists(src):
                        failures.append((n, f"the block did not create {fig[0]}"))
                        continue
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(src, dst)
                    output = output.rstrip("\r\n") + ("\n" if output.strip() else "") + "\\image(%s)" % fig[1]
                lines[at + 1:at + 1] = format_results(output)

    for n, (lines, _) in docs.items():
        with open(os.path.join(out_dir, n[:-4] + ".mepml"), "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"converted {len(names)} files into {out_dir}")
    for n, err in failures:
        print(f"\n--- {n}: block failed\n{err[-2000:]}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} block(s) failed to compile or run", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
