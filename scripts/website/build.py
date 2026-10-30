#!/usr/bin/env python3
"""Build the documentation website.

The site is the project's own; its pieces come from:
  - Doxygen's XML (docs/Doxyfile, the GenerateDocsXml target): the API reference;
  - mep (`mep-mepml`): the mepml reports in docs/mepml/examples, optionally run
    first (--run) so their results are current, with a page saying how every
    report's code blocks went;
  - Markdown: README.md as the home page and the guides docs/website.toml lists.

Everything is written under OUT as static files with relative links, so the
site works from disk (file://) or from any web server and path.

usage: scripts/website/build.py --xml DOXYGEN_XML_DIR --out OUT
                                [--mep-mepml PATH] [--run [--force] [--timeout S] [-j N] [--no-pch]]
                                [--status-dir DIR] [--allow-failures] [--no-reports]
                                [--config docs/website.toml] [REPORT.mepml...]
Exits 1 when a report's code block failed (unless --allow-failures), after
writing the whole site.
"""
import argparse
import datetime
import glob
import html
import json
import os
import re
import shutil
import sys
import time
import tomllib
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import apiref  # noqa: E402
import highlight as hl  # noqa: E402
import pages  # noqa: E402
import reports as rep  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
E = html.escape


def project_info(root):
    """(name, version) from the root CMakeLists.txt's project() call."""
    with open(os.path.join(root, "CMakeLists.txt")) as f:
        m = re.search(r"project\(\s*(\S+)\s+VERSION\s+([^\s)]+)", f.read())
    return (m.group(1), m.group(2)) if m else ("project", "")


def namespace_aliases(root, ns):
    """Names the headers declare for the root namespace (`namespace dm = datamunge;`)."""
    out = set()
    pat = re.compile(r"^\s*namespace\s+(\w+)\s*=\s*(?:::)?" + re.escape(ns) + r"\s*;", re.M)
    for path in glob.glob(os.path.join(root, "include", "**", "*.h*"), recursive=True):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                out.update(pat.findall(f.read()))
        except OSError:
            pass
    return sorted(out)


def rel(target, page):
    """`target` (site-root relative) as seen from the page `page`."""
    return os.path.relpath(target.split("#")[0], os.path.dirname(page) or ".").replace(os.sep, "/") + (
        "#" + target.split("#", 1)[1] if "#" in target else "")


def base_of(page):
    depth = page.count("/")
    return "../" * depth


class Site:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        with open(args.config, "rb") as f:
            self.config = tomllib.load(f)
        name, version = project_info(ROOT)
        self.root_ns = args.namespace or name
        self.title = self.config.get("title") or args.project_name or name
        self.version = args.version or version
        self.repo = (self.config.get("repository") or "").rstrip("/")
        self.branch = self.config.get("branch") or "main"
        self.aliases = namespace_aliases(ROOT, self.root_ns)
        self.env = Environment(loader=FileSystemLoader(os.path.join(HERE, "templates")),
                               autoescape=select_autoescape(["html"]))
        self.search = []
        self.search_seen = set()
        self.used_in = defaultdict(dict)    # API page -> {report url: title}
        self.report_uses = defaultdict(dict)  # report url -> {API url: label}
        self.current_report = None
        self.written = 0

    # -- helpers ---------------------------------------------------------------
    def source_url(self, path, line=0):
        if not self.repo or not path:
            return ""
        return f"{self.repo}/blob/{self.branch}/{path}" + (f"#L{line}" if line else "")

    def repo_file_url(self, path):
        if not self.repo or not os.path.exists(os.path.join(ROOT, path)):
            return None
        kind = "tree" if os.path.isdir(os.path.join(ROOT, path)) else "blob"
        return f"{self.repo}/{kind}/{self.branch}/{path}"

    def write(self, page, title, content, section, sidebar="", math=False, wide=False, description=""):
        base = base_of(page)
        tpl = self.env.get_template("base.html")
        logo = self.config.get("logo")
        text = tpl.render(
            base=base, page=page, title=title, content=content, section=section, sidebar=sidebar, math=math,
            wide=wide, description=description, site_title=self.title, version=self.version, repo=self.repo,
            logo=("assets/" + logo) if logo and os.path.exists(os.path.join(ROOT, logo)) else "",
            year=datetime.date.today().year, built=datetime.date.today().isoformat(),
        )
        path = os.path.join(self.out, page)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.written += 1

    def add_search(self, title, kind, url, detail="", summary=""):
        if (title, kind) in self.search_seen:  # overloads: the first one
            return
        self.search_seen.add((title, kind))
        summary = re.sub(r"\s+", " ", (summary or "").replace("\\(", "").replace("\\)", "")).strip()
        if len(summary) > 110:
            summary = summary[:107].rsplit(" ", 1)[0] + "…"
        self.search.append({"t": title, "k": kind, "u": url, "d": detail, "s": summary})

    # -- build -------------------------------------------------------------------
    def build(self):
        t0 = time.time()
        if os.path.isdir(self.out):
            # Pages and assets are rewritten; stale ones (renamed reports) go.
            for name in os.listdir(self.out):
                p = os.path.join(self.out, name)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        os.makedirs(self.out, exist_ok=True)
        self.copy_static()
        print("reading the API reference (Doxygen XML)", flush=True)
        self.api = apiref.Api(self.args.xml, self.root_ns)
        self.linker = hl.Linker(self.api.resolve, self.root_ns, self.aliases, on_link=self.on_link)
        self.doc = apiref.DocRenderer(self.api, lambda code, lang: hl.highlight(code, lang, self.linker))
        self.guides = [dict(g) for g in self.config.get("guides", [])]
        for g in self.guides:
            g["url"] = f"guides/{g['slug']}.html"
        failures = self.build_reports()
        self.build_api()
        self.build_guides()
        self.build_home()
        self.write_search()
        print(f"website: {self.written} pages in {self.out} ({time.time() - t0:.0f} s)")
        return failures

    def copy_static(self):
        dest = os.path.join(self.out, "static")
        shutil.copytree(os.path.join(HERE, "static"), dest)
        with open(os.path.join(dest, "site.css"), "a", encoding="utf-8") as f:
            f.write("\n" + hl.stylesheet())
        logo = self.config.get("logo")
        if logo and os.path.exists(os.path.join(ROOT, logo)):
            os.makedirs(os.path.dirname(os.path.join(self.out, "assets", logo)), exist_ok=True)
            shutil.copyfile(os.path.join(ROOT, logo), os.path.join(self.out, "assets", logo))

    def on_link(self, url):
        if self.current_report is None:
            return
        r = self.current_report
        page = url.split("#")[0]
        self.used_in[page][r.url] = r.title
        label = page.rsplit("/", 1)[-1].replace(".html", "").replace("class.", "").replace("struct.", "") \
            .replace("fn.", "")
        if page.endswith("/index.html"):
            label = page[len("api/"):-len("/index.html")].replace("/", "::")
        self.report_uses[r.url][page] = label

    # -- reports -----------------------------------------------------------------
    def build_reports(self):
        a = self.args
        examples = os.path.join(ROOT, "docs", "mepml", "examples")
        files = a.reports or sorted(glob.glob(os.path.join(examples, "**", "*.mepml"), recursive=True))
        self.reports = []
        if a.no_reports or not files:
            return []
        status_dir = a.status_dir or os.path.join(os.path.dirname(os.path.abspath(self.out)), "mepml-status")
        print(f"{len(files)} reports" + (" (running the changed ones with mep)" if a.run else ""), flush=True)
        def status_path(path):
            slug_name = re.sub(r"[/.\-]", "_", os.path.splitext(rep.report_rel(path, examples))[0])
            return os.path.join(status_dir, slug_name + ".json")

        if a.run:
            stale = [(p, status_path(p)) for p in files if rep.is_stale(p, status_path(p), a.force)]
            if stale:
                stale = rep.with_prerequisites(stale, status_path)
                rep.validate_cpp_toolchain([p for p, _ in stale])
                env = {} if a.no_pch else rep.toolchain(os.path.join(os.path.dirname(status_dir), "mepml-pch"), files,
                                                         a.mep_mepml)
                rep.run_reports(a.mep_mepml, stale, timeout=a.timeout, jobs=a.jobs or rep.default_jobs(), env=env)
        for path in files:
            r = rep.Report(path, rep.report_rel(path, examples))
            r.status = rep.load_status(status_path(path))
            self.current_report = r
            self.linker.base = base_of(r.url)
            rep.export(a.mep_mepml, r, self.out, lambda code, lang: hl.highlight(code, lang, self.linker))
            self.current_report = None
            self.reports.append(r)
        groups = defaultdict(list)
        for r in self.reports:
            groups[r.group].append(r)
        self.report_groups = sorted(groups.items(), key=lambda kv: (kv[0] != "", kv[0]))
        for r in self.reports:
            self.write_report(r)
            self.add_search(r.title, "report", r.url, r.subtitle, r.summary)
        self.write_reports_index()
        self.write_status()
        broken = [r for r in self.reports if r.broken]
        never = sum(1 for r in self.reports if not r.status)
        print(f"reports: {len(self.reports)}, with failed code blocks: {len(broken)}"
              + (f", never run by mep: {never}" if never else "") + "; see reports/status.html")
        for r in broken:
            print(f"  failed: {r.rel}", file=sys.stderr)
        return broken

    @staticmethod
    def group_title(group):
        if not group:
            return "Reports"
        name = group.split("/")[-1]
        if name == "nn":
            return "Neural networks"
        t = re.sub(r"^[\d.]+_", "", name).replace("_", " ")
        return t[:1].upper() + t[1:]

    def reports_sidebar(self, page, current=None):
        out = ['<input class="side-filter" type="search" placeholder="Filter reports" aria-label="Filter reports">',
               f'<a class="side-top" href="{rel("reports/index.html", page)}">All reports</a>',
               f'<a class="side-top" href="{rel("reports/status.html", page)}">Build status</a>']
        for group, items in self.report_groups:
            depth = group.count("/") + (1 if group else 0)
            title = " / ".join(self.group_title("/".join(group.split("/")[:i + 1]))
                               for i in range(len(group.split("/")))) if group else "General"
            out.append(f'<div class="side-group"><div class="side-heading">{E(title)}</div><ul>')
            for r in sorted(items, key=lambda r: r.title.lower()):
                cur = ' class="current"' if r is current else ""
                mark = ' <span class="dot-fail" title="code blocks failed"></span>' if r.broken else ""
                out.append(f'<li><a{cur} href="{rel(r.url, page)}">{E(r.title)}</a>{mark}</li>')
            out.append("</ul></div>")
        return "".join(out)

    def status_banner(self, r, page):
        st = r.status
        if not st:
            return ('<div class="banner banner-info">mep has not run this report\'s code yet; its results are the ones '
                    'stored in the report.</div>')
        if st.get("error"):
            return f'<div class="banner banner-fail">{E(st["error"])}. <a href="{rel("reports/status.html", page)}">Build status</a></div>'
        if st["failed"]:
            return (f'<div class="banner banner-fail">{st["failed"]} of {len(st["blocks"])} code blocks failed the last '
                    f'time mep ran this report; their results are from an earlier run. '
                    f'<a href="{rel("reports/status.html", page)}#{E(slug_id(r.rel))}">Details</a></div>')
        return ""

    def write_report(self, r):
        page = r.url
        uses = self.report_uses.get(r.url, {})
        tail = []
        if uses:
            items = "".join(f'<a class="chip" href="{rel(u, page)}"><code>{E(label)}</code></a>'
                            for u, label in sorted(uses.items(), key=lambda kv: kv[1].lower()))
            tail.append(f'<h2 id="api-used">API used here</h2><div class="chips">{items}</div>')
        src = self.source_url("docs/mepml/examples/" + r.rel)
        st = r.status
        meta = []
        if st and not st.get("error"):
            n = len(st["blocks"])
            ok = n - st["failed"] - st["skipped"]
            when = datetime.datetime.fromtimestamp(st["when"]).strftime("%Y-%m-%d")
            meta.append(f"{ok} of {n} code blocks ran on {when}")
        if src:
            meta.append(f'<a href="{E(src)}">source</a>')
        footer = f'<p class="report-meta">{" · ".join(meta)}</p>' if meta else ""
        content = (self.status_banner(r, page) + f'<article class="report">{r.body}</article>'
                   + "".join(tail) + footer)
        self.write(page, r.title, content, "reports", self.reports_sidebar(page, r), math=True,
                   description=r.subtitle)

    def write_reports_index(self):
        page = "reports/index.html"
        out = ['<h1>Reports</h1>',
               '<p class="lead">Literate reports written in mepml, the document format of the mep editor: the '
               'method and its mathematics, small C++ programs built on the library, and each program\'s tables and '
               'figures. mep runs their code when the site is built.</p>',
               f'<p><a href="{rel("reports/status.html", page)}">Build status of every report</a></p>']
        for group, items in self.report_groups:
            title = self.group_title(group) if group else "General"
            if group:
                title = " / ".join(self.group_title("/".join(group.split("/")[:i + 1]))
                                   for i in range(len(group.split("/"))))
            out.append(f'<h2 id="{E(slug_id(group or "general"))}">{E(title)}</h2><div class="cards">')
            for r in sorted(items, key=lambda r: r.title.lower()):
                badge = '<span class="badge badge-fail">failed blocks</span>' if r.broken else ""
                out.append(f'<a class="card" href="{rel(r.url, page)}"><div class="card-title">{E(r.title)}{badge}</div>'
                           f'<div class="card-text">{E(r.subtitle or r.summary)}</div></a>')
            out.append("</div>")
        self.write(page, "Reports", "".join(out), "reports", self.reports_sidebar(page), math=True)

    def write_status(self):
        page = "reports/status.html"
        ran = [r for r in self.reports if r.status]
        blocks = sum(len(r.status["blocks"]) for r in ran)
        failed = sum(r.status["failed"] for r in ran)
        skipped = sum(r.status["skipped"] for r in ran)
        broken = [r for r in self.reports if r.broken]
        never = [r for r in self.reports if not r.status]
        out = ['<h1>Report build status</h1>',
               '<p class="lead">How each report\'s code blocks went the last time mep ran them (<code>mep-mepml run</code>, '
               'which <code>just docs</code> and <code>just docs-mepml-run</code> call). A report is rerun when its source '
               'changes and, until it passes, after it failed; a failed block keeps its earlier results.</p>',
               '<div class="stats">',
               stat(len(self.reports), "reports"), stat(blocks, "code blocks run"),
               stat(blocks - failed - skipped, "finished"), stat(failed, "failed", failed > 0),
               stat(skipped, "skipped"), stat(len(never), "not run yet"), '</div>',
               '<div class="table-wrap"><table class="status"><tr><th>Report</th><th>Blocks</th><th>Failed</th>'
               '<th>Skipped</th><th>Run time</th><th>Last run</th></tr>']
        for r in sorted(self.reports, key=lambda r: (not r.broken, r.rel)):
            link = f'<a href="{rel(r.url, page)}">{E(r.title)}</a><div class="muted small">{E(r.rel)}</div>'
            st = r.status
            if not st:
                out.append(f'<tr><td>{link}</td><td colspan="5" class="muted">not run</td></tr>')
                continue
            when = datetime.datetime.fromtimestamp(st["when"]).strftime("%Y-%m-%d %H:%M")
            secs = sum(b.get("seconds", 0) for b in st["blocks"])
            fail = f'<span class="fail">{st["failed"]}</span>' if st["failed"] else "0"
            if st.get("error"):
                fail = f'<span class="fail">{E(st["error"])}</span>'
            out.append(f'<tr><td>{link}</td><td>{len(st["blocks"])}</td><td>{fail}</td><td>{st["skipped"]}</td>'
                       f'<td>{secs:.0f} s</td><td>{when}</td></tr>')
        out.append("</table></div>")
        if broken:
            out.append('<h2 id="failures">Failures</h2>')
            for r in broken:
                out.append(f'<h3 id="{E(slug_id(r.rel))}"><a href="{rel(r.url, page)}">{E(r.title)}</a></h3>')
                if r.status.get("error"):
                    out.append(f"<p>{E(r.status['error'])}</p>")
                for b in r.status["blocks"]:
                    if b.get("status") != "failed":
                        continue
                    out.append(f'<p>Block {b["index"]} (line {b["line"]}, {E(b["lang"])}): '
                               f'<strong>{E(b.get("reason") or "failed")}</strong></p>')
                    if b.get("output"):
                        out.append(f'<pre class="output">{E(b["output"])}</pre>')
        self.write(page, "Report build status", "".join(out), "reports", self.reports_sidebar(page))

    # -- API -----------------------------------------------------------------
    def api_sidebar(self, page, current_ns=None):
        out = [f'<a class="side-top" href="{rel("api/index.html", page)}">API overview</a>',
               f'<a class="side-top" href="{rel("api/classes.html", page)}">All classes</a>',
               '<div class="side-heading">Namespaces</div><ul class="tree">']
        for ns in self.api.namespaces():
            depth = ns.name.count("::")
            label = ns.name.split("::")[-1] if depth else ns.name
            cur = ' class="current"' if current_ns is ns else ""
            out.append(f'<li style="--depth:{depth}"><a{cur} href="{rel(ns.url, page)}">{E(label)}</a></li>')
        out.append("</ul>")
        return "".join(out)

    def crumbs(self, comp, page, kind_label, include_self=False):
        parts = []
        ns = comp if comp.kind == "namespace" else self.api.namespace_of(comp)
        chain = []
        c = ns
        while c is not None:
            chain.append(c)
            c = c.parent
        chain.reverse()
        for c in chain:
            if c is comp and not include_self:
                continue
            parts.append(f'<a href="{rel(c.url, page)}">{E(c.short)}</a>')
        return f'<div class="kicker"><span class="kind">{E(kind_label)}</span> {"::".join(parts)}</div>'

    def header_line(self, path, line, page, link=True):
        if not path:
            return ""
        inc = path[len("include/"):] if path.startswith("include/") else path
        src = self.source_url(path, line)
        anchor = f' <a class="src" href="{E(src)}">source</a>' if src and link else ""
        return f'<div class="meta"><code>#include &lt;{E(inc)}&gt;</code>{anchor}</div>'

    def used_block(self, page):
        uses = self.used_in.get(page)
        if not uses:
            return ""
        items = "".join(f'<a class="chip" href="{rel(u, page)}">{E(t)}</a>'
                        for u, t in sorted(uses.items(), key=lambda kv: kv[1].lower()))
        return f'<h2 id="used-in-reports">Used in reports</h2><div class="chips">{items}</div>'

    def member_section(self, m, page):
        self.doc.base = base_of(page)
        self.linker.base = base_of(page)
        sig = self.doc.signature(m)
        body = self.doc.description(m.brief) + self.doc.description(m.detail)
        if m.kind == "enum":
            body += self.doc.enum_values(m)
        src = self.source_url(m.file, m.line)
        meta = f'<a class="src" href="{E(src)}">source</a>' if src else ""
        prot = '<span class="tag">protected</span>' if m.prot == "protected" else ""
        return (f'<section class="member" id="{E(m.anchor)}"><div class="member-head">'
                f'<a class="anchor" href="#{E(m.anchor)}" aria-label="Link to this">#</a>{prot}{meta}</div>'
                f'<pre class="hl sig"><code>{sig}</code></pre><div class="doc">{body}</div></section>')

    def build_api(self):
        api = self.api
        print(f"API: {len(api.namespaces())} namespaces, {len(api.types())} classes, "
              f"{len(api.functions)} free functions", flush=True)
        for ns in api.namespaces():
            self.write_namespace(ns)
        for c in api.types():
            self.write_class(c)
        for (ns_id, name), overloads in sorted(api.functions.items(), key=lambda kv: kv[0][1]):
            self.write_function(api.compounds[ns_id], name, overloads)
        self.write_api_index()
        self.write_classes_index()

    def summary_rows(self, items, page):
        rows = []
        for label, url, brief in items:
            rows.append(f'<tr><td><a href="{rel(url, page)}"><code>{E(label)}</code></a></td><td>{brief}</td></tr>')
        return f'<div class="table-wrap"><table class="summary">{"".join(rows)}</table></div>' if rows else ""

    def write_namespace(self, ns):
        page = ns.url
        self.doc.base = self.linker.base = base_of(page)
        out = [self.crumbs(ns, page, "namespace"), f'<h1><code>{E(ns.name)}</code></h1>']
        if ns.name == self.root_ns and self.aliases:
            out.append(f'<p class="muted">Also spelled <code>{"</code>, <code>".join(E(a) for a in self.aliases)}</code> '
                       f'(a namespace alias).</p>')
        out.append(f'<div class="doc">{self.doc.description(ns.brief)}{self.doc.description(ns.detail)}</div>')
        subs = [c for c in ns.inner if c.kind == "namespace"]
        if subs:
            out.append('<h2 id="namespaces">Namespaces</h2>')
            out.append(self.summary_rows([(c.name, c.url, self.doc.brief(c.brief)) for c in sorted(subs, key=lambda c: c.name)], page))
        # What it holds, by the header that declares it.
        by_header = defaultdict(lambda: {"types": [], "functions": {}, "other": []})
        for c in ns.inner:
            if c.kind != "namespace":
                by_header[c.file]["types"].append(c)
        for m in ns.members:
            if m.kind == "function":
                by_header[m.file]["functions"].setdefault(m.name, m)
            else:
                by_header[m.file]["other"].append(m)
        headers = sorted(by_header)
        defs = []
        for h in headers:
            g = by_header[h]
            inc = h[len("include/"):] if h.startswith("include/") else h
            out.append(f'<h2 id="{E(slug_id(inc))}" class="header-title"><code>{E(inc)}</code></h2>')
            if g["types"]:
                out.append('<h4>Classes</h4>' + self.summary_rows(
                    [(c.local, c.url, self.doc.brief(c.brief)) for c in sorted(g["types"], key=lambda c: c.local.lower())], page))
            if g["functions"]:
                rows = []
                for name, m in sorted(g["functions"].items(), key=lambda kv: kv[0].lower()):
                    n = len(api_overloads(self.api, ns, name))
                    label = name + ("()" if n == 1 else f"() ×{n}")
                    rows.append((label, m.url.split("#")[0], self.doc.brief(m.brief)))
                out.append('<h4>Functions</h4>' + self.summary_rows(rows, page))
            if g["other"]:
                kinds = {"enum": "enum", "typedef": "type", "variable": "variable"}
                out.append('<h4>Types and constants</h4>' + self.summary_rows(
                    [(f"{m.name}", m.url, f'<span class="tag">{kinds[m.kind]}</span> ' + self.doc.brief(m.brief))
                     for m in sorted(g["other"], key=apiref.Member.sort_key)], page))
                defs.extend(g["other"])
        if defs:
            out.append('<h2 id="definitions">Types and constants in full</h2>')
            out.extend(self.member_section(m, page) for m in sorted(defs, key=apiref.Member.sort_key))
            for m in defs:
                self.add_search(f"{ns.name}::{m.name}", m.kind, m.url, ns.name, self.doc.plain(m.brief))
        self.write(page, ns.name, "".join(out), "api", self.api_sidebar(page, ns), math=True,
                   description=self.doc.plain(ns.brief))
        self.add_search(ns.name, "namespace", page, "", self.doc.plain(ns.brief))

    def write_class(self, c):
        page = c.url
        self.doc.base = self.linker.base = base_of(page)
        kind = c.kind
        out = [self.crumbs(c, page, kind)]
        tpl = self.doc.template_line(c.el)
        out.append(f'<h1><code>{E(c.local or c.short)}</code></h1>')
        bases = []
        for b in c.el.findall("basecompoundref"):
            name = E((b.text or "").strip())
            href = self.doc.href(b.get("refid")) if b.get("refid") else None
            bases.append(f'{E(b.get("prot", "public"))} ' + (f'<a href="{E(href)}">{name}</a>' if href else name))
        decl = (tpl + "\n" if tpl else "") + f'<span class="k">{kind}</span> <span class="nc">{E(c.short)}</span>'
        if bases:
            decl += " : " + ", ".join(bases)
        out.append(f'<pre class="hl sig"><code>{decl}</code></pre>')
        out.append(self.header_line(c.file, c.line, page))
        if kind == "concept":
            init = c.el.find("initializer")
            if init is not None:
                code = "".join(self.doc.code_text(h) for cl in init.iter("codeline") for h in cl) or "".join(init.itertext())
                out.append(f'<pre class="hl"><code>{hl.highlight(code, "cpp", self.linker)}</code></pre>')
        out.append(f'<div class="doc">{self.doc.description(c.brief)}{self.doc.description(c.detail)}</div>')
        derived = []
        for d in c.el.findall("derivedcompoundref"):
            href = self.doc.href(d.get("refid")) if d.get("refid") else None
            if href:
                derived.append(f'<a href="{E(href)}"><code>{E((d.text or "").strip())}</code></a>')
        if derived:
            out.append(f'<p class="muted">Derived classes: {", ".join(derived)}</p>')
        out.append(self.used_block(page))
        # Members, by category.
        cats = [("Types", []), ("Constructors", []), ("Static functions", []), ("Functions", []),
                ("Operators", []), ("Static data", []), ("Fields", [])]
        index = dict(cats)
        nested = sorted((x for x in c.inner if x.kind != "namespace"), key=lambda x: x.short.lower())
        for m in c.members:
            short = apiref.strip_template_args(c.short)
            if m.kind in ("typedef", "enum"):
                index["Types"].append(m)
            elif m.kind == "variable":
                index["Static data" if m.static else "Fields"].append(m)
            elif m.name in (short, "~" + short):
                index["Constructors"].append(m)
            elif m.is_operator:
                index["Operators"].append(m)
            elif m.static:
                index["Static functions"].append(m)
            else:
                index["Functions"].append(m)
        if nested:
            out.append('<h2 id="nested">Nested classes</h2>' + self.summary_rows(
                [(x.short, x.url, self.doc.brief(x.brief)) for x in nested], page))
        summary = []
        for title, items in cats:
            if not items:
                continue
            seen = {}
            for m in items:
                if m.name not in seen:
                    seen[m.name] = [m]
                else:
                    seen[m.name].append(m)
            rows = []
            for name, ms in seen.items():
                label = name + ("()" if ms[0].kind == "function" else "") + (f" ×{len(ms)}" if len(ms) > 1 else "")
                rows.append((label, f"{page}#{ms[0].anchor}", self.doc.brief(ms[0].brief)))
            summary.append(f"<h4>{title}</h4>" + self.summary_rows(rows, page))
        if summary:
            out.append('<h2 id="members">Members</h2>' + "".join(summary))
        for title, items in cats:
            if not items:
                continue
            out.append(f'<h2 id="{slug_id(title)}">{title}</h2>')
            last = None
            for m in items:
                if m.name != last:
                    out.append(f'<h3 class="member-name"><code>{E(m.name)}</code></h3>')
                    last = m.name
                out.append(self.member_section(m, page))
                self.add_search(f"{c.name}::{m.name}", "member " + m.kind, m.url, c.name, self.doc.plain(m.brief))
        self.write(page, c.name, "".join(out), "api", self.api_sidebar(page, self.api.namespace_of(c)), math=True,
                   description=self.doc.plain(c.brief))
        self.add_search(c.name, kind, page, "", self.doc.plain(c.brief))

    def write_function(self, ns, name, overloads):
        page = overloads[0].url.split("#")[0]
        self.doc.base = self.linker.base = base_of(page)
        out = [self.crumbs(ns, page, "function", include_self=True), f'<h1><code>{E(name)}</code></h1>']
        if len(overloads) > 1:
            out.append(f'<p class="muted">{len(overloads)} overloads.</p>')
        for m in overloads:
            out.append(self.header_line(m.file, m.line, page, link=False) if m is overloads[0] or m.file != overloads[0].file else "")
            out.append(self.member_section(m, page))
        out.append(self.used_block(page))
        brief = self.doc.plain(overloads[0].brief)
        self.write(page, f"{ns.name}::{name}", "".join(out), "api", self.api_sidebar(page, ns), math=True,
                   description=brief)
        self.add_search(f"{ns.name}::{name}", "function", page, "", brief)

    def write_api_index(self):
        page = "api/index.html"
        self.doc.base = base_of(page)
        api = self.api
        out = ['<h1>API reference</h1>',
               f'<p class="lead">Every public namespace, class and function of <code>{E(self.root_ns)}</code>, from the '
               'documentation comments in its headers (read by Doxygen).'
               + (f' <code>{E(self.aliases[0])}</code> is an alias of <code>{E(self.root_ns)}</code>: '
                  f'<code>#include &lt;{E(self.aliases[0])}/{E(self.aliases[0])}.hpp&gt;</code> brings in everything.'
                  if self.aliases and os.path.exists(os.path.join(ROOT, "include", self.aliases[0], self.aliases[0] + ".hpp"))
                  else "") + '</p>',
               '<div class="stats">', stat(len(api.namespaces()), "namespaces"), stat(len(api.types()), "classes"),
               stat(len(api.functions), "functions"), '</div>',
               '<div class="table-wrap"><table class="summary"><tr><th>Namespace</th><th>Classes</th><th>Functions</th><th></th></tr>']
        for ns in api.namespaces():
            types = sum(1 for c in ns.inner if c.kind != "namespace")
            fns = len({m.name for m in ns.members if m.kind == "function"})
            out.append(f'<tr><td><a href="{rel(ns.url, page)}"><code>{E(ns.name)}</code></a></td><td>{types}</td>'
                       f'<td>{fns}</td><td>{self.doc.brief(ns.brief)}</td></tr>')
        out.append("</table></div>")
        self.write(page, "API reference", "".join(out), "api", self.api_sidebar(page))

    def write_classes_index(self):
        page = "api/classes.html"
        self.doc.base = base_of(page)
        by_letter = defaultdict(list)
        for c in self.api.types():
            by_letter[c.short[:1].upper()].append(c)
        letters = sorted(by_letter)
        out = ['<h1>All classes</h1>', '<p class="letters">' + " ".join(
            f'<a href="#letter-{E(l)}">{E(l)}</a>' for l in letters) + "</p>"]
        for l in letters:
            out.append(f'<h2 id="letter-{E(l)}">{E(l)}</h2>')
            out.append(self.summary_rows([(c.name.split("::", 1)[-1] if "::" in c.name else c.name, c.url,
                                           self.doc.brief(c.brief))
                                          for c in sorted(by_letter[l], key=lambda c: c.short.lower())], page))
        self.write(page, "All classes", "".join(out), "api", self.api_sidebar(page))

    # -- guides and home -----------------------------------------------------
    def page_of(self, repo_path):
        for g in self.guides:
            if os.path.normpath(g["source"]) == os.path.normpath(repo_path):
                return g["url"]
        if os.path.normpath(repo_path) == os.path.normpath(self.config.get("home", "README.md")):
            return "index.html"
        return None

    def markdown(self):
        return pages.Markdown(ROOT, self.out, self.page_of, self.repo_file_url,
                              lambda code, lang: hl.highlight(code, lang, self.linker))

    def guides_sidebar(self, page, current=None):
        items = "".join(f'<li><a{" class=current" if g is current else ""} href="{rel(g["url"], page)}">{E(g["title"])}</a></li>'
                        for g in self.guides)
        return (f'<a class="side-top" href="{rel("guides/index.html", page)}">All guides</a>'
                f'<div class="side-heading">Guides</div><ul>{items}</ul>')

    def build_guides(self):
        md = self.markdown()
        for g in self.guides:
            if not os.path.exists(os.path.join(ROOT, g["source"])):
                print(f"guide not found: {g['source']}", file=sys.stderr)
                continue
            self.linker.base = base_of(g["url"])
            body, title = md.render(g["source"], g["url"])
            src = self.source_url(g["source"])
            foot = f'<p class="report-meta"><a href="{E(src)}">source</a></p>' if src else ""
            self.write(g["url"], g["title"], f'<article class="prose">{body}</article>{foot}', "guides",
                       self.guides_sidebar(g["url"], g), math=True)
            self.add_search(g["title"], "guide", g["url"], "", "")
        page = "guides/index.html"
        cards = "".join(f'<a class="card" href="{rel(g["url"], page)}"><div class="card-title">{E(g["title"])}</div>'
                        f'<div class="card-text muted">{E(g["source"])}</div></a>' for g in self.guides)
        self.write(page, "Guides", f'<h1>Guides</h1><div class="cards">{cards}</div>', "guides",
                   self.guides_sidebar(page))

    def build_home(self):
        page = "index.html"
        self.linker.base = ""
        body, _ = self.markdown().render(self.config.get("home", "README.md"), page)
        n_reports = len(self.reports)
        logo = self.config.get("logo")
        logo_html = f'<img class="hero-logo" src="assets/{E(logo)}" alt="">' if logo and os.path.exists(os.path.join(ROOT, logo)) else ""
        cards = [
            ("api/index.html", "API reference",
             f"{len(self.api.namespaces())} namespaces, {len(self.api.types())} classes and "
             f"{len(self.api.functions)} functions, from the headers' documentation."),
            ("reports/index.html", "Reports",
             f"{n_reports} literate reports: methods, their mathematics and worked examples, run by mep."),
            ("guides/index.html", "Guides", ", ".join(g["title"] for g in self.guides) + "."),
        ]
        hero = (f'<section class="hero">{logo_html}<div><h1>{E(self.title)}</h1>'
                f'<p class="lead">{E(self.config.get("tagline", ""))}</p></div></section>'
                '<div class="cards cards-3">' + "".join(
                    f'<a class="card" href="{u}"><div class="card-title">{E(t)}</div><div class="card-text">{E(d)}</div></a>'
                    for u, t, d in cards) + "</div>")
        # README starts with its own centred title and logo, shown in the hero instead.
        body = re.sub(r"^\s*(<p align=\"center\">.*?</p>\s*)+", "", body, flags=re.S)
        self.write(page, self.title, hero + f'<article class="prose">{body}</article>', "home", wide=True, math=True)

    def write_search(self):
        with open(os.path.join(self.out, "search-index.js"), "w", encoding="utf-8") as f:
            f.write("window.SEARCH_INDEX=")
            json.dump(self.search, f, separators=(",", ":"), ensure_ascii=False)
            f.write(";\n")


def api_overloads(api, ns, name):
    return api.functions.get((ns.id, name), [])


def slug_id(text):
    return re.sub(r"[^\w-]+", "-", text.lower()).strip("-") or "x"


def stat(n, label, bad=False):
    return f'<div class="stat{" stat-bad" if bad else ""}"><div class="stat-n">{n}</div><div class="stat-l">{E(label)}</div></div>'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xml", required=True, help="Doxygen's XML output directory")
    ap.add_argument("--out", required=True, help="where the site is written (emptied first)")
    ap.add_argument("--config", default=os.path.join(ROOT, "docs", "website.toml"))
    ap.add_argument("--mep-mepml", default="mep-mepml", help="mep's mepml tool")
    ap.add_argument("--run", action="store_true", help="run each changed report's code blocks with mep first")
    ap.add_argument("--force", action="store_true", help="with --run, rerun every report")
    ap.add_argument("--timeout", type=float, default=0, help="with --run, the longest a block may run (s)")
    ap.add_argument("--jobs", "-j", type=int, default=0,
                    help="with --run, how many reports run at once (default: one per core, as memory allows)")
    ap.add_argument("--no-pch", action="store_true",
                    help="with --run, compile the blocks with plain g++, not clang++ with a precompiled dm.hpp")
    ap.add_argument("--status-dir", help="where each report's last run status is kept (default: OUT/../mepml-status)")
    ap.add_argument("--allow-failures", action="store_true", help="exit 0 even when a code block failed")
    ap.add_argument("--no-reports", action="store_true", help="leave the reports out")
    ap.add_argument("--project-name", help="the site's title (default: the CMake project name)")
    ap.add_argument("--version", help="the version shown (default: the CMake project version)")
    ap.add_argument("--namespace", help="the C++ root namespace (default: the CMake project name)")
    ap.add_argument("reports", nargs="*", help="the reports (default: docs/mepml/examples/**/*.mepml)")
    args = ap.parse_args()
    site = Site(args)
    try:
        failures = site.build()
    except rep.ToolchainError as error:
        ap.exit(1, str(error) + "\n")
    return 1 if failures and not args.allow_failures else 0


if __name__ == "__main__":
    sys.exit(main())
