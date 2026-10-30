"""The API reference: Doxygen's XML read into a model, and its documentation as HTML.

Doxygen (docs/Doxyfile, XML only) parses the C++; this module decides what the
website shows and where. Shown: the project's namespaces and the classes,
functions, types and variables declared in headers under include/, public and
protected. Hidden: anonymous and detail namespaces (`detail`, `*_detail`,
`impl`, `internal`), anything only in a .cpp, private members.

URLs (relative to the site root):
  api/<ns path>/index.html          a namespace: what it holds, grouped by header,
                                    with its enums, typedefs and variables in full
  api/<ns path>/class.<Name>.html   a class (struct.<Name> for a struct, nested
                                    classes as Outer.Inner), every member in full
  api/<ns path>/fn.<name>.html      a free function, every overload
Doxygen ids (refid) map to those URLs, so its cross-references become links.
"""
import html
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

HIDDEN_NS = re.compile(r"(^|::)(anonymous_namespace\{[^}]*\}|detail|\w+_detail|impl|internal)(::|$)")
COMPOUND_KINDS = {"namespace", "class", "struct", "union", "concept"}
MEMBER_KINDS = {"function", "variable", "typedef", "enum"}
# Doxygen's character elements.
CHARS = {
    "ndash": "–", "mdash": "—", "nonbreakablespace": " ", "lsquo": "‘", "rsquo": "’",
    "ldquo": "“", "rdquo": "”", "copy": "©", "trademark": "™", "registered": "®",
    "sp": " ", "zwj": "", "zwnj": "", "hellip": "…", "times": "×", "minus": "−", "deg": "°",
    "plusmn": "±", "middot": "·", "laquo": "«", "raquo": "»", "nbsp": " ",
}
BLOCK_TAGS = {
    "parameterlist", "simplesect", "programlisting", "itemizedlist", "orderedlist", "table", "verbatim",
    "heading", "xrefsect", "blockquote", "preformatted", "variablelist", "hruler", "sect1", "sect2", "sect3",
    "sect4", "details", "para", "image", "dot", "dotfile", "htmlonly", "manonly", "rtfonly", "latexonly",
    "docbookonly", "xmlonly",
}
SECTION_TITLES = {
    "return": "Returns", "see": "See also", "note": "Note", "warning": "Warning", "attention": "Attention",
    "remark": "Remark", "pre": "Precondition", "post": "Postcondition", "invariant": "Invariant",
    "since": "Since", "version": "Version", "author": "Author", "authors": "Authors", "date": "Date",
    "copyright": "Copyright", "deprecated": "Deprecated", "todo": "To do", "important": "Important",
}
CALLOUTS = {"note", "warning", "attention", "remark", "deprecated", "todo", "important"}
PARAM_TITLES = {"param": "Parameters", "templateparam": "Template parameters", "exception": "Throws",
                "retval": "Return values"}
OP_CHARS = {
    "<": "lt", ">": "gt", "=": "eq", "!": "not", "+": "plus", "-": "minus", "*": "star", "/": "slash",
    "%": "mod", "&": "and", "|": "or", "^": "xor", "~": "tilde", "[": "index", "]": "", "(": "call", ")": "",
    ",": "comma", " ": "", '"': "lit",
}


def slug(name):
    """A file-name-safe form of a C++ name (`operator<<` is `operator-lt-lt`)."""
    out = []
    for c in name:
        if c.isalnum() or c in "_.":
            out.append(c)
        elif c == ":":
            out.append(".")
        elif c in OP_CHARS:
            if OP_CHARS[c]:
                out.append("-" + OP_CHARS[c])
        else:
            out.append("-")
    return re.sub(r"\.+", ".", "".join(out)).strip(".") or "_"


def strip_template_args(name):
    """`Foo<T, U>::Bar` without its template arguments: `Foo::Bar`."""
    out, depth = [], 0
    for c in name:
        if c == "<":
            depth += 1
        elif c == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(c)
    return "".join(out).replace(" ", "")


class Member:
    """A function, variable, typedef or enum, in a namespace or a class."""

    def __init__(self, el, owner):
        self.el = el
        self.owner = owner
        self.id = el.get("id")
        self.kind = el.get("kind")
        self.prot = el.get("prot", "public")
        self.static = el.get("static") == "yes"
        self.name = (el.findtext("name") or "").strip()
        self.qualname = (el.findtext("qualifiedname") or f"{owner.name}::{self.name}").strip()
        loc = el.find("location")
        # Where it is declared (a header), not where Doxygen met its definition.
        self.file = (loc.get("declfile") or loc.get("file", "")) if loc is not None else ""
        self.line = int(loc.get("declline") or loc.get("line") or 0) if loc is not None else 0
        self.brief = el.find("briefdescription")
        self.detail = el.find("detaileddescription")
        self.url = ""      # page#anchor
        self.anchor = ""

    @property
    def is_operator(self):
        return self.kind == "function" and re.match(r"operator\W", self.name + " ") is not None

    def sort_key(self):
        return (self.name.lower(), self.line)


class Compound:
    """A namespace, class, struct, union or concept."""

    def __init__(self, el, refid, kind):
        self.el = el
        self.id = refid
        self.kind = kind
        self.name = (el.findtext("compoundname") or "").strip()
        loc = el.find("location")
        self.file = loc.get("file", "") if loc is not None else ""
        self.line = int(loc.get("line", "0") or 0) if loc is not None else 0
        self.brief = el.find("briefdescription")
        self.detail = el.find("detaileddescription")
        self.members = []
        self.inner = []       # visible nested classes / namespaces (Compound)
        self.parent = None    # enclosing Compound
        self.url = ""
        self.local = ""       # name relative to its namespace (Outer::Inner)
        self.visible = False

    @property
    def short(self):
        return self.name.split("::")[-1]

    @property
    def is_type(self):
        return self.kind != "namespace"


class Api:
    """Every compound and member Doxygen found, what is shown, and where."""

    def __init__(self, xml_dir, root_ns, public_dirs=("include/",)):
        self.xml_dir = xml_dir
        self.root_ns = root_ns
        self.public_dirs = tuple(public_dirs)
        self.compounds = {}           # refid -> Compound
        self.urls = {}                # refid -> url (compounds and members)
        self.by_name = {}             # qualified name (no template args) -> url
        self.short_names = defaultdict(set)  # last name part -> urls of project symbols
        self.functions = {}           # (namespace refid, name) -> [Member]
        self.member_by_id = {}
        self._load()
        self._decide()

    # -- reading -----------------------------------------------------------
    def _load(self):
        index = ET.parse(os.path.join(self.xml_dir, "index.xml")).getroot()
        for c in index.findall("compound"):
            kind = c.get("kind")
            if kind not in COMPOUND_KINDS:
                continue
            refid = c.get("refid")
            path = os.path.join(self.xml_dir, refid + ".xml")
            if not os.path.exists(path):
                continue
            cd = ET.parse(path).getroot().find("compounddef")
            if cd is None:
                continue
            comp = Compound(cd, refid, kind)
            self.compounds[refid] = comp
        for comp in self.compounds.values():
            for tag in ("innerclass", "innernamespace", "innerconcept"):
                for inner in comp.el.findall(tag):
                    child = self.compounds.get(inner.get("refid"))
                    if child is None:
                        continue
                    if inner.get("prot", "public") == "private":
                        child.private = True
                    child.parent = comp
            for sec in comp.el.findall("sectiondef"):
                for m in sec.findall("memberdef"):
                    member = Member(m, comp)
                    comp.members.append(member)
                    self.member_by_id[member.id] = member

    def public_file(self, path):
        return path.startswith(self.public_dirs)

    def _ns_visible(self, name):
        return (name == self.root_ns or name.startswith(self.root_ns + "::")) and not HIDDEN_NS.search(name)

    def _decide(self):
        """Which compounds and members are shown, and their URLs."""
        for comp in self.compounds.values():
            if comp.kind == "namespace":
                comp.visible = self._ns_visible(comp.name)
        # Types: their enclosing scope shown, declared in a public header, not private.
        changed = True
        while changed:
            changed = False
            for comp in self.compounds.values():
                if comp.visible or comp.kind == "namespace":
                    continue
                parent = comp.parent
                ok = (parent is not None and parent.visible and not getattr(comp, "private", False)
                      and self.public_file(comp.file) and "@" not in comp.name
                      and not HIDDEN_NS.search(comp.name))
                if ok:
                    comp.visible = True
                    changed = True
        for comp in self.compounds.values():
            if not comp.visible:
                continue
            ns = self.namespace_of(comp)
            ns_dir = "api/" + "/".join(slug(p) for p in ns.name.split("::"))
            if comp.kind == "namespace":
                comp.url = ns_dir + "/index.html"
            else:
                comp.local = comp.name[len(ns.name) + 2:] if comp.name.startswith(ns.name + "::") else comp.short
                comp.url = f"{ns_dir}/{comp.kind}.{slug(comp.local)}.html"
            self.urls[comp.id] = comp.url
            self._index_name(comp.name, comp.url)
            if comp.parent is not None and comp.parent.visible:
                comp.parent.inner.append(comp)
        # Members.
        for comp in self.compounds.values():
            if not comp.visible:
                continue
            seen = defaultdict(int)
            shown = []
            for m in comp.members:
                if m.kind not in MEMBER_KINDS or m.prot == "private" or m.name.startswith("@"):
                    continue
                if not self.public_file(m.file):
                    continue
                shown.append(m)
            comp.members = shown
            for m in shown:
                seen[m.name] += 1
                n = seen[m.name]
                if comp.kind == "namespace" and m.kind == "function":
                    self.functions.setdefault((comp.id, m.name), []).append(m)
                    base = comp.url.rsplit("/", 1)[0] + f"/fn.{slug(m.name)}.html"
                    m.anchor = f"o{n}"
                    m.url = base + "#" + m.anchor
                    page = base
                else:
                    prefix = {"function": "", "variable": "var.", "typedef": "type.", "enum": "enum."}[m.kind] \
                        if comp.kind == "namespace" else ""
                    m.anchor = prefix + slug(m.name) + (f"-{n}" if n > 1 else "")
                    m.url = comp.url + "#" + m.anchor
                    page = m.url
                self.urls[m.id] = m.url
                if n == 1:
                    # (A class's members are found by their qualified name only.)
                    self._index_name(m.qualname, page, short=comp.kind == "namespace")
                if m.kind == "enum":
                    for v in m.el.findall("enumvalue"):
                        self.urls[v.get("id")] = m.url
                        vname = (v.findtext("name") or "").strip()
                        self._index_name(f"{m.qualname}::{vname}", m.url, short=False)

    def _index_name(self, qualname, url, short=True):
        key = strip_template_args(qualname)
        self.by_name.setdefault(key, url)
        if short:
            self.short_names[key.split("::")[-1]].add(url)

    # -- lookups -------------------------------------------------------------
    def namespace_of(self, comp):
        while comp is not None and comp.kind != "namespace":
            comp = comp.parent
        return comp

    def namespaces(self):
        return sorted((c for c in self.compounds.values() if c.visible and c.kind == "namespace"),
                      key=lambda c: c.name)

    def types(self):
        return sorted((c for c in self.compounds.values() if c.visible and c.is_type), key=lambda c: c.name)

    def url_of(self, refid):
        return self.urls.get(refid)

    def resolve(self, parts):
        """The URL of a qualified name given as parts, if the reference shows it."""
        key = "::".join(parts)
        url = self.by_name.get(key)
        if url:
            return url
        # `dm::RPlot`, re-exported into the root namespace by a using-declaration:
        # the one project symbol of that name, when there is exactly one.
        if len(parts) == 2 and parts[0] == self.root_ns:
            urls = self.short_names.get(parts[1], set())
            if len(urls) == 1:
                return next(iter(urls))
        return None


# -- documentation markup -------------------------------------------------------
class DocRenderer:
    """Doxygen's description markup as HTML, its references as links.

    `base` is the prefix from the page being written to the site root (set by
    the caller for each page). `highlight(code, lang)` highlights code."""

    def __init__(self, api, highlight):
        self.api = api
        self.highlight = highlight
        self.base = ""

    def href(self, refid):
        url = self.api.url_of(refid)
        return self.base + url if url else None

    # Block level.
    def description(self, el):
        if el is None:
            return ""
        return "".join(self.block(c) for c in el).strip()

    def is_empty(self, el):
        return el is None or not "".join(el.itertext()).strip()

    def block(self, el):
        tag = el.tag
        if tag == "para":
            return self.para(el)
        if tag in ("sect1", "sect2", "sect3", "sect4"):
            level = {"sect1": 3, "sect2": 4, "sect3": 5, "sect4": 6}[tag]
            title = el.find("title")
            out = f"<h{level}>{self.inline_children(title)}</h{level}>" if title is not None else ""
            return out + "".join(self.block(c) for c in el if c.tag != "title")
        if tag == "internal":
            return ""
        if tag == "simplesect":
            return self.simplesect(el)
        if tag == "parameterlist":
            return self.parameterlist(el)
        if tag == "programlisting":
            lang = (el.get("filename") or ".cpp").lstrip(".") or "cpp"
            lines = ["".join(self.code_text(h) for h in cl) for cl in el.findall("codeline")]
            code = "\n".join(lines).rstrip("\n")
            return f'<pre class="hl"><code>{self.highlight(code, lang)}</code></pre>'
        if tag in ("verbatim", "preformatted"):
            return f"<pre>{html.escape(''.join(el.itertext()))}</pre>"
        if tag in ("itemizedlist", "orderedlist"):
            t = "ul" if tag == "itemizedlist" else "ol"
            items = "".join(f"<li>{''.join(self.block(p) for p in li)}</li>" for li in el.findall("listitem"))
            return f"<{t}>{items}</{t}>"
        if tag == "variablelist":
            out = []
            for entry in el:
                if entry.tag == "varlistentry":
                    out.append(f"<dt>{self.inline_children(entry.find('term'))}</dt>")
                elif entry.tag == "listitem":
                    out.append(f"<dd>{''.join(self.block(p) for p in entry)}</dd>")
            return f"<dl>{''.join(out)}</dl>"
        if tag == "table":
            rows = []
            for row in el.findall("row"):
                cells = []
                for e in row.findall("entry"):
                    t = "th" if e.get("thead") == "yes" else "td"
                    cells.append(f"<{t}>{''.join(self.block(p) for p in e)}</{t}>")
                rows.append(f"<tr>{''.join(cells)}</tr>")
            return f'<div class="table-wrap"><table>{"".join(rows)}</table></div>'
        if tag == "heading":
            level = min(6, int(el.get("level", "3")) + 2)
            return f"<h{level}>{self.inline_children(el)}</h{level}>"
        if tag == "xrefsect":
            title = el.findtext("xreftitle") or ""
            xd = el.find("xrefdescription")
            body = "".join(self.block(c) for c in xd) if xd is not None else ""
            return self.section(title, body, callout=True, kind=slug(title.lower()))
        if tag == "blockquote":
            return f"<blockquote>{''.join(self.block(c) for c in el)}</blockquote>"
        if tag == "hruler":
            return "<hr>"
        if tag == "details":
            summary = el.find("summary")
            body = "".join(self.block(c) for c in el if c.tag != "summary")
            return f"<details><summary>{self.inline_children(summary)}</summary>{body}</details>"
        if tag in ("image", "dot", "dotfile", "htmlonly", "manonly", "rtfonly", "latexonly", "docbookonly", "xmlonly"):
            return ""
        return self.inline(el)

    def para(self, p):
        """A paragraph, which in Doxygen's XML can hold blocks (lists, sections)."""
        out, run = [], [html.escape(p.text or "")]

        def flush():
            text = "".join(run).strip()
            if text:
                out.append(f"<p>{text}</p>")

        for c in p:
            if c.tag in BLOCK_TAGS:
                flush()
                run = []
                out.append(self.block(c))
            else:
                run.append(self.inline(c))
            run.append(html.escape(c.tail or ""))
        flush()
        return "".join(out)

    def section(self, title, body, callout=False, kind=""):
        cls = f"doc-sect doc-{kind}" + (" callout" if callout else "")
        return f'<div class="{cls}"><div class="doc-sect-title">{title}</div><div class="doc-sect-body">{body}</div></div>'

    def simplesect(self, el):
        kind = el.get("kind", "par")
        if kind == "par":
            title_el = el.find("title")
            title = self.inline_children(title_el) if title_el is not None else ""
            body = "".join(self.block(c) for c in el if c.tag != "title")
            return self.section(title, body, kind="par")
        body = "".join(self.block(c) for c in el)
        return self.section(SECTION_TITLES.get(kind, kind.title()), body, callout=kind in CALLOUTS, kind=kind)

    def parameterlist(self, el):
        kind = el.get("kind", "param")
        rows = []
        for item in el.findall("parameteritem"):
            names = []
            for pn in item.iter("parametername"):
                d = pn.get("direction")
                names.append(f"<code>{self.inline_children(pn)}</code>" + (f' <span class="dir">[{d}]</span>' if d else ""))
            for pt in item.iter("parametertype"):
                names.append(f'<code class="ptype">{self.inline_children(pt)}</code>')
            pd = item.find("parameterdescription")
            desc = "".join(self.block(c) for c in pd) if pd is not None else ""
            rows.append(f"<dt>{', '.join(names)}</dt><dd>{desc}</dd>")
        return self.section(PARAM_TITLES.get(kind, kind.title()), f'<dl class="params">{"".join(rows)}</dl>',
                            kind="params")

    # Inline level.
    def inline_children(self, el):
        if el is None:
            return ""
        out = [html.escape(el.text or "")]
        for c in el:
            out.append(self.block(c) if c.tag in BLOCK_TAGS and c.tag != "para" else
                       (self.para_inline(c) if c.tag == "para" else self.inline(c)))
            out.append(html.escape(c.tail or ""))
        return "".join(out)

    def para_inline(self, p):
        return self.inline_children(p)

    def inline(self, el):
        tag = el.tag
        if tag in CHARS:
            return CHARS[tag]
        if tag == "ref":
            href = self.href(el.get("refid"))
            inner = self.inline_children(el)
            return f'<a href="{html.escape(href)}">{inner}</a>' if href else inner
        if tag == "ulink":
            return f'<a href="{html.escape(el.get("url", ""))}">{self.inline_children(el)}</a>'
        if tag == "computeroutput":
            return f"<code>{self.inline_children(el)}</code>"
        if tag == "bold":
            return f"<strong>{self.inline_children(el)}</strong>"
        if tag in ("emphasis", "italic"):
            return f"<em>{self.inline_children(el)}</em>"
        if tag == "strike" or tag == "del":
            return f"<s>{self.inline_children(el)}</s>"
        if tag in ("subscript", "superscript"):
            t = "sub" if tag == "subscript" else "sup"
            return f"<{t}>{self.inline_children(el)}</{t}>"
        if tag == "linebreak":
            return "<br>"
        if tag == "anchor":
            return f'<span id="{html.escape(el.get("id", ""))}"></span>'
        if tag == "formula":
            return self.formula(el.text or "")
        if tag in ("htmlonly", "latexonly", "rtfonly", "manonly", "xmlonly", "docbookonly", "image", "dot"):
            return ""
        return self.inline_children(el)

    def formula(self, tex):
        tex = tex.strip()
        if tex.startswith("$") and tex.endswith("$"):
            return f'<span class="math">\\({html.escape(tex[1:-1])}\\)</span>'
        if tex.startswith("\\[") or tex.startswith("\\begin"):
            return f'<span class="math display">{html.escape(tex)}</span>'
        return f'<span class="math">\\({html.escape(tex)}\\)</span>'

    def code_text(self, el):
        """The plain text of a code line's highlight element (Pygments highlights it again)."""
        out = [el.text or ""]
        for c in el:
            out.append(" " if c.tag == "sp" else self.code_text(c))
            out.append(c.tail or "")
        return "".join(out)

    def plain(self, el):
        """A description's text, for search and summaries."""
        if el is None:
            return ""
        return re.sub(r"\s+", " ", "".join(el.itertext())).strip()

    def brief(self, el):
        """A brief description as inline HTML (no <p>)."""
        if el is None:
            return ""
        paras = [p for p in el if p.tag == "para"]
        return " ".join(self.inline_children(p).strip() for p in paras).strip()

    # Signatures.
    def linked(self, el):
        """A type (or other code text with references) as HTML."""
        if el is None:
            return ""
        return self.inline_children(el).strip()

    def template_line(self, el):
        tpl = el.find("templateparamlist")
        if tpl is None:
            return ""
        params = []
        for p in tpl.findall("param"):
            s = self.linked(p.find("type"))
            dn = (p.findtext("declname") or "").strip()
            if dn and not s.endswith(dn):
                s += " " + html.escape(dn)
            dv = p.find("defval")
            if dv is not None:
                s += " = " + self.linked(dv)
            params.append(s)
        return f'<span class="k">template</span>&lt;{", ".join(params)}&gt;'

    def signature(self, m, name_html=None):
        """A member's declaration as highlighted, linked HTML."""
        el = m.el
        name = name_html or f'<span class="nf">{html.escape(m.name)}</span>'
        pre = []
        if el.get("virt") in ("virtual", "pure-virtual"):
            pre.append("virtual")
        if m.static:
            pre.append("static")
        if el.get("explicit") == "yes":
            pre.append("explicit")
        if el.get("constexpr") == "yes":
            pre.append("constexpr")
        elif el.get("consteval") == "yes":
            pre.append("consteval")
        if el.get("inline") == "yes" and m.kind == "variable":
            pre.append("inline")
        lead = "".join(f'<span class="k">{p}</span> ' for p in pre)
        tpl = self.template_line(el)
        tpl = tpl + "\n" if tpl else ""
        if m.kind == "function":
            ret = self.linked(el.find("type"))
            params = []
            for p in el.findall("param"):
                s = self.linked(p.find("type"))
                dn = (p.findtext("declname") or "").strip()
                if dn:
                    s += " " + f'<span class="nv">{html.escape(dn)}</span>'
                arr = p.findtext("array")
                if arr:
                    s += html.escape(arr)
                dv = p.find("defval")
                if dv is not None:
                    s += " = " + self.linked(dv)
                params.append(s)
            args = el.findtext("argsstring") or ""
            close = args.rfind(")")
            suffix = html.escape(args[close + 1:].strip()) if close >= 0 else ""
            head = f"{lead}{ret + ' ' if ret else ''}{name}("
            flat = head + ", ".join(params) + ")"
            plain_len = len(re.sub(r"<[^>]+>", "", html.unescape(flat)))
            if plain_len > 88 and len(params) > 1:
                body = head + "\n" + ",\n".join("    " + p for p in params) + ")"
            else:
                body = flat
            return tpl + body + (" " + suffix if suffix else "")
        if m.kind == "variable":
            init = self.linked(el.find("initializer"))
            return f"{tpl}{lead}{self.linked(el.find('type'))} {name}{(' ' + init) if init else ''}"
        if m.kind == "typedef":
            definition = (el.findtext("definition") or "").strip()
            if definition.startswith("using"):
                return f'{tpl}<span class="k">using</span> {name} = {self.linked(el.find("type"))}'
            return f'<span class="k">typedef</span> {self.linked(el.find("type"))} {name}{html.escape(el.findtext("argsstring") or "")}'
        if m.kind == "enum":
            strong = "class " if el.get("strong") == "yes" else ""
            under = self.linked(el.find("type"))
            return f'<span class="k">enum {strong}</span>{name}{" : " + under if under else ""}'
        return name

    def enum_values(self, m):
        rows = []
        for v in m.el.findall("enumvalue"):
            name = html.escape((v.findtext("name") or "").strip())
            init = self.linked(v.find("initializer"))
            desc = self.description(v.find("briefdescription")) + self.description(v.find("detaileddescription"))
            rows.append(f"<tr><td><code>{name}</code></td><td><code>{init}</code></td><td>{desc}</td></tr>")
        if not rows:
            return ""
        return ('<div class="table-wrap"><table class="enum-values"><tr><th>Enumerator</th><th>Value</th><th>Description</th></tr>'
                + "".join(rows) + "</table></div>")
