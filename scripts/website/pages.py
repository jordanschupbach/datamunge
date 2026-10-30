"""Markdown pages (the home page's README and the guides) as HTML.

CommonMark with tables and strikethrough (markdown-it-py), headings given ids
for the contents list, code fences highlighted, and links fixed up: a link to
another page of the site goes there, a link to any other file of the
repository goes to it on the repository's web page, and a picture is copied
into the site.
"""
import html
import os
import re
import shutil

from markdown_it import MarkdownIt


def heading_id(text, used):
    base = re.sub(r"[^\w\- ]", "", text.lower()).strip().replace(" ", "-") or "section"
    base = re.sub(r"-+", "-", base)
    n, out = 1, base
    while out in used:
        n += 1
        out = f"{base}-{n}"
    used.add(out)
    return out


class Markdown:
    """Renders the repository's Markdown files for the site.

    `page_of(repo_path)` gives the site URL of a repository file that is a page
    of the site (or None); `repo_url(repo_path)` its URL on the repository's web
    page (or None); `highlight(code, lang)` highlights a fence."""

    def __init__(self, root, out_root, page_of, repo_url, highlight):
        self.root = root
        self.out_root = out_root
        self.page_of = page_of
        self.repo_url = repo_url
        self.highlight = highlight
        self.md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
        self.md.renderer.rules["fence"] = self._fence

    def _fence(self, tokens, idx, options, env):
        tok = tokens[idx]
        lang = tok.info.strip().split()[0] if tok.info.strip() else ""
        lang = lang.strip("{}.").split(",")[0]
        return f'<pre class="hl"><code>{self.highlight(tok.content.rstrip(chr(10)), lang)}</code></pre>\n'

    def render(self, source, page_url):
        """(HTML, title) of the Markdown file `source` (repository-relative), for the page at `page_url`."""
        with open(os.path.join(self.root, source), encoding="utf-8") as f:
            text = f.read()
        tokens = self.md.parse(text)
        used, title = set(), None
        for i, tok in enumerate(tokens):
            if tok.type == "heading_open":
                inline = tokens[i + 1]
                plain = "".join(c.content for c in (inline.children or []) if c.type in ("text", "code_inline"))
                tok.attrSet("id", heading_id(plain, used))
                if title is None and tok.tag == "h1":
                    title = plain.strip()
            if tok.type == "inline":
                for child in tok.children or []:
                    if child.type == "link_open":
                        child.attrSet("href", self._link(child.attrGet("href") or "", source, page_url))
                    elif child.type == "image":
                        child.attrSet("src", self._picture(child.attrGet("src") or "", source, page_url))
        body = self.md.renderer.render(tokens, self.md.options, {})
        # Raw HTML in the Markdown (README's centred logo): its pictures too.
        body = re.sub(r'(<img\b[^>]*\ssrc=")([^"]+)(")',
                      lambda m: m.group(1) + html.escape(self._picture(html.unescape(m.group(2)), source, page_url))
                      + m.group(3), body)
        return body, title

    def _relative(self, target_url, page_url):
        return os.path.relpath(target_url, os.path.dirname(page_url) or ".").replace(os.sep, "/")

    def _link(self, href, source, page_url):
        if not href or re.match(r"^[a-z][a-z0-9+.-]*:|^#|^//", href):
            return href
        path, _, frag = href.partition("#")
        repo_path = os.path.normpath(os.path.join(os.path.dirname(source), path)).replace(os.sep, "/")
        site = self.page_of(repo_path)
        if site:
            return self._relative(site, page_url) + ("#" + frag if frag else "")
        remote = self.repo_url(repo_path)
        if remote:
            return remote + ("#" + frag if frag else "")
        return href

    def _picture(self, src, source, page_url):
        if not src or re.match(r"^[a-z][a-z0-9+.-]*:|^//|^data:", src):
            return src
        repo_path = os.path.normpath(os.path.join(os.path.dirname(source), src))
        full = os.path.join(self.root, repo_path)
        if not os.path.isfile(full):
            return src
        dest_rel = "assets/" + repo_path.replace(os.sep, "/")
        dest = os.path.join(self.out_root, dest_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(full):
            shutil.copyfile(full, dest)
        return self._relative(dest_rel, page_url)
