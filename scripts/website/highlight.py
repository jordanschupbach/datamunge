"""Syntax highlighting for the website, with C++ names linked to the API reference.

Code is tokenized with Pygments and written as `<span class="TOKEN">` runs (the
short Pygments class names, which the stylesheet colours). In C/C++ code, a
qualified name that starts with the project's namespace or one of its aliases
(`dm::graphics::RPlot::create`) becomes a link to its API page: the longest
prefix the reference knows is linked, so `dm::graphics::RPlot::create` links to
`create` on RPlot's page, and `dm::graphics::RPlot` alone to the class.
"""
import html

from pygments.lexers import get_lexer_by_name
from pygments.token import STANDARD_TYPES, Name, Operator, Punctuation, Token
from pygments.util import ClassNotFound

# mepml/Markdown language names Pygments spells differently.
LANG_ALIASES = {"c++": "cpp", "sh": "bash", "shell": "bash", "r": "r", "py": "python", "js": "javascript"}
C_LIKE = {"cpp", "c", "c++", "cxx", "hpp"}


def token_class(ttype):
    """The short CSS class Pygments gives a token type ('' for plain text)."""
    while ttype not in STANDARD_TYPES and ttype.parent is not None:
        ttype = ttype.parent
    return STANDARD_TYPES.get(ttype, "")


def _merge_scope(tokens):
    """Join the lexer's ':' ':' pairs into one '::' token."""
    out = []
    for ttype, value in tokens:
        if value == ":" and out and out[-1][1] == ":" and out[-1][0] in Operator:
            out[-1] = (Operator, "::")
        else:
            out.append((ttype, value))
    return out


class Linker:
    """Turns qualified C++ names into links.

    `resolve(parts)` gives the (site-root-relative) URL of a fully qualified name (a list of its
    parts, `['datamunge', 'graphics', 'RPlot']`) or None. Names start with
    `root` or one of `aliases` (each standing for `root`). `on_link(url)` is
    told about every link made, so pages can list where a symbol is used."""

    def __init__(self, resolve, root, aliases=(), on_link=None):
        self.base = ""  # prefix from the page being written to the site root
        self.resolve = resolve
        self.root = root
        self.aliases = set(aliases)
        self.on_link = on_link

    def longest(self, parts):
        """(url, number of parts it covers) for the longest known prefix of `parts`."""
        if not parts or (parts[0] != self.root and parts[0] not in self.aliases):
            return None, 0
        full = [self.root] + parts[1:]
        for n in range(len(full), 1, -1):
            url = self.resolve(full[:n])
            if url:
                return url, n
        return None, 0


def highlight(code, lang, linker=None):
    """`code` as highlighted HTML (the inside of a <pre>), names linked when it is C/C++."""
    lang = (lang or "").lower()
    lang = LANG_ALIASES.get(lang, lang)
    try:
        lexer = get_lexer_by_name(lang) if lang else None
    except ClassNotFound:
        lexer = None
    if lexer is None:
        return html.escape(code)
    lexer.stripnl = False
    lexer.ensurenl = False
    tokens = list(lexer.get_tokens(code))
    if lang in C_LIKE:
        tokens = _merge_scope(tokens)
    out = []
    i = 0
    while i < len(tokens):
        ttype, value = tokens[i]
        if linker and lang in C_LIKE and ttype in Name:
            # A chain name (:: name)*, as token indices of its names.
            names = [i]
            j = i + 1
            while j + 1 < len(tokens) and tokens[j][1] == "::" and tokens[j + 1][0] in Name:
                names.append(j + 1)
                j += 2
            parts = [tokens[k][1] for k in names]
            url, n = linker.longest(parts)
            if url:
                end = names[n - 1]
                inner = "".join(_span(tokens[k]) for k in range(i, end + 1))
                out.append(f'<a class="sym" href="{html.escape(linker.base + url)}">{inner}</a>')
                if linker.on_link:
                    linker.on_link(url)
                i = end + 1
                continue
        out.append(_span((ttype, value)))
        i += 1
    return "".join(out)


def _span(tok):
    ttype, value = tok
    cls = token_class(ttype)
    text = html.escape(value)
    return f'<span class="{cls}">{text}</span>' if cls and value.strip() else text


def stylesheet():
    """Token colours for the light and the dark theme (Pygments styles, our backgrounds)."""
    from pygments.formatters import HtmlFormatter
    from pygments.styles import get_all_styles

    def rules(style, scope):
        css = HtmlFormatter(style=style).get_style_defs(scope)
        keep = []
        for line in css.splitlines():
            line = line.strip()
            # Our own background and line-number rules win.
            if not line or "background" in line.split("{")[0] or line.startswith(("pre", "td.", "span.linenos", ".linenos")):
                continue
            if line.split("{")[0].strip() == scope.strip():
                continue
            keep.append(line.replace("background-color", "--unused"))
        return "\n".join(keep)

    dark = "github-dark" if "github-dark" in set(get_all_styles()) else "monokai"
    return (
        "/* Generated from Pygments' 'default' and '" + dark + "' styles. */\n"
        + rules("default", ".hl") + "\n"
        + rules(dark, ':root[data-theme="dark"] .hl') + "\n"
    )
