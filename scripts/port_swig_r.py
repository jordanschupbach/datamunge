#!/usr/bin/env python3
"""One-off porter: rewrite SWIG-style R calls to the rosetta R6 API.

Usage: port_swig_r.py GENERATED_R FILE...

Uses the generated R6 interface (R/datamunger.R) to know classes, their (inherited) methods,
fields and statics, then rewrites, outside strings and comments:
  Cls(...)                 -> Cls$new(...)
  Cls_static(...)          -> Cls$static(...)
  Cls_method(obj, ...)     -> obj$method(...)
  Cls_field_get(obj)       -> obj$field
  Cls_field_set(obj, v)    -> obj$field <- v
  delete_Cls(obj)          -> obj$close()
"""
import re
import sys

ALIASES = {"BSR": "BSplineRegression", "PSR": "PSplineRegression", "RGBA": "RGB"}
IDENT = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")


def load(path):
    text = open(path).read()
    classes = {}
    current = None
    section = None
    for line in text.splitlines():
        m = re.match(r'^`([^`]+)` <- R6::R6Class\("[^"]+"(?:, inherit = `([^`]+)`)?', line)
        if m:
            current = {"parent": m.group(2), "methods": set(), "fields": set(), "statics": set()}
            classes[m.group(1)] = current
            section = "public"
            continue
        if current is not None:
            if line.startswith("  active = list("):
                section = "active"
            elif line.startswith("  public = list("):
                section = "public"
            elif line.startswith("))") or line.startswith(")"):
                current = None
                continue
            m = re.match(r"^\s+`([^`]+)` = function", line)
            if m and current is not None:
                (current["fields"] if section == "active" else current["methods"]).add(m.group(1))
        m = re.match(r"^`([^`]+)`\$`([^`]+)` <- function", line)
        if m and m.group(1) in classes:
            classes[m.group(1)]["statics"].add(m.group(2))
    for alias, target in ALIASES.items():
        classes.setdefault(alias, classes.get(target))
    return classes


def members(classes, name, kind):
    out = set()
    while name and name in classes and classes[name]:
        out |= classes[name][kind]
        name = classes[name]["parent"]
    return out


def tokens(src):
    """Yield (kind, start, end): kind is 'code', 'str' or 'comment'."""
    i, n, start = 0, len(src), 0
    while i < n:
        c = src[i]
        if c in "\"'`":
            if start < i:
                yield "code", start, i
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            yield ("code" if c == "`" else "str"), i, j + 1
            i = start = j + 1
        elif c == "#":
            if start < i:
                yield "code", start, i
            j = src.find("\n", i)
            j = n if j < 0 else j
            yield "comment", i, j
            i = start = j
        else:
            i += 1
    if start < n:
        yield "code", start, n


def mask(src):
    """Copy of src with string and comment contents blanked, so code matching ignores them."""
    chars = list(src)
    for kind, a, b in tokens(src):
        if kind in ("str", "comment"):
            for k in range(a + 1, b - (1 if kind == "str" else 0)):
                if chars[k] != "\n":
                    chars[k] = " "
    return "".join(chars)


def call_args(masked, open_paren):
    """Return (close_index, [(start, end) of top-level args])."""
    depth, args, start = 0, [], open_paren + 1
    for k in range(open_paren, len(masked)):
        c = masked[k]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                if masked[start:k].strip() or args:
                    args.append((start, k))
                return k, args
        elif c == "," and depth == 1:
            args.append((start, k))
            start = k + 1
    raise ValueError("unbalanced call")


SIMPLE = re.compile(r"^[A-Za-z.][A-Za-z0-9._]*(\$[A-Za-z_.][A-Za-z0-9._]*|\[\[[^\]]+\]\]|\[[^\]]+\]|\([^()]*\))*$")


def receiver(text):
    text = text.strip()
    return text if SIMPLE.match(text) or re.match(r"^[A-Za-z.][\w.$]*\(.*\)$", text, re.S) else f"({text})"


def resolve(classes, name):
    """Split Cls_member into (class, member, kind), longest class prefix first."""
    name = re.sub(r"__SWIG_\d+$", "", name)
    parts = name.split("_")
    for cut in range(len(parts) - 1, 0, -1):
        cls, member = "_".join(parts[:cut]), "_".join(parts[cut:])
        if cls not in classes or not classes[cls]:
            continue
        for suffix, kind in (("_get", "get"), ("_set", "set")):
            if member.endswith(suffix) and member[: -len(suffix)] in members(classes, cls, "fields"):
                return cls, member[: -len(suffix)], kind
        if member in members(classes, cls, "statics"):
            return cls, member, "static"
        if member in members(classes, cls, "methods"):
            return cls, member, "method"
    return None


def port(src, classes):
    changed = True
    while changed:
        changed = False
        masked = mask(src)
        for m in IDENT.finditer(masked):
            name, a = m.group(0), m.start()
            b = m.end()
            if b >= len(masked) or masked[b] != "(":
                continue
            prev = masked[:a].rstrip()[-1:] if masked[:a].rstrip() else ""
            if prev in ("$", "@") or masked[max(0, a - 2):a] == "::":
                continue
            close, args = call_args(masked, b)
            argtext = [src[s:e] for s, e in args]
            replacement = None
            ctor = re.sub(r"__SWIG_\d+$", "", name)
            if ctor in classes and classes[ctor] is not None:
                replacement = f"{ctor}$new(" + src[b + 1:close] + ")"
            elif name.startswith("delete_") and name[7:] in classes and len(args) == 1:
                replacement = f"{receiver(argtext[0])}$close()"
            else:
                found = resolve(classes, name)
                if found:
                    cls, member, kind = found
                    if kind == "static":
                        replacement = f"{cls}${member}(" + src[b + 1:close] + ")"
                    elif kind == "get" and len(args) == 1:
                        replacement = f"{receiver(argtext[0])}${member}"
                    elif kind == "set" and len(args) == 2:
                        replacement = f"{receiver(argtext[0])}${member} <- {argtext[1].strip()}"
                    elif kind == "method" and args:
                        rest = ",".join(argtext[1:]).strip()
                        replacement = f"{receiver(argtext[0])}${member}({rest})"
            if replacement is not None:
                src = src[:a] + replacement + src[close + 1:]
                changed = True
                break
    return src


def main():
    classes = load(sys.argv[1])
    for path in sys.argv[2:]:
        old = open(path).read()
        new = port(old, classes)
        if new != old:
            open(path, "w").write(new)
            print("ported", path)


if __name__ == "__main__":
    main()
