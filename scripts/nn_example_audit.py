#!/usr/bin/env python3
"""Audit where each canonical NN example keeps its training workflow.

The target shape of every example is:

* the C++ example file owns its orchestration (option parsing, data setup, the
  epoch/batch loop, the per-step forward/loss/gradient/optimizer logic,
  evaluation, checkpointing, reporting) in ``main`` or an example-specific
  function of the same file -- never behind a generic ``Run``/``Train`` harness,
  whether that harness lives in a shared header or was pasted into the file;
* shared ``examples/nn/common`` headers only provide components (model forward
  functions, data readers, metrics, checks) and are bound through SWIG
  (``examples/bindings/nn_components.i``) so the R and Python examples can call
  the same C++;
* the R and Python examples implement the same loop over those bindings.

This script classifies all canonical examples and writes
``examples/nn/example_audit.json``.  ``--markdown PATH`` also writes a report,
``--check`` exits nonzero when a shared header owns a training loop or when an
example listed in ``--require-compliant`` is not compliant.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nn_migration_status import sources  # noqa: E402

COMMON = ROOT / "examples/nn/common"
COMPONENTS_I = ROOT / "examples/bindings/nn_components.i"
AUDIT = ROOT / "examples/nn/example_audit.json"
LEDGER = ROOT / "examples/nn/language_status.json"

LITERALS = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
)
INCLUDE = re.compile(r'^\s*#\s*include\s*"common/([^"]+)"', re.M)
# A candidate function header: optional template/qualifiers, a return type, a
# name and an opening parenthesis.  The parameter list is matched by hand so
# default arguments such as ``= {}`` or ``= nn::Shape({1, 2})`` work.
HEADER = re.compile(
    r"(?:^|\n)[ \t]*(?:template\s*<[^{};]*>\s*)?(?:(?:inline|static|constexpr|virtual|explicit|friend)\s+)*"
    r"(?P<ret>[A-Za-z_][\w:<>,\s\*&\[\]]*?)\s+(?P<name>~?[A-Za-z_]\w*)\s*\(",
    re.S,
)
CONTROL = {"if", "for", "while", "switch", "return", "else", "do", "catch", "sizeof", "static_cast",
           "reinterpret_cast", "const_cast", "dynamic_cast", "new", "delete", "throw", "case", "operator"}
LOOP = re.compile(r"\b(?:for|while)\s*\(")
# Environment ``env.step(action)`` calls are not updates; optimizer ``.step(`` calls are.
UPDATE = re.compile(
    r"\.minimize\(|optimizer\.|\boptimizer\b|(?:opt\w*|adam\w*|sgd\w*)\.step\(|\.apply\(|\bApply\(|\.Update\(|\bUpdate\(|"
    r"\.Train\(|\.TrainStep\(|\.train_step\(|\.Minimize\(|\.update\(|\.Fit\(|\.fit\(|\.Optimize\(|\bOptimize\w*\("
)
TRAINING_CONTEXT = re.compile(r"nn::|optimizer|Optimizer|ParameterStore|\.minimize\(|backward|\.step\(|\.Update\(|\.Train\(")
GRAD = re.compile(r"nn::grad\(|\.backward\(|\.minimize\(|optimizer\.|nn::Optimizer\b|Gradient\(|\.Gradient\(")
HEADER_STEP_METHOD = re.compile(r"\b[a-z_]\w*(?:\.|->)(?:Run|Update|Train|Gradient|Step|Objective|TrainStep|Fit)\(")
CALLBACK_PARAM = re.compile(
    r"std::function|Builder\b|Hook\b|Factory\b|const\s+(?:[\w:]+::)?(?:MarkedDemo|Spec|Demo\w*)\s*&|Make\w*\b"
)
# A loop header that declares an epoch/step/episode/update counter, iterates a batch order, or is `for (;;)`.
LOOP_VARIABLE = re.compile(
    r"\(\s*(?:(?:const\s+)?(?:int|long|unsigned|std::\w+|size_t|auto&?)\s+)?(?:epoch|step|episode|update|iteration|iter|round)\w*\s*[=:]"
    r"|\(\s*;\s*;\s*\)|begin_epoch|\b(?:batches|order)\.next\("
)


def mask(text):
    return LITERALS.sub(lambda m: " " * len(m[0]), text)


def matching(masked, start, open_char, close_char):
    """Index just past the bracket closing the one at ``start`` (or -1)."""
    depth = 0
    for i in range(start, len(masked)):
        if masked[i] == open_char:
            depth += 1
        elif masked[i] == close_char:
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def function_bodies(text):
    """Yield (name, params, body, start, end) for every brace-delimited function in ``text``."""
    masked = mask(text)
    for match in HEADER.finditer(masked):
        name = match.group("name")
        ret = match.group("ret").strip()
        words = ret.split()
        if name in CONTROL or (words and words[-1] in CONTROL) or any(w in CONTROL for w in words[:1]):
            continue
        params_end = matching(masked, match.end() - 1, "(", ")")
        if params_end < 0:
            continue
        tail = re.match(r"\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?(?:->\s*[\w:<>]+\s*)?\{",
                        masked[params_end:])
        if not tail:
            continue
        body_start = params_end + tail.end()
        end = matching(masked, body_start - 1, "{", "}")
        if end < 0:
            continue
        yield name, text[match.end(): params_end - 1], text[body_start: end - 1], match.start(), end


def training_loop(body):
    """A function owns a training loop when it iterates (over epochs/steps/episodes/updates, or
    around an update call) and performs optimizer/gradient updates in a tensor context -- the
    update may sit in a local lambda the loop calls."""
    masked = mask(body)
    if not ((UPDATE.search(masked) or GRAD.search(masked)) and TRAINING_CONTEXT.search(masked)):
        return False
    for loop in LOOP.finditer(masked):
        header_end = matching(masked, loop.end() - 1, "(", ")")
        if header_end < 0:
            continue
        if LOOP_VARIABLE.search(masked[loop.start(): header_end]):
            return True
        rest = masked[header_end:]
        brace = re.match(r"\s*\{", rest)
        if brace:
            end = matching(masked, header_end + brace.end() - 1, "{", "}")
        else:
            semicolon = rest.find(";")
            end = header_end + semicolon + 1 if semicolon >= 0 else len(masked)
        if end >= 0 and UPDATE.search(masked[loop.start(): end]):
            return True
    return False


def normalized(body):
    return re.sub(r"\s+", " ", mask(body)).strip()


def transitive_headers(text, cache):
    seen, stack = [], INCLUDE.findall(text)
    while stack:
        header = stack.pop()
        if header in seen:
            continue
        seen.append(header)
        if header not in cache:
            path = COMMON / header
            cache[header] = INCLUDE.findall(path.read_text()) if path.exists() else []
        stack.extend(cache[header])
    return sorted(seen)


def header_harnesses():
    """Shared-header functions that own a training loop: {header: [function names]}."""
    result = {}
    for path in sorted(COMMON.rglob("*.hpp")):
        text = path.read_text()
        names = [name for name, _params, body, _s, _e in function_bodies(text) if training_loop(body)]
        if names:
            result[path.relative_to(COMMON).as_posix()] = sorted(set(names))
    return result


def bound_headers():
    if not COMPONENTS_I.exists():
        return set()
    text = COMPONENTS_I.read_text()
    return set(re.findall(r'%include\s+"common/([^"]+)"', text)) | set(
        re.findall(r'^//\s*bound-header:\s*(\S+)', text, re.M))


def binding_adapters():
    """Explicit model-component facades; unused transitive helper types need no wrapper.

    A declaration names a header actually parsed by SWIG. It does not excuse
    hidden training steps or missing language implementations/validation.
    Keep the unwrapped-header inventory even when a facade covers an example.
    """
    if not COMPONENTS_I.exists():
        return {}
    text = COMPONENTS_I.read_text()
    included = set(re.findall(r'%include\s+"common/([^"]+)"', text))
    result = dict(re.findall(r'^//\s*binding-adapter:\s*(\w+)\s+(\S+)\s*$', text, re.M))
    for name, header in result.items():
        if header not in included:
            raise ValueError(f"{name}: component adapter {header} is not parsed by SWIG")
    return result


def reachable(functions, start_body):
    """Local functions called (transitively) from ``start_body``."""
    by_name = collections.defaultdict(list)
    for item in functions:
        by_name[item[0]].append(item)
    seen, stack, order = set(), [start_body], []
    while stack:
        body = stack.pop()
        masked = mask(body)
        # plain calls and explicit template calls: Name(...) or Name<Args>(...)
        for name in set(re.findall(r"\b([A-Za-z_]\w*)\s*(?:<[^<>;{}]*>\s*)?\(", masked)):
            if name in by_name and name not in seen and name != "main":
                seen.add(name)
                for item in by_name[name]:
                    order.append(item)
                    stack.append(item[2])
    return order


def audit():
    header_cache = {}
    harnesses = header_harnesses()
    harness_names = {name for names in harnesses.values() for name in names}
    bound = bound_headers()
    adapters = binding_adapters()
    ledger = json.loads(LEDGER.read_text())["examples"] if LEDGER.exists() else {}
    examples = {}
    bodies_by_file = {}
    for source, relative in sources():
        text = source.read_text()
        functions = list(function_bodies(text))
        bodies_by_file[source.stem] = functions
    # Harness copies: identical normalized bodies in two or more example files.
    body_owners = collections.defaultdict(set)
    for stem, functions in bodies_by_file.items():
        for name, _params, body, _s, _e in functions:
            if len(body) > 400:
                body_owners[(name, normalized(body))].add(stem)
    pasted = {key for key, owners in body_owners.items() if len(owners) > 1}

    for source, relative in sources():
        text = source.read_text()
        functions = bodies_by_file[source.stem]
        main = next((f for f in functions if f[0] == "main"), None)
        main_body = main[2] if main else ""
        local = reachable([f for f in functions if f[0] != "main"], main_body)
        loop_owner, loop_functions = "none", []
        if training_loop(main_body):
            loop_owner = "main"
        for name, params, body, _s, _e in local:
            if not training_loop(body):
                continue
            loop_functions.append(name)
            kind = "local"
            if (name, normalized(body)) in pasted or CALLBACK_PARAM.search(params):
                kind = "pasted_harness"
            # A visible loop in main does not excuse another reachable loop
            # hidden behind a callback/spec harness (for example TF replay).
            if loop_owner == "none" or kind == "pasted_harness":
                loop_owner = kind
        reachable_text = main_body + "".join(f[2] for f in local)
        masked_reachable = mask(reachable_text)
        local_names = {f[0] for f in functions}
        header_calls = sorted({m for m in re.findall(r"(?<![.>\w])([A-Za-z_]\w*)\s*(?:<[^<>;{}]*>\s*)?\(", masked_reachable)
                               if m in harness_names and m not in local_names})
        if loop_owner == "none" and header_calls:
            loop_owner = "header"
        own_step = bool(GRAD.search(masked_reachable) or re.search(r"\.minimize\(|optimizer\.", masked_reachable))
        header_step = bool(HEADER_STEP_METHOD.search(masked_reachable))
        if own_step and header_step:
            step_owner = "mixed"
        elif own_step:
            step_owner = "example"
        elif header_step:
            step_owner = "header_method"
        else:
            step_owner = "none"
        headers = transitive_headers(text, header_cache)
        unbound = [h for h in headers if h not in bound]
        record = ledger.get(source.stem, {})
        languages = {
            language: (record.get(language) or {}).get("implementation", "missing")
            for language in ("python", "r")
        }
        validated = {
            language: bool((record.get(language) or {}).get("validation"))
            for language in ("python", "r")
        }
        compliant = loop_owner in ("main", "local", "none")
        opaque_step = step_owner in ("header_method", "mixed")
        examples[source.stem] = dict(
            cpp=source.relative_to(ROOT).as_posix(),
            section=relative.parts[0],
            loop_owner=loop_owner,
            loop_functions=loop_functions,
            header_harness_calls=header_calls,
            step_owner=step_owner,
            common_headers=headers,
            unbound_headers=unbound,
            binding_adapter=adapters.get(source.stem),
            python=languages["python"],
            r=languages["r"],
            cpp_compliant=compliant,
            opaque_step_methods=opaque_step,
            bindings_complete=not unbound or source.stem in adapters,
            python_validated=validated["python"],
            r_validated=validated["r"],
            cpp_validated=bool(record.get("cpp_reference_validation")),
        )
    return dict(schema=1, total=len(examples), header_harnesses=harnesses,
                bound_headers=sorted(bound), examples=examples)


def summary(report):
    lines = []
    examples = report["examples"]
    counts = collections.Counter(e["loop_owner"] for e in examples.values())
    steps = collections.Counter(e["step_owner"] for e in examples.values())
    lines.append(f"Canonical NN examples: {report['total']}")
    lines.append("loop owner: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    lines.append("step owner: " + ", ".join(f"{k}={v}" for k, v in sorted(steps.items())))
    compliant = sum(e["cpp_compliant"] for e in examples.values())
    complete = sum(e["bindings_complete"] for e in examples.values())
    opaque = sum(e["opaque_step_methods"] for e in examples.values())
    lines.append(f"C++ loop ownership compliant: {compliant}/{report['total']}; per-step logic hidden in a header "
                 f"method: {opaque}; component bindings complete: {complete}/{report['total']}")
    lines.append(f"shared headers owning training loops: {sum(len(v) for v in report['header_harnesses'].values())} "
                 f"in {len(report['header_harnesses'])} headers")
    for language in ("python", "r"):
        native = sum(e[language] == "native" for e in examples.values())
        validated = sum(e[language + "_validated"] for e in examples.values())
        lines.append(f"{language}: native={native}/{report['total']}, validated={validated}/{report['total']}")
    by_section = collections.defaultdict(collections.Counter)
    for e in examples.values():
        by_section[e["section"]]["total"] += 1
        by_section[e["section"]]["compliant"] += e["cpp_compliant"]
        by_section[e["section"]]["opaque"] += e["opaque_step_methods"]
        by_section[e["section"]]["bound"] += e["bindings_complete"]
        by_section[e["section"]]["python"] += e["python"] == "native"
        by_section[e["section"]]["r"] += e["r"] == "native"
    lines.append("section: total loop_compliant opaque_step bound python_native r_native")
    for section in sorted(by_section):
        c = by_section[section]
        lines.append(f"  {section}: {c['total']} {c['compliant']} {c['opaque']} {c['bound']} {c['python']} {c['r']}")
    return "\n".join(lines)


def markdown(report):
    examples = report["examples"]
    out = ["# NN example workflow audit", "",
           "Generated by `scripts/nn_example_audit.py` (`just nn-audit`). The target shape is "
           "described in `docs/nn-language-migration.md`: the C++ example owns its main loop and "
           "per-step logic, shared headers provide SWIG-bound components, and the R/Python "
           "examples implement the same loop over those bindings.", "", "```", summary(report), "```", ""]
    if report["header_harnesses"]:
        out += ["## Shared headers that still own a training loop", ""]
        for header, names in sorted(report["header_harnesses"].items()):
            out.append(f"- `{header}`: {', '.join(names)}")
        out.append("")
    out += ["## Examples needing C++ work", "",
            "| example | section | loop owner | harness | step owner | python | r |", "|---|---|---|---|---|---|---|"]
    for name, e in sorted(examples.items(), key=lambda kv: (kv[1]["section"], kv[0])):
        if e["cpp_compliant"]:
            continue
        harness = ", ".join(e["loop_functions"] + e["header_harness_calls"])
        out.append(f"| {name} | {e['section']} | {e['loop_owner']} | {harness} | {e['step_owner']} | {e['python']} | {e['r']} |")
    out += ["", "## Unbound shared headers by number of dependent examples", ""]
    usage = collections.Counter(h for e in examples.values() for h in e["unbound_headers"])
    for header, count in usage.most_common():
        out.append(f"- `{header}`: {count}")
    out += ["", "## Per-example status", "",
            "| example | loop owner | step owner | unbound headers | python | r |", "|---|---|---|---|---|---|"]
    for name, e in sorted(examples.items()):
        out.append(f"| {name} | {e['loop_owner']} | {e['step_owner']} | {len(e['unbound_headers'])} | {e['python']} | {e['r']} |")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markdown", metavar="PATH", help="also write a Markdown report")
    parser.add_argument("--check", action="store_true", help="fail when a shared header owns a training loop")
    parser.add_argument("--require-compliant", nargs="*", default=[], metavar="NAME",
                        help="with --check, these examples must be C++ compliant")
    parser.add_argument("--show", nargs="*", default=[], metavar="NAME", help="print these examples' records")
    parser.add_argument("--require-complete", action="store_true",
                        help="fail unless every example owns its workflow, has bound components, and has validated Python/R ports")
    args = parser.parse_args()
    report = audit()
    text = json.dumps(report, indent=1, sort_keys=False) + "\n"
    if not AUDIT.exists() or AUDIT.read_text() != text:
        AUDIT.write_text(text)
    if args.markdown:
        Path(args.markdown).write_text(markdown(report))
    print(summary(report))
    for name in args.show:
        print(json.dumps({name: report["examples"][name]}, indent=1))
    if args.check or args.require_complete:
        failures = []
        if report["header_harnesses"]:
            failures.append("shared headers own training loops: " + ", ".join(sorted(report["header_harnesses"])))
        for name in args.require_compliant:
            if not report["examples"][name]["cpp_compliant"]:
                failures.append(f"{name} is not compliant ({report['examples'][name]['loop_owner']})")
        if args.require_complete:
            requirements = {
                "C++ workflow": lambda e: e["cpp_compliant"] and not e["opaque_step_methods"],
                "component bindings": lambda e: e["bindings_complete"],
                "validated C++ reference": lambda e: e["cpp_validated"],
                "validated native Python": lambda e: e["python"] == "native" and e["python_validated"],
                "validated native R": lambda e: e["r"] == "native" and e["r_validated"],
            }
            for label, predicate in requirements.items():
                missing = [name for name, entry in report["examples"].items() if not predicate(entry)]
                if missing:
                    failures.append(f"{label}: {len(missing)} unfinished ({', '.join(missing[:8])}"
                                    + (", ..." if len(missing) > 8 else "") + ")")
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
