# Native Python NN example. CSV parsing and training-only numeric/categorical feature columns.
"""CMD: just python-example csv_features
CSV parsing and training-only numeric/categorical feature columns.

Port of tflow's csv_features. Read a table containing quoted commas, escaped
quotes, embedded newlines, CRLF, missing values and held-out categories.
Fit numeric means/scales and sorted categorical vocabularies on four
training rows, then apply that fixed transformation to two validation rows.
Numeric missingness and categorical missing/unknown values have separate
indicator columns. Column lookup uses header names, so a reordered input
table produces the same model features.

Check exact numeric moments and one-hot positions, verify validation values
cannot change fitted state, and feed the dense float32 matrix to an nn
linear prediction. This example demonstrates preprocessing, not model
generalization. The loader also accepts a caller-selected delimiter.

Differences from TF: the parser and feature columns (examples/nn/common/csv.hpp,
used only by this example) are written out below and return nn tensors; the
prediction is an eager nn.matmul instead of a session graph. The fixture is
written to output/nn/python/<mode>/fixtures/tabular/features.csv.

CSV rules: RFC 4180 quoting (doubled quotes, embedded delimiters and
newlines), CRLF or LF records, optional UTF-8 BOM, preserved whitespace, a
unique nonempty header and rectangular records. Feature statistics and
vocabularies are fitted ONLY on training rows. Numeric columns emit a
standardized value plus a missing indicator; empty numeric fields (after ASCII
trimming) are imputed to the training mean; population variance of observed
training values; constant/all-missing columns use scale 1 (all-missing mean 0).
Categorical columns keep exact strings with separate missing and unknown
buckets followed by sorted known values.
"""
import bisect
import copy
import math
import re
import sys
from dataclasses import dataclass, field

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

checks = helper("checks")
binary = helper("binary")

FLOAT_MAX = 3.4028234663852886e38
NUMBER = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


# ---- CSV table (common/csv.hpp) -------------------------------------------------------


@dataclass
class CsvTable:
    header: list = field(default_factory=list)
    rows: list = field(default_factory=list)

    def column(self, name):
        for i, value in enumerate(self.header):
            if value == name:
                return i
        raise ValueError("missing CSV column: " + name)

    def at(self, row, column):
        return self.rows[row][column]


def parse_csv(text, delimiter=","):
    if delimiter in ('"', "\n", "\r", "\0"):
        raise ValueError("invalid CSV delimiter")
    if text.startswith("﻿"):
        text = text[1:]
    records, row, chars = [], [], []
    quoted = closed = started = False

    def end_field():
        nonlocal closed, started
        row.append("".join(chars))
        chars.clear()
        closed = started = False

    i = 0
    while i < len(text):
        c = text[i]
        if c == "\0":
            raise ValueError("NUL in CSV")
        if quoted:
            if c == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    chars.append('"')
                    i += 1
                else:
                    quoted, closed = False, True
            else:
                chars.append(c)
            i += 1
            continue
        if c == delimiter:
            end_field()
        elif c in "\n\r":
            if c == "\r":
                if i + 1 == len(text) or text[i + 1] != "\n":
                    raise ValueError("bare CR outside quoted field")
                i += 1
            end_field()
            records.append(row)
            row = []
        elif c == '"' and not started and not closed:
            quoted = started = True
        else:
            if closed or c == '"':
                raise ValueError("invalid CSV quote placement")
            chars.append(c)
            started = True
        i += 1
    if quoted:
        raise ValueError("unterminated CSV quote")
    if started or closed or row:
        end_field()
        records.append(row)
    if not records:
        raise ValueError("CSV requires a header")
    result = CsvTable(header=records[0])
    if any(not name for name in result.header) or len(set(result.header)) != len(result.header):
        raise ValueError("empty or duplicate CSV header")
    for i, record in enumerate(records[1:], start=1):
        if len(record) != len(result.header):
            raise ValueError(f"CSV record width mismatch at record {i + 1}")
        result.rows.append(record)
    return result


def load_csv(path, delimiter=","):
    data = binary.read_binary(path, 128 * 1024 * 1024)
    return parse_csv(data.decode("utf-8", "surrogateescape"), delimiter)


def numeric(text):
    """None for an empty (ASCII-blank) cell, else a strict finite decimal value."""
    text = text.strip(" \t\r\n")
    if not text:
        return None
    if text[0] == "+":
        text = text[1:]
        if not text or text[0] in "+-":
            raise ValueError("invalid numeric sign")
    if not NUMBER.fullmatch(text):
        raise ValueError("invalid finite numeric field: " + text)
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("invalid finite numeric field: " + text)
    return value


# ---- fitted feature columns --------------------------------------------------------------

NUMERIC, CATEGORICAL = "numeric", "categorical"


@dataclass
class FittedFeature:
    name: str
    kind: str
    mean: float = 0.0
    scale: float = 1.0
    vocabulary: list = field(default_factory=list)


class FeatureColumns:
    def __init__(self):
        self.features, self.names = [], []

    @staticmethod
    def fit(table, training_rows, specs):
        if not training_rows or not specs:
            raise ValueError("feature fitting needs training rows and columns")
        result = FeatureColumns()
        unique = set()
        for name, kind in specs:
            if name in unique:
                raise ValueError("duplicate feature column")
            unique.add(name)
            column = table.column(name)
            fit = FittedFeature(name, kind)
            if kind == NUMERIC:
                count, m2 = 0, 0.0
                for row in training_rows:
                    value = numeric(table.at(row, column))
                    if value is not None:
                        count += 1
                        delta = value - fit.mean
                        fit.mean += delta / count
                        m2 += delta * (value - fit.mean)
                if count and m2 > 0:
                    fit.scale = math.sqrt(m2 / count)
                if not math.isfinite(fit.mean) or not math.isfinite(fit.scale):
                    raise ValueError("numeric statistics overflow")
                result.names += [name + ":value", name + ":missing"]
            elif kind == CATEGORICAL:
                values = sorted({table.at(row, column) for row in training_rows} - {""})
                fit.vocabulary = values
                result.names += [name + ":missing", name + ":unknown"]
                result.names += [name + ":category=" + value for value in values]
            else:
                raise ValueError("invalid feature kind")
            result.features.append(fit)
        return result

    def transform(self, table, rows):
        if not self.features:
            raise RuntimeError("feature columns are not fitted")
        width = len(self.names)
        result = [0.0] * (len(rows) * width)
        offset = 0
        for feature in self.features:
            column = table.column(feature.name)
            for r, row in enumerate(rows):
                value = table.at(row, column)
                if feature.kind == NUMERIC:
                    number = numeric(value)
                    normalized = (number - feature.mean) / feature.scale if number is not None else 0.0
                    if not math.isfinite(normalized) or abs(normalized) > FLOAT_MAX:
                        raise ValueError("normalized feature overflows float32")
                    result[r * width + offset] = normalized
                    result[r * width + offset + 1] = 0.0 if number is not None else 1.0
                else:
                    index = 0 if value == "" else 1
                    vocabulary = feature.vocabulary
                    at = bisect.bisect_left(vocabulary, value)
                    if value and at < len(vocabulary) and vocabulary[at] == value:
                        index = 2 + at
                    result[r * width + offset + index] = 1.0
            offset += 2 if feature.kind == NUMERIC else len(feature.vocabulary) + 2
        return nn.reshape(nn.from_vector(result, [len(rows) * width]), [len(rows), width])


# ---- checks -----------------------------------------------------------------------------


def rejects(function, value):
    try:
        function(value)
    except ValueError:
        return True
    return False


def parser_checks():
    quoted = parse_csv('a,b\r\n"one, two","say ""hi"""\r\n"line1\nline2",""\r\n')
    checks.require(
        len(quoted.rows) == 2
        and quoted.at(0, 0) == "one, two"
        and quoted.at(0, 1) == 'say "hi"'
        and quoted.at(1, 0) == "line1\nline2"
        and quoted.at(1, 1) == "",
        "quoted CSV fields",
    )
    checks.require(parse_csv("﻿a;b\n1;2", ";").at(0, 1) == "2", "BOM and alternate delimiter")
    checks.require(parse_csv("a,b\n,\n").rows == [["", ""]], "empty fields and terminal newline")
    checks.require(parse_csv("a\n  x  ").at(0, 0) == "  x  ", "preserve whitespace")
    rejected = 0
    for malformed in ("", "a,a\n1,2", ",b\n1,2", "a,b\n1", "a\n1,2", 'a\n"open', 'a\nraw"quote', 'a\n"closed"tail', "a\r1"):
        checks.require(rejects(parse_csv, malformed), "malformed CSV must fail")
        rejected += 1
    for invalid in ("1x", "NaN", "inf", "1e999", "+-1"):
        checks.require(rejects(numeric, invalid), "invalid/nonfinite numeric must fail")
        rejected += 1
    checks.near(numeric(" +1.25e2 "), 125, 0, "numeric whitespace/sign/exponent")
    checks.require(numeric(" \t") is None, "empty numeric cell is missing")
    print("  quoting, multiline fields, CRLF/LF, BOM, delimiters and empty cells passed")
    print(f"  {rejected} malformed CSV/numeric cases rejected")


def features_check():
    text = (
        "age,income,city,target,note\r\n"
        "20,100,Denver,0,first\r\n"
        '30,200,Boulder,1,"said ""hello"""\r\n'
        ',300,Denver,0,"two\nlines"\r\n'
        "50,400,Austin,1,last training row\r\n"
        '1000,500,"Rome, IT",1,validation only\r\n'
        ",,,0,validation missing\r\n"
    )
    path = checks.output_dir("fixtures/tabular") / "features.csv"
    binary.write_binary(path, text.encode("utf-8"))
    table = load_csv(path)
    specs = [("age", NUMERIC), ("income", NUMERIC), ("city", CATEGORICAL)]
    features = FeatureColumns.fit(table, [0, 1, 2, 3], specs)
    checks.near(features.features[0].mean, 100.0 / 3, 1e-12, "training age mean excludes validation")
    checks.near(features.features[0].scale, math.sqrt(1400.0 / 9), 1e-12, "observed population age deviation")
    checks.near(features.features[1].mean, 250, 0, "training income mean")
    checks.near(features.features[1].scale, math.sqrt(12500.0), 1e-12, "population income deviation")
    checks.require(features.features[2].vocabulary == ["Austin", "Boulder", "Denver"], "training-only sorted vocabulary")
    training, validation = features.transform(table, [0, 1, 2, 3]), features.transform(table, [4, 5])
    checks.require(
        tuple(training.shape()) == (4, 9) and tuple(validation.shape()) == (2, 9), "feature dimensions"
    )
    tr, va = training.values(), validation.values()
    checks.near(tr[2 * 9 + 0], 0, 0, "impute standardized mean")
    checks.near(tr[2 * 9 + 1], 1, 0, "numeric missing indicator")
    checks.near(va[0], (1000 - 100.0 / 3) / math.sqrt(1400.0 / 9), 1e-5, "held-out numeric transform")
    checks.near(va[5], 1, 0, "unknown category bucket")
    checks.near(va[9 + 4], 1, 0, "missing category bucket")
    reordered = copy.deepcopy(table)
    reordered.header[0], reordered.header[2] = reordered.header[2], reordered.header[0]
    for row in reordered.rows:
        row[0], row[2] = row[2], row[0]
    transformed = features.transform(reordered, [4, 5])
    checks.require(transformed.values() == va, "column reorder must preserve features")
    sparse = parse_csv("constant,missing\n4,\n4,\n")
    fixed = FeatureColumns.fit(sparse, [0, 1], [("constant", NUMERIC), ("missing", NUMERIC)])
    zeros = fixed.transform(sparse, [0, 1]).values()
    for row in range(2):
        checks.near(zeros[row * 4 + 0], 0, 0, "constant column")
        checks.near(zeros[row * 4 + 2], 0, 0, "all-missing imputation")
        checks.near(zeros[row * 4 + 3], 1, 0, "all-missing indicator")
    weights = nn.full([9, 1], 0.1)
    out = nn.matmul(validation, weights).values()
    for row in range(2):
        expected = 0.0
        for col in range(9):
            expected += va[row * 9 + col] * 0.1
        checks.near(out[row], expected, 1e-6, "nn consumes feature matrix")
    print(f"  training rows=4, validation rows=2; features={len(features.names)}")
    for name in features.names:
        print("    " + name)
    print(
        f"  training age mean={'%g' % features.features[0].mean}, scale={'%g' % features.features[0].scale}"
        f"; held-out age 1000 -> {'%g' % va[0]}"
    )
    print("  unknown/missing buckets, fixed statistics and reordered columns passed")
    print(f"  nn validation predictions=({'%g' % out[0]}, {'%g' % out[1]})")


def main(arguments=None):
    try:
        nn.set_num_threads(2)
        print("== CSV parser checks")
        parser_checks()
        print("\n== Fitted feature columns")
        features_check()
        print("All CSV and feature-column checks passed.")
    except Exception as error:  # noqa: BLE001 - mirrors the C++ catch-all
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
