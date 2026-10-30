# Native Python NN example. Whitespace, Unicode characters, learned BPE and greedy WordPiece.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.4_data_loading_and_preprocessing/text_tokenization.py
"""Whitespace, Unicode characters, learned BPE and greedy WordPiece.

Port of tflow's text_tokenization. Learn ten BPE merges from a weighted toy
vocabulary, inspect segmentation of unseen word combinations, and reload a
saved merge model. Test Unicode scalar boundaries and invalid UTF-8,
deterministic tie-breaking, WordPiece continuation markers, longest matches
and whole-word unknown fallback.

Convert WordPiece tokens to stable IDs, truncate/right-pad a batch and feed
it to an nn embedding lookup with masked mean pooling. An empty sentence
must produce a zero vector even when the PAD embedding is nonzero.
Preprocessing preserves case/punctuation and uses ASCII whitespace, without
Unicode normalization. This is not a full BERT basic tokenizer.
References: https://arxiv.org/abs/1508.07909 and the original BERT tokenizer
at https://github.com/google-research/bert/blob/master/tokenization.py.

Differences from TF: the tokenizers (examples/nn/common/tokenization.hpp, used
only by this example) are written out below and pad_tokens returns nn
tensors; the lookup and pooling are eager nn ops (gather, mul, sum, maximum,
div). The BPE model is written to output/nn/python/<mode>/fixtures/text/bpe.model.

Tokenizer rules: whitespace splitting uses ASCII space/tab/newline characters
and preserves case and punctuation. Character tokens are Unicode scalar values;
malformed, overlong and surrogate UTF-8 is rejected. BPE learns weighted
adjacent-pair frequencies within words including an explicit </w> marker, ties
break lexicographically, merges apply in ranked order and unseen characters
become [UNK]. WordPiece is longest-match-first with ## continuations and
replaces a whole word by [UNK] if any suffix fails. IDs: PAD=0, UNK=1.
"""
import re
import sys

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

checks = helper("checks")
binary = helper("binary")

ASCII_SPACE = " \t\r\n\f\v"


# ---- tokenizers (common/tokenization.hpp) -------------------------------------------------


def characters(text):
    """Unicode scalar tokens of UTF-8 text (str, or bytes to validate raw input)."""
    data = text.encode("utf-8", "surrogatepass") if isinstance(text, str) else bytes(text)
    tokens, i = [], 0
    while i < len(data):
        lead = data[i]
        if lead < 0x80:
            count, value, minimum = 1, lead, 0
        elif 0xC2 <= lead <= 0xDF:
            count, value, minimum = 2, lead & 31, 0x80
        elif 0xE0 <= lead <= 0xEF:
            count, value, minimum = 3, lead & 15, 0x800
        elif 0xF0 <= lead <= 0xF4:
            count, value, minimum = 4, lead & 7, 0x10000
        else:
            raise ValueError("invalid UTF-8 leading byte")
        if len(data) - i < count:
            raise ValueError("truncated UTF-8")
        for j in range(1, count):
            byte = data[i + j]
            if (byte & 0xC0) != 0x80:
                raise ValueError("invalid UTF-8 continuation")
            value = (value << 6) | (byte & 63)
        if value < minimum or value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
            raise ValueError("invalid Unicode scalar")
        tokens.append(data[i : i + count].decode("utf-8"))
        i += count
    return tokens


def whitespace(text):
    characters(text)  # validate UTF-8 even though splitting operates on ASCII bytes
    return re.findall("[^" + re.escape(ASCII_SPACE) + "]+", text)


def quoted(value):
    """std::quoted output: surrounding quotes, backslash-escaped quotes/backslashes."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Stream:
    """The istream reads Bpe::Load uses: >> word, >> std::quoted(word), >> std::ws."""

    def __init__(self, text):
        self.text, self.at = text, 0

    def skip_space(self):
        while self.at < len(self.text) and self.text[self.at] in ASCII_SPACE:
            self.at += 1

    def word(self):
        self.skip_space()
        start = self.at
        while self.at < len(self.text) and self.text[self.at] not in ASCII_SPACE:
            self.at += 1
        if start == self.at:
            raise ValueError("stream read failed")
        return self.text[start : self.at]

    def quoted(self):
        self.skip_space()
        if self.at >= len(self.text):
            raise ValueError("stream read failed")
        if self.text[self.at] != '"':
            return self.word()
        self.at += 1
        result = []
        while self.at < len(self.text):
            c = self.text[self.at]
            self.at += 1
            if c == "\\" and self.at < len(self.text):
                result.append(self.text[self.at])
                self.at += 1
            elif c == '"':
                return "".join(result)
            else:
                result.append(c)
        raise ValueError("stream read failed")

    def eof(self):
        self.skip_space()
        return self.at >= len(self.text)


class Bpe:
    def __init__(self):
        self.alphabet, self.merges = set(), []

    @staticmethod
    def merge(symbols, pair):
        result, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                result.append(pair[0] + pair[1])
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        return result

    @staticmethod
    def train(frequencies, max_merges, min_count=2):
        if not frequencies or not min_count:
            raise ValueError("BPE needs a corpus and positive minimum count")
        model = Bpe()
        words = []
        for key in sorted(frequencies):  # std::map key order
            count = frequencies[key]
            if not count or whitespace(key) != [key]:
                raise ValueError("BPE corpus keys must be nonempty words")
            symbols = characters(key)
            model.alphabet.update(symbols)
            words.append((symbols + ["</w>"], count))
        for _ in range(max_merges):
            counts = {}
            for symbols, count in words:
                for i in range(1, len(symbols)):
                    pair = (symbols[i - 1], symbols[i])
                    counts[pair] = counts.get(pair, 0) + count
            if not counts:
                break
            best = None
            for pair in sorted(counts):  # first maximum in lexicographic pair order
                if best is None or counts[pair] > counts[best]:
                    best = pair
            if counts[best] < min_count:
                break
            model.merges.append(best)
            words = [(Bpe.merge(symbols, best), count) for symbols, count in words]
        return model

    def encode_word(self, word):
        symbols = characters(word)
        if not symbols:
            return []
        symbols = [symbol if symbol in self.alphabet else "[UNK]" for symbol in symbols] + ["</w>"]
        for pair in self.merges:
            symbols = Bpe.merge(symbols, pair)
        return symbols

    def save(self):
        lines = [f"TFLOW_BPE_V1 {len(self.alphabet)} {len(self.merges)}"]
        lines += [quoted(symbol) for symbol in sorted(self.alphabet)]
        lines += [quoted(first) + " " + quoted(second) for first, second in self.merges]
        return "\n".join(lines) + "\n"

    @staticmethod
    def load(serialized):
        stream = Stream(serialized)
        try:
            magic, chars, merges = stream.word(), int(stream.word()), int(stream.word())
        except ValueError:
            raise ValueError("invalid BPE model header") from None
        if magic != "TFLOW_BPE_V1" or not chars or chars > 1000000 or merges > 1000000:
            raise ValueError("invalid BPE model header")
        model = Bpe()
        for _ in range(chars):
            try:
                symbol = stream.quoted()
            except ValueError:
                raise ValueError("invalid BPE alphabet") from None
            if len(characters(symbol)) != 1 or symbol in model.alphabet:
                raise ValueError("invalid BPE alphabet")
            model.alphabet.add(symbol)
        known = set(model.alphabet) | {"</w>"}
        for _ in range(merges):
            try:
                pair = (stream.quoted(), stream.quoted())
            except ValueError:
                raise ValueError("invalid BPE merge dependency") from None
            if pair[0] not in known or pair[1] not in known:
                raise ValueError("invalid BPE merge dependency")
            known.add(pair[0] + pair[1])
            model.merges.append(pair)
        if not stream.eof():
            raise ValueError("trailing BPE model data")
        return model


class WordPiece:
    def __init__(self, vocabulary, max_chars=100):
        if len(vocabulary) < 2 or vocabulary[0] != "[PAD]" or vocabulary[1] != "[UNK]" or not max_chars:
            raise ValueError("WordPiece needs PAD=0, UNK=1 and positive max length")
        self.vocabulary, self.max_chars, self.ids = list(vocabulary), max_chars, {}
        for i, token in enumerate(self.vocabulary):
            characters(token)
            if not token or token in self.ids:
                raise ValueError("empty/duplicate WordPiece token")
            self.ids[token] = i

    def encode_word(self, word):
        chars = characters(word)
        if len(chars) > self.max_chars:
            return ["[UNK]"]
        tokens, start = [], 0
        while start < len(chars):
            found, end = "", len(chars)
            while end > start:
                candidate = ("##" if start else "") + "".join(chars[start:end])
                if candidate in self.ids:
                    found = candidate
                    break
                end -= 1
            if not found:
                return ["[UNK]"]
            tokens.append(found)
            start = end
        return tokens

    def encode(self, sentence):
        return [self.ids[token] for word in whitespace(sentence) for token in self.encode_word(word)]


def pad_tokens(sequences, max_length):
    """Right-truncated, right-padded IDs int32 [N, max_length] (PAD = 0) and mask float32."""
    if not sequences or not max_length:
        raise ValueError("token batching needs sequences and positive max length")
    ids = [0] * (len(sequences) * max_length)
    mask = [0.0] * (len(sequences) * max_length)
    for i, sequence in enumerate(sequences):
        for j in range(min(max_length, len(sequence))):
            if sequence[j] < 0:
                raise ValueError("negative token ID")
            ids[i * max_length + j] = sequence[j]
            mask[i * max_length + j] = 1.0
    shape = [len(sequences), max_length]
    return (
        nn.cast(nn.from_vector_int64(ids, shape), nn.DType.Int32),
        nn.from_vector(mask, shape),
    )


# ---- checks -----------------------------------------------------------------------------


def show(tokens):
    print("".join(f"[{value}] " for value in tokens))


def rejects(function, value):
    try:
        function(value)
    except ValueError:
        return True
    return False


def basic():
    words = whitespace("  café\tnewer\nlowest  ")
    checks.require(words == ["café", "newer", "lowest"], "whitespace boundaries")
    chars = characters("café🙂")
    checks.require(chars == ["c", "a", "f", "é", "🙂"], "Unicode scalar tokenization")
    checks.require(len(characters("é")) == 2, "combining marks remain distinct scalars")
    failures = 0
    for invalid in (b"\xc0\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x82", b"\x80", b"\xe2x\x80"):
        checks.require(rejects(characters, invalid), "malformed UTF-8 must fail")
        failures += 1
    print("  whitespace: ", end="")
    show(words)
    print("  characters: ", end="")
    show(chars)
    print(f"  {failures} invalid UTF-8 cases rejected")


def bpe_checks():
    bpe = Bpe.train({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 10)
    checks.require(
        len(bpe.merges) == 10 and bpe.merges[0] == ("e", "s"), "weighted pair counts and deterministic tie break"
    )
    print("  learned merges:" + "".join(f" ({first},{second})" for first, second in bpe.merges))
    path = checks.output_dir("fixtures/text") / "bpe.model"
    serialized = bpe.save()
    binary.write_binary(path, serialized.encode("utf-8"))
    restored = Bpe.load(binary.read_binary(path).decode("utf-8"))
    checks.require(restored.save() == serialized, "BPE persistence preserves merge ranks/alphabet")
    for word in ("low", "lower", "lowest", "newer", "widest"):
        pieces = bpe.encode_word(word)
        checks.require(restored.encode_word(word) == pieces, "reloaded BPE segmentation")
        checks.require(
            "".join(pieces) == word + "</w>", "BPE preserves known character sequence and word boundary"
        )
        print(f"  {word}: ", end="")
        show(pieces)
    checks.require(bpe.encode_word("λ") == ["[UNK]", "</w>"], "BPE unseen character fallback")
    checks.require(rejects(Bpe.load, serialized + " garbage"), "invalid saved model rejected")


def wordpiece_and_batch():
    wp = WordPiece(["[PAD]", "[UNK]", "low", "##est", "lower", "new", "##er", "wide", "##st", "caf", "##é", "##s"])
    checks.require(wp.encode_word("lowest") == ["low", "##est"], "WordPiece continuations")
    checks.require(wp.encode_word("lower") == ["lower"], "WordPiece longest match")
    checks.require(wp.encode_word("lowerx") == ["[UNK]"], "whole word fallback after failed suffix")
    checks.require(wp.encode_word("café") == ["caf", "##é"], "WordPiece Unicode boundaries")
    checks.require(wp.encode_word("a" * 101) == ["[UNK]"], "maximum word length")
    for word in ("lowest", "lower", "newer", "café", "lowerx"):
        print(f"  {word}: ", end="")
        show(wp.encode_word(word))
    batch_ids, batch_mask = pad_tokens([wp.encode("lowest newer"), wp.encode("unknown"), wp.encode("")], 3)
    checks.require(tuple(batch_ids.shape()) == (3, 3), "padded token shape")
    ids = batch_ids.values()
    checks.require(ids[0] == 2 and ids[1] == 3 and ids[2] == 5, "right truncation and stable IDs")
    checks.near(batch_mask.values()[3 + 1], 0, 0, "padding mask")
    vocab = len(wp.vocabulary)
    table = []
    for i in range(vocab):
        table += [float(i), float(100 + i)]
    embeddings = nn.reshape(nn.from_vector(table, [vocab * 2]), [vocab, 2])
    lookup = nn.gather(embeddings, batch_ids, 0)
    mask = nn.expand_dims(batch_mask, 2)
    total = nn.sum(nn.mul(lookup, mask), [1])
    denominator = nn.maximum(nn.sum(mask, [1]), nn.scalar(1))
    pooled = nn.div(total, denominator).values()
    checks.near(pooled[0], 10.0 / 3, 1e-6, "truncated mean embedding")
    checks.near(pooled[1], 100 + 10.0 / 3, 1e-5, "embedding feature two")
    checks.near(pooled[2], 1, 0, "unknown embedding")
    for c in range(2):
        checks.near(pooled[4 + c], 0, 0, "empty sentence ignores PAD embedding")
    print("  token batch [3,3], retained lengths=(3,1,0)")
    print(f"  masked mean embeddings=({'%g' % pooled[0]}, {'%g' % pooled[1]}), (1,101), (0,0)")


def main(arguments=None):
    try:
        nn.set_num_threads(2)
        print("== Whitespace and characters")
        basic()
        print("\n== Learned BPE")
        bpe_checks()
        print("\n== WordPiece and nn token batching")
        wordpiece_and_batch()
        print("All tokenization checks passed.")
    except Exception as error:  # noqa: BLE001 - mirrors the C++ catch-all
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
