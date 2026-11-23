"""
compression.py — Phase 3: compressing the posting lists.

Step 1 — gaps. Posting lists are increasing, so store the DIFFERENCES:
         docs [3, 17, 42, 43] → gaps [3, 14, 25, 1]
         Frequent terms have many small gaps, and small numbers need few bits.
         Positions inside a document are gap-encoded the same way.

Step 2 — a variable-length code for each gap (all three from the course):

    Variable-byte (VB)   7 bits of the number per byte; the first bit of a
                         byte says "this is the last byte".  5 → 10000101
    γ (gamma)      length in unary, then the number in binary without
                         its leading 1.                     5 → 110 01
    Golomb (parameter b) quotient (n-1)//b in unary, remainder (n-1)%b in
                         "truncated binary". b ≈ 0.69 × the average gap is
                         optimal when gaps are random.     5, b=3 → 10 10

The sizes of the WHOLE index are computed with each code's length formula
(gamma_bits, golomb_bits, vb_bytes): fast, and no huge bit strings in memory.
The real encoders and decoders are run on a sample of lists to check that
decode(encode(x)) == x, that the formulas give the true lengths, and to time
decoding.

Run:
    python compression.py          # writes analysis/compression.md
"""

import random
import sys
import time
from pathlib import Path

from index import load

ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"

def gaps(numbers) -> list[int]:
    """[3, 17, 42] → [3, 14, 25]. The first number is kept as is."""
    return [b - a for a, b in zip([0] + list(numbers[:-1]), numbers)]

def ungaps(gap_list: list[int]) -> list[int]:
    out, total = [], 0
    for g in gap_list:
        total += g
        out.append(total)
    return out

# ── Variable-byte ─────────────────────────────────────────────────────────────

def vb_encode_number(n: int) -> bytes:
    groups = []
    while True:
        groups.insert(0, n % 128)          # 7 bits at a time, most significant first
        if n < 128:
            break
        n //= 128
    groups[-1] += 128                      # high bit set = last byte of this number
    return bytes(groups)

def vb_encode(numbers) -> bytes:
    return b"".join(vb_encode_number(n) for n in numbers)

def vb_decode(data: bytes) -> list[int]:
    numbers, n = [], 0
    for byte in data:
        if byte < 128:
            n = n * 128 + byte             # more bytes follow
        else:
            numbers.append(n * 128 + byte - 128)
            n = 0
    return numbers

def vb_bytes(n: int) -> int:
    """Length of n in variable-byte, without encoding it."""
    return max(1, (n.bit_length() + 6) // 7)

# ── Elias gamma (numbers ≥ 1) ─────────────────────────────────────────────────

def gamma_encode_number(n: int) -> str:
    offset = bin(n)[3:]                    # binary without "0b" and the leading 1
    return "1" * len(offset) + "0" + offset

def gamma_encode(numbers) -> str:
    return "".join(gamma_encode_number(n) for n in numbers)

def gamma_decode(bits: str) -> list[int]:
    numbers, i = [], 0
    while i < len(bits):
        length = 0
        while bits[i] == "1":              # unary part: how many offset bits follow
            length += 1; i += 1
        i += 1                             # skip the 0
        numbers.append(int("1" + bits[i:i + length], 2))
        i += length
    return numbers

def gamma_bits(n: int) -> int:
    return 2 * (n.bit_length() - 1) + 1

# ── Golomb (numbers ≥ 1, parameter b ≥ 1) ─────────────────────────────────────

def golomb_parameter(gap_list: list[int]) -> int:
    """b ≈ 0.69 × mean gap (optimal for geometrically distributed gaps)."""
    return max(1, round(0.69 * sum(gap_list) / len(gap_list)))

def _truncated_binary_split(b: int) -> tuple[int, int]:
    """For remainders 0..b-1: the first u remainders get k bits, the rest k+1 bits."""
    k = b.bit_length() - 1                 # floor(log2 b)
    u = 2 ** (k + 1) - b
    return k, u

def golomb_encode_number(n: int, b: int) -> str:
    q, r = divmod(n - 1, b)
    k, u = _truncated_binary_split(b)
    if r < u:
        tail = format(r, "b").zfill(k) if k else ""
    else:
        tail = format(r + u, "b").zfill(k + 1)
    return "1" * q + "0" + tail

def golomb_encode(numbers, b: int) -> str:
    return "".join(golomb_encode_number(n, b) for n in numbers)

def golomb_decode(bits: str, b: int) -> list[int]:
    k, u = _truncated_binary_split(b)
    numbers, i = [], 0
    while i < len(bits):
        q = 0
        while bits[i] == "1":              # unary quotient
            q += 1; i += 1
        i += 1
        x = int(bits[i:i + k], 2) if k else 0
        i += k
        if x >= u:                         # long remainder: one more bit
            x = 2 * x + int(bits[i]) - u
            i += 1
        numbers.append(q * b + x + 1)
    return numbers

def golomb_bits(n: int, b: int) -> int:
    q, r = divmod(n - 1, b)
    k, u = _truncated_binary_split(b)
    return q + 1 + (k if r < u else k + 1)

# ── Measuring the whole index ─────────────────────────────────────────────────

def all_gap_lists(index):
    """For every term: its doc-number gaps, and the position gaps of each posting."""
    for term, plist in index.postings.items():
        position_gaps = []
        for k in range(len(plist)):
            position_gaps += gaps(plist.positions_of(k))
        yield term, gaps(plist.docs), position_gaps

def measure(index) -> dict:
    sizes = {kind: {"numbers": 0, "raw": 0, "vb": 0, "gamma": 0, "golomb": 0}
             for kind in ("docs", "positions")}
    for _, doc_gaps, pos_gaps in all_gap_lists(index):
        for kind, gap_list in (("docs", doc_gaps), ("positions", pos_gaps)):
            b = golomb_parameter(gap_list)
            s = sizes[kind]
            s["numbers"] += len(gap_list)
            s["raw"]     += 32 * len(gap_list)                      # 4-byte integers
            s["vb"]      += 8 * sum(vb_bytes(g) for g in gap_list)
            s["gamma"]   += sum(gamma_bits(g) for g in gap_list)
            s["golomb"]  += sum(golomb_bits(g, b) for g in gap_list)
    return sizes

def check_and_time(index, sample_size: int = 300) -> dict:
    """Round-trip and length checks, and decoding speed, on a random sample of terms."""
    random.seed(0)
    terms = random.sample(sorted(index.postings), sample_size)
    numbers, seconds = 0, {"vb": 0.0, "gamma": 0.0, "golomb": 0.0}
    for term in terms:
        g = gaps(index.postings[term].docs)
        b = golomb_parameter(g)
        vb, gm, gb = vb_encode(g), gamma_encode(g), golomb_encode(g, b)
        assert len(vb) == sum(vb_bytes(x) for x in g)
        assert len(gm) == sum(gamma_bits(x) for x in g)
        assert len(gb) == sum(golomb_bits(x, b) for x in g)
        for name, decode in (("vb", lambda: vb_decode(vb)), ("gamma", lambda: gamma_decode(gm)),
                             ("golomb", lambda: golomb_decode(gb, b))):
            start = time.perf_counter()
            assert decode() == g, (name, term)
            seconds[name] += time.perf_counter() - start
        assert ungaps(g) == list(index.postings[term].docs)
        numbers += len(g)
    return {name: numbers / s / 1e6 for name, s in seconds.items()} | {"numbers": numbers}

# ── Report ────────────────────────────────────────────────────────────────────

def mb(bits: int) -> str:
    return f"{bits / 8 / 1e6:,.1f} MB"

def write_report(index, sizes: dict, speed: dict, path: Path) -> None:
    example = index.postings["sedit"]                          # "sedition", a mid-size list
    ex_docs = list(example.docs[:6]); ex_gaps = gaps(ex_docs); b = golomb_parameter(gaps(example.docs))
    lines = [
        "# Posting-list compression (Phase 3)",
        "",
        f"Index: `data/index_{index.method}.pkl`, {index.num_docs:,} documents, "
        f"{len(index.postings):,} terms. Generated by `compression.py`.",
        "",
        "| What | Numbers | Uncompressed (32-bit) | Variable-byte | Elias γ | Golomb |",
        "|---|---|---|---|---|---|",
    ]
    for kind, label in (("docs", "Doc-number gaps"), ("positions", "Position gaps")):
        s = sizes[kind]
        lines.append(f"| {label} | {s['numbers']:,} | {mb(s['raw'])} | "
                     + " | ".join(f"{mb(s[c])} ({s[c] / s['raw']:.0%})" for c in ("vb", "gamma", "golomb"))
                     + " |")
    total = {c: sizes["docs"][c] + sizes["positions"][c] for c in ("raw", "vb", "gamma", "golomb")}
    lines.append(f"| **Total** | | **{mb(total['raw'])}** | "
                 + " | ".join(f"**{mb(total[c])} ({total[c] / total['raw']:.0%})**"
                              for c in ("vb", "gamma", "golomb")) + " |")
    lines += [
        "",
        f"Bits per doc-number gap: VB {sizes['docs']['vb'] / sizes['docs']['numbers']:.1f}, "
        f"γ {sizes['docs']['gamma'] / sizes['docs']['numbers']:.1f}, "
        f"Golomb {sizes['docs']['golomb'] / sizes['docs']['numbers']:.1f}.",
        "",
        f"Decoding speed (pure Python, {speed['numbers']:,} doc gaps from 300 random terms): "
        f"VB {speed['vb']:.1f}, γ {speed['gamma']:.1f}, Golomb {speed['golomb']:.1f} million numbers/s. "
        "All round trips decode(encode(x)) == x passed, and the length formulas "
        "matched the real encodings.",
        "",
        "## Example: the first postings of `sedit` (\"sedition\")",
        "",
        f"- doc numbers: {ex_docs}",
        f"- gaps: {ex_gaps}",
        f"- variable-byte: {' '.join(format(x, '08b') for x in vb_encode(ex_gaps))}",
        f"- γ: {' '.join(gamma_encode_number(g) for g in ex_gaps)}",
        f"- Golomb (b = {b}): {' '.join(golomb_encode_number(g, b) for g in ex_gaps)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    idx = load("porter")
    print("[compression] checking encoders on a sample ...")
    speed = check_and_time(idx)
    print("[compression] measuring the whole index ...")
    sizes = measure(idx)
    ANALYSIS_DIR.mkdir(exist_ok=True)
    write_report(idx, sizes, speed, ANALYSIS_DIR / "compression.md")
    print((ANALYSIS_DIR / "compression.md").read_text(encoding="utf-8"))
