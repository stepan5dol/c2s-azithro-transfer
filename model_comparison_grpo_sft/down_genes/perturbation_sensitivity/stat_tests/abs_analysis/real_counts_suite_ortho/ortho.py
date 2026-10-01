#!/usr/bin/env python3
"""
ortho.py — the one place in this suite that maps rat gene symbols to human.

Everything reads ortholog_pairs.tsv, built by build_ortholog_pairs.py from HCOP
(HGNC Comparison of Orthology Predictions) plus the RGD/HGNC curated ortholog list.
It replaces the seventeen private copies of

    omap = {r: h for r, h in zip(csv.rat_symbol, csv.human_symbol) if isinstance(h, str)}
    human = omap.get(rat, rat.upper())

that used to live in each script, reading a 167-symbol reward-function list and
uppercasing the other 73-84%.

ACCEPTANCE RULE. A pair is kept if the nomenclature bodies curated it, or if at
least MIN_SUPPORT of HCOP's nine predictors back it:

    rgd_curated OR support_n >= 4

The threshold is not a guess. HCOP publishes no official cutoff, so
build_ortholog_pairs.py calibrates it against the curated list and prints the
table: agreement peaks at >=3 and >=4 (F1 0.912 / 0.913), collapses below 3
(precision 0.44 at >=1), and above 5 loses real orthologs without buying
precision. 4 is the stricter of the two optima, which is what a set comparison
wants -- a false pair drags a foreign gene into a panel. Curation passes
regardless of votes because it is a decision by RGD and HGNC rather than a
predictor's opinion, and it is what keeps Rps7->RPS7 (support 4, curated) ahead of
Rps7->COLEC11 (support 5, not curated).

This is also why there is no longer a symbol-identity override or an HGNC fallback
tier here. Both were patches for using a single tree-based predictor: Ensembl
Compara alone put ~90 orthologs on neighbouring loci (Ahrr->EXOC3, Tgm2->KIAA1755,
Chmp2a->UBE2M), and rather than adjudicate those by hand the consensus count does
it -- every one of them collapses to support 1 or disappears, while the
same-named gene carries 7-9 and RGD curation.

WHAT CALLERS GET. `.upper()` is not a fallback. A rat gene either has an accepted
counterpart or it does not, and the ones that do not are RETURNED, not silently
turned into a plausible-looking symbol:

    m, unmapped = ortho.rat_to_human_flat(rat_genes)
    print(ortho.coverage_line(rat_genes, m, unmapped))

ONE-TO-MANY. Use `rat_to_human` where the analysis can carry several human symbols
for one rat gene (set intersections, gene-set membership); it returns a list. Use
`rat_to_human_flat` only where a single symbol is structurally required -- a column
lookup in a human matrix -- and read `ambiguous` to see what was collapsed. The
flat form prefers curated, then higher support, then the identically spelled
partner, then alphabetical, so it never depends on row order in the file.

NOMENCLATURE DRIFT is normalised on both sides before joining: human through HGNC
(approved, then alias, then previous), rat through RGD (symbol, then old symbol).
GSE275938 still says H3F3A, H1F0, ADSS where HGNC now says H3-3A, H1-0, ADSS2, so
an exact string match against the data axis silently drops real genes. Symbols are
returned in the spelling the data uses, since that is what the matrices index by.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PAIRS_TSV = HERE / "ortholog_pairs.tsv"
PROVENANCE = HERE / "ortholog_pairs.provenance.json"
HGNC_CACHE = HERE / "hgnc_complete_set.tsv"
RGD_CACHE = HERE / "genes_rat_rgd.tsv"

MIN_SUPPORT = 4          # calibrated against RGD curation; see the module docstring

_cache: dict[tuple, pd.DataFrame] = {}
_canon_map: dict[str, str] | None = None
_canon_rat_map: dict[str, str] | None = None
_checked = False


def provenance() -> dict:
    """What built ortholog_pairs.tsv — HCOP and RGD snapshots, support histogram.

    A tsv has nowhere to carry this and both sources are live files, so without the
    sidecar a table found on disk in six months is unattributable.
    """
    if not PROVENANCE.exists():
        raise SystemExit(f"{PROVENANCE.name} is missing — run build_ortholog_pairs.py")
    return json.loads(PROVENANCE.read_text())


def _check_once() -> None:
    """Refuse a table whose bytes do not match what the sidecar recorded."""
    global _checked
    if _checked:
        return
    prov = provenance()
    actual = hashlib.md5(PAIRS_TSV.read_bytes()).hexdigest()
    if actual != prov["pairs_md5"]:
        raise SystemExit(
            f"{PAIRS_TSV.name} does not match its provenance record:\n"
            f"  on disk   {actual}\n"
            f"  recorded  {prov['pairs_md5']} (built {prov['built_at']})\n"
            f"Re-run build_ortholog_pairs.py, or restore the recorded table.")
    _checked = True


def _build_canon(current: list, alias_fields: list[tuple], sep: str) -> dict[str, str]:
    """symbol/alias/previous -> current symbol, dropping ambiguous aliases.

    AN ALIAS THAT NAMES TWO GENES NAMES NEITHER. Rat `Car1` is an OLD_SYMBOL of
    BOTH Cxadr (historically "CAR") and Ca1 (carbonic anhydrase 1); human `A1`
    is an alias of ATP6V0A1, RFC1, RFC2 and RFC4. 653 rat old symbols and 1599
    human aliases are ambiguous like this. Resolving them by `setdefault` -- the
    previous behaviour -- silently picked whichever row the file listed first,
    which is how `Car1` in the rat matrix became `Cxadr` and then collected
    CXADR's ortholog alongside its own CA1.

    So an alias is only usable when it points at exactly one current symbol.
    Ambiguous ones are dropped and the symbol is left as the data spells it,
    which costs a join and never invents one. A symbol that is itself current
    always wins over any alias claim on it.
    """
    m = {s.upper(): s for s in current if isinstance(s, str)}
    claims: dict[str, set] = {}
    for row_symbol, fields in alias_fields:
        if not isinstance(row_symbol, str):
            continue
        for field in fields:
            if not isinstance(field, str):
                continue
            for a in field.split(sep):
                a = a.strip().upper()
                if a and a not in m:
                    claims.setdefault(a, set()).add(row_symbol)
    for a, owners in claims.items():
        if len(owners) == 1:
            m[a] = next(iter(owners))
    return m


def _canon() -> dict[str, str]:
    """HGNC symbol/alias/previous -> approved symbol, approved filled first."""
    global _canon_map
    if _canon_map is None:
        h = pd.read_csv(HGNC_CACHE, sep="\t", low_memory=False,
                        usecols=["symbol", "alias_symbol", "prev_symbol"])
        _canon_map = _build_canon(
            list(h["symbol"]),
            list(zip(h["symbol"], zip(h["alias_symbol"], h["prev_symbol"]))),
            "|")
    return _canon_map


def _canon_rat() -> dict[str, str]:
    """RGD symbol/old symbol -> current rat symbol. The rat mirror of _canon()."""
    global _canon_rat_map
    if _canon_rat_map is None:
        d = pd.read_csv(RGD_CACHE, sep="\t", low_memory=False,
                        usecols=["SYMBOL", "OLD_SYMBOL"])
        _canon_rat_map = _build_canon(
            list(d["SYMBOL"]),
            [(s, (o,)) for s, o in zip(d["SYMBOL"], d["OLD_SYMBOL"])],
            ";")
    return _canon_rat_map


def _axis_lookup(axis: set[str], species: str = "human") -> dict[str, str]:
    """canonical symbol -> the spelling this dataset actually uses.

    A gene axis can carry TWO spellings of the same gene, because these matrices
    were built against annotations of different ages: GSE275938 has both
    BHLHE40 and DEC1 as separate columns, and 105 more such pairs; the rat axis
    has 11, including Cxadr/Car1 and Lgals9/Lgals5. Both spellings canonicalise
    to one symbol, so only one of them can be the value here.

    The choice must not depend on iteration order. It used to: this was
    `for g in axis: out.setdefault(...)` over a SET, so which column won was
    decided by string hashing, and Bhlhe40 came back as BHLHE40 or DEC1
    depending on PYTHONHASHSEED -- the same script, the same pinned table, two
    different answers. Sorting alone would fix the reproducibility and still
    pick DEC1 over BHLHE40, so the approved symbol is preferred first and the
    tie broken alphabetically.
    """
    canon = _canon() if species == "human" else _canon_rat()
    groups: dict[str, list[str]] = {}
    for g in sorted(map(str, axis)):
        groups.setdefault(canon.get(g.upper(), g), []).append(g)
    return {c: min(v, key=lambda s: (s != c, s)) for c, v in groups.items()}


def axis_collisions(axis: set[str], species: str = "human") -> dict[str, list[str]]:
    """Genes the axis spells more than one way; only one spelling is used.

    Worth printing when an analysis reports per-gene numbers off such an axis:
    the discarded column is a real column of counts that no longer participates.
    """
    canon = _canon() if species == "human" else _canon_rat()
    groups: dict[str, list[str]] = {}
    for g in sorted(map(str, axis)):
        groups.setdefault(canon.get(g.upper(), g), []).append(g)
    return {c: v for c, v in groups.items() if len(v) > 1}


def load_pairs(min_support: int = MIN_SUPPORT,
               curated_always: bool = True,
               human_axis: set[str] | None = None,
               rat_axis: set[str] | None = None) -> pd.DataFrame:
    """Accepted pairs, optionally restricted to the genes present in the data.

    min_support=1, curated_always=False gives every HCOP prediction including the
    single-predictor noise; that is the view to use when asking how much the
    threshold removed, not for analysis.
    """
    _check_once()
    key = (min_support, curated_always, id(human_axis), id(rat_axis))
    if key not in _cache:
        df = pd.read_csv(PAIRS_TSV, sep="\t")
        keep = df["support_n"] >= min_support
        if curated_always:
            keep |= df["rgd_curated"].astype(bool)
        df = df[keep]
        if human_axis is not None:
            canon, look = _canon(), _axis_lookup(human_axis)
            key_col = df["human_symbol"].map(lambda s: canon.get(str(s).upper(), str(s)))
            df = df[key_col.isin(look)].copy()
            df["human_symbol"] = key_col[key_col.isin(look)].map(look)
        if rat_axis is not None:
            canon_r, look_r = _canon_rat(), _axis_lookup(rat_axis, "rat")
            key_col = df["rat_symbol"].map(lambda s: canon_r.get(str(s).upper(), str(s)))
            df = df[key_col.isin(look_r)].copy()
            df["rat_symbol"] = key_col[key_col.isin(look_r)].map(look_r)
        _cache[key] = df.reset_index(drop=True)
    return _cache[key]


def rat_to_human(genes, human_axis=None, **kw) -> tuple[dict[str, list[str]], set[str]]:
    """{rat: [human, ...]} plus the rat genes with no accepted counterpart.

    Keys come back in the caller's own spelling even when the join happened on the
    modernised symbol, so a workbook written years ago still indexes its own rows.
    """
    genes = {str(g) for g in genes}
    canon_r = _canon_rat()
    as_given: dict[str, list[str]] = {}
    for g in genes:
        as_given.setdefault(canon_r.get(g.upper(), g), []).append(g)
    pairs = load_pairs(human_axis=human_axis, **kw)
    key_col = pairs["rat_symbol"].map(lambda s: canon_r.get(str(s).upper(), str(s)))
    sel = key_col.isin(as_given)
    out: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for k, human in zip(key_col[sel], pairs.loc[sel, "human_symbol"]):
        for original in as_given[k]:
            if human not in seen.setdefault(original, set()):
                seen[original].add(human)
                out.setdefault(original, []).append(human)
    return out, genes - set(out)


def human_to_rat(genes, rat_axis=None, **kw) -> tuple[dict[str, list[str]], set[str]]:
    """The same join read the other way; the table is symmetric by construction."""
    genes = {str(g) for g in genes}
    canon_h = _canon()
    as_given: dict[str, list[str]] = {}
    for g in genes:
        as_given.setdefault(canon_h.get(g.upper(), g), []).append(g)
    pairs = load_pairs(rat_axis=rat_axis, **kw)
    key_col = pairs["human_symbol"].map(lambda s: canon_h.get(str(s).upper(), str(s)))
    sel = key_col.isin(as_given)
    out: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for k, rat in zip(key_col[sel], pairs.loc[sel, "rat_symbol"]):
        for original in as_given[k]:
            if rat not in seen.setdefault(original, set()):
                seen[original].add(rat)
                out.setdefault(original, []).append(rat)
    return out, genes - set(out)


def rat_to_human_flat(genes, human_axis=None, **kw
                      ) -> tuple[dict[str, str], set[str], dict[str, list[str]]]:
    """{rat: human} for call sites that need exactly one symbol.

    Returns (mapping, unmapped, ambiguous). `ambiguous` lists every rat gene that
    had more than one accepted ortholog, so the collapse is visible, never silent.
    """
    pairs = load_pairs(human_axis=human_axis, **kw)
    rank = {(r, h): (not c, -n) for r, h, n, c in
            zip(pairs["rat_symbol"], pairs["human_symbol"],
                pairs["support_n"], pairs["rgd_curated"].astype(bool))}
    multi, unmapped = rat_to_human(genes, human_axis=human_axis, **kw)
    flat, ambiguous = {}, {}
    for rat, humans in multi.items():
        if len(humans) > 1:
            ambiguous[rat] = sorted(humans)
        # curated first, then support, then the identically spelled partner
        flat[rat] = min(humans, key=lambda h: (rank.get((rat, h), (True, 0)),
                                               h != rat.upper(), h))
    return flat, unmapped, ambiguous


def join(rat_genes, human_genes, human_axis=None, rat_axis=None, **kw) -> pd.DataFrame:
    """Map rat->human and human->rat separately, then keep only what both agree on.

    One row per (rat_symbol, human_symbol) surviving BOTH directions. Everything
    that compares a rat list to a human list should use this rather than mapping
    one side and intersecting strings.
    """
    r2h, _ = rat_to_human(rat_genes, human_axis=human_axis, **kw)
    h2r, _ = human_to_rat(human_genes, rat_axis=rat_axis, **kw)
    fwd = {(r, h) for r, hs in r2h.items() for h in hs}
    rev = {(r, h) for h, rs in h2r.items() for r in rs}
    pairs = load_pairs(human_axis=human_axis, rat_axis=rat_axis, **kw)
    meta = {(r, h): (n, c) for r, h, n, c in
            zip(pairs["rat_symbol"], pairs["human_symbol"],
                pairs["support_n"], pairs["rgd_curated"])}
    rows = [{"rat_symbol": r, "human_symbol": h,
             "support_n": meta.get((r, h), (None, None))[0],
             "rgd_curated": meta.get((r, h), (None, None))[1]}
            for (r, h) in sorted(fwd & rev)]
    return pd.DataFrame(rows, columns=["rat_symbol", "human_symbol",
                                       "support_n", "rgd_curated"])


def join_report(rat_genes, human_genes, **kw) -> dict:
    """Counts for the join, including what each direction found on its own."""
    r2h, r_un = rat_to_human(rat_genes, human_axis=kw.get("human_axis"),
                             **{k: v for k, v in kw.items() if k != "human_axis"
                                and k != "rat_axis"})
    h2r, h_un = human_to_rat(human_genes, rat_axis=kw.get("rat_axis"),
                             **{k: v for k, v in kw.items() if k != "human_axis"
                                and k != "rat_axis"})
    j = join(rat_genes, human_genes, **kw)
    fwd = {(r, h) for r, hs in r2h.items() for h in hs}
    rev = {(r, h) for h, rs in h2r.items() for r in rs}
    return {"rat_in": len(set(map(str, rat_genes))), "rat_mapped": len(r2h),
            "rat_unmapped": len(r_un),
            "human_in": len(set(map(str, human_genes))), "human_mapped": len(h2r),
            "human_unmapped": len(h_un),
            "pairs_rat_side": len(fwd), "pairs_human_side": len(rev),
            "pairs_both": len(j),
            "rat_genes_joined": j["rat_symbol"].nunique() if len(j) else 0,
            "human_genes_joined": j["human_symbol"].nunique() if len(j) else 0}


def coverage_line(genes, mapping, unmapped, label: str = "") -> str:
    """One-line coverage report; print it so the loss is never hidden."""
    n = len(set(str(g) for g in genes))
    return (f"[ortho] {label}{len(mapping)}/{n} rat genes mapped "
            f"({100 * len(mapping) / n:.0f}%), {len(unmapped)} without an ortholog")


if __name__ == "__main__":
    pr = provenance()
    print(f"built {pr['built_at']} | {pr['n_pairs']} pairs | md5 {pr['pairs_md5']}")
    p = load_pairs()
    print(f"accepted at support>={MIN_SUPPORT} or curated: {len(p)}")
    demo = ["Rps7", "Ahrr", "Tgm2", "Eif1a", "Ftl1", "Car4", "Sept7", "NotAGene"]
    flat, unmapped, ambiguous = rat_to_human_flat(demo)
    print("\nflat:", flat)
    print("unmapped:", unmapped)
    print("ambiguous:", ambiguous)
