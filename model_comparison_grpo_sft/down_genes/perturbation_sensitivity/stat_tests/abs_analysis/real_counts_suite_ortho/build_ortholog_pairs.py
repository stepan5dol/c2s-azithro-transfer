#!/usr/bin/env python3
"""
build_ortholog_pairs.py — builds ortholog_pairs.tsv, the rat<->human join for this
suite, from HCOP plus the RGD curated list.

WHY NOT ENSEMBL COMPARA ALONE, WHICH THIS USED TO DO. Compara is one tree-based
predictor, and on this rat annotation it puts the ortholog on a neighbouring or
overlapping locus often enough to matter: Ahrr->EXOC3, Rps7->COLEC11,
Tgm2->KIAA1755, Chmp2a->UBE2M, Bloc1s1->RDH5, Efnb3->WRAP53 -- roughly 90 of the
15000 rat genes on our axes, 9 of them inside the rescue workbooks. Ensembl REST
returns the same calls, so it is the database's position rather than an extraction
fault. Patching it with hand-written rules (a symbol-identity override, an HGNC
fallback tier) meant inventing adjudication logic and then maintaining it.

WHAT REPLACES IT.

  HCOP -- HGNC Comparison of Orthology Predictions, the human_rat 15-column file.
      One row per predicted pair with a `support` field naming every resource that
      backs it: Ensembl, OMA, OrthoDB, PANTHER, PhylomeDB, Treefam, Inparanoid,
      NCBI, HGNC. A lone bad call from one predictor is support 1; a real ortholog
      carries 6-9. Consensus is then a single threshold instead of a rule we wrote.

  RGD_ORTHOLOGS -- the rat<->human list curated by RGD with HGNC, i.e. by the
      nomenclature bodies rather than by a gene tree. Carried as a boolean column
      so a caller can demand curation rather than votes.

Neither `symbol_hgnc` nor `symbol_identity` exists any more: both were workarounds
for having a single predictor, and the support count subsumes them.

Output: ortholog_pairs.tsv + ortholog_pairs.provenance.json.
Both sources are live files, so the provenance sidecar records what was used and
ortho.py refuses a table whose bytes disagree with it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
OUT = HERE / "ortholog_pairs.tsv"
PROVENANCE = HERE / "ortholog_pairs.provenance.json"
HCOP_CACHE = HERE / "human_rat_hcop.tsv"
RGD_ORTHO_CACHE = HERE / "rgd_orthologs.tsv"
HGNC_CACHE = HERE / "hgnc_complete_set.tsv"
RGD_GENES_CACHE = HERE / "genes_rat_rgd.tsv"

HCOP_URL = ("https://storage.googleapis.com/public-download-files/hcop/"
            "human_rat_hcop_fifteen_column.txt.gz")
RGD_ORTHO_URL = "https://download.rgd.mcw.edu/data_release/orthologs/RGD_ORTHOLOGS.txt"


def _pinned(path: Path, url: str, refresh: bool, gz: bool = False,
            skip_hash_comments: bool = False) -> pd.DataFrame:
    """Read a pinned snapshot. Downloading is opt-in, never automatic.

    HCOP and RGD republish on their own schedule, so a silent re-fetch would move
    the table while the build still looked reproducible. A missing file is an
    error; --refresh moves to the current release deliberately.
    """
    if path.exists() and not refresh:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if not path.exists() and not refresh:
        raise SystemExit(
            f"{path.name} is missing. It is a pinned input, not something to\n"
            f"re-fetch silently -- a newer release changes the table. Restore the\n"
            f"file, or run with --refresh to move to the current release on purpose.")
    r = requests.get(url, timeout=900)
    r.raise_for_status()
    text = gzip.decompress(r.content).decode() if gz else r.text
    if skip_hash_comments:
        text = "\n".join(l for l in text.split("\n") if not l.startswith("#"))
    d = pd.read_csv(io.StringIO(text), sep="\t", low_memory=False)
    d.to_csv(path, sep="\t", index=False)
    print(f"[fetch] refreshed -> {path.name}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="download current HCOP / RGD instead of the pinned copies")
    args = ap.parse_args()

    hcop = _pinned(HCOP_CACHE, HCOP_URL, args.refresh, gz=True)
    hcop = hcop[(hcop["rat_symbol"] != "-") & (hcop["human_symbol"] != "-")]
    hcop = hcop.dropna(subset=["rat_symbol", "human_symbol"])
    print(f"[hcop] {len(hcop)} pairs with symbols on both sides")

    rgd = _pinned(RGD_ORTHO_CACHE, RGD_ORTHO_URL, args.refresh, skip_hash_comments=True)
    rat_c = next(c for c in rgd.columns if "RAT" in c.upper() and "SYMBOL" in c.upper())
    hum_c = next(c for c in rgd.columns if "HUMAN" in c.upper() and "SYMBOL" in c.upper())
    curated = {(str(r).upper(), str(h).upper())
               for r, h in zip(rgd[rat_c], rgd[hum_c])
               if isinstance(r, str) and isinstance(h, str)}
    print(f"[rgd]  {len(curated)} curated rat<->human pairs ({rat_c} / {hum_c})")

    out = pd.DataFrame({
        "rat_symbol": hcop["rat_symbol"].astype(str),
        "human_symbol": hcop["human_symbol"].astype(str),
        "rat_ensembl": hcop["rat_ensembl_gene"].astype(str),
        "human_ensembl": hcop["human_ensembl_gene"].astype(str),
        "support": hcop["support"].astype(str),
    })
    out["support_n"] = out["support"].map(lambda s: len([x for x in s.split(",") if x and x != "-"]))
    out["rgd_curated"] = [(r.upper(), h.upper()) in curated
                          for r, h in zip(out["rat_symbol"], out["human_symbol"])]
    out = (out.sort_values(["rat_symbol", "human_symbol"])
              .drop_duplicates(["rat_symbol", "human_symbol"])
              .reset_index(drop=True))
    out.to_csv(OUT, sep="\t", index=False)

    print(f"\n{len(out)} pairs -> {OUT.name}")
    print("support_n distribution:")
    print(out["support_n"].value_counts().sort_index().to_string())
    print(f"RGD-curated: {int(out['rgd_curated'].sum())}")

    # HCOP publishes no official cutoff, so calibrate it instead of picking one:
    # treat the RGD/HGNC curated list as the reference and see which support_n
    # threshold agrees with it best. Precision = of the pairs a threshold accepts,
    # how many are curated; recall = of the curated pairs, how many it keeps.
    print("\nsupport_n calibrated against the RGD curated list:")
    print(f"{'>=n':>4s} {'accepted':>9s} {'curated kept':>13s} {'precision':>10s} {'recall':>8s} {'F1':>6s}")
    n_cur = int(out["rgd_curated"].sum())
    best = (0, -1.0)
    for n in range(1, int(out["support_n"].max()) + 1):
        sel = out[out["support_n"] >= n]
        if not len(sel):
            continue
        tp = int(sel["rgd_curated"].sum())
        prec, rec = tp / len(sel), tp / n_cur
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        best = max(best, (n, f1), key=lambda t: t[1])
        print(f"{n:>4d} {len(sel):>9d} {tp:>13d} {prec:>10.3f} {rec:>8.3f} {f1:>6.3f}")
    print(f"best agreement at support_n >= {best[0]} (F1 {best[1]:.3f})")

    print("\nthe genes Compara alone got wrong:")
    for g in ["Rps7", "Ahrr", "Tgm2", "Atp5mf", "Chmp2a", "Bloc1s1", "Efnb3", "Eif1a"]:
        rows = out[out["rat_symbol"] == g]
        if not len(rows):
            print(f"  {g:9s} absent from HCOP")
            continue
        best = rows.sort_values("support_n", ascending=False)
        shown = "; ".join(f"{h} (support {n}{', RGD' if c else ''})"
                          for h, n, c in zip(best["human_symbol"].head(3),
                                             best["support_n"].head(3),
                                             best["rgd_curated"].head(3)))
        print(f"  {g:9s} {shown}")

    prov = {
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pairs_tsv": OUT.name,
        "pairs_md5": hashlib.md5(OUT.read_bytes()).hexdigest(),
        "n_pairs": int(len(out)),
        "support_n": {str(k): int(v) for k, v in
                      out["support_n"].value_counts().sort_index().items()},
        "rgd_curated_pairs": int(out["rgd_curated"].sum()),
        "sources": {
            "hcop": {"file": HCOP_CACHE.name,
                     "md5": hashlib.md5(HCOP_CACHE.read_bytes()).hexdigest()},
            "rgd_orthologs": {"file": RGD_ORTHO_CACHE.name,
                              "md5": hashlib.md5(RGD_ORTHO_CACHE.read_bytes()).hexdigest()},
            "hgnc": {"file": HGNC_CACHE.name,
                     "md5": hashlib.md5(HGNC_CACHE.read_bytes()).hexdigest()}
                     if HGNC_CACHE.exists() else None,
            "rgd_genes": {"file": RGD_GENES_CACHE.name,
                          "md5": hashlib.md5(RGD_GENES_CACHE.read_bytes()).hexdigest()}
                          if RGD_GENES_CACHE.exists() else None,
        },
    }
    PROVENANCE.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"\nmd5 {prov['pairs_md5']}  ->  {PROVENANCE.name}")


if __name__ == "__main__":
    main()
