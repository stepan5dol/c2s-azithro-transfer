#!/usr/bin/env python3
"""
limma_sets.py — limma's camera and fry on the BPD7mo disease arm, per cell
type, called through rpy2. Nothing is reimplemented: these are the reference
implementations from limma (Wu & Smyth, NAR 2012 for camera; Giner & Smyth
2016 for fry).

WHY, GIVEN THE SUITE ALREADY HAS GSEA. Both GSEA prerank and the per-gene
Mann-Whitney used elsewhere here treat genes as independent. Genes inside a
pathway are correlated, so both are anticonservative by an amount neither
reports. camera asks the same competitive question -- "is this set shifted
more than the rest of the transcriptome" -- with the variance inflated by the
inter-gene correlation estimated from the data:

    VIF = 1 + (m - 1) * rho

Two runs: rho ESTIMATED per set from the residual space of the same design
(`inter.gene.cor=NA` -- camera's default is the FIXED 0.01, not estimation),
and rho at that fixed 0.01, because the whole correction rides on this one
number and the gap between the two runs should be visible.

WHAT EACH STATISTIC ANSWERS, given a pathway holds genes moving both ways:

    camera            competitive, directional. Fixes the correlation problem,
                      NOT the mixing one -- it still nets a bidirectional set.
    fry PValue        self-contained, directional.
    fry PValue.Mixed  self-contained, UNDIRECTED: is the set moved at all, in
                      either direction. The one standard statistic that does
                      not cancel on a set holding both directions.

READ fry's p-values WITH THE COLUMN COUNT IN HAND. fry is self-contained, so
its p-value falls with the number of columns, and the columns here are 600 to
2600 cells. It will call nearly everything significant; that is the column
count talking, not the biology. Its useful output is the RANKING of Mixed
p-values across terms, not whether they clear a threshold. camera, being
competitive, is the one that stays interpretable at this n -- which is why
both are run.

    python export_for_limma.py && python limma_sets.py
        -> autophagy_program/reports/bpd7mo_limma_sets.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INP = HERE / "reports" / "limma_input"
OUT = HERE / "autophagy_program" / "reports"
# Cohorts other than BPD7mo are exported by export_for_limma.py --condition into
# their own input directory and land in their own table; BPD7mo keeps the paths
# it has always used, so re-running it overwrites the same file as before.
INP_OF = {"BPD7mo": INP}
DEST_OF = {"BPD7mo": "bpd7mo_limma_sets.csv"}
CTS = ["gCap", "aCap", "Pericyte", "VEC"]
MIN_SET = 2           # whole terms, disease arm -- same reasoning
MIN_SET_HALF = 2      # NOT a judgement call: two genes is the minimum for an
                      # inter-gene correlation to exist. Nothing else is
                      # filtered. An earlier version used 10, then 5 -- and 5
                      # was picked after seeing that 10 dropped CMA, which is
                      # choosing a threshold to produce a result. Set size is
                      # printed on every mark instead, so a reader discounts a
                      # small set directly rather than trusting a cutoff.
FIXED_RHO = 0.01


def main():
    global INP
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="BPD7mo",
                    choices=["BPD7mo", "BPDPH7mo", "Acute26"],
                    help="which export_for_limma.py run to read")
    cond = ap.parse_args().condition
    INP = INP_OF.get(cond, HERE / "reports" / f"limma_input_{cond.lower()}")
    dest_name = DEST_OF.get(cond, f"{cond.lower()}_limma_sets.csv")
    print(f"[cohort] {cond} <- {INP.name} -> {dest_name}")

    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr

    cv = ro.default_converter + numpy2ri.converter + pandas2ri.converter
    importr("limma")
    print(f"[R] limma {ro.r('as.character(packageVersion(\'limma\'))')[0]}")

    sets = {k: list(v) for k, v in json.loads((INP / "sets.json").read_text()).items()}
    frames = []

    stems = [(ct, ct) for ct in CTS] + [(ct, f"{ct}_azi") for ct in CTS]
    for ct, stem in stems:
        if not (INP / f"{stem}.json").exists():
            continue
        meta = json.loads((INP / f"{stem}.json").read_text())
        ng, nc = meta["n_genes"], meta["n_cells"]
        Y = np.fromfile(INP / f"{stem}.bin", dtype=np.float32).reshape(ng, nc).astype(np.float64)
        genes = list(meta["genes"])
        group = np.array(meta["group"])

        print(f"\n=== {ct} [{meta['arm']}]: {ng} genes x {nc} cells "
              f"({meta['n_ref']} {meta['label_ref']} / {meta['n_test']} {meta['label_test']})")

        with cv.context():
            ro.globalenv["y"] = Y                   # numpy2ri keeps the shape
        ro.globalenv["gene_names"] = ro.StrVector(genes)
        ro.r("rownames(y) <- gene_names")
        ro.globalenv["grp"] = ro.FactorVector(
            ro.StrVector(["ref" if g == 0 else "test" for g in group]),
            levels=ro.StrVector(["ref", "test"]))
        ro.r("design <- model.matrix(~ grp)")

        # Split each term by what DISEASE did to the member, and run the halves as
        # separate sets. Under the AZI prompt the two halves must move opposite
        # ways, so camera over the whole term nets them out -- the same
        # cancellation the NES version of the deck figure had. The split is not
        # circular: members are grouped by the disease log2FC (an independent
        # variable) and the outcome measured is the AZI contrast. On the disease
        # arm it WOULD be circular, so there the whole term is the only option.
        use = sets
        if meta["arm"] == "azi":
            use = {}
            lfc = dict(zip(meta["genes"], meta["stat"]))
            for t, members in sets.items():
                for half, sel in (("dn", lambda v: v < -0.25), ("up", lambda v: v > 0.25)):
                    h = [m for m in members if m in lfc and sel(lfc[m])]
                    if len(h) >= MIN_SET_HALF:
                        use[f"{t} @@{half}"] = h
        ro.globalenv["sets_r"] = ro.ListVector({k: ro.StrVector(v) for k, v in use.items()})
        ro.r("idx <- ids2indices(sets_r, rownames(y), remove.empty=TRUE)")
        floor_ = MIN_SET_HALF if meta["arm"] == "azi" else MIN_SET
        ro.r(f"idx <- idx[sapply(idx, length) >= {floor_}]")
        n_sets = int(ro.r("length(idx)")[0])
        print(f"  {n_sets} sets with >= {floor_} members on this axis")

        ro.r("cam <- camera(y, idx, design, contrast=2, inter.gene.cor=NA)")

        # cameraPR on the SAME log2FC whose median the figures plot, so that the
        # position of a mark and its significance are the same quantity. camera
        # above ranks on the moderated t (each gene weighted by its own
        # variance), which is a different quantity and can disagree in sign with
        # the median -- it did, in 2 of 20 cells. Ranking cameraPR on the log2FC
        # removes that mismatch at the cost of the variance weighting.
        # inter.gene.cor: limma's own default for BOTH camera and cameraPR is
        # the FIXED value below (?camera: "with the default value
        # inter.gene.cor=0.01, camera will rank biologically interpretable
        # sets more highly... a useful compromise"), not a per-set estimate --
        # per-set estimation is the NA mode camera() above still runs for
        # its own Correlation/FDR columns, and it is the one the docs warn is
        # unstable at small n. Standard settings here, nothing invented.
        a, b = Y[:, group == 0], Y[:, group == 1]
        lfc = (np.log2(np.expm1(b).mean(axis=1) + 1)
               - np.log2(np.expm1(a).mean(axis=1) + 1))      # Seurat's log2FC
        ro.globalenv["stat"] = ro.FloatVector(lfc)
        ro.r("names(stat) <- gene_names")
        ro.r(f"""
            pr <- do.call(rbind, lapply(names(idx), function(nm) {{
                r <- cameraPR(stat, idx[nm], use.ranks=TRUE,
                              inter.gene.cor={FIXED_RHO}, sort=FALSE)
                data.frame(term=nm, prDirection=r$Direction, prPValue=r$PValue)
            }}))
            pr$prFDR <- p.adjust(pr$prPValue, method="BH")
        """)
        ro.r(f"cam01 <- camera(y, idx, design, contrast=2, inter.gene.cor={FIXED_RHO})")
        ro.r("fr  <- fry(y, idx, design, contrast=2)")

        with cv.context():
            cam = ro.r("data.frame(term=rownames(cam), cam, row.names=NULL)")
            cam01 = ro.r("data.frame(term=rownames(cam01), cam01, row.names=NULL)")
            fr = ro.r("data.frame(term=rownames(fr), fr, row.names=NULL)")
            pr = ro.r("pr")

        cam = cam.rename(columns={"Direction": "camera_direction", "PValue": "camera_p",
                                  "FDR": "camera_fdr", "Correlation": "camera_rho"})
        cam01 = cam01[["term", "PValue", "FDR"]].rename(
            columns={"PValue": "camera_p_rho01", "FDR": "camera_fdr_rho01"})
        fr = fr.rename(columns={"Direction": "fry_direction", "PValue": "fry_p",
                                "FDR": "fry_fdr", "PValue.Mixed": "fry_mixed_p",
                                "FDR.Mixed": "fry_mixed_fdr"})
        keep_cam = [c for c in ("term", "NGenes", "camera_rho", "camera_direction",
                                "camera_p", "camera_fdr") if c in cam.columns]
        keep_fry = [c for c in ("term", "fry_direction", "fry_p", "fry_fdr",
                                "fry_mixed_p", "fry_mixed_fdr") if c in fr.columns]
        pr = pr.rename(columns={"prDirection": "camerapr_direction",
                                "prPValue": "camerapr_p", "prFDR": "camerapr_fdr"})
        m = (cam[keep_cam].merge(cam01, on="term").merge(fr[keep_fry], on="term")
             .merge(pr, on="term"))
        m.insert(0, "arm", meta["arm"])
        m.insert(0, "cell_type", ct)
        frames.append(m)

        if "camera_rho" in m.columns:
            print(f"  estimated inter-gene correlation: "
                  f"{m.camera_rho.min():.4f}–{m.camera_rho.max():.4f}")
        print(f"  camera FDR<0.05: {int((m.camera_fdr < 0.05).sum())}/{len(m)}   "
              f"fry Mixed FDR<0.05: {int((m.fry_mixed_fdr < 0.05).sum())}/{len(m)}")

    o = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / dest_name
    o.to_csv(dest, index=False)
    print(f"\n-> {dest}  ({len(o)} rows)")


if __name__ == "__main__":
    main()
