#!/usr/bin/env python3
"""
build_all_panel_variants.py — the two rat arms of the real three-armed
experiment, on the unfiltered rescued panel:

    injury arm      room air  -> hyperoxia
    treatment arm   hyperoxia -> hyperoxia + azithromycin

Each is built twice: all four cell types on one figure, then gCap alone on its
own. Hyperoxia-suppressed and hyperoxia-induced genes are separate violins; see
updown_core.py for the delta construction and panel_sets.py for the panel.

Unlike the human side, the treatment arm here is measured, not predicted, so
real and model blocks are both present in it.

Run a single combination instead (gate may be "gated" to override):
    python build_all_panel_variants.py azi full_list

Only T=1.0 / test_inference_results3.jsonl (see real_counts_suite/README.md).
"""
import sys

import panel_sets as PSets
import updown_core as UC


GATES = ("full_list",)   # PSets.GATES also offers the differential-expression
                         # filtered variant; it is not part of this analysis
SOLO_CT = ("gCap",)      # cell types that additionally get a figure of their own


def main():
    if len(sys.argv) == 3:
        combos = [(sys.argv[1], sys.argv[2])]
    else:
        combos = [(arm, gate) for arm in UC.ARMS for gate in GATES]
    for arm, gate in combos:
        print(f"\n{'=' * 78}\n=== arm={arm}  gate={gate}  all cell types\n{'=' * 78}")
        UC.ONLY_CT = None
        UC.run(arm, gate)
        # same construction, same numbers, one cell type on its own figure
        for ct in SOLO_CT:
            print(f"\n{'=' * 78}\n=== arm={arm}  gate={gate}  {ct} alone\n{'=' * 78}")
            UC.ONLY_CT = ct
            UC.run(arm, gate)
        UC.ONLY_CT = None


if __name__ == "__main__":
    main()
