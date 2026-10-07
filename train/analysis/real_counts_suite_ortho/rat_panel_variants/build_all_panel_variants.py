#!/usr/bin/env python3
"""
Rat rescued-gene delta figures for both arms of the experiment:

    injury arm      room air  -> hyperoxia
    treatment arm   hyperoxia -> hyperoxia + azithromycin

Each arm is drawn for all four cell types and for gCap alone, with the
full_list panel (panel_sets.py); the delta construction is in updown_core.py.
Predictions: test_inference_results_t10.jsonl (temperature 1.0).

    python build_all_panel_variants.py              both arms, full_list
    python build_all_panel_variants.py azi gated    one arm and gate
"""
import sys

import panel_sets as PSets
import updown_core as UC


GATES = ("full_list",)
SOLO_CT = ("gCap",)


def main():
    if len(sys.argv) == 3:
        combos = [(sys.argv[1], sys.argv[2])]
    else:
        combos = [(arm, gate) for arm in UC.ARMS for gate in GATES]
    for arm, gate in combos:
        print(f"\n{'=' * 78}\n=== arm={arm}  gate={gate}  all cell types\n{'=' * 78}")
        UC.ONLY_CT = None
        UC.run(arm, gate)
        for ct in SOLO_CT:
            print(f"\n{'=' * 78}\n=== arm={arm}  gate={gate}  {ct} alone\n{'=' * 78}")
            UC.ONLY_CT = ct
            UC.run(arm, gate)
        UC.ONLY_CT = None


if __name__ == "__main__":
    main()
