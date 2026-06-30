"""
Read synthetic_results.json and patch the FILL placeholders in the paper.
Usage: python scripts/fill_paper_results.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "synthetic" / "synthetic_results.json"
PAPER = ROOT / "paper" / "hormetic_ib_slot.tex"

SCHEDULE_ORDER = [
    "hormetic_sigmoid",
    "hormetic_cosine",
    "linear",
    "reverse",
    "random_permutation",
    "fixed_beta",
]
DISPLAY_NAMES = {
    "hormetic_sigmoid":   r"\textbf{Hormetic-Sigmoid}",
    "hormetic_cosine":    r"\textbf{Hormetic-Cosine}",
    "linear":             "Linear",
    "reverse":            "Reverse",
    "random_permutation": r"Random Perm.",
    "fixed_beta":         r"Fixed-$\beta$",
}
K_VALS = [2, 4, 6]


def fmt(mean, std):
    if mean is None:
        return "N/A"
    return f"${mean:.3f} \\pm {std:.3f}$"


def main():
    with open(RESULTS) as f:
        data = json.load(f)

    agg = data.get("aggregated", {})
    if not agg:
        print("No 'aggregated' key in results. Run run_synthetic.py first.")
        return

    rows = []
    for name in SCHEDULE_ORDER:
        if name not in agg:
            continue
        a = agg[name]
        disp = DISPLAY_NAMES.get(name, name)
        ir_cells = []
        for k in K_VALS:
            m = a["ir"].get(str(k), {}).get("mean")
            s = a["ir"].get(str(k), {}).get("std")
            ir_cells.append(fmt(m, s))
        collapse = fmt(a["collapse_mean"], a["collapse_std"])
        stab = f"${a['stability_mean']:.3f} \\pm {a['stability_std']:.3f}$"
        rows.append(f"    {disp} & {' & '.join(ir_cells)} & {collapse} & {stab} \\\\")

    table_body = "\n".join(rows)

    tex = PAPER.read_text()

    # Replace the FILL rows with real data
    import re
    pattern = re.compile(
        r"(\\textbf\{Hormetic-Sigmoid\}.*?Fixed-\$\\beta\$.*?\\\\)",
        re.DOTALL
    )
    new_tex = pattern.sub(lambda m: table_body, tex)

    if new_tex == tex:
        # Try a simpler approach: find the block between \midrule and \bottomrule
        pattern2 = re.compile(
            r"(\\midrule\n)(.*?)(\\bottomrule)",
            re.DOTALL
        )
        new_tex = pattern2.sub(
            lambda m: m.group(1) + table_body + "\n" + m.group(3),
            tex
        )

    PAPER.write_text(new_tex)
    print("Paper updated with real results.")
    print("\nTable rows written:")
    for r in rows:
        print(" ", r)


if __name__ == "__main__":
    main()
