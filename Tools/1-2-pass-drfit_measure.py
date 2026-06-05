#!/usr/bin/env python3

import json
import statistics
import re
import sys
from collections import Counter


def chapter_num(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def load_matrix(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_reverse_best(matrix):
    reverse = {}

    for jp_name, scores in matrix.items():

        if not scores:
            continue

        best_en = max(
            scores.items(),
            key=lambda x: x[1]
        )[0]

        reverse[best_en] = jp_name

    return reverse


def main(path):

    matrix = load_matrix(path)

    reverse_best = build_reverse_best(matrix)

    best_scores = []
    margins = []
    drifts = []

    exact_matches = 0
    near_1_matches = 0
    near_2_matches = 0

    large_drift = 0
    extreme_drift = 0

    reciprocal_matches = 0

    drift_rows = []
    margin_rows = []

    for jp_name, scores in matrix.items():

        jp_num = chapter_num(jp_name)

        if jp_num is None:
            continue

        ordered = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if not ordered:
            continue

        best_en, best_score = ordered[0]

        en_num = chapter_num(best_en)

        if en_num is None:
            continue

        drift = abs(jp_num - en_num)

        drifts.append(drift)
        best_scores.append(best_score)

        if len(ordered) > 1:
            margin = best_score - ordered[1][1]
            margins.append(margin)
        else:
            margin = 1.0

        drift_rows.append({
            "jp": jp_name,
            "en": best_en,
            "score": best_score,
            "drift": drift,
            "margin": margin
        })

        margin_rows.append({
            "jp": jp_name,
            "en": best_en,
            "score": best_score,
            "drift": drift,
            "margin": margin
        })

        if drift == 0:
            exact_matches += 1

        if drift <= 1:
            near_1_matches += 1

        if drift <= 2:
            near_2_matches += 1

        if drift > 5:
            large_drift += 1

        if drift > 20:
            extreme_drift += 1

        if reverse_best.get(best_en) == jp_name:
            reciprocal_matches += 1

    total = len(drifts)

    print("=" * 70)
    print("MATCH QUALITY")
    print("=" * 70)

    print(
        f"Total Chapters                : {total}"
    )

    print(
        f"Exact Match Rate              : "
        f"{100*exact_matches/total:.2f}% "
        f"({exact_matches}/{total})"
    )

    print(
        f"Within ±1 Chapter             : "
        f"{100*near_1_matches/total:.2f}% "
        f"({near_1_matches}/{total})"
    )

    print(
        f"Within ±2 Chapters            : "
        f"{100*near_2_matches/total:.2f}% "
        f"({near_2_matches}/{total})"
    )

    print()

    print("=" * 70)
    print("DRIFT STATISTICS")
    print("=" * 70)

    print(
        f"Average Drift                 : "
        f"{statistics.mean(drifts):.2f}"
    )

    print(
        f"Median Drift                  : "
        f"{statistics.median(drifts):.2f}"
    )

    print(
        f"Max Drift                     : "
        f"{max(drifts)}"
    )

    if len(drifts) > 1:
        print(
            f"Drift StdDev                  : "
            f"{statistics.stdev(drifts):.2f}"
        )

    print()

    print(
        f"Large Drift (>5)              : "
        f"{large_drift}"
    )

    print(
        f"Extreme Drift (>20)           : "
        f"{extreme_drift}"
    )

    print()

    print("=" * 70)
    print("SIMILARITY STATISTICS")
    print("=" * 70)

    print(
        f"Average Best Score            : "
        f"{statistics.mean(best_scores):.4f}"
    )

    print(
        f"Median Best Score             : "
        f"{statistics.median(best_scores):.4f}"
    )

    if len(best_scores) > 1:
        print(
            f"Score StdDev                  : "
            f"{statistics.stdev(best_scores):.4f}"
        )

    print()

    print(
        f"Average Margin                : "
        f"{statistics.mean(margins):.4f}"
    )

    print(
        f"Median Margin                 : "
        f"{statistics.median(margins):.4f}"
    )

    print()

    ambiguous = sum(
        1 for x in margins
        if x < 0.02
    )

    print(
        f"Ambiguous Matches (<0.02)     : "
        f"{ambiguous}"
    )

    print()

    print("=" * 70)
    print("RECIPROCAL MATCHES")
    print("=" * 70)

    print(
        f"Reciprocal Match Rate         : "
        f"{100*reciprocal_matches/total:.2f}% "
        f"({reciprocal_matches}/{total})"
    )

    print()

    print("=" * 70)
    print("TOP DRIFT OUTLIERS")
    print("=" * 70)

    for row in sorted(
        drift_rows,
        key=lambda x: x["drift"],
        reverse=True
    )[:30]:

        print(
            f"{row['jp']:15s}"
            f" -> "
            f"{row['en']:15s}"
            f" drift={row['drift']:4d}"
            f" score={row['score']:.4f}"
            f" margin={row['margin']:.4f}"
        )

    print()

    print("=" * 70)
    print("LOWEST MARGIN MATCHES")
    print("=" * 70)

    for row in sorted(
        margin_rows,
        key=lambda x: x["margin"]
    )[:50]:

        print(
            f"{row['jp']:15s}"
            f" -> "
            f"{row['en']:15s}"
            f" score={row['score']:.4f}"
            f" margin={row['margin']:.4f}"
            f" drift={row['drift']}"
        )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python drift_measure.py matrix.json"
        )
        sys.exit(1)

    main(sys.argv[1])