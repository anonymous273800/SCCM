from __future__ import annotations

import math
import statistics
from collections import defaultdict
from itertools import combinations

try:
    from scipy.stats import t as student_t
    from scipy.stats import rankdata, wilcoxon
except Exception:  # pragma: no cover - documented fallback
    student_t = None
    rankdata = None
    wilcoxon = None


def as_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def descriptive(values):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": "", "std": "", "se": "", "ci95_low": "", "ci95_high": "", "median": ""}
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    if n > 1:
        critical = float(student_t.ppf(0.975, n - 1)) if student_t is not None else 1.96
        half = critical * se
    else:
        half = 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "median": statistics.median(clean),
    }


def rank_biserial(differences):
    nonzero = [float(d) for d in differences if float(d) != 0.0]
    if not nonzero:
        return 0.0
    absolute = [abs(d) for d in nonzero]
    if rankdata is not None:
        ranks = list(rankdata(absolute, method="average"))
    else:
        order = sorted(range(len(absolute)), key=absolute.__getitem__)
        ranks = [0.0] * len(absolute)
        for rank, index in enumerate(order, start=1):
            ranks[index] = float(rank)
    positive = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    negative = sum(rank for rank, diff in zip(ranks, nonzero) if diff < 0)
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def paired_wilcoxon(a_values, b_values):
    pairs = [
        (float(a), float(b))
        for a, b in zip(a_values, b_values)
        if a is not None and b is not None
        and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    differences = [a - b for a, b in pairs]
    nonzero = [d for d in differences if d != 0.0]
    if not pairs:
        return {"n_pairs": 0, "n_nonzero": 0, "statistic": "", "p_value": "", "effect_rank_biserial": "", "mean_difference": "", "median_difference": "", "wins_a": 0, "ties": 0, "wins_b": 0}
    if not nonzero:
        statistic, p_value = 0.0, 1.0
    elif wilcoxon is not None:
        result = wilcoxon(
            [a for a, _ in pairs],
            [b for _, b in pairs],
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    else:
        # Exact two-sided sign-test fallback. It is conservative and keeps the script usable without SciPy.
        positive = sum(d > 0 for d in nonzero)
        n = len(nonzero)
        tail = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1)) / (2 ** n)
        statistic, p_value = float(min(positive, n - positive)), min(1.0, 2.0 * tail)
    return {
        "n_pairs": len(pairs),
        "n_nonzero": len(nonzero),
        "statistic": statistic,
        "p_value": p_value,
        "effect_rank_biserial": rank_biserial(differences),
        "mean_difference": statistics.fmean(differences),
        "median_difference": statistics.median(differences),
        "wins_a": sum(d > 0 for d in differences),
        "ties": sum(d == 0 for d in differences),
        "wins_b": sum(d < 0 for d in differences),
    }


def holm_adjust(rows, group_fields=("scope", "metric"), alpha=0.05):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        p = as_float(row.get("p_value"))
        if p is not None:
            grouped[tuple(row.get(field) for field in group_fields)].append((index, p))
    for members in grouped.values():
        ordered = sorted(members, key=lambda item: item[1])
        m = len(ordered)
        running_max = 0.0
        for rank, (index, p) in enumerate(ordered):
            adjusted = min(1.0, (m - rank) * p)
            running_max = max(running_max, adjusted)
            rows[index]["p_holm"] = running_max
            rows[index]["significant_holm_0_05"] = running_max < alpha
    return rows
