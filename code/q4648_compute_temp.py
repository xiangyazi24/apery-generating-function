#!/usr/bin/env python3
"""One-off exact Q4648 computation through n=200.

Prompt notation:
  a_0=0, a_1=6 is the rational companion solution;
  b_0=1, b_1=5 is the integer Apéry-number solution.
"""

from __future__ import annotations

import csv
import json
import math
from fractions import Fraction
from pathlib import Path

from sympy import factorint
import matplotlib.pyplot as plt

N = 200
OUT = Path("q4648-temp-out")
OUT.mkdir(exist_ok=True)


def primes_below(limit: int) -> list[int]:
    sieve = [True] * limit
    sieve[:2] = [False, False]
    for p in range(2, math.isqrt(limit - 1) + 1):
        if sieve[p]:
            for k in range(p * p, limit, p):
                sieve[k] = False
    return [p for p in range(limit) if sieve[p]]


def vp(x: int, p: int) -> int:
    x = abs(x)
    e = 0
    while x % p == 0:
        x //= p
        e += 1
    return e


def slope(xs: list[int], ys: list[float]) -> float:
    xb = sum(xs) / len(xs)
    yb = sum(ys) / len(ys)
    return sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)


def fmt_factor(f: dict[int, int]) -> str:
    if not f:
        return "1"
    return " * ".join(str(p) if e == 1 else f"{p}^{e}" for p, e in sorted(f.items()))


a = [Fraction(0), Fraction(6)]
b = [Fraction(1), Fraction(5)]
for n in range(1, N):
    P = 34 * n**3 + 51 * n**2 + 27 * n + 5
    den = (n + 1) ** 3
    a.append((P * a[n] - n**3 * a[n - 1]) / den)
    b.append((P * b[n] - n**3 * b[n - 1]) / den)
assert all(x.denominator == 1 for x in b)

rows: list[dict[str, object]] = []
L = 1
for n in range(1, N + 1):
    L = math.lcm(L, n)
    d = L**3
    da = d * a[n]
    db = d * b[n]
    assert da.denominator == db.denominator == 1
    da_i, db_i = int(da), int(db)
    g = math.gcd(da_i, db_i)
    assert d % a[n].denominator == 0
    base = d // a[n].denominator
    residual = math.gcd(abs(a[n].numerator), abs(int(b[n])))
    assert g == base * residual
    rows.append({"n": n, "a_num": a[n].numerator, "a_den": a[n].denominator,
                 "b": int(b[n]), "lcm": L, "d": d, "da": da_i, "db": db_i,
                 "g": g, "base": base, "residual": residual,
                 "ratio": math.log(g) / n})

for n in range(1, N + 1):
    assert a[n] * b[n - 1] - a[n - 1] * b[n] == Fraction(6, n**3)

with (OUT / "q4648_results.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "a_num", "a_den", "b_n", "lcm_1_n", "d_n", "d_n_a_n",
                     "d_n_b_n", "gcd", "d_over_den_a", "gcd_num_a_b", "log_gcd_over_n"])
    for r in rows:
        writer.writerow([r["n"], r["a_num"], r["a_den"], r["b"], r["lcm"], r["d"],
                         r["da"], r["db"], r["g"], r["base"], r["residual"],
                         f'{r["ratio"]:.17g}'])

xs = [int(r["n"]) for r in rows]
ratios = [float(r["ratio"]) for r in rows]
fig, ax = plt.subplots(figsize=(9, 5.4))
ax.plot(xs, ratios, linewidth=1.2)
ax.set_xlabel("n")
ax.set_ylabel("log(gcd(d_n a_n, d_n b_n))/n")
ax.set_title("Apéry gcd diagnostic through n=200")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "q4648_log_gcd_over_n.png", dpi=180)
plt.close(fig)

small_factors = {n: {int(p): int(e) for p, e in factorint(int(rows[n-1]["g"])).items()}
                 for n in range(1, 31)}
primes = primes_below(100)
valuations = {p: [vp(int(r["g"]), p) for r in rows] for p in primes}
with (OUT / "q4648_valuations.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["n", *[f"v_{p}" for p in primes]])
    for i, n in enumerate(range(1, N + 1)):
        writer.writerow([n, *[valuations[p][i] for p in primes]])

valuation_summary = []
tail_x = list(range(101, 201))
for p in primes:
    vals = valuations[p]
    valuation_summary.append({"p": p, "v100": vals[99], "v200": vals[199], "max": max(vals),
                              "tail_increment": vals[199]-vals[99],
                              "tail_ols_slope": slope(tail_x, [float(v) for v in vals[100:]]),
                              "max_v_over_n": max(v/n for n, v in enumerate(vals, start=1))})

sc_fail_ge5 = []
sc_total_ge5 = 0
sc_fail_2_3 = []
sc_total_2_3 = 0
for p in primes_below(N + 1):
    mod = p**3
    for m in range(1, N // p + 1):
        residue = (int(b[m*p]) - int(b[m])) % mod
        record = {"p": p, "m": m, "mp": m*p, "residue": residue}
        if p >= 5:
            sc_total_ge5 += 1
            if residue:
                sc_fail_ge5.append(record)
        else:
            sc_total_2_3 += 1
            if residue:
                sc_fail_2_3.append(record)

a_rat_total = 0
a_rat_skipped = 0
a_rat_fail = []
for p in primes_below(N + 1):
    mod = p**3
    for m in range(1, N // p + 1):
        x, y = a[m*p], a[m]
        if math.gcd(x.denominator * y.denominator, p) != 1:
            a_rat_skipped += 1
            continue
        a_rat_total += 1
        lhs = x.numerator * pow(x.denominator, -1, mod) % mod
        rhs = y.numerator * pow(y.denominator, -1, mod) % mod
        residue = (lhs - rhs) % mod
        if residue:
            a_rat_fail.append({"p": p, "m": m, "mp": m*p, "residue": residue})

window_summary = []
for lo, hi in [(1,50),(51,100),(101,150),(151,200)]:
    seg = rows[lo-1:hi]
    xw = [int(r["n"]) for r in seg]
    logs = [math.log(int(r["g"])) for r in seg]
    rw = [float(r["ratio"]) for r in seg]
    window_summary.append({"window": f"{lo}-{hi}", "log_g_ols_slope": slope(xw, logs),
                           "min_ratio": min(rw), "max_ratio": max(rw)})

selected = []
for n in [1,2,3,4,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]:
    r = rows[n-1]
    selected.append({"n": n, "g_digits": len(str(r["g"])), "ratio": r["ratio"],
                     "a_den_digits": len(str(r["a_den"])), "d_digits": len(str(r["d"]))})

summary = {
    "N": N, "all_b_integral": True, "all_scaled_integral": True, "wronskian_verified": True,
    "endpoint_gcd": str(rows[-1]["g"]), "endpoint_gcd_digits": len(str(rows[-1]["g"])),
    "endpoint_ratio": rows[-1]["ratio"], "min_ratio": min(ratios), "max_ratio": max(ratios),
    "residual_exceptions": [{"n": int(r["n"]), "residual": int(r["residual"])}
                            for r in rows if int(r["residual"]) != 1],
    "selected": selected, "windows": window_summary,
    "small_factors": {str(n): fmt_factor(f) for n, f in small_factors.items()},
    "valuation_summary": valuation_summary,
    "supercongruence": {"tests_p_ge_5": sc_total_ge5, "failures_p_ge_5": sc_fail_ge5,
                            "tests_p_2_3": sc_total_2_3, "failures_p_2_3": sc_fail_2_3,
                            "rational_a_admissible": a_rat_total, "rational_a_skipped": a_rat_skipped,
                            "rational_a_failures": a_rat_fail},
    "first_terms": [{"n": n, "a": str(a[n]), "b": int(b[n]), "g": int(rows[n-1]["g"]),
                      "ratio": rows[n-1]["ratio"]} for n in range(1,11)]
}
(OUT / "q4648_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print("Q4648_COMPUTATION_OK")
print(json.dumps(summary, sort_keys=True))
