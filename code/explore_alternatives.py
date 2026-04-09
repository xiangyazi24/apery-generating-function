"""
Explore alternative series/generating functions for ζ(3) that might
yield ODEs without problematic singular points.

Goal: find a series Σ c_n x^n = rational multiple of ζ(3)
where the ODE for the generating function has:
  - No singular point on [0,1], OR
  - Only simple indicial roots at singular points

Candidates:
1. Apéry original: Σ (-1)^{n-1}/(n³ C(2n,n)) x^n  [double root — BLOCKED]
2. Related: Σ 1/(n² C(2n,n)) x^n = 2[arcsin(√x/2)]²  [known closed form]
3. Zudilin-type: other Apéry-like series with different central binomial structure
4. Hypergeometric: ₃F₂ representations of ζ(3)
5. Modified generating functions: weight by different factors
"""

import numpy as np
from scipy.special import comb
from fractions import Fraction

ZETA3 = 1.2020569031595942

# ============================================================
# Candidate 1: Σ x^n / (n³ C(2n,n))  (no alternating sign)
# ============================================================
print("=== Candidate 1: Σ x^n/(n³ C(2n,n)) ===")
def series1(x, N=300):
    return sum(x**n / (n**3 * comb(2*n, n, exact=True)) for n in range(1, N+1))

val = series1(4.0)  # x=4 is boundary of convergence
print(f"  S(4) = {val:.15f}")
print(f"  Known: S(4) = π²ln2/4 - 7ζ(3)/16 ??? (checking)")
# Actually, Σ 4^n/(n³ C(2n,n)) = π² ln2/2 - 7ζ(3)/8... let me check
# From literature: Σ_{n=1}^∞ 1/(n³ C(2n,n)) = π²/18 · √3 - ... no
# Let me just check S(1)
val1 = series1(1.0)
print(f"  S(1) = {val1:.15f}")

# Check if this relates to ζ(3) by a rational multiple
ratios = [val1/ZETA3, val1*2/ZETA3, val1*5/ZETA3, val1*10/ZETA3]
print(f"  S(1)/ζ(3) = {val1/ZETA3:.10f}")
# Not obviously rational


# ============================================================
# Candidate 2: Σ (-1)^{n-1} x^n / (n³ C(2n,n)²)
# Double central binomial coefficient
# ============================================================
print("\n=== Candidate 2: Σ (-1)^{n-1} x^n/(n³ C(2n,n)²) ===")
def series2(x, N=200):
    s = 0.0
    for n in range(1, N+1):
        cn = comb(2*n, n, exact=True)
        s += ((-1)**(n-1)) * x**n / (n**3 * cn**2)
    return s

# Radius of convergence: C(2n,n)² ~ 16^n/(πn), so R = 16
val2_1 = series2(1.0)
val2_16 = series2(16.0)
print(f"  S(1) = {val2_1:.15f}")
print(f"  S(16) = {val2_16:.15f}")
print(f"  S(1)/ζ(3) = {val2_1/ZETA3:.10f}")

# ============================================================
# Candidate 3: Apéry numbers generating function
# a_n satisfies: (n+1)³ a_{n+1} = (2n+1)(17n²+17n+5)a_n - n³ a_{n-1}
# with a_0=1, a_1=5
# The GF Σ a_n x^n satisfies a third-order ODE
# ============================================================
print("\n=== Candidate 3: Apéry numbers a_n ===")
def apery_numbers(N=30):
    a = [Fraction(1), Fraction(5)]
    for n in range(1, N):
        a_next = ((2*n+1)*(17*n**2+17*n+5)*a[n] - n**3*a[n-1]) / (n+1)**3
        a.append(a_next)
    return a

a_n = apery_numbers(20)
print("  First few Apéry numbers:")
for i in range(8):
    print(f"    a_{i} = {a_n[i]}")

# The Apéry proof uses: ζ(3) = Σ (-1)^{n-1} b_n / a_n² where b_n are related
# But let's look at the GF: A(x) = Σ a_n x^n
# This satisfies: [(1-34x+x²)D² + (x²-34x+1)D·x·D + ... ] complicated
# The ODE is: x²(1-34x+x²)y''' + 3x(1-51x+2x²)y'' + (1-112x+6x²)y' + (-5+3x)y = 0
# Wait, I need to check this.

# Actually, the GF of Apéry numbers satisfies a known ODE.
# Let me compute it numerically and check for a rational relation to ζ(3)

A_vals = [float(a_n[n]) for n in range(21)]
# GF converges for |x| < (1+√2)^{-4} ≈ 0.0294... (very small radius!)

# Sum A(x) for small x
def apery_gf(x, N=20):
    return sum(float(a_n[n]) * x**n for n in range(N+1))

x_test = 0.02
print(f"\n  A({x_test}) = {apery_gf(x_test):.15f}")
# Radius is very small, not useful for computing ζ(3) at a nice point.


# ============================================================
# Candidate 4: ₃F₂ hypergeometric
# ζ(3) = (7/2) Σ_{n=1}^∞ (-1)^{n+1}/(n³ C(2n,n)) · (4x)^n  at x=1/4???
# Or: ζ(3) relates to ₃F₂(1,1,1; 2,2; -1) and similar
# ============================================================
print("\n=== Candidate 4: Hypergeometric series ===")

# ₃F₂(1,1,1; 2,2; z) = Σ (1)_n³/[(2)_n² n!] z^n = Σ z^n/(n+1)²
# This is related to Li₂ type, not ζ(3)

# Gosper-type: ζ(3) = (5/2) Σ_{n=0}^∞ (-1)^n (n!)²/(2n+1)!  ... check
def gosper_series(N=100):
    s = 0.0
    for n in range(N):
        # (n!)² / (2n+1)! = 1/[(2n+1)C(2n,n) · 2^{2n}] ... let me just compute
        import math
        s += (-1)**n * (math.factorial(n))**2 / math.factorial(2*n+1)
    return s

gs = gosper_series()
print(f"  Gosper-type: (5/2)·Σ(-1)^n(n!)²/(2n+1)! = {2.5*gs:.15f}")
print(f"  ζ(3) = {ZETA3:.15f}")
print(f"  Match: {abs(2.5*gs - ZETA3) < 1e-10}")

# ============================================================
# Candidate 5: Σ (-1)^n/(2n+1)³  = π³/32 (Dirichlet beta)
# NOT ζ(3), but related Euler sums?
# ============================================================

# ============================================================
# Candidate 6: The "other" Apéry series
# ζ(3) = (5/2) Σ_{n=1}^∞ (-1)^{n-1} / (n³ C(2n,n))
# But also: ζ(3) = Σ_{n=0}^∞ (-1)^n [Σ_{k=0}^n C(n,k)²C(n+k,k)²] x / ...
# ============================================================

# ============================================================
# Candidate 7: Series without central binomial coefficients
# Try: weight the power series differently to avoid the x² singularity
#
# If F(x) = Σ c_n x^n satisfies an ODE, the indicial equation at x=0
# depends on the GROWTH RATE of c_n.
#
# For c_n ~ C^n/(n^α · something), the radius of convergence is 1/C
# and the indicial roots relate to the "something" factor.
#
# c_n = (-1)^{n-1}/(n³ C(2n,n)): C(2n,n) ~ 4^n/√(πn)
# so c_n ~ (-1)^{n-1}·√(πn)/(n³·4^n) = (-1)^{n-1}/(n^{5/2}·4^n)·√π
# This gives R = 4, and the n^{-5/2} decay is what creates the
# specific indicial structure.
#
# What if we use c_n = (-1)^{n-1}/(n² C(2n,n))? (one less power of n)
# ============================================================
print("\n=== Candidate 7: Σ (-1)^{n-1} x^n/(n² C(2n,n)) ===")
def series7(x, N=300):
    return sum(((-1)**(n-1)) * x**n / (n**2 * comb(2*n, n, exact=True)) for n in range(1, N+1))

# This is known: Σ x^n/(n² C(2n,n)) = 2[arcsin(√x/2)]²
# Let's verify:
import math
val7 = series7(1.0)
arcsin_val = 2 * math.asin(0.5)**2  # 2[arcsin(1/2)]² = 2(π/6)² = π²/18
print(f"  S(1) = {val7:.15f}")
print(f"  2[arcsin(1/2)]² = {arcsin_val:.15f}")
# With alternating signs it's different
# Actually Σ (-1)^{n-1}/(n² C(2n,n)) x^n at x=1...
print(f"  (This is the n² version, related to arcsin, not ζ(3))")

# What ODE does this satisfy?
# If f(x) = Σ (-1)^{n-1}/(n�� C(2n,n)) x^n, then
# xf'(x) = Σ (-1)^{n-1}/(n C(2n,n)) x^n
# x(xf')' = ... let me find the ODE numerically

# ============================================================
# Candidate 8: Σ (-1)^{n-1} H_n / (n² C(2n,n)) x^n
# where H_n = 1 + 1/2 + ... + 1/n (harmonic numbers)
# This appears in Apéry's proof
# ============================================================
print("\n=== Candidate 8: Series with harmonic numbers ===")
def series8(x, N=200):
    H = 0.0
    s = 0.0
    for n in range(1, N+1):
        H += 1.0/n
        s += ((-1)**(n-1)) * H * x**n / (n**2 * comb(2*n, n, exact=True))
    return s

val8 = series8(1.0)
print(f"  Σ (-1)^{{n-1}} H_n/(n² C(2n,n)) at x=1 = {val8:.15f}")
print(f"  ζ(3)/2 = {ZETA3/2:.15f}")
print(f"  Ratio to ζ(3): {val8/ZETA3:.10f}")

# ============================================================
# Candidate 9: Direct Apéry approach with modified weights
# Try Σ c_n x^n where c_n = (-1)^{n-1}/(n³ C(2n,n)) · (2n+1)
# The factor (2n+1) might simplify the ODE
# ============================================================
print("\n=== Candidate 9: Σ (-1)^{n-1}(2n+1)/(n³ C(2n,n)) x^n ===")
def series9(x, N=300):
    return sum(((-1)**(n-1)) * (2*n+1) * x**n / (n**3 * comb(2*n, n, exact=True)) for n in range(1, N+1))

val9 = series9(1.0)
print(f"  S(1) = {val9:.15f}")
print(f"  S(1)/ζ(3) = {val9/ZETA3:.10f}")

# ============================================================
# Candidate 10: Guillera-type series (fast converging)
# Σ_{n=0}^∞ (-1)^n (6n+1) (1/2)_n³ / (n!)³ / 4^n
# This converges much faster (Ramanujan-type)
# ============================================================
print("\n=== Candidate 10: Ramanujan/Guillera-type ===")
def pochhammer(a, n):
    """Rising factorial (a)_n = a(a+1)...(a+n-1)"""
    result = 1.0
    for k in range(n):
        result *= (a + k)
    return result

# Zudilin: ζ(3) = (1/4) Σ_{n=0}^∞ (-1)^n (2n+1) [(2n)!/(2^n n!)]^3 / [(2n)!·(3n+1)!] ?
# This is getting complicated. Let me focus on simpler candidates.

# ============================================================
# KEY INSIGHT: What ODE properties we need
# ============================================================
print("\n" + "="*60)
print("WHAT WE NEED")
print("="*60)
print("""
For ζ(3) ∈ R_RTCRN via generating function, we need Σ c_n x^n where:

1. The generating function satisfies a LINEAR ODE with polynomial coeffs
   (guaranteed if c_n satisfies a polynomial recurrence — "D-finite")

2. The ODE has NO singular point at x=0, OR the singular point has
   only simple indicial roots

3. c_n has rational terms (so partial sums are rational)

4. The sum at some rational x₀ equals a rational multiple of ζ(3)

5. x₀ is within the radius of convergence

The Apéry series c_n = (-1)^{n-1}/(n³ C(2n,n)) satisfies 1,3,4,5
but FAILS 2 (double indicial root at x=0).

Question: does there exist ANY D-finite series for ζ(3) satisfying
all five conditions?
""")

# ============================================================
# Compute the ODE for some candidate series by finding the recurrence
# ============================================================
print("=== Recurrence analysis ===")

# For Apéry series: c_n = (-1)^{n-1}/(n³ C(2n,n))
# c_{n+1}/c_n = -n³/[(n+1)³] · C(2n,n)/C(2n+2,n+1)
# C(2n+2,n+1)/C(2n,n) = (2n+2)(2n+1)/[(n+1)(n+1)] = 2(2n+1)/(n+1)
# So c_{n+1}/c_n = -n³/[(n+1)³] · (n+1)/[2(2n+1)] = -n³/[(n+1)²·2(2n+1)]

# Recurrence: (n+1)²·2(2n+1)·c_{n+1} + n³·c_n = 0
# Or: 2(n+1)²(2n+1)c_{n+1} = -n³ c_n

# This is a FIRST-ORDER recurrence in n. The ODE will be of order
# equal to the order of the recurrence (1st order gives 1st order ODE
# for Σ c_n x^n... wait, that's not right).

# Actually, for a first-order recurrence p(n)c_{n+1} + q(n)c_n = 0,
# the generating function satisfies a first-order ODE if we can
# relate Σ n^k c_n x^n to derivatives of f(x).

# Let's find the minimal ODE for F(x) = Σ c_n x^n using the recurrence.
# Using the "Ore algebra" approach: n → xD on generating functions.

# 2(n+1)²(2n+1)c_{n+1} + n³ c_n = 0

# In GF terms: Σ 2(n+1)²(2n+1)c_{n+1}x^n + Σ n³ c_n x^n = 0

# First sum: Σ 2(n+1)²(2n+1)c_{n+1}x^n
# = 2 Σ (n+1)²(2n+1)c_{n+1}x^n
# Change index m = n+1: = 2 Σ_{m=1} m²(2m-1) c_m x^{m-1}
# = (2/x) Σ m²(2m-1) c_m x^m
# = (2/x) Σ (2m³ - m��) c_m x^m
# = (2/x) [2 Σ m³ c_m x^m - Σ m² c_m x^m]

# And Σ m^k c_m x^m = (xD)^k F(x) where θ = xD.

# So: (2/x)[2θ³F - θ²F] + θ³F = 0
# (2/x)θ²(2θ-1)F + θ³F = 0
# Multiply by x: 2θ²(2θ-1)F + xθ³F = 0

# Hmm, this gives an operator equation. Let me expand θ = xD:
# θ²(2θ-1) = 2θ³ - θ² → in terms of D: 2x³D³+... (complicated)

# Actually the original ODE x²(4+x)F''' + x(10+3x)F'' + (2+x)F' = 1
# is the INHOMOGENEOUS version. The recurrence gives the homogeneous part.

# The key structure: the ODE is third order because the recurrence
# 2(n+1)²(2n+1)c_{n+1} + n³ c_n = 0 has polynomial coefficients of
# degree 3 in n, and θ = xD satisfies θⁿF = Σ nⁿ c_k x^k.

# The degree of the polynomial in n determines the ORDER of the ODE.
# degree 3 → order 3 ODE.

# For a LOWER order ODE, we'd need a recurrence with LOWER degree in n.
# But the recurrence is determined by c_n — we can't change it.

print("The Apéry series has recurrence degree 3 → third-order ODE.")
print("To get a lower-order (simpler) ODE, need a DIFFERENT series.")
print()

# ============================================================
# Search: series c_n with low-degree recurrence AND Σc_n ∝ ζ(3)
# ============================================================

# The simplest series for ζ(3): c_n = 1/n³
# Recurrence: (n+1)³ c_{n+1} = n³ c_n ... no, c_{n+1}/c_n = n³/(n+1)³
# So (n+1)³ c_{n+1} - n³ c_n = 0... wait, c_n = 1/n³, c_{n+1} = 1/(n+1)³
# (n+1)³·1/(n+1)³ = 1, n³·1/n³ = 1. So c_{n+1}(n+1)³ = c_n·n³? No.
# c_n = 1/n³ satisfies n³c_n = 1 for all n. Not a recurrence relating c_{n+1} to c_n.

# Actually: (n+1)³c_{n+1} = 1 = n³c_n? No, n³c_n = n³/n³ = 1 and (n+1)³c_{n+1} = 1.
# So (n+1)³c_{n+1} = n³c_n. This IS a first-order recurrence! Degree 3 in n.

# For F(x) = Σ_{n=1}^∞ x^n/n³ = Li₃(x):
# The ODE: x Li₃' = Li₂(x), x Li₂' = -ln(1-x), (1-x)[-ln(1-x)]' = 1
# System of three first-order ODEs. Singularity at x=0 (from Li₃' = Li₂/x).

# So Li₃ has the SAME type of singularity!
print("Li₃(x) = Σ x^n/n³: ODE has x·Li���' = Li₂ → 1/x singularity at x=0")
print("AND singularity at x=1 (from ln(1-x)). WORSE than Apéry.")
print()

# ============================================================
# The fundamental question: is there a D-finite function equal to
# a rational multiple of ζ(3) at a rational point, whose ODE has
# no singular points (or only simple indicial roots) on the path
# from initial conditions to the evaluation point?
# ============================================================

# Let me check: what about evaluating at x where the ORIGINAL ODE
# has no singularity? The Apéry ODE x²(4+x)F''' + ... has singular
# points at x=0 and x=-4. For x ∈ (0, ∞), there's no singular point!

# So if we could start at x=1 (where F(1)=(2/5)ζ(3)) and integrate
# to x=2 or somewhere else... but we don't know F(1) (that's what
# we're trying to compute!).

# What about integrating from x=-4 to x=1? At x=-4, the ODE is singular.
# F(-4) = Σ(-1)^{n-1}(-4)^n/(n³C(2n,n)) = Σ 4^n/(n³C(2n,n))
# This is at the boundary of convergence and likely transcendental.

# From x=4 to x=1 (backwards)? x=4 is also boundary.

print("SUMMARY: All known series for ζ(3) seem to have ODEs with")
print("singular points at the natural starting point (x=0).")
print("The Apéry series is already one of the BEST — it moved the")
print("singularity from both endpoints to just x=0, with simple")
print("rational ICs. The obstruction is the double indicial root.")
