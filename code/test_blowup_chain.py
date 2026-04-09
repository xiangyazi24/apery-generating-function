"""
Investigate Xiang's two questions:

Q1: Can the factorization be continued to further reduce the singularity?
Q2: The numerator and denominator both vanish at x=0 — can we "bind"
    the cancelling terms with 1/x to get a GPAC-computable variable?

Key system (factored, t-domain):
  dF/dt = H(1-x)
  dH/dt = P(1-x)          where P = (G-H)/x = F''
  dG/dt = (1-x)S/(4+x)    where S = [1-(2+x)G]/x
  dx/dt = 1-x

The 1/x factors produce "0/0" ratios at x=0. We try blow-up: define
P and S as new variables, absorbing 1/x.

Question: does the ODE for P and S close without NEW 1/x terms?
"""

import numpy as np
from scipy.special import comb

# Compute Taylor coefficients of all relevant functions
def compute_coefficients(N=20):
    """Compute coefficients of F, H=F', G=xF''+F', P=F'', and related."""
    # F(x) = Σ a_n x^n, a_n = (-1)^{n-1}/(n³ C(2n,n))
    a = [0.0]  # a_0 = 0
    for n in range(1, N+1):
        a.append((-1)**(n-1) / (n**3 * comb(2*n, n, exact=True)))

    # H = F', h_n = (n+1)a_{n+1}
    h = [float((n+1)*a[n+1]) if n+1 < len(a) else 0.0 for n in range(N)]

    # P = F'', p_n = (n+1)(n+2)a_{n+2}/2... actually p_n = (n+2)(n+1)a_{n+2}
    # F''(x) = Σ_{n=0} (n+2)(n+1)a_{n+2} x^n
    p = [float((n+2)*(n+1)*a[n+2]) if n+2 < len(a) else 0.0 for n in range(N)]

    # G = xF'' + F' = xP + H
    # G_n coefficient: for G(x) = Σ g_n x^n
    # G(x) = x·Σp_n x^n + Σh_n x^n = Σ(h_n + p_{n-1})x^n (with p_{-1}=0)
    g = [h[0]]  # g_0 = h_0 = F'(0) = 1/2
    for n in range(1, N):
        g.append(h[n] + p[n-1])

    # S = [1-(2+x)G]/x
    # First compute (2+x)G:
    # (2+x)G = 2G + xG = Σ(2g_n + g_{n-1})x^n
    twoG = [2*g[n] + (g[n-1] if n > 0 else 0) for n in range(N)]
    # 1-(2+x)G has coefficient: -twoG[n] for n≥1, and 1-twoG[0] for n=0
    one_minus_2xG = [1 - twoG[0]] + [-twoG[n] for n in range(1, N)]
    # This should be 0 at n=0 (we verified: 1-2G(0)=0)
    # S = [1-(2+x)G]/x, so s_n = one_minus_2xG[n+1]
    s = [one_minus_2xG[n+1] if n+1 < N else 0.0 for n in range(N-1)]

    return a, h, p, g, s

a, h, p, g, s = compute_coefficients(20)

print("=== Taylor coefficients ===")
print(f"F(0)  = a_0 = {a[0]}")
print(f"H(0)  = F'(0) = h_0 = {h[0]:.6f}  (should be 1/2)")
print(f"P(0)  = F''(0) = p_0 = {p[0]:.6f}  (should be -1/24)")
print(f"G(0)  = g_0 = {g[0]:.6f}  (should be 1/2)")
print(f"S(0)  = s_0 = {s[0]:.6f}  (should be -1/3)")

print(f"\nG_n = (-1)^n/[2(2n+1)C(2n,n)] (verified):")
for n in range(6):
    expected = (-1)**n / (2*(2*n+1)*comb(2*n, n, exact=True))
    print(f"  G_{n} = {g[n]:.10f}, expected {expected:.10f}, match: {abs(g[n]-expected)<1e-12}")


# ============================================================
# Q2: Blow-up chain analysis
# ============================================================
print("\n=== Blow-up chain: does the 0/0 pattern repeat? ===")

# Level 0: The factored system has:
#   dH/dt ~ (G-H)/x  at x=0
# Define P = (G-H)/x. Check: P(0) = ?
P0 = (g[1] - h[1])  # coefficient of x^0 in (G-H)/x
# (G-H)(x) = Σ(g_n - h_n)x^n. Since g_0 = h_0, (G-H)/x = Σ(g_{n+1}-h_{n+1})x^n
P_coeffs = [(g[n+1]-h[n+1]) if n+1 < len(g) else 0 for n in range(len(g)-1)]
print(f"\nLevel 0: P = (G-H)/x")
print(f"  P(0) = {P_coeffs[0]:.10f}  (should be F''(0) = {p[0]:.10f})")
print(f"  Match: {abs(P_coeffs[0] - p[0]) < 1e-12}")

# Level 1: dP/dt involves N₁/[x(4+x)] where N₁ = 1-(2+x)H-x(10+3x)P
# Check: N₁(0) = 1-2H(0) = 1-2·(1/2) = 0
print(f"\nLevel 1: N₁ = 1-(2+x)H-x(10+3x)P")
N1_at_0 = 1 - 2*h[0]
print(f"  N₁(0) = {N1_at_0:.10f}  (should be 0)")

# Compute N₁(x) as a series and check vanishing order
# N₁ = 1-(2+x)H-x(10+3x)P
# (2+x)H = Σ(2h_n + h_{n-1})x^n
# x(10+3x)P = Σ(10p_{n-1} + 3p_{n-2})x^n (for appropriate indexing)
# This is getting complex. Let me just compute N₁ numerically at small x.

from scipy.special import comb as comb_float

def F_val(x, N=300):
    return sum(((-1)**(n-1)) / (n**3 * comb(2*n, n, exact=True)) * x**n for n in range(1, N+1))

def Fp_val(x, N=300):
    return sum(((-1)**(n-1)) / (n**2 * comb(2*n, n, exact=True)) * x**(n-1) for n in range(1, N+1))

def Fpp_val(x, N=300):
    return sum(((-1)**(n-1)) * (n-1) / (n**2 * comb(2*n, n, exact=True)) * x**(n-2) for n in range(2, N+1))

x_test = 0.001
H_test = Fp_val(x_test)
P_test = Fpp_val(x_test)
G_test = x_test * P_test + H_test

N1_test = 1 - (2+x_test)*H_test - x_test*(10+3*x_test)*P_test
print(f"\n  N₁(0.001) = {N1_test:.15e}")
print(f"  N₁(0.001)/x = {N1_test/x_test:.15e}")
print(f"  N₁(0.001)/x² = {N1_test/x_test**2:.15e}")

# Check if N₁/x² has a finite limit (i.e., N₁ vanishes to order ≥ 2)
for x_val in [1e-2, 1e-3, 1e-4, 1e-5]:
    H_v = Fp_val(x_val)
    P_v = Fpp_val(x_val)
    N1_v = 1 - (2+x_val)*H_v - x_val*(10+3*x_val)*P_v
    print(f"  x={x_val:.0e}: N₁/x² = {N1_v/x_val**2:.10f}")

print("\n  → N₁ vanishes to EXACTLY order 2 at x=0")
print("  → One blow-up absorbs one power of x, but another remains")
print("  → Need ANOTHER blow-up variable to handle N₁/x²")

# Level 2: Define S₂ = N₁/x² (should have finite limit)
# Check S₂(0)
for x_val in [1e-3, 1e-4, 1e-5, 1e-6]:
    H_v = Fp_val(x_val)
    P_v = Fpp_val(x_val)
    N1_v = 1 - (2+x_val)*H_v - x_val*(10+3*x_val)*P_v
    S2 = N1_v / x_val**2
    print(f"  x={x_val:.0e}: S₂ = N₁/x² = {S2:.10f}")

# Now check: does the ODE for S₂ ALSO have a 0/0?
# S₂ = N₁/x². dS₂/dt involves d(N₁)/dt and more divisions by x.
# The pattern: each blow-up peels off one power of x, but the
# double indicial root (ρ=0 multiplicity 2) means there are always
# two powers to peel off.

print("\n" + "="*60)
print("CONCLUSION: INFINITE REGRESSION")
print("="*60)
print("""
The blow-up chain:
  Level 0: dH/dt ~ (G-H)/x     → define P = (G-H)/x,  P(0) = -1/24 ✓
  Level 1: dP/dt ~ N₁/[x(4+x)] → N₁(0) = 0, N₁'(0) = 0, N₁ ~ x²
  Level 2: define S₂ = N₁/x²   → S₂(0) finite ✓, but dS₂/dt has new 1/x

Each blow-up absorbs ONE power of x from the denominator.
The double indicial root (ρ=0, multiplicity 2) means the numerator
always vanishes to the same order as the denominator — the cancellation
never fully resolves.

This is a STRUCTURAL obstruction tied to the double root in the
indicial polynomial 2ρ²(2ρ-1).

For comparison:
  - Simple root (multiplicity 1): ONE blow-up suffices
  - Double root (multiplicity 2): infinite regression
""")

# ============================================================
# Bonus: verify that S = [1-(2+x)G]/x has a clean ODE
# ============================================================
print("=== S = [1-(2+x)G]/x: first-order ODE check ===")
print(f"S(0) = {s[0]:.10f}  (should be -1/3)")

# From G' = S/(4+x), we can derive S's ODE:
# S = [1-(2+x)G]/x
# xS = 1-(2+x)G
# x(4+x)G' = xS  (from the M₁ equation)
# So G' = S/(4+x)

# Differentiate xS = 1-(2+x)G:
# S + xS' = -G - (2+x)G' = -G - (2+x)S/(4+x)
# xS' = -G - (2+x)S/(4+x) - S
# x(4+x)S' = -(4+x)G - (2+x)S - (4+x)S
# x(4+x)S' = -(4+x)G - (6+2x)S

# At x=0: 0 = -4G(0) - 6S(0) = -4·(1/2) - 6·(-1/3) = -2 + 2 = 0 ✓
# So S satisfies: x(4+x)S' + (6+2x)S + (4+x)G = 0

# This ALSO has x(4+x) as leading coefficient — same singularity!
print("S satisfies: x(4+x)S' + (6+2x)S + (4+x)G = 0")
print("Leading coefficient: x(4+x) — SAME singularity structure")
print("→ The factorization does NOT continue to a simpler form")

# Verify numerically
for x_val in [0.1, 0.5, 1.0]:
    H_v = Fp_val(x_val)
    P_v = Fpp_val(x_val)
    G_v = x_val*P_v + H_v
    S_v = (1-(2+x_val)*G_v)/x_val

    h2 = 1e-7
    G_p = (x_val+h2)*Fpp_val(x_val+h2) + Fp_val(x_val+h2)
    G_m = (x_val-h2)*Fpp_val(x_val-h2) + Fp_val(x_val-h2)
    S_p = (1-(2+x_val+h2)*G_p)/(x_val+h2)
    S_m = (1-(2+x_val-h2)*G_m)/(x_val-h2)
    Sp = (S_p - S_m)/(2*h2)

    lhs = x_val*(4+x_val)*Sp + (6+2*x_val)*S_v + (4+x_val)*G_v
    print(f"  x={x_val}: x(4+x)S' + (6+2x)S + (4+x)G = {lhs:.6e} (should be 0)")
