"""
Operator factorization of the Apéry ODE.

The ODE: x²(4+x)F''' + x(10+3x)F'' + (2+x)F' = 1

Since there's no F term (only derivatives), set H = F' to get second-order:
  M[H] = x²(4+x)H'' + x(10+3x)H' + (2+x)H = 1

Try to factor M = M₁ ∘ M₂ where:
  M₂[H] = xH' + αH    (first-order, Euler-type)
  M₁[G] = c(x)G' + d(x)G  (first-order)

If G = M₂[H], then M₁[G] = 1.

Matching coefficients with M[H] = x²(4+x)H'' + x(10+3x)H' + (2+x)H:
  c(x)·x = x²(4+x)  →  c(x) = x(4+x)
  c(x)(1+α) + d(x)x = x(10+3x)
  d(x)·α = 2+x

From d(x)·α = 2+x → d(x) = (2+x)/α

From the second equation:
  x(4+x)(1+α) + x·(2+x)/α = x(10+3x)
  (4+x)(1+α) + (2+x)/α = 10+3x

Constant terms: 4(1+α) + 2/α = 10
x terms: (1+α) + 1/α = 3

From x terms: α + 1 + 1/α = 3 → α + 1/α = 2 → α² - 2α + 1 = 0 → (α-1)² = 0 → α = 1
Check constant: 4·2 + 2·1 = 10 ✓

So: M₂[H] = xH' + H,  M₁[G] = x(4+x)G' + (2+x)G

The first-order ODE for G = xF'' + F':
  x(4+x)G' + (2+x)G = 1

Singularity: x(4+x) vanishes at x=0 with simple zero (order 1, not 2!)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import comb

ZETA3 = 1.2020569031595942
TARGET = (2.0/5) * ZETA3

# --- Verify the factorization numerically ---
def F_series(x, N=300):
    return sum(((-1)**(n-1)) / (n**3 * comb(2*n, n, exact=True)) * x**n for n in range(1, N+1))

def Fp_series(x, N=300):
    return sum(((-1)**(n-1)) / (n**2 * comb(2*n, n, exact=True)) * x**(n-1) for n in range(1, N+1))

def Fpp_series(x, N=300):
    return sum(((-1)**(n-1)) * (n-1) / (n**2 * comb(2*n, n, exact=True)) * x**(n-2) for n in range(2, N+1))

# G(x) = xF''(x) + F'(x) should satisfy x(4+x)G' + (2+x)G = 1
print("=== Verify factorization ===")
for x_test in [0.0, 0.1, 0.5, 1.0]:
    H = Fp_series(x_test)
    Hpp = Fpp_series(x_test)
    G = x_test * Hpp + H
    print(f"  x={x_test:.1f}: G = {G:.10f}", end="")
    if x_test > 0:
        h = 1e-7
        G_plus = (x_test+h)*Fpp_series(x_test+h) + Fp_series(x_test+h)
        G_minus = (x_test-h)*Fpp_series(x_test-h) + Fp_series(x_test-h)
        Gp = (G_plus - G_minus)/(2*h)
        lhs = x_test*(4+x_test)*Gp + (2+x_test)*G
        print(f", x(4+x)G' + (2+x)G = {lhs:.10f} (should be 1)")
    else:
        print(f", (2+0)*G(0) = {2*G:.10f} (should be 1)")

# --- G series: G_n = (-1)^n / [2(2n+1)C(2n,n)] ---
print("\n=== G series coefficients ===")
print("G(x) = Σ G_n x^n where G_n = (-1)^n / [2(2n+1)C(2n,n)]")
for n in range(6):
    cn = comb(2*n, n, exact=True)
    Gn = (-1)**n / (2*(2*n+1)*cn)
    # Verify via recurrence: G_n = -n*G_{n-1}/[2(2n+1)]
    if n > 0:
        cn_prev = comb(2*(n-1), n-1, exact=True)
        Gn_prev = (-1)**(n-1) / (2*(2*(n-1)+1)*cn_prev)
        Gn_rec = -n * Gn_prev / (2*(2*n+1))
        print(f"  G_{n} = {Gn:.10f} (recurrence: {Gn_rec:.10f})")
    else:
        print(f"  G_{n} = {Gn:.10f}")

G_at_1 = sum((-1)**n / (2*(2*n+1)*comb(2*n, n, exact=True)) for n in range(300))
print(f"\nG(1) = {G_at_1:.15f}")
print(f"F'(1) + F''(1) = G(1) = {G_at_1:.15f}")


# ============================================================
# System using factorization (t-domain, reparametrized by 1/x)
#
# Variables: F, H=F', G=xF''+F', w=1/(4+x), x
#
# dF/dτ = H·x·(1-x)         — polynomial ✓
# dH/dτ = (G-H)·(1-x)       — polynomial ✓ (1/x absorbed by reparam)
# dG/dτ = (1-x)·w·[1-(2+x)G] — polynomial ✓
# dw/dτ = -w²·x·(1-x)       — polynomial ✓
# dx/dτ = x·(1-x)            — polynomial ✓ (but x=0 is fixed point)
#
# ICs: F=0, H=1/2, G=1/2, w=1/4, x=0  — all rational ✓
# ============================================================

print("\n=== System test: factored + reparametrized ===")

def rhs_factored_reparam(tau, Y):
    F, H, G, w, x = Y
    one_mx = 1 - x

    dF = H * x * one_mx
    dH = (G - H) * one_mx
    dG = one_mx * w * (1 - (2+x)*G)
    dw = -w**2 * x * one_mx
    dx = x * one_mx

    return [dF, dH, dG, dw, dx]

# Test from x=0 (will be frozen)
Y0_zero = [0.0, 0.5, 0.5, 0.25, 0.0]
sol0 = solve_ivp(rhs_factored_reparam, [0, 100], Y0_zero,
                 method='RK45', rtol=1e-12, atol=1e-15)
print(f"From x=0: x(τ=100) = {sol0.y[4,-1]:.15f} (frozen? {abs(sol0.y[4,-1]) < 1e-10})")
print(f"  F(τ=100) = {sol0.y[0,-1]:.15f}")

# Test from x=ε (should converge)
eps = 0.01
Y0_eps = [F_series(eps), Fp_series(eps), eps*Fpp_series(eps)+Fp_series(eps),
          1/(4+eps), eps]
sol_eps = solve_ivp(rhs_factored_reparam, [0, 500], Y0_eps,
                    method='RK45', rtol=1e-12, atol=1e-15)
print(f"\nFrom x=0.01: x(τ_final) = {sol_eps.y[4,-1]:.15f}")
print(f"  F(τ_final) = {sol_eps.y[0,-1]:.15f}")
print(f"  Target     = {TARGET:.15f}")
print(f"  Error      = {abs(sol_eps.y[0,-1] - TARGET):.6e}")


# ============================================================
# Test: original t-domain (non-reparametrized) from x=ε
# Uses the factored system but without reparametrization
#
# dF/dt = H·(1-x)
# dH/dt = (G-H)·(1-x)/x    ← has 1/x
# dG/dt = (1-x)·[1-(2+x)G]/[x(4+x)]  ← has 1/[x(4+x)]
# dx/dt = 1-x                ← NO fixed point!
# ============================================================

print("\n=== Factored system in t-domain (no reparam, from t=ε) ===")

def rhs_factored_t(t, Y):
    F, H, G, x = Y
    one_mx = 1 - x

    if abs(x) < 1e-14:
        # At x=0: use limiting values
        dH = -one_mx / 24  # (G-H)/x → -1/24
        dG = 0  # [1-(2+x)G]/[x(4+x)] has finite limit but complex
        dx = one_mx
        dF = H * one_mx
        return [dF, dH, dG, dx]

    dF = H * one_mx
    dH = (G - H) * one_mx / x
    dG = one_mx * (1 - (2+x)*G) / (x*(4+x))
    dx = one_mx

    return [dF, dH, dG, dx]

# Start from small t (x ≈ 0.01)
t_start = 0.01
x_s = 1 - np.exp(-t_start)
Y0_t = [F_series(x_s), Fp_series(x_s), x_s*Fpp_series(x_s)+Fp_series(x_s), x_s]

sol_t = solve_ivp(rhs_factored_t, [t_start, 30], Y0_t,
                  method='RK45', rtol=1e-12, atol=1e-14)
print(f"  t_final = {sol_t.t[-1]:.1f}, x = {sol_t.y[3,-1]:.15f}")
print(f"  F = {sol_t.y[0,-1]:.15f}")
print(f"  Target = {TARGET:.15f}")
print(f"  Error  = {abs(sol_t.y[0,-1] - TARGET):.6e}")

# ============================================================
# KEY INSIGHT: The factorization reduced singularity from order 2 to order 1
# Original: 1/x² → After factoring: two layers of 1/x
# This is structural progress but doesn't resolve the PIVP question.
# ============================================================

print("\n" + "="*60)
print("SUMMARY OF FACTORIZATION")
print("="*60)
print("""
Original operator: x²(4+x)D³ + x(10+3x)D² + (2+x)D
Factors as: D ∘ M₁ ∘ M₂  where
  M₂ = xD + 1  (Euler operator θ+1)
  M₁ = x(4+x)D + (2+x)

Intermediate variable: G = xF'' + F' satisfies
  x(4+x)G' + (2+x)G = 1  (first-order ODE)

Singularity reduction:
  Original: leading coeff x²(4+x), indicial roots ρ = 0, 0, 1/2
  G equation: leading coeff x(4+x), "indicial root" σ = -1/2
  → Singularity order reduced from 2 to 1

Polynomial system (5 variables, reparametrized by 1/x):
  dF/dτ = Hx(1-x)           ICs: F=0      ✓ polynomial, ✓ rational
  dH/dτ = (G-H)(1-x)        ICs: H=1/2    ✓ polynomial, ✓ rational
  dG/dτ = (1-x)w[1-(2+x)G]  ICs: G=1/2    ✓ polynomial, ✓ rational
  dw/dτ = -w²x(1-x)         ICs: w=1/4    ✓ polynomial, ✓ rational
  dx/dτ = x(1-x)            ICs: x=0      ✓ polynomial, ✓ rational

Remaining obstruction: x=0 is a fixed point of dx/dτ = x(1-x).
Singularity order went 2→1, but 1→0 seems blocked.
""")
