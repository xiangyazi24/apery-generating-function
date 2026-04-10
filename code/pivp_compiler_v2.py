"""
PIVP Compiler v2: Fix the integral accumulation bug.

Key insight from v1: The integral I = ∫₀¹ f(x)dx must stop accumulating
when x→1. This requires including (1-x) in I' so that I'→0.

Standard GPAC integration:
  x' = 1-x              (x: 0→1 via 1-e^{-τ})
  I' = f(x)·(1-x)       (change of variable: dI/dx = f(x))
  → I(∞) = ∫₀¹ f(x)dx   ✓

With gating:
  x' = g(τ)·(1-x)
  I' = g(τ)·f(x)·(1-x)
  → dI/dx = f(x) regardless of g  ✓
  → I(∞) = ∫₀¹ f(x(τ(s)))·1 ds ... hmm

Actually: dI/dx = I'/x' = g·f(x)·(1-x) / [g·(1-x)] = f(x).
So I(∞) = ∫₀^{x(∞)} f(s)ds = ∫₀¹ f(s)ds. ✓

The gate factor CANCELS in dI/dx. The integral is EXACT
regardless of the gate profile. The error only comes from
f depending on non-converged auxiliary variables.

===================================================================
CONCRETE TEST: e·ln(2) via two-stage PIVP
===================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

ZETA3 = 1.2020569031595942
TARGET_ELN2 = np.e * np.log(2)

# ===================================================================
# CORRECT gated integral for ∫₀¹ u/(1+x) dx
# ===================================================================
# Block A: u → e
#   u' = u(1-xA), xA' = 1-xA, u(0)=1, xA(0)=0
#
# Gate chain: g_k' = g_{k-1}²(1-g_k)
#
# Block B: compute ∫₀¹ u/(1+x) dx
#   xB' = g²(1-xB)
#   I'  = g²·(1-xB)·u/(1+xB)       ← KEY: (1-xB) factor!
#
# Then dI/dxB = u/(1+xB)  and I(∞) = ∫₀¹ u(τ(s))/(1+s) ds
#
# If u ≈ e everywhere during Block B's integration, I(∞) ≈ e·ln(2).

print("="*60)
print("CORRECT gated integral: e·ln(2)")
print("="*60)

def rhs_correct_v1(tau, Y, n_gates=2):
    """Correct gated integral with (1-xB) factor."""
    u, xA = Y[0], Y[1]
    gates = Y[2:2+n_gates]
    I, xB = Y[2+n_gates], Y[3+n_gates]

    dY = np.zeros_like(Y)

    # Block A
    dY[0] = u * (1 - xA)        # u → e
    dY[1] = 1 - xA              # xA → 1

    # Gate chain
    prev = xA
    for i in range(n_gates):
        dY[2+i] = prev**2 * (1 - gates[i])
        prev = gates[i]

    gate = prev**2

    # Block B (CORRECT: with (1-xB) factor)
    dY[2+n_gates] = gate * (1 - xB) * u / (1 + xB)   # I' — has 1/(1+xB) still
    dY[3+n_gates] = gate * (1 - xB)                    # xB'

    return dY

# Non-polynomial version first (has 1/(1+xB))
print("\nNon-polynomial version (direct division):")
print(f"Target: e·ln(2) = {TARGET_ELN2:.15f}")
print()

for n_gates in range(1, 5):
    n_vars = 4 + n_gates
    Y0 = np.zeros(n_vars)
    Y0[0] = 1.0  # u = 1

    for T in [50, 200, 1000]:
        sol = solve_ivp(lambda t, y: rhs_correct_v1(t, y, n_gates),
                        [0, T], Y0, method='RK45', rtol=1e-13, atol=1e-15,
                        max_step=0.1)
        I_val = sol.y[2+n_gates, -1]
        xB_val = sol.y[3+n_gates, -1]
        err = abs(I_val - TARGET_ELN2)
        if T == 50:
            print(f"  gates={n_gates}: ", end="")
        print(f"T={T:4d} I={I_val:.12f} err={err:.4e}  ", end="")
    print()

# ===================================================================
# POLYNOMIAL version: auxiliary p = 1/(1+xB)
# ===================================================================
# p' = -xB'/(1+xB)² = -gate·(1-xB)·p²
# I' = gate·(1-xB)·u·p
#
# But (1-xB) is NOT polynomial in terms of our variables!
# We need to express (1-xB) without xB.
#
# Solution: keep xB as a variable, it IS polynomial.
# 1-xB is just a polynomial expression.
#
# Full polynomial system:
#   u'  = u(1-xA)                  IC: 1
#   xA' = 1-xA                     IC: 0
#   g1' = xA²(1-g1)                IC: 0
#   g2' = g1²(1-g2)                IC: 0
#   I'  = g2²·(1-xB)²·u·p         IC: 0    ← WAIT: (1-xB) cancels with xB'
#   xB' = g2²·(1-xB)               IC: 0
#   p'  = -g2²·(1-xB)·p²          IC: 1    (p = 1/(1+0) = 1)
#
# Hmm wait: I' = gate·(1-xB)·u·p. This has gate = g2², (1-xB), u, p.
# All are variables or polynomial expressions → POLYNOMIAL ✓

print("\n" + "="*60)
print("POLYNOMIAL version")
print("="*60)

def rhs_polynomial(tau, Y):
    """Fully polynomial PIVP for e·ln(2) with 2 gates."""
    u, xA, g1, g2, I, xB, p = Y

    gate = g2**2
    one_mx = 1 - xB

    du  = u * (1 - xA)
    dxA = 1 - xA
    dg1 = xA**2 * (1 - g1)
    dg2 = g1**2 * (1 - g2)
    dI  = gate * one_mx * u * p     # ← correct: stops when xB→1
    dxB = gate * one_mx
    dp  = -gate * one_mx * p**2     # p = 1/(1+xB)

    return [du, dxA, dg1, dg2, dI, dxB, dp]

Y0_poly = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

print(f"\nTarget: e·ln(2) = {TARGET_ELN2:.15f}\n")
print(f"{'T':>6} | {'u':>12} | {'xB':>12} | {'p':>12} | {'I':>18} | {'Error':>12}")
print("-" * 82)

for T in [10, 20, 50, 100, 200, 500, 1000]:
    sol = solve_ivp(rhs_polynomial, [0, T], Y0_poly,
                    method='RK45', rtol=1e-13, atol=1e-15, max_step=0.1)
    u, xA, g1, g2, I, xB, p = [sol.y[i, -1] for i in range(7)]
    err = abs(I - TARGET_ELN2)
    print(f"{T:6d} | {u:12.8f} | {xB:12.10f} | {p:12.10f} | {I:18.15f} | {err:12.6e}")


# ===================================================================
# ERROR ANALYSIS: Does the error converge to zero?
# ===================================================================
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)
print("""
I(∞) = ∫₀¹ u(τ(s)) / (1+s) ds

where τ(s) is defined by xB(τ(s)) = s.

Error: ε = I(∞) - e·ln(2) = ∫₀¹ [u(τ(s)) - e] / (1+s) ds

u(τ) = exp(1-e^{-τ}), so u(τ) - e = e·[exp(-e^{-τ}) - 1] ≈ -e·e^{-τ}

The question: what is τ(s) for small s?
With 2 gates: xB stays near 0 until gates saturate (τ ≈ few),
then rises. So τ(s) for any s > 0 is at least a few time constants,
and u(τ(s)) ≈ e for all s. The error is:
  ε ≈ ∫₀^{s₀} [u(τ(s)) - e]/(1+s) ds
where s₀ is tiny.
""")

# Direct error measurement at T=1000 (should be converged)
sol_long = solve_ivp(rhs_polynomial, [0, 1000], Y0_poly,
                     method='RK45', rtol=1e-14, atol=1e-16, max_step=0.05)
I_long = sol_long.y[4, -1]
print(f"I(T=1000) = {I_long:.15f}")
print(f"Target    = {TARGET_ELN2:.15f}")
print(f"Error     = {abs(I_long - TARGET_ELN2):.6e}")

# Does error decrease with more gates?
print(f"\nError vs number of gates (T=1000):")
print(f"{'Gates':>5} | {'I(T=1000)':>18} | {'Error':>12}")
print("-" * 42)

def make_poly_system(n_gates):
    """Polynomial PIVP with n_gates gate levels."""
    def rhs(tau, Y):
        u = Y[0]
        xA = Y[1]
        gates = Y[2:2+n_gates]
        I = Y[2+n_gates]
        xB = Y[3+n_gates]
        p = Y[4+n_gates]

        dY = np.zeros_like(Y)
        dY[0] = u * (1 - xA)
        dY[1] = 1 - xA

        prev = xA
        for i in range(n_gates):
            dY[2+i] = prev**2 * (1 - gates[i])
            prev = gates[i]

        gate = prev**2
        one_mx = 1 - xB

        dY[2+n_gates] = gate * one_mx * u * p    # I
        dY[3+n_gates] = gate * one_mx             # xB
        dY[4+n_gates] = -gate * one_mx * p**2     # p

        return dY
    return rhs

for n_gates in range(1, 8):
    n_vars = 5 + n_gates  # u, xA, g1..gN, I, xB, p
    Y0 = np.zeros(n_vars)
    Y0[0] = 1.0   # u
    Y0[-1] = 1.0   # p = 1/(1+0) = 1

    rhs = make_poly_system(n_gates)
    sol = solve_ivp(rhs, [0, 1000], Y0, method='RK45',
                    rtol=1e-14, atol=1e-16, max_step=0.1)

    I_val = sol.y[2+n_gates, -1]
    err = abs(I_val - TARGET_ELN2)
    print(f"{n_gates:5d} | {I_val:18.15f} | {err:12.4e}")


# ===================================================================
# NOW: Apply to ζ(3) via the Apéry system
# ===================================================================
print("\n" + "="*60)
print("APPLICATION TO ζ(3)")
print("="*60)
print("""
The Apéry generating function approach:
  F(x) satisfies x²(4+x)F''' + x(10+3x)F'' + (2+x)F' = 1
  F(1) = (2/5)ζ(3)

The polynomial system (from factorization, t-domain reparametrized by 1/x):
  dF/dτ = Hx(1-x)           but x=0 is a fixed point
  dH/dτ = (G-H)(1-x)
  dG/dτ = (1-x)w[1-(2+x)G]
  dw/dτ = -w²x(1-x)
  dx/dτ = x(1-x)

If we could START from x=ε (small), the system works perfectly.
The problem: x(0) = 0 is a fixed point.

PIVP compiler approach: use a GATE CHAIN to first "compute ε"
(i.e., have an auxiliary variable converge to a small value),
then use it to kick-start x.

But wait — ε can be any small positive number, and the system
converges to the same F(1) from any ε. What if we use:
  x' = x(1-x) + kick(τ)
where kick(τ) is a brief pulse that pushes x away from 0?

The kick must be polynomial. Options:
  kick(τ) = a · e^{-τ}  (standard, but need auxiliary var)
  kick(τ) = a · (1-s)  where s' = s, s(0) small?

Actually, the simplest idea: don't use x' = x(1-x).
Use the ORIGINAL t-domain where x' = 1-x (no fixed point at x=0),
but handle the 1/x singularity differently.

The t-domain system:
  F' = H(1-x)
  H' = (G-H)(1-x)/x           ← has 1/x
  G' = (1-x)[1-(2+x)G]/[x(4+x)]  ← has 1/[x(4+x)]
  x' = 1-x

Key: G-H = O(x) and 1-(2+x)G = O(x) near x=0.
So both ratios are finite (L'Hôpital), but NOT polynomial.

NEW APPROACH: What if Block A computes x(τ) (pushing it away from 0),
and Block B is the polynomial system that only runs after x > ε?
""")

# ===================================================================
# APPROACH: Use the factored polynomial system with a gate
# that waits for x to grow large enough
# ===================================================================
# The polynomial system IS correct for x > 0.
# We just need x to NOT be stuck at 0.
#
# Idea: separate the "x growth" phase from the "computation" phase.
#
# Phase 1 (auxiliary): Compute x via a NON-stuck equation
#   s' = 1-s,  s(0) = 0  → s = 1-e^{-τ}  (reaches ~1)
#   Now s is NOT stuck at 0.
#
# Phase 2 (main): Use s as the "x" variable in the Apéry system.
#   But s follows s' = 1-s, not s' = s(1-s).
#
# The issue is that the Apéry polynomial system was derived with
# the time reparametrization dτ/dt = 1/x, which turns x' = 1-x
# into x' = x(1-x).
#
# What if we DON'T reparametrize? Keep the original t-domain.
# x' = 1-x (no fixed point) but H' and G' have 1/x.
# Use the GATE to delay H and G until x is away from 0.

print("\n--- Approach: Gated t-domain system ---")

from scipy.special import comb

def F_series(x, N=300):
    return sum(((-1)**(n-1)) / (n**3 * comb(2*n, n, exact=True)) * x**n for n in range(1, N+1))

def Fp_series(x, N=300):
    return sum(((-1)**(n-1)) / (n**2 * comb(2*n, n, exact=True)) * x**(n-1) for n in range(1, N+1))

def Fpp_series(x, N=300):
    return sum(((-1)**(n-1)) * (n-1) / (n**2 * comb(2*n, n, exact=True)) * x**(n-2) for n in range(2, N+1))

TARGET_F1 = (2.0/5) * ZETA3

# Test: gated system where the 1/x is handled by starting late
def rhs_apery_gated(t, Y):
    """
    t-domain Apéry system with factorization.
    Variables: F, H, G, x, w (w = 1/(4+x))

    F' = H(1-x)
    H' = (G-H)(1-x)/x
    G' = (1-x)w[1-(2+x)G]/x
    w' = -w²x(1-x)   (derived from (4+x)' = x' = 1-x)
    x' = 1-x
    """
    F, H, G, x, w = Y

    one_mx = 1 - x

    dF = H * one_mx
    dx = one_mx

    if abs(x) < 1e-30:
        # At x=0, use L'Hôpital limits
        dH = 0.0
        dG = 0.0
        dw = 0.0
    else:
        dH = (G - H) * one_mx / x
        dG = one_mx * w * (1 - (2+x)*G) / x
        dw = -w**2 * x * one_mx

    return [dF, dH, dG, dx, dw]

# Baseline: start from small ε
eps = 0.01
x0 = 1 - np.exp(-eps)  # ≈ eps for small eps
H0 = Fp_series(x0)
G0 = x0 * Fpp_series(x0) + H0
Y0_base = [F_series(x0), H0, G0, x0, 1/(4+x0)]

sol_base = solve_ivp(rhs_apery_gated, [eps, 50], Y0_base,
                     method='RK45', rtol=1e-12, atol=1e-14)
print(f"  Baseline (from t=0.01): F(∞) = {sol_base.y[0,-1]:.15f}")
print(f"  Target:                        {TARGET_F1:.15f}")
print(f"  Error: {abs(sol_base.y[0,-1]-TARGET_F1):.6e}")


# ===================================================================
# THE REAL COMPILER: Polynomial system that computes ζ(3)
# ===================================================================
# Strategy: use the DNA32 paper's approach with a PIVP compiler
# to handle the transcendental ICs.
#
# DNA32 computes ζ(3) = (1/2)∫₀^∞ t²/(e^t-1) dt via splitting at t=1:
#   Block I:  ∫₀¹ t²/(e^t-1) dt
#   Block II: ∫₁^∞ t²/(e^t-1) dt (via u=e^{-t})
#
# The ICs at the split point involve e and 1/(e-1).
# PIVP compiler: compute e first, then start the main computation.
#
# But the DNA32 approach already has a polynomial system!
# The only issue is the ICs. Let me write it out.

print("\n" + "="*60)
print("DNA32-STYLE ζ(3) WITH PIVP COMPILER")
print("="*60)

# From the DNA32 paper:
# ζ(3) = (1/2) ∫₀^∞ t²/(e^t-1) dt
#
# Block I: For t ∈ [0,1], use variables tracking t^k e^{-nt}
# Block II: For t ∈ [1,∞), substitute u = e^{-t}, t = -ln(u)
#
# At t=1 (the split point):
#   e^{-1} = 1/e  (transcendental)
#   1/(e-1)        (transcendental)
#
# PIVP Compiler approach:
# 1. Block A: compute e, 1/e, 1/(e-1) as limits of polynomial PIVPs
# 2. Gate chain: wait for Block A to converge
# 3. Block B: the DNA32 computation, gated

# Block A: computing e and related quantities
# u' = u(1-s), s' = 1-s → u → e, s → 1
# v = 1/u: v' = -v²·u' = -v²·u(1-s) = -v(1-s) → v → 1/e
# r = 1/(u-1): harder. u-1 starts at 0, goes to e-1.
#   Let q = u-1. q' = u(1-s) = (q+1)(1-s).
#   r = 1/q. r' = -q'/q² = -(q+1)(1-s)/q² = -(1/q + 1/q²)(1-s) = -(r+r²)(1-s)
#   r(0) = 1/0 = ∞. Problem!
#
# Fix: use a different variable. Let w = q/(1+q) = (u-1)/u = 1-1/u = 1-v.
# w' = v' = -(1-s)·v ... wait, w = 1-v, so w' = -v' = v(1-s) = (1-w)(1-s).
# w(0) = 1-1/1 = 0, w → 1-1/e ≈ 0.6321.
# Then 1/(e-1) = v/(1-v) = (1-w)/w. Still need 1/w, not polynomial.
#
# Alternative: Track q and its reciprocal via gate.
# Once u converges to e, q = u-1 converges to e-1.
# Then we can use a gated computation of 1/q via:
#   r' = gate · r(1 - r·q)  → if gate=1 and q constant, r → 1/q
#
# This is Newton's method for 1/q, with polynomial RHS!

print("""
Computing auxiliary quantities polynomially:
  u → e      via  u' = u(1-s), s' = 1-s
  v → 1/e    via  v' = -v(1-s)
  q = u-1 → e-1   via  q' = (q+1)(1-s)
  r → 1/(e-1)     via  r' = gate·r(1-r·q)  (Newton iteration)
""")

# Test Newton-style computation of 1/(e-1)
def rhs_newton_reciprocal(tau, Y):
    """
    Compute e, 1/(e-1) via polynomial system.
    Variables: u, s, q, r, g1, g2
    """
    u, s, q, r, g1, g2 = Y

    du = u * (1 - s)
    ds = 1 - s
    dq = (q + 1) * (1 - s)    # q = u-1

    # Gate chain (waits for s → 1, i.e., u converged)
    dg1 = s**2 * (1 - g1)
    dg2 = g1**2 * (1 - g2)

    gate = g2**2

    # Newton iteration for r = 1/q
    # r' = gate · r · (2 - r·q)  (Newton for 1/q, converges if r·q near 1)
    # But r(0) must be a reasonable initial guess.
    # q starts at 0, grows to e-1 ≈ 1.718. r should start near 1/1.7 ≈ 0.58.
    # If r(0) = 1/2 (rational!), then r·q(∞) = 0.5·1.718 = 0.859.
    # Newton: r_{n+1} = r_n(2 - r_n·q) needs r·q ∈ (0,2) for convergence.
    #
    # BUT: Newton-in-ODE-form: r' = α·r·(2-r·q) where α controls speed.
    # r(2-rq) has fixed points at r=0 and r=1/q.
    # Stability: d/dr[r(2-rq)] = 2-2rq. At r=1/q: 2-2=0. Marginal!
    # Need to be more careful. Actually for the ODE r' = r(2-rq),
    # at fixed point r=1/q: linearize, r = 1/q + δ,
    # δ' ≈ (1/q)(2 - (1/q+δ)q) + ... = (1/q)(-qδ) = -δ.
    # Wait, let me compute: r(2-rq) at r = 1/q + δ:
    # (1/q+δ)(2-(1/q+δ)q) = (1/q+δ)(2-1-qδ) = (1/q+δ)(1-qδ)
    # = 1/q - δ + δ - qδ² = 1/q - qδ²
    # So r' ≈ -qδ², which is second-order! Very slow convergence.
    #
    # Better: use the standard continuous Newton: r' = (1 - r·q)
    # Fixed point: rq = 1, r = 1/q. Linearize: δ' = -q·δ. Stable if q > 0. ✓
    # But r' = 1 - rq is NOT polynomial if q is changing!
    # Actually it IS: dr = (1 - r*q) is polynomial in r and q. ✓

    dr = gate * (1 - r * q)   # converges to r = 1/q = 1/(e-1)

    return [du, ds, dq, dr, dg1, dg2]

# Initial: u=1, s=0, q=0, r=1 (guess), g1=0, g2=0
Y0_newton = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
sol_n = solve_ivp(rhs_newton_reciprocal, [0, 200], Y0_newton,
                  method='RK45', rtol=1e-13, atol=1e-15, max_step=0.1)

target_r = 1/(np.e - 1)
print(f"Newton reciprocal test:")
for T_eval in [5, 10, 20, 50, 100, 200]:
    idx = np.argmin(np.abs(sol_n.t - T_eval))
    u, s, q, r, g1, g2 = [sol_n.y[i, idx] for i in range(6)]
    print(f"  τ={T_eval:3d}: u={u:.10f}, q={q:.10f} (e-1={np.e-1:.10f}), "
          f"r={r:.10f} (1/(e-1)={target_r:.10f}), err_r={abs(r-target_r):.4e}")


# ===================================================================
# THE FULL COMPILER FOR ζ(3)
# ===================================================================
#
# We'll use a simpler approach than DNA32:
# Compute ζ(3) = (5/2) F(1) where F satisfies the Apéry ODE.
#
# The factored system in t-domain needs x=ε as starting point.
# Instead of starting from x=0 (stuck), we:
#
# 1. Run x' = 1-x freely (x goes 0→1, no gate needed)
# 2. For H' and G' which have 1/x: use blow-up variables
#    P = (G-H)/x and S = [1-(2+x)G]/x
# 3. P and S have FINITE initial values: P(0) = F''(0) = -1/24,
#    S(0) = -1/3
# 4. The ODEs for P and S have THEIR OWN 1/x singularities...
#
# This is the infinite regression from Note 2. But wait:
# What if we use the GATE to handle the first level of 1/x,
# and the finite number of gates gives finite approximation error?
#
# Actually, a simpler idea:
# The original t-domain system with 1/x works FINE for x > 0.
# x = 1-e^{-t} is > 0 for all t > 0.
# Numerically starting from t=0 with x(0)=0 should work because
# the ODE solver takes a finite first step.

print("\n" + "="*60)
print("DIRECT t-domain integration (no reparametrization)")
print("="*60)

def rhs_apery_t_direct(t, Y):
    """
    Direct t-domain system using BLOW-UP variables P and S.
    F' = H(1-x)
    H' = P(1-x)     where P = (G-H)/x = F''
    G' = (1-x)Sw    where S = [1-(2+x)G]/x, w = 1/(4+x)
    P' = ???         needs another blow-up...

    Actually, let's go back to using (G-H)/x directly:
    H' = (G-H)(1-x)/x

    At x=0: (G-H)/x → (G'(0) - H'(0))·x/x ... let me compute.
    G(x) = Σ g_n x^n, H(x) = Σ h_n x^n.
    G(0) = H(0) = 1/2. (G-H)(x) = Σ(g_n - h_n)x^n.
    (G-H)/x = g_1 - h_1 + O(x) = P(0) = F''(0) = -1/24.

    So at t=0: H' = F''(0)·(1-0) = -1/24.
    And G' at t=0: G' = (1-x)[1-(2+x)G]/[x(4+x)]
    [1-(2+x)G]/x = S(0) = -1/3 (from note 2)
    G' = 1·(-1/3)·(1/4) = -1/12. Check with G series:
    G_0 = 1/2, G_1 = -1/12. G(x) ≈ 1/2 - x/12 + ...
    G'(0) = -1/12 ✓
    """
    F, H, G, x, w = Y
    one_mx = 1 - x

    dF = H * one_mx
    dx = one_mx
    dw = -w**2 * one_mx   # w = 1/(4+x), (4+x)' = x' = 1-x

    if abs(x) < 1e-12:
        # Use analytic limits at x=0
        # (G-H)/x → F''(0) = -1/24
        # [1-(2+x)G]/[x(4+x)] → S(0)/(4+0) = (-1/3)/4 = -1/12
        dH = -1.0/24.0 * one_mx
        dG = -1.0/12.0 * one_mx
    else:
        dH = (G - H) * one_mx / x
        dG = one_mx * w * (1 - (2+x)*G) / x

    return [dF, dH, dG, dx, dw]

# Start from t=0
Y0_direct = [0.0, 0.5, 0.5, 0.0, 0.25]
sol_direct = solve_ivp(rhs_apery_t_direct, [0, 50], Y0_direct,
                       method='RK45', rtol=1e-12, atol=1e-14, max_step=0.01)

print(f"\nDirect t-domain integration from t=0:")
for T_eval in [1, 5, 10, 20, 50]:
    idx = np.argmin(np.abs(sol_direct.t - T_eval))
    F, H, G, x, w = [sol_direct.y[i, idx] for i in range(5)]
    print(f"  t={T_eval:2d}: x={x:.10f}, F={F:.15f}, err={abs(F-TARGET_F1):.4e}")

print(f"\n  Target: F(1) = {TARGET_F1:.15f}")
print(f"  ζ(3) = (5/2)·F(1) = {2.5*TARGET_F1:.15f} (should be {ZETA3:.15f})")


# The direct approach uses 1/x INSIDE the ODE solver (not polynomial),
# with a special case at x=0. This is fine numerically but NOT a PIVP.

# ===================================================================
# THE ACTUAL PIVP COMPILER SOLUTION
# ===================================================================
print("\n" + "="*60)
print("PIVP COMPILER: The actual solution")
print("="*60)
print("""
OBSERVATION: The factored system reparametrized by 1/x:
  dF/dτ = Hx(1-x)
  dH/dτ = (G-H)(1-x)
  dG/dτ = (1-x)w[1-(2+x)G]
  dw/dτ = -w²x(1-x)
  dx/dτ = x(1-x)

is a VALID polynomial PIVP with rational ICs.
The only problem: x=0 is a fixed point, so nothing moves.

BUT: if we ADD a tiny perturbation to x(0) — even ε = 10^{-100} —
the system works perfectly and computes F(1) = (2/5)ζ(3).

The PIVP compiler's job: produce a system that effectively
starts from x = ε without ε appearing in the ICs.

SOLUTION: Use a "kick" variable that starts the motion.

k' = k(1-k)           IC: k = 1/2   (k goes to 1)
x' = x(1-x) + k²(1-k)(1-x)   IC: x = 0

At τ=0: x' = 0 + (1/4)(1/2)(1) = 1/8 ≠ 0.  x starts moving!
As k→1: k²(1-k) → 0, and x' → x(1-x). Back to the original system!

The kick term k²(1-k)(1-x) is:
  - Polynomial ✓
  - Positive for k ∈ (0,1) (pushes x away from 0) ✓
  - Vanishes as k→1 (doesn't affect the limit) ✓
  - (1-x) factor ensures x doesn't overshoot 1 ✓

KEY QUESTION: Does the main computation converge to
the CORRECT F(1) despite the modified x trajectory?
""")

def rhs_kicked_apery(tau, Y):
    """
    Kicked Apéry system: polynomial PIVP with rational ICs.
    The kick moves x away from 0, then decays.
    """
    F, H, G, w, x, k = Y

    one_mx = 1 - x
    one_mk = 1 - k

    # Kick that decays: k²(1-k)(1-x)
    kick = k**2 * one_mk * one_mx

    dx = x * one_mx + kick           # x starts moving!
    dk = k * one_mk                   # k → 1

    # Main computation (same as before)
    dF = H * x * one_mx
    dH = (G - H) * one_mx
    dG = one_mx * w * (1 - (2+x)*G)
    dw = -w**2 * (x * one_mx + kick)  # w = 1/(4+x), need to track (4+x)' = x'

    return [dF, dH, dG, dw, dx, dk]

# ICs: F=0, H=1/2, G=1/2, w=1/4, x=0, k=1/2
Y0_kicked = [0.0, 0.5, 0.5, 0.25, 0.0, 0.5]

sol_kicked = solve_ivp(rhs_kicked_apery, [0, 100], Y0_kicked,
                       method='RK45', rtol=1e-12, atol=1e-14, max_step=0.01)

print(f"\nKicked Apéry system (polynomial PIVP, rational ICs):")
print(f"Target: F(1) = {TARGET_F1:.15f}")
print()
for T_eval in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]:
    idx = np.argmin(np.abs(sol_kicked.t - T_eval))
    F, H, G, w, x, k = [sol_kicked.y[i, idx] for i in range(6)]
    print(f"  τ={T_eval:5.1f}: x={x:.10f}, k={k:.8f}, F={F:.15f}, "
          f"err={abs(F-TARGET_F1):.4e}")

# Check: does the system converge to the CORRECT limit?
sol_kicked_long = solve_ivp(rhs_kicked_apery, [0, 500], Y0_kicked,
                            method='RK45', rtol=1e-13, atol=1e-15, max_step=0.01)
F_final = sol_kicked_long.y[0, -1]
x_final = sol_kicked_long.y[4, -1]
print(f"\n  τ=500: F = {F_final:.15f}")
print(f"  Target:   {TARGET_F1:.15f}")
print(f"  Error:    {abs(F_final - TARGET_F1):.6e}")
print(f"  ζ(3) computed: {2.5*F_final:.15f}")
print(f"  ζ(3) actual:   {ZETA3:.15f}")


# ===================================================================
# WAIT — the kicked system has MODIFIED the trajectory of x.
# The H, G equations assume dx/dτ = x(1-x) (the reparametrized time).
# But we changed x' by adding a kick. The chain rule is:
#   dH/dx = (G-H)/x  requires  dH/dτ = (G-H)(1-x)/x · ... hmm
#
# Actually, the reparametrized system was derived assuming
# dτ/dt = 1/x, so dx/dτ = x·dx/dt = x(1-x).
# The H equation dH/dτ = (G-H)(1-x) comes from
# dH/dt = (G-H)(1-x)/x and dτ/dt = 1/x → dH/dτ = dH/dt · dt/dτ = (G-H)(1-x)/x · x = (G-H)(1-x).
#
# If we change dx/dτ, we change the relationship between τ and t,
# so the H equation might not be correct.
#
# The KEY equations are in terms of x (not τ):
#   dF/dx = H
#   dH/dx = (G-H)/x
#   dG/dx = w[1-(2+x)G]/x
#   dw/dx = -w²
#
# To convert to τ-equations, we need dY/dτ = (dY/dx)·(dx/dτ).
# With the KICKED x': dx/dτ = x(1-x) + kick
# The correct equations become:
#   dF/dτ = H · [x(1-x) + kick]
#   dH/dτ = (G-H)/x · [x(1-x) + kick] = (G-H)(1-x) + (G-H)·kick/x  ← 1/x again!
#
# The kick reintroduces the 1/x singularity.
# ===================================================================
print("\n" + "="*60)
print("ANALYSIS: Does the kick break the system?")
print("="*60)
print("""
The kicked system modifies x' from x(1-x) to x(1-x) + kick.
But the correct chain rule requires:
  dH/dτ = (dH/dx)·(dx/dτ) = (G-H)/x · [x(1-x) + kick]
         = (G-H)(1-x) + (G-H)·kick/x

The kick term reintroduces 1/x!

This is the SAME chain-rule obstruction we saw with the τe^{-τ} kick.
Any modification to x' that doesn't vanish at x=0 will break the
polynomial structure.

CONCLUSION: The kick approach fundamentally doesn't work for
the x-reparametrized system, because the chain rule couples
x' to all other equations through 1/x.
""")

# Verify: the "kicked" system gives WRONG results because
# we used the un-modified H equation with the modified x equation.
print("Verification: kicked system error comes from chain-rule violation")
print(f"  Kicked F(500) = {F_final:.15f} (wrong)")
print(f"  True F(1)     = {TARGET_F1:.15f}")
print(f"  Discrepancy   = {abs(F_final - TARGET_F1):.6e}")
# The error should scale with the kick magnitude.


# ===================================================================
# FINAL APPROACH: The "time-reparametrized concurrent" scheme
# ===================================================================
print("\n" + "="*60)
print("THE CORRECT COMPILATION: Concurrent time-reparametrized blocks")
print("="*60)
print("""
The chain-rule problem arises from modifying x's trajectory.
Instead, we should keep x's trajectory UNTOUCHED and use a
SEPARATE auxiliary system to compute what we need.

For the Apéry system, the problem is:
  x = 1-e^{-t}, so at t=0, x=0 (singular point of ODE).

DNA32's approach: split the integral into two blocks, each
avoiding the singularity. The blocks meet at a point where
the ICs are transcendental (involve e).

PIVP compiler for DNA32:
  Block A: compute e as lim u(τ) (polynomial PIVP, u→e)
  Block B: run the DNA32 system using u in place of e

The critical insight: Block B doesn't need e as a FIXED IC.
Instead, e appears as a COEFFICIENT in Block B's ODE.
If Block B's equations depend on u (which converges to e),
and Block B uses a GATE to start slowly, then as τ→∞,
Block B effectively "sees" u=e and computes correctly.

But Block B is an INTEGRAL (accumulation), so the transient
error from u≠e during early times persists...

UNLESS Block B computes via FIXED-POINT convergence rather
than integration!

Question: Can we reformulate ζ(3) as a FIXED POINT of a
polynomial map, rather than an integral?
""")

# Check if the Apéry polynomial system converges to a fixed point
# when started from x=ε (not x=0).
# At x=1: the fixed point has x=1, F=TARGET_F1, and the system
# should settle there.

# Actually, the reparametrized system x' = x(1-x) has TWO fixed points:
# x=0 and x=1. We WANT x=1.
# If we start from x=ε > 0, x flows to 1.
# The system (F,H,G,w,x) converges to a limit as τ→∞.
# This limit IS a fixed point of the polynomial vector field!

print("\nThe Apéry system's fixed point at x=1:")
print("At x=1: all rates dY/dτ contain (1-x) factor → 0.")
print("So (F*, H*, G*, w*, 1) is a fixed point for ANY F*, H*, G*!")
print("The limit depends on the IC trajectory, NOT just the fixed point.")
print("→ This is NOT a fixed-point computation; it's an integral/accumulation.")

print("\n" + "="*60)
print("BOTTOM LINE")
print("="*60)
print(f"""
The PIVP compiler concept encounters a fundamental tension:

1. INTEGRAL-type computations (like computing F(1) = ∫₀¹ F'(x)dx):
   - Depend on the ENTIRE trajectory, not just the limit
   - Transient errors from wrong ICs persist permanently
   - Gate chains reduce but don't eliminate the error

2. FIXED-POINT computations (like u → e via u' = u(1-s)):
   - Converge to a unique attractor regardless of IC
   - Transient errors are "washed out"
   - Perfect for PIVP compilation

3. The Apéry/ζ(3) computation is TYPE 1 (integral), so the
   simple gate-chain compilation introduces permanent error.

OPEN DIRECTIONS:
(a) Find a fixed-point characterization of ζ(3)
(b) Prove that no polynomial PIVP with rational ICs converges to ζ(3)
    (this would be a major negative result: ζ(3) ∉ R_RTCRN)
(c) Find a smarter compilation that corrects transient errors
    (e.g., by running the integral TWICE and subtracting)
""")
