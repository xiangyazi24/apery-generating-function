"""
Multiple integrals → GPAC: theory and examples.

Xiang's question: is there a unified method to convert k-fold integrals
into polynomial PIVP systems?

===================================================================
GENERAL METHOD: Parametric Cube + Leibniz Cascade
===================================================================

Given: I = ∫₀¹...∫₀¹ R(x₁,...,xₖ) dx₁...dxₖ  (R rational)

Step 1: Define I(t) = ∫₀ᵗ...∫₀ᵗ R(x₁,...,xₖ) dx₁...dxₖ
        Goal: I(1) = I.

Step 2: Differentiate by Leibniz rule. I'(t) is a sum of k boundary
        integrals, each a (k-1)-fold integral.

Step 3: Each boundary integral becomes a new variable. Differentiate
        again to get (k-2)-fold integrals. Continue until 0-fold
        (point evaluations).

Step 4: The resulting system is a cascade of ODEs. If R is rational,
        the point evaluations are rational functions of t, and the
        system can be made polynomial by introducing auxiliary
        variables for 1/(denominator).

===================================================================
EXAMPLE: ζ(3) = ∫₀¹∫₀¹∫₀¹ 1/(1-xyz) dx dy dz
===================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp, dblquad, tplquad

ZETA3 = 1.2020569031595942

# === Verify: ∫∫∫ 1/(1-xyz) = ζ(3) ===
print("=== Verify triple integral ===")
# Numerical integration (slow but correct)
val, err = tplquad(lambda z, y, x: 1/(1-x*y*z + 1e-15),
                    0, 0.999, 0, 0.999, 0, 0.999)
print(f"  ∫∫∫ 1/(1-xyz) �� {val:.10f}")
print(f"  ζ(3) = {ZETA3:.10f}")
print(f"  Error: {abs(val-ZETA3):.6e}")

# === The parametric cube approach ===
# I₃(t) = ∫₀ᵗ∫₀ᵗ∫₀ᵗ 1/(1-xyz) dxdydz
#
# By symmetry in x,y,z and Leibniz rule:
#   I₃'(t) = 3 · A₂(t)
# where A₂(t) = ∫₀ᵗ∫₀ᵗ 1/(1-xyt) dxdy
#
# Computing A₂: inner integral ∫₀ᵗ dx/(1-xyt) = -ln(1-yt²)/(yt)
# So A₂(t) = ∫₀ᵗ -ln(1-yt²)/(yt) dy = Li₂(t³)/t
#
# Therefore: I₃'(t) = 3Li₂(t³)/t  → has 1/t!
#
# The polylog singularity reappears.

print("\n=== Parametric cube reduces to polylog ===")
print("I₃(t) = Li₃(t³)")
print("I₃'(t) = 3Li₂(t³)/t  — has 1/t singularity")
print("This is the SAME obstruction as the direct polylog approach.")

# ===================================================================
# ALTERNATIVE: Use a DIFFERENT parametrization to avoid 1/t
# ===================================================================
# Instead of I(t) = ∫₀ᵗ∫₀ᵗ∫₀ᵗ, what about:
# J(t) = ∫₀¹∫₀¹∫₀¹ 1/(1-txyz) dxdydz  (parameter in integrand)
#
# This is exactly Li₃(t) (parameter in front).
# J'(t) = ∫₀¹∫₀¹∫₀¹ xyz/(1-txyz)² dxdydz
#
# Can we express this as a function of J and t?

print("\n=== Alternative: parameter in integrand ===")
print("J(t) = ∫∫∫ 1/(1-txyz) = Li₃(t)")
print("J'(t) = ∫∫∫ xyz/(1-txyz)² dxdydz")
print("J(1) = ζ(3)")

# Actually: d/dt[1/(1-txyz)] = xyz/(1-txyz)²
# And ∫∫∫ xyz/(1-txyz)² dxdydz = d/dt[Li₃(t)] = Li₂(t)/t
# So J'(t) = Li₂(t)/t — 1/t again.

# ===================================================================
# KEY INSIGHT: For ∫∫∫ 1/(1-xyz), the iterated structure always
# gives Li₃, whose ODE tower inherently has 1/x singularities.
#
# But what if we use a DIFFERENT integrand for ζ(3)?
# ===================================================================

# ===================================================================
# CANDIDATE: ζ(3) via FINITE integrals without polylog singularity
# ===================================================================

# Idea: use ζ(3) = ∫₀¹∫₀¹ ln(x)ln(y)/(1-xy) ... no, still has poles.

# Better idea: use BEUKERS' integral
# ζ(3) = ∫₀¹∫₀¹ -ln(xy)/(1-xy) dxdy  ... nope, = 2ζ(3) actually
# Check: ∫₀¹∫₀¹ -ln(xy)/(1-xy) dxdy = 2ζ(3)

print("\n=== Beukers-type double integral ===")
# ζ(3) - 2 · (rational) = ... Beukers' proof used:
# ∫₀¹∫₀¹ -ln(xy)/(1-xy) · P(x,y) dx dy
# where P is a Legendre polynomial, giving rational approximations.

# For GPAC purposes, the simplest useful integral is:

# ζ(3) = Σ 1/n³ = ∫₀¹ Li₂(x)/x dx = ∫₀¹ [-ln(1-x)]²/(2x) dx ... nah

# ===================================================================
# COMPLETELY DIFFERENT APPROACH: Split-integral method
# (This is what the DNA32 paper does)
# ===================================================================
print("\n=== Split-integral method (DNA32 approach) ===")
print("""
The DNA32 paper computes ζ(3) = ∫₀^∞ t²/(e^t-1) dt via:

Block I:  ∫₀¹ t²/(e^t-1) dt    (exponential substitution → poly ODE)
Block II: ∫₁^∞ t²/(e^t-1) dt   (substitution u=e^{-t} → [0,1/e])

Both blocks are polynomial ODE systems. BUT: the ICs involve e and 1/(e-1).

The ICs are transcendental because we split at t=1, and the
exponential function at t=1 is e (transcendental).
""")

# ===================================================================
# NEW IDEA: What if we split at t=0 instead of t=1?
# Or: what if we use a different change of variables?
# ===================================================================

# The integral ∫₀^∞ t²/(e^t-1) dt:
# Near t=0: t²/(e^t-1) ≈ t²/t = t (bounded, integrable)
# Near t=∞: decays exponentially

# Substitution: let u = 1-e^{-t}, t = -ln(1-u), dt = du/(1-u)
# When t: 0→∞, u: 0→1
# ∫₀¹ [-ln(1-u)]² / (1/(1-u) - 1) · du/(1-u)
# = ∫₀¹ [-ln(1-u)]² / (u/(1-u)) · du/(1-u)
# = ∫₀¹ [-ln(1-u)]² (1-u)² / (u(1-u)) du
# = ∫₀¹ [ln(1-u)]² (1-u) / u du

# So ζ(3) = (1/2) ∫₀¹ [ln(1-u)]²(1-u)/u du

print("\n=== Substitution u = 1-e^{-t} ===")
from scipy.integrate import quad

def integrand_v1(u):
    if u < 1e-15:
        return 0
    return (np.log(1-u))**2 * (1-u) / u

val_v1, _ = quad(integrand_v1, 0, 1-1e-10)
print(f"  (1/2)∫₀¹ [ln(1-u)]²(1-u)/u du = {val_v1/2:.15f}")
print(f"  ζ(3) = {ZETA3:.15f}")
print(f"  Match: {abs(val_v1/2 - ZETA3) < 1e-6}")

# For GPAC: I(t) = (1/2)∫₀ᵗ [ln(1-u)]²(1-u)/u du
# Define v = -ln(1-t), v' = 1/(1-t), v(0) = 0
# Then [ln(1-t)]² = v²
# I'(t) = (1/2) v²(1-t)/t — has 1/t!

# Same 1/t singularity...

# ===================================================================
# The 1/t singularity seems UNIVERSAL for integrals computing ζ(3)
# ===================================================================

# Let me try yet another approach: DOUBLE integral with polynomial measure
# ζ(3) = (some factor) · ∫₀¹∫₀¹ P(x,y)/(1-xy) dx dy
# where P is chosen to make the ODE nice.

# With P = 1: ∫∫ 1/(1-xy) dxdy = Σ 1/(n+1)² = π²/6 - 1 (NOT ζ(3))
# With P = -ln(xy): gives ζ(3) (Euler)
# With P = x(1-x)y(1-y): Beukers' proof, gives ζ(3) - rational

# What about: ∫₀¹∫₀¹ x^a y^b / (1-xy)^c dx dy ?
# This is a Beta-function integral, = Γ stuff.

# ===================================================================
# EXPLORATION: Use the substitution x = sin²θ
# Generating function F(x) evaluated at x = sin²(π/6) = 1/4
# ===================================================================

# Our Apéry F: F(1) = (2/5)ζ(3)
# What about F(1/4)?
from scipy.special import comb as comb_fn
def apery_F(x, N=300):
    return sum(((-1)**(n-1)) / (n**3 * comb_fn(2*n, n, exact=True)) * x**n for n in range(1, N+1))

print(f"\n=== Apéry F at special points ===")
print(f"  F(1/4) = {apery_F(0.25):.15f}")
print(f"  F(1/4)/ζ(3) = {apery_F(0.25)/ZETA3:.15f}")
print(f"  F(1) = {apery_F(1.0):.15f}")
print(f"  F(2) = {apery_F(2.0):.15f}")
print(f"  F(4) = {apery_F(4.0):.15f}")
print(f"  F(4)/ζ(3) = {apery_F(4.0)/ZETA3:.15f}")

# What is F(4)? At x=4 (boundary of convergence):
# F(4) = Σ (-1)^{n-1} 4^n / (n³ C(2n,n))
# C(2n,n) ~ 4^n/√(πn), so terms ~ (-1)^{n-1} √(πn)/n³ → 0
# The series converges (alternating, terms → 0)

# Is F(4) a known constant? F(4)/ζ(3) ≈ ?
ratio4 = apery_F(4.0) / ZETA3
print(f"\n  F(4)/ζ(3) = {ratio4:.15f}")
# Check some simple fractions:
for p in range(1, 20):
    for q in range(1, 20):
        if abs(ratio4 - p/q) < 1e-6:
            print(f"  !! F(4)/ζ(3) ≈ {p}/{q}")

# ===================================================================
# SYSTEMATIC SEARCH for integral representations with "nice" ODEs
# ===================================================================
print("\n" + "="*60)
print("SUMMARY: Multiple Integrals → GPAC")
print("="*60)
print("""
General method (Parametric Cube + Leibniz Cascade):
  - Works in principle for any rational integrand
  - Reduces k-fold integral to cascade of ODEs
  - System is polynomial (after auxiliary variables for denominators)

For ζ(3) specifically:
  - Triple integral ∫∫∫ 1/(1-xyz) reduces to Li₃(t³)
  - ALL known integral representations have 1/t singularity at t=0
  - This seems to be a UNIVERSAL feature of ζ(3) integrals

The 1/t singularity appears because:
  - ζ(3) involves 1/n³ in its series
  - The generating function starts at n=1 (not n=0)
  - This "gap" translates to a zero at x=0, creating 1/x in the ODE

Open question: Is there an integral for ζ(3) over [0,1]^k with
a rational integrand whose GPAC ODE system has:
  (a) No singular points on the integration path, AND
  (b) Rational initial conditions?

The DNA32 approach achieves (a) but not (b).
The Apéry approach achieves (b) but not (a).
""")
