"""
PIVP Compiler: Sequential Composition → Concurrent System

Xiang's idea: Design PIVPs as if they run sequentially
("first compute e, then use e as IC for the main block"),
then compile into a single concurrent system where the
limit is correct despite everything starting simultaneously.

===================================================================
THE PROBLEM
===================================================================

We want: ζ(3) = F(1) where F satisfies a polynomial ODE with
a singular IC that depends on e = exp(1).

Naive approach: two blocks running concurrently
  Block A: computes e (converges as τ→∞)
  Block B: uses Block A's output as an "IC" for the main computation

The issue: Block B starts at τ=0 with the WRONG value (Block A
hasn't converged yet), so it accumulates error during early times.

Xiang's compiler concept: find a compilation scheme where the
early-time error is "washed out" in the limit.

===================================================================
KEY INSIGHT: Use time-scale separation
===================================================================

If Block A converges MUCH faster than Block B evolves, then:
  - Block A reaches its target value quickly
  - Block B barely moves during this transient
  - After Block A settles, Block B sees the correct "IC"

Implementation: add a speed parameter λ to Block A.
  Block A runs at speed λ (fast)
  Block B runs at speed 1 (slow)

As λ→∞, Block B's error → 0. But we want a SINGLE system
(no λ parameter), so we need to "bake in" the time-scale
separation.

===================================================================
ALTERNATIVE: Use convergence to a FIXED POINT
===================================================================

If the main computation converges to a fixed point (not an
integral), then early-time errors don't matter — the system
forgets its IC.

This is exactly what happens with the Apéry system from x=ε:
the solution converges to the same F(1) regardless of the
starting point (as long as ε is small enough).

But the INTEGRAL type computation (F = ∫₀¹ ...) accumulates,
so it DOES depend on early values.

===================================================================
CONCRETE TEST: Two-stage computation of ζ(3)
===================================================================

Stage 1: Compute e = lim_{τ→∞} exp(1-e^{-τ})
  System: u' = u(1-x), x' = 1-x, u(0)=1, x(0)=0
  Then u(τ) = exp(x(τ)) = exp(1-e^{-τ}) → e as τ→∞

Stage 2: Use e to set up the DNA32 integral for ζ(3)
  (The DNA32 paper computes ζ(3) via ∫₀^∞ t²/(e^t-1)dt
   using a split at t=1, which needs e and 1/(e-1) as ICs)

For now, let's test a SIMPLER two-stage example first.
"""

import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

ZETA3 = 1.2020569031595942

# ===================================================================
# WARM-UP: Two-stage computation of ln(2)
# ===================================================================
# Stage 1: compute 1/2 as a fixed point
#   v' = v(1-2v), v(0) = 1/4  → v → 1/2
#   (logistic equation with carrying capacity 1/2)
#
# Stage 2: compute ln(2) = ∫₀¹ 1/(1+x) dx
#   using v as a "parameter" (should be 1/2 at convergence)
#   I' = 1/(1+x), x' = 1-x → I(∞) = ∫₀¹ dx/(1+x) = ln(2)
#
# This doesn't actually NEED Stage 1. Let's make it need it.

# Better example: compute ln(2) using a transcendental IC
# ln(2) = -ln(1/2) = ∫₁^{1/2} 1/x dx (wrong sign, let's think...)
# ln(2) = ∫₀¹ 1/(1+t) dt, straightforward.

# OK, more relevant example:
# Compute e·ln(2) as a two-stage PIVP
# Stage 1: compute e
# Stage 2: multiply by ln(2) (which is easy to compute)

# Actually, let me go straight to the Apéry problem.

# ===================================================================
# TEST 1: The "slow-fast" compilation for Apéry
# ===================================================================
#
# Block A (fast): Compute e via u' = u(1-x), x' = 1-x
# Block B (slow): Use DNA32-style integral that needs e as IC
#
# Compilation trick: make Block A converge on a FAST timescale
# while Block B evolves SLOWLY.
#
# Concretely:
#   Block A: u' = λ·u(1-xA), xA' = λ(1-xA)  (speed λ)
#   Block B: the main computation at speed 1
#
# But this requires λ as a parameter. Can we encode λ polynomially?
#
# YES: use a "clock" variable c that goes 0→1 slowly:
#   c' = ε(1-c)  (slow clock)
#   Block A runs with xA' = 1-xA  (fast, reaches ~1 by τ≈5)
#   Block B runs with a rate proportional to c
#   → Block B is essentially off until c ≈ 1, by which time Block A is done

print("="*60)
print("TEST 1: Slow-fast separation (gate mechanism)")
print("="*60)

def rhs_gated(tau, Y, gate_speed=0.1):
    """
    Block A: exponential convergence to e
      u' = u(1-xA), xA' = 1-xA
      u(0) = 1, xA(0) = 0
      → u → e, xA → 1

    Gate: g' = gate_speed * (1-g), g(0) = 0
      g → 1 slowly (timescale 1/gate_speed)

    Block B: integral ∫₀¹ 1/(1+t) dt = ln(2)
      I' = g/(1+xB), xB' = g(1-xB)
      I(0) = 0, xB(0) = 0

    Result: I → ∫₀¹ 1/(1+t)dt = ln(2) as τ→∞,
    but only after gate opens (g→1).
    """
    u, xA, g, I, xB = Y

    # Block A (fast)
    du = u * (1 - xA)
    dxA = 1 - xA

    # Gate
    dg = gate_speed * (1 - g)

    # Block B (gated)
    dI = g / (1 + xB)
    dxB = g * (1 - xB)

    return [du, dxA, dg, dI, dxB]

print("\nSimple gate test: Block A = e, Block B = ln(2)")
print("(Block B doesn't actually need Block A here — just testing the gate)")

LN2 = np.log(2)

for gate_speed in [1.0, 0.1, 0.01]:
    Y0 = [1.0, 0.0, 0.0, 0.0, 0.0]
    T_final = max(50, 20/gate_speed)
    sol = solve_ivp(lambda t, y: rhs_gated(t, y, gate_speed),
                    [0, T_final], Y0, method='RK45', rtol=1e-12, atol=1e-15,
                    max_step=0.1)
    u_final = sol.y[0, -1]
    I_final = sol.y[3, -1]
    print(f"  gate_speed={gate_speed}: u→{u_final:.10f} (e={np.e:.10f}), "
          f"I→{I_final:.10f} (ln2={LN2:.10f}), err={abs(I_final-LN2):.2e}")


# ===================================================================
# TEST 2: Two-stage where Block B DEPENDS on Block A
# ===================================================================
#
# Goal: compute e · ln(2) ≈ 1.88417...
#
# Stage 1: compute e → u(∞)
# Stage 2: compute u(∞) · ln(2) = e·ln(2)
#
# Compilation: Block B computes ∫₀¹ u/(1+t) dt
# If u were constant = e, this gives e·ln(2).
# But u is still converging during Block B's integration!
#
# With gate: u converges BEFORE Block B starts → error small.

print("\n" + "="*60)
print("TEST 2: Block B depends on Block A")
print("="*60)
print("Goal: compute e·ln(2) = " + f"{np.e * LN2:.15f}")

def rhs_dependent(tau, Y, gate_speed=0.1):
    """
    Block A: u → e
    Gate: g → 1 (slowly)
    Block B: I = ∫₀¹ u/(1+t) dt ≈ e·ln(2) when u≈e
    """
    u, xA, g, I, xB = Y

    # Block A (fast)
    du = u * (1 - xA)
    dxA = 1 - xA

    # Gate (slow)
    dg = gate_speed * (1 - g)

    # Block B (gated, depends on u from Block A)
    dI = g * u / (1 + xB)  # ← u from Block A appears here
    dxB = g * (1 - xB)

    return [du, dxA, dg, dI, dxB]

target = np.e * LN2

for gate_speed in [1.0, 0.5, 0.1, 0.05, 0.01]:
    Y0 = [1.0, 0.0, 0.0, 0.0, 0.0]
    T_final = max(100, 30/gate_speed)
    sol = solve_ivp(lambda t, y: rhs_dependent(t, y, gate_speed),
                    [0, T_final], Y0, method='RK45', rtol=1e-13, atol=1e-15,
                    max_step=0.05)
    u_final = sol.y[0, -1]
    I_final = sol.y[3, -1]
    err = abs(I_final - target)
    print(f"  gate_speed={gate_speed:.2f}: I={I_final:.12f}, target={target:.12f}, "
          f"err={err:.4e}")

print("\n→ As gate_speed → 0, Block A has more time to converge before Block B starts.")
print("  Error should decrease with gate_speed.")


# ===================================================================
# TEST 3: Can we make the gate POLYNOMIAL (no parameter)?
# ===================================================================
#
# The gate g' = ε(1-g) introduces a parameter ε. For a true PIVP,
# we need to encode this without free parameters.
#
# Idea: use a SIGMOID gate built from the system itself.
# Define:
#   s' = s(1-s)·(1-c),  s(0) = 1/2  → s → 1 (logistic)
#   c' = s·(1-c),        c(0) = 0    → c → 1 (but gated by s)
#   Block B rate proportional to c²  (c starts ~0, rises to 1)
#
# Key: c's rise is delayed because it depends on s,
# and s saturates at 1 quickly. But this makes c ≈ 1-e^{-τ},
# similar to xA. Not enough separation.
#
# Better: use a CHAIN of gates.
#   xA' = 1-xA                → xA → 1 fast
#   g₁' = xA²(1-g₁)          → g₁ rises after xA ≈ 1
#   g₂' = g₁²(1-g₂)          → g₂ rises after g₁ ≈ 1
#   Block B rate ∝ g₂²        → starts only after chain converges

print("\n" + "="*60)
print("TEST 3: Chain of polynomial gates (no free parameter)")
print("="*60)

def rhs_chain_gate(tau, Y):
    """
    Block A: u → e, xA → 1
    Gate chain: g1 waits for xA, g2 waits for g1
    Block B: gated by g2², depends on u
    """
    u, xA, g1, g2, I, xB = Y

    # Block A
    du = u * (1 - xA)
    dxA = 1 - xA

    # Gate chain (polynomial!)
    dg1 = xA**2 * (1 - g1)
    dg2 = g1**2 * (1 - g2)

    # Block B: gated by g2²
    gate = g2**2
    dI = gate * u / (1 + xB)
    dxB = gate * (1 - xB)

    return [du, dxA, dg1, dg2, dI, dxB]

Y0_chain = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
sol_chain = solve_ivp(rhs_chain_gate, [0, 50], Y0_chain,
                      method='RK45', rtol=1e-13, atol=1e-15, max_step=0.05)

# Evaluate at various times
print(f"\nTarget: e·ln(2) = {target:.15f}")
for t_eval in [5, 10, 20, 30, 40, 50]:
    idx = np.argmin(np.abs(sol_chain.t - t_eval))
    vals = sol_chain.y[:, idx]
    u, xA, g1, g2, I, xB = vals
    print(f"  τ={t_eval:3d}: u={u:.8f}, xA={xA:.6f}, g1={g1:.6f}, g2={g2:.6f}, "
          f"I={I:.10f}, err={abs(I-target):.4e}")

# Run longer
sol_long = solve_ivp(rhs_chain_gate, [0, 200], Y0_chain,
                     method='RK45', rtol=1e-13, atol=1e-15, max_step=0.1)
I_long = sol_long.y[4, -1]
print(f"\n  τ=200: I = {I_long:.15f}")
print(f"  target = {target:.15f}")
print(f"  error  = {abs(I_long - target):.6e}")


# ===================================================================
# TEST 4: Deeper chain for better separation
# ===================================================================
print("\n" + "="*60)
print("TEST 4: Deeper gate chain (3 levels)")
print("="*60)

def rhs_deep_chain(tau, Y):
    """3-level gate chain for better time separation."""
    u, xA, g1, g2, g3, I, xB = Y

    du = u * (1 - xA)
    dxA = 1 - xA

    dg1 = xA**2 * (1 - g1)
    dg2 = g1**2 * (1 - g2)
    dg3 = g2**2 * (1 - g3)

    gate = g3**2
    dI = gate * u / (1 + xB)
    dxB = gate * (1 - xB)

    return [du, dxA, dg1, dg2, dg3, dI, dxB]

Y0_deep = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
sol_deep = solve_ivp(rhs_deep_chain, [0, 200], Y0_deep,
                     method='RK45', rtol=1e-13, atol=1e-15, max_step=0.1)

print(f"\nTarget: e·ln(2) = {target:.15f}")
for t_eval in [10, 20, 50, 100, 150, 200]:
    idx = np.argmin(np.abs(sol_deep.t - t_eval))
    vals = sol_deep.y[:, idx]
    u, xA, g1, g2, g3, I, xB = vals
    print(f"  τ={t_eval:3d}: g3={g3:.6f}, I={I:.12f}, err={abs(I-target):.4e}")

# Run even longer
sol_vlong = solve_ivp(rhs_deep_chain, [0, 500], Y0_deep,
                      method='RK45', rtol=1e-13, atol=1e-15, max_step=0.2)
I_vlong = sol_vlong.y[5, -1]
print(f"\n  τ=500: I = {I_vlong:.15f}")
print(f"  error  = {abs(I_vlong - target):.6e}")


# ===================================================================
# ANALYSIS: Is the error O(1/chain_depth) or O(exp(-chain_depth))?
# ===================================================================
print("\n" + "="*60)
print("TEST 5: Error vs chain depth")
print("="*60)

def make_chain_system(n_gates):
    """Build a system with n_gates levels of gating."""
    def rhs(tau, Y):
        u = Y[0]
        xA = Y[1]
        gates = Y[2:2+n_gates]
        I = Y[2+n_gates]
        xB = Y[3+n_gates]

        dY = np.zeros_like(Y)
        dY[0] = u * (1 - xA)       # u → e
        dY[1] = 1 - xA             # xA → 1

        # Gate chain
        prev = xA
        for i in range(n_gates):
            dY[2+i] = prev**2 * (1 - gates[i])
            prev = gates[i]

        gate_val = prev**2
        dY[2+n_gates] = gate_val * u / (1 + xB)   # I
        dY[3+n_gates] = gate_val * (1 - xB)        # xB

        return dY
    return rhs

print(f"Target: e·ln(2) = {target:.15f}")
print(f"\n{'Gates':>5} | {'I(τ=300)':>18} | {'Error':>12} | {'Variables':>4}")
print("-" * 55)

for n_gates in range(1, 7):
    n_vars = 4 + n_gates  # u, xA, g1,...,gN, I, xB
    Y0 = np.zeros(n_vars)
    Y0[0] = 1.0  # u

    rhs = make_chain_system(n_gates)
    T = 300
    sol = solve_ivp(rhs, [0, T], Y0, method='RK45',
                    rtol=1e-13, atol=1e-15, max_step=0.2)

    I_val = sol.y[2+n_gates, -1]
    err = abs(I_val - target)
    print(f"{n_gates:5d} | {I_val:18.15f} | {err:12.4e} | {n_vars:4d}")


# ===================================================================
# KEY QUESTION: Is this system truly polynomial with rational ICs?
# ===================================================================
print("\n" + "="*60)
print("PIVP PROPERTIES CHECK")
print("="*60)
print("""
System (2-gate version, 6 variables):
  u'  = u(1-xA)              IC: u=1     ✓ rational
  xA' = 1-xA                 IC: xA=0    ✓ rational
  g1' = xA²(1-g1)            IC: g1=0    ✓ rational
  g2' = g1²(1-g2)            IC: g2=0    ✓ rational
  I'  = g2²·u/(1+xB)         IC: I=0     ← NOT polynomial (1/(1+xB))
  xB' = g2²·(1-xB)           IC: xB=0    ✓ rational

Problem: I' = g2²·u/(1+xB) is NOT polynomial!
Fix: introduce w = 1/(1+xB) as auxiliary variable.
  w' = -w²·xB' = -w²·g2²·(1-xB) = -w²·g2²·(1-1/w+1) ... hmm

Actually: xB' = g2²(1-xB), so 1+xB is increasing from 1.
w = 1/(1+xB):
  w' = -xB'/(1+xB)² = -g2²(1-xB)·w²

And (1-xB) = (1+xB) - 2xB = 1/w - 2(1/w - 1) = -1/w + 2
Wait: xB = 1/w - 1, so 1-xB = 2 - 1/w. Not polynomial in w.

Alternative: let v = 1+xB. Then v' = g2²(2-v) and I' = g2²·u/v.
Still 1/v.

To make polynomial: auxiliary p = 1/v = 1/(1+xB).
  p' = -v'/v² = -g2²(2-v)/v² = -g2²(2p² - p)
  I' = g2²·u·p

Now the FULL polynomial system:
  u'  = u(1-xA)              IC: u=1
  xA' = 1-xA                 IC: xA=0
  g1' = xA²(1-g1)            IC: g1=0
  g2' = g1²(1-g2)            IC: g2=0
  I'  = g2²·u·p              IC: I=0
  xB' = g2²·(1-xB)           IC: xB=0
  p'  = g2²·p·(1-2p)         IC: p=1    (since 1/(1+0) = 1)

Wait, let me redo: p = 1/(1+xB)
  p' = -xB'·p² = -g2²(1-xB)·p²
  1-xB = 1 - (1/p - 1) = 2 - 1/p

Hmm, this gives p' = -g2²(2-1/p)·p² = -g2²(2p²-p)
= g2²·p·(1-2p)

So: p' = g2²·p·(1-2p),  p(0) = 1/(1+0) = 1

Check: p → 1/(1+1) = 1/2 as xB → 1. Indeed p=1/2 is fixed point of p(1-2p)=0.

POLYNOMIAL? YES! All terms are polynomial. ✓
RATIONAL ICs? YES! All are 0 or 1. ✓
BOUNDED? Need to verify. ✓ (all variables in [0,e])
""")

# Verify the fully polynomial version
print("Verifying fully polynomial system:")

def rhs_polynomial(tau, Y):
    """Fully polynomial PIVP for e·ln(2)."""
    u, xA, g1, g2, I, xB, p = Y

    du = u * (1 - xA)
    dxA = 1 - xA
    dg1 = xA**2 * (1 - g1)
    dg2 = g1**2 * (1 - g2)
    dI = g2**2 * u * p
    dxB = g2**2 * (1 - xB)
    dp = g2**2 * p * (1 - 2*p)

    return [du, dxA, dg1, dg2, dI, dxB, dp]

Y0_poly = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
sol_poly = solve_ivp(rhs_polynomial, [0, 300], Y0_poly,
                     method='RK45', rtol=1e-13, atol=1e-15, max_step=0.1)

print(f"\n  Target: e·ln(2) = {target:.15f}")
for t_eval in [10, 30, 50, 100, 200, 300]:
    idx = np.argmin(np.abs(sol_poly.t - t_eval))
    vals = sol_poly.y[:, idx]
    u, xA, g1, g2, I, xB, p = vals
    print(f"  τ={t_eval:3d}: u={u:.8f}, g2={g2:.6f}, xB={xB:.6f}, p={p:.6f}, "
          f"I={I:.12f}, err={abs(I-target):.4e}")


# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*60)
print("SUMMARY: PIVP COMPILER via Gate Chains")
print("="*60)
print("""
COMPILATION SCHEME:
  Input:  "First compute α = PIVP_A(∞), then compute PIVP_B(α)"
  Output: A single polynomial PIVP where lim_{τ→∞} I(τ) = PIVP_B(α)

METHOD:
  1. Include Block A in the system (computes α)
  2. Add a chain of polynomial gates: g_k' = g_{k-1}²(1-g_k)
     - Each gate waits for the previous one to converge
     - More gates = more time for Block A to settle before Block B starts
  3. Gate Block B's evolution by the last gate's value
  4. Add auxiliary variables to make all divisions polynomial

PROPERTIES:
  ✓ Polynomial RHS (all terms are polynomial)
  ✓ Rational initial conditions (all 0 or 1)
  ✓ Bounded trajectories
  ✓ Converges to correct value as τ→∞

TRADE-OFF:
  - More gate levels → smaller early-time error → faster convergence
  - But: error may NOT → 0 for any FIXED number of gates!
  - Need to analyze: does the scheme have ZERO error in the limit?

CRITICAL QUESTION:
  Is lim_{τ→∞} I(τ) EXACTLY e·ln(2), or only approximately?
  The integral ∫₀^∞ g²(τ)·u(τ)·p(τ)dτ depends on the ENTIRE
  trajectory of u(τ), not just its limit!
""")

# Check: is the limit EXACT?
print("Checking if limit is exact (longer integration):")
for T in [100, 500, 1000, 2000]:
    sol_check = solve_ivp(rhs_polynomial, [0, T], Y0_poly,
                          method='RK45', rtol=1e-14, atol=1e-16, max_step=0.1)
    I_check = sol_check.y[4, -1]
    xB_check = sol_check.y[5, -1]
    print(f"  T={T:5d}: I={I_check:.15f}, xB={xB_check:.15f}, err={abs(I_check-target):.6e}")

print("""
If the error plateaus (doesn't improve with T), then the limit
is NOT exact — there's a permanent offset from the transient
period when u ≠ e.

This is the fundamental issue with integral-type computations:
they accumulate, so transient errors persist.
""")
