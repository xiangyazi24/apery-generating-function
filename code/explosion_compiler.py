"""
PIVP Compiler via Explosions (Xiang's idea)

Core concept:
  - An explosion (finite-time blow-up) acts as "infinity" for a sub-computation
  - First ω: explosion at t=1 → Block A runs to "∞" by t=1
  - Second ω: explosion at t=2 → Block B runs to "∞" by t=2
  - DNA32 resolution removes the explosion → bounded system
  - All compiled into a single polynomial ODE on t ∈ [0, ∞)

Key mechanism:
  v' = v² with v(0) = 1 → v = 1/(1-t), explosion at t=1
  Define effective time s₁ = ∫₀ᵗ v(τ)dτ = -ln(1-t) → ∞ as t→1⁻
  Any computation coupled to v has "infinite time" by t=1.

Resolution (DNA32): replace v by w = 1/v = 1-t.
  w' = -1 (smooth). The resolved system is smooth across t=1.
"""

import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

ZETA3 = 1.2020569031595942
E = np.e

# ===================================================================
# TEST 1: Computing e via explosion at t=1
# ===================================================================
print("="*60)
print("TEST 1: Computing e via explosion at t=1")
print("="*60)
print("""
In s₁-time (s₁ = -ln(1-t)):
  du/ds₁ = u(1-φ),  φ' = 1-φ,  u(0)=1, φ(0)=0
  → φ = 1-e^{-s₁} → 1
  → u = exp(1-e^{-s₁}) → e  ✓

Resolved (t-time, using w = 1-t instead of v = 1/(1-t)):
  u' = u(1-φ)/w = u·w/w = u     (since φ = 1-w = t)
  φ' = (1-φ)/w = 1               (since 1-φ = w)
  w' = -1

So u(t) = eᵗ, φ(t) = t, w(t) = 1-t.
At t=1: u = e ✓. But for t > 1: u = eᵗ > e. DOESN'T FREEZE.
""")

def rhs_explosion_basic(t, Y):
    """Basic explosion-based computation of e."""
    u, phi, w = Y
    # w = 1-t (resolved explosion variable)
    # Resolved equations:
    du = u  # u = e^t
    dphi = 1.0  # phi = t
    dw = -1.0  # w = 1-t
    return [du, dphi, dw]

Y0 = [1.0, 0.0, 1.0]
sol = solve_ivp(rhs_explosion_basic, [0, 3], Y0, method='RK45',
                rtol=1e-14, atol=1e-16, max_step=0.01)

print(f"{'t':>5} | {'u':>14} | {'u/e':>10} | {'w=1-t':>8}")
print("-" * 48)
for t_eval in [0.0, 0.5, 0.9, 0.99, 1.0, 1.01, 1.5, 2.0, 3.0]:
    idx = np.argmin(np.abs(sol.t - t_eval))
    u, phi, w = sol.y[0, idx], sol.y[1, idx], sol.y[2, idx]
    print(f"{t_eval:5.2f} | {u:14.8f} | {u/E:10.6f} | {w:8.4f}")

print("\n→ u passes through e at t=1 but doesn't freeze.")
print("  This is the core problem with simple explosion resolution.")


# ===================================================================
# TEST 2: Can we FREEZE u at e using a fixed-point mechanism?
# ===================================================================
print("\n" + "="*60)
print("TEST 2: Fixed-point freezing at the explosion")
print("="*60)
print("""
Idea: use u' = f(u)·v where f has a stable fixed point at e.
  In s₁-time: du/ds₁ = f(u). If f(e)=0, u → e and stays.
  In t-time (resolved): u' = f(u)/w. At w=0: 0/0 → 0 (L'Hôpital).
  For t > 1: f(u) = f(e) = 0, so u' = 0. u frozen at e! ✓

PROBLEM: f(e) = 0 requires a polynomial vanishing at e.
Since e is transcendental, no polynomial vanishes at e.

BUT: what if f involves OTHER variables?
  f(u, E) = E - u,  with E → e via a separate computation.
  Then f(e, e) = 0. ✓

This requires E to ALSO converge to e... which is what we're trying
to achieve. Circular? Not if E converges faster than u.
""")

# Logistic approach: u' = u(K-u)·v, where K will converge to e
# In s₁-time: du/ds₁ = u(K-u). Fixed point at u=K (stable if K > 0).

# We need K to converge to e. Let K satisfy:
#   K' = K·v = K/w → K = e^t (same as before)
# K doesn't freeze either!

# What if K' = K·(1-φ)·v = K(1-φ)/w = K·w/w = K → K = e^t. Same.

# Alternative: use K = e^φ where φ → 1.
# K = e^φ, K' = K·φ' = K·(1-φ)/w = K·w/w = K. Still K = e^t.

# The root cause: ANY variable satisfying dy/ds₁ = f(y) where f has
# unbounded growth has dy/dt = f(y)·v which doesn't freeze after resolution.

# EXCEPT: if f makes the variable converge to a SELF-DETERMINED fixed point.

# Example: logistic equation du/ds₁ = u(1-u). Fixed point: u = 1.
# Resolved: u' = u(1-u)/w.
# At u=1, f=0. u freezes at 1 for t > 1. ✓
# But u → 1, not e.

# For u → e, we need f(u) = 0 at u = e.
# f(u) = u(e - u) → polynomial only if e is replaced by a variable.

print("Test: logistic u' = u(1-u)·v → u → 1 and freezes")

def rhs_logistic(t, Y):
    u, w = Y
    if abs(w) < 1e-14:
        du = 0.0  # L'Hôpital limit
    else:
        du = u * (1 - u) / w
    dw = -1.0
    return [du, dw]

Y0_log = [0.5, 1.0]  # u(0) = 0.5, w = 1-t
sol_log = solve_ivp(rhs_logistic, [0, 3], Y0_log, method='RK45',
                    rtol=1e-14, atol=1e-16, max_step=0.001)

print(f"\n{'t':>5} | {'u':>14} | {'w':>10}")
print("-" * 35)
for t_eval in [0.0, 0.5, 0.9, 0.99, 1.0, 1.01, 1.5, 2.0, 3.0]:
    idx = np.argmin(np.abs(sol_log.t - t_eval))
    u, w = sol_log.y[0, idx], sol_log.y[1, idx]
    print(f"{t_eval:5.2f} | {u:14.10f} | {w:10.6f}")

print("\n→ u → 1 at t=1 and STAYS at 1 for t > 1! ✓")
print("  The logistic fixed point 'freezes' the variable.")
print("  But fixed point is 1, not e.")


# ===================================================================
# TEST 3: Two-variable "implicit" approach
# ===================================================================
print("\n" + "="*60)
print("TEST 3: Coupled system to freeze u at e")
print("="*60)
print("""
System:
  E' = E        (E = eᵗ, provides "target")
  u' = (E-u)/w  (u tracks E; at w=0, u=E=e; for t>1, u frozen?)

In s₁-time: du/ds₁ = E(s₁) - u where E(s₁) = exp(1-e^{-s₁}).
This is a LINEAR ODE: u converges to E as s₁ → ∞.
At s₁ = ∞ (t=1): u = E = e. ✓

For t > 1:
  u' = (E-u)/w = (eᵗ - u)/(1-t)
  At t=1: (e-e)/0 = 0/0. L'Hôpital: u' → -u'/(−1) = u'?
  Actually: d/dt[E-u] = E-u', d/dt[w] = -1.
  L'Hôpital: lim (E-u)/w = lim (E'-u')/(-1) = -(E'-u').
  So u' = -(E'-u'), i.e., 2u' = E' = e. u' = e/2 at t=1.
  u MOVES after t=1!
""")

def rhs_coupled(t, Y):
    E, u, w = Y
    dE = E  # E = e^t
    dw = -1.0
    if abs(w) < 1e-14:
        # L'Hôpital: u' = -(E' - u')/(-1) → 2u' = E' → u' = E'/2
        du = dE / 2.0
    else:
        du = (E - u) / w
    return [dE, du, dw]

Y0_c = [1.0, 0.0, 1.0]  # E=1, u=0, w=1
sol_c = solve_ivp(rhs_coupled, [0, 3], Y0_c, method='RK45',
                  rtol=1e-14, atol=1e-16, max_step=0.001)

print(f"\n{'t':>5} | {'E':>14} | {'u':>14} | {'u-e':>12} | {'w':>8}")
print("-" * 62)
for t_eval in [0.0, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 1.5, 2.0]:
    idx = np.argmin(np.abs(sol_c.t - t_eval))
    E_v, u_v, w_v = sol_c.y[0, idx], sol_c.y[1, idx], sol_c.y[2, idx]
    print(f"{t_eval:5.2f} | {E_v:14.8f} | {u_v:14.8f} | {u_v-E:12.6e} | {w_v:8.4f}")

print("\n→ u reaches e at t=1 but DOESN'T freeze — drifts away for t>1.")
print("  The coupled approach fails because E keeps changing after t=1.")


# ===================================================================
# TEST 4: Self-referential fixed point (the real test)
# ===================================================================
print("\n" + "="*60)
print("TEST 4: Can we build a fixed-point equation for e?")
print("="*60)
print("""
We need u → e and u frozen after explosion. This requires:
  u' = f(u) / w  where f(e) = 0 and f is POLYNOMIAL in u alone.

Since e is transcendental, no nonzero polynomial f has f(e) = 0.
This seems to be a FUNDAMENTAL OBSTRUCTION.

BUT: what if we use MULTIPLE explosions?

Explosion at t=1: computes c₁ = some algebraic function of e
Explosion at t=2: uses c₁ to compute c₂ = better approximation
...

Each explosion computes one "digit" of e?
No — each step would need to know e to set up the fixed point.

ALTERNATIVE: Abandon freezing. Instead, use the explosion
to create an EXACT snapshot of y(1) = e.

The snapshot: at t=1, define Z = y(1) - u where u → y.
Then Z → 0 at t=1. If Z stays 0 for t > 1, then u = y always.
u' = y' = E → u = e^t for t > 1. u doesn't freeze, but u = E always.

This means: u = E at all times. So if Block B reads u, it gets E(t) = e^t.
Block B sees the CURRENT E, not the snapshot E(1) = e.
""")

# ===================================================================
# TEST 5: The real question — what computation STRUCTURE works?
# ===================================================================
print("\n" + "="*60)
print("TEST 5: Using the explosion for INSTANTANEOUS computation")
print("="*60)
print("""
KEY INSIGHT: The explosion makes Block B run "infinitely fast" at t=1.
At that INSTANT, y(1) = e exactly. Block B reads the correct value!

Example: Block B computes ∫₀¹ 1/(1+s) ds = ln(2) using the
"second explosion" v₂ = 1/(2-t), which blows up at t=2.

BUT: we want Block B's result to depend on Block A's output (e).

Concrete goal: compute e·ln(2) using two explosions.

System:
  Phase 1 (effective t < 1): Compute E = e^t → e at t=1.
  Phase 2 (effective t ∈ (1,2)): Compute ∫₀¹ E/(1+s) ds using
    the explosion at t=2 as Phase 2's "infinity".

The "Phase 2 explosion" v₂ = 1/(2-t):
  Effective time s₂ = -ln(2-t).
  At t=1: s₂ = 0 (Phase 2 starts).
  At t=2⁻: s₂ → ∞ (Phase 2 "finishes").

Phase 2 in s₂-time:
  dI/ds₂ = E(t(s₂)) · (1-xB) / (1+xB)
  dxB/ds₂ = 1-xB
  → xB = 1-e^{-s₂} → 1
  → I = ∫₀^{xB} E(t(s)) / (1+s) ds

The problem: E(t) = e^t is changing during Phase 2.
At t=1: E = e. At t=2: E = e². E ≈ e·e^{t-1} during Phase 2.

If Phase 2 is "fast" (explosion at t=2 compresses infinite time into
the instant before t=2), then E is approximately constant at E(2) = e².
Wrong value!

Actually, the explosion at t=2 means Phase 2's effective time
is concentrated near t=2, where E = e². So Block B computes
e²·ln(2), not e·ln(2).

To get the correct value, Block B must run at t=1, not t=2.
→ Use the SAME explosion (t=1) for both blocks,
   but with sequential sub-computations.
""")

# ===================================================================
# TEST 6: Sequential sub-computations within ONE explosion
# ===================================================================
print("="*60)
print("TEST 6: Sequential sub-computations in one explosion")
print("="*60)
print("""
Within the explosion at t=1, the effective time s₁ → ∞.
We can sequence TWO computations in s₁-time:

Sub-computation A (s₁ ∈ [0, ∞)):
  E → e (converges in finite s₁-time)

Sub-computation B (starts after A converges):
  Uses E ≈ e to compute Block B

But how to "sequence" within s₁-time? Use a gate chain!
  g' = E²(1-g)   (gate opens after E converges)
  Block B gated by g

Wait — this is exactly the gate chain approach from before,
just in s₁-time instead of τ-time. And it has the SAME problem:
gate chains give approximate, not exact results.

Unless... we use ANOTHER explosion within s₁-time?

Meta-explosion: within s₁-time, have a variable that blows up
at s₁ = T₁, creating a "second level infinity" within the first.

In t-time: s₁ = -ln(1-t). A variable blowing up at s₁ = T₁ means
it blows up at t = 1-e^{-T₁}. This is a REAL explosion at a time
before t=1!

System:
  w₁ = 1-t (explosion 1 at t=1)
  w₂ = (1-e^{-T₁}) - t (explosion 2 at t=t₂ < 1)

  In t-time: w₂ = t₂ - t, w₂' = -1. Smooth.
  At t=t₂: Block A finishes (second-level "infinity").
  For t ∈ (t₂, 1): Block B runs, with infinite effective time (from w₁).
  At t=1: Block B finishes.

BUT: t₂ must be chosen. It depends on when Block A converges.
And the PIVP can't adaptively choose t₂.

Unless t₂ is a FIXED rational number. E.g., t₂ = 1/2.

Explosion at t=1/2: Block A computes E → exp(1/2) = √e ≈ 1.6487
Explosion at t=1: Block B uses E to compute...

But E at t=1/2 is not e, it's √e. Not useful.

HOWEVER: if we track E = e^t:
  E(1/2) = √e
  We ALSO track E² = e^{2t}: at t=1/2, E² = e. ✓!

So E² reaches e at t=1/2. If we use the explosion at t=1/2 to
freeze E² at e, then Block B (running in [1/2, 1] effective time)
can use the frozen E² = e. ✓?

Let's test: E' = E, E(0)=1, and V = E² satisfies V' = 2EE' = 2E² = 2V.
V(t) = e^{2t}. V(1/2) = e. V(1) = e². Doesn't freeze.

For freezing at t=1/2: need f(V)/w₂ where f(e) = 0, w₂ = 1/2-t.
Same problem: f polynomial, f(e) = 0 impossible.
""")

# ===================================================================
# NUMERICAL TEST: Two-explosion system (t=1/2 and t=1)
# ===================================================================
print("\n--- Numerical test: two explosions ---")

def rhs_two_explosions(t, Y):
    """
    Two explosions: at t=1/2 (w₁=1/2-t) and t=1 (w₂=1-t).

    E = e^{2t} → e at t=1/2, e² at t=1.

    Phase 1 (t ∈ [0, 1/2)): E converges to e in effective time of 1st explosion.
    Resolved: E' = 2E (smooth, but doesn't freeze).

    We test whether gating Block B by the first explosion helps.
    Block B: compute ∫₀¹ E/(1+x) dx = E·ln(2) at the instant E = e.
    """
    E, I, xB, w1, w2 = Y

    dE = 2 * E   # E = e^{2t}
    dw1 = -1.0   # w1 = 1/2 - t
    dw2 = -1.0   # w2 = 1 - t

    # Block B runs using second explosion (w2 = 1-t)
    # Rate factor: 1/w2 (infinite at t=1)
    # But we also want to delay until after t=1/2
    # Use w1 as a "phase indicator": w1 > 0 for t < 1/2, w1 < 0 for t > 1/2

    # Polynomial rate that's ~0 for t < 1/2 and ~1/w2 for t > 1/2:
    # Can't do exact step function with polynomials.

    # Just run Block B with explosion at t=1:
    if abs(w2) < 1e-14:
        dI = 0.0
        dxB = 0.0
    else:
        one_mxB = 1 - xB
        dI = E * one_mxB / ((1 + xB) * w2)
        dxB = one_mxB / w2

    return [dE, dI, dxB, dw1, dw2]

Y0_2e = [1.0, 0.0, 0.0, 0.5, 1.0]
sol_2e = solve_ivp(rhs_two_explosions, [0, 0.9999], Y0_2e,
                   method='RK45', rtol=1e-13, atol=1e-15, max_step=0.001)

target = E * np.log(2)
print(f"\nTarget: e·ln(2) = {target:.15f}")
print(f"\n{'t':>6} | {'E=e^2t':>12} | {'xB':>10} | {'I':>14} | {'err(I-e·ln2)':>14}")
print("-" * 65)
for t_eval in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]:
    idx = np.argmin(np.abs(sol_2e.t - t_eval))
    E_v, I_v, xB_v = sol_2e.y[0, idx], sol_2e.y[1, idx], sol_2e.y[2, idx]
    err = abs(I_v - target)
    print(f"{t_eval:6.4f} | {E_v:12.6f} | {xB_v:10.6f} | {I_v:14.10f} | {err:14.6e}")

# What does Block B actually compute?
# dI/dxB = I'/xB' = E/(1+xB). Since E = e^{2t} and t is changing:
# I = ∫₀^{xB} E(t(s))/(1+s) ds where t(s) is the time at which xB = s.
# Since xB = 1-e^{-s₂} where s₂ = -ln(1-t), we have xB(t) = t.
# Wait, xB' = (1-xB)/(1-t), so dxB/(1-xB) = dt/(1-t),
# ln(1-xB) = ln(1-t), so xB = t. ← xB = t!
# So I = ∫₀ᵗ e^{2s}/(1+s) ds. At t→1⁻: I = ∫₀¹ e^{2s}/(1+s) ds.

I_exact = np.exp(2) * np.log(2)  # Wrong, let me compute properly
from scipy.integrate import quad
I_actual, _ = quad(lambda s: np.exp(2*s)/(1+s), 0, 1)
print(f"\n  I(1⁻) converges to: ∫₀¹ e^{{2s}}/(1+s) ds = {I_actual:.15f}")
print(f"  This is NOT e·ln(2) = {target:.15f}")
print(f"  Because E = e^{{2t}} varies during integration!")

# The CORRECT integral requires E = e (constant) during integration:
# ∫₀¹ e/(1+s) ds = e·ln(2)
# But E = e^{2t} varies from 1 to e² during t ∈ [0,1].

print("""
DIAGNOSIS: Block B reads E(t) = e^{2t}, which varies during the
integration period. Even with the explosion giving "infinite time"
at t=1, the integration happens BEFORE t=1 (as xB grows from 0 to 1).

The integral is ∫₀¹ E(t(s))/(1+s) ds where t(s) varies,
so E(t(s)) ≠ e throughout.

CORE OBSTRUCTION: In a concurrent ODE system, Block B cannot
read a "snapshot" of E at a specific time — it reads E's
CURRENT value, which changes during Block B's integration.
""")


# ===================================================================
# FINAL ANALYSIS: What structure would work?
# ===================================================================
print("="*60)
print("ANALYSIS: Requirements for zero-error compilation")
print("="*60)
print("""
For Block B to compute EXACTLY using Block A's converged value:

OPTION 1: Freeze Block A's output.
  Requires: a polynomial function f with f(e) = 0.
  Impossible: e is transcendental.

OPTION 2: Make Block B's computation INSTANTANEOUS.
  Block B runs "infinitely fast" at the exact instant when E = e.
  This IS what the explosion provides! The explosion at t=1 gives
  infinite effective time concentrated at the INSTANT t=1.

  BUT: Block B's integral accumulates over a range, not an instant.
  Even with "infinite speed" at t=1, the INTEGRATION VARIABLE xB
  sweeps from 0 to 1 DURING the explosion, and E changes during
  this sweep.

OPTION 3: Decouple the integration variable from real time.
  If Block B's integration variable xB is INDEPENDENT of the
  explosion variable, then xB can sweep 0→1 while E stays at e.

  This requires: xB' does NOT depend on the explosion speed v.
  But then Block B's integral doesn't benefit from the explosion.

OPTION 4: Design Block B as a FIXED-POINT computation.
  Block B converges to e·ln(2) as a fixed point, not an integral.
  The fixed point depends on E's current value.
  Near t=1 (explosion), Block B has infinite time to reach
  the fixed point e·ln(2) (with E = e at that instant).
  After t=1, E changes, but Block B is already at the fixed point
  and tracks the new one... which is different from e·ln(2).

  Unless Block B's fixed point is INDEPENDENT of E for t > 1.
  This requires Block B's coupling to E to turn off after t=1.
  Which requires a polynomial switch — back to Option 1.

CONCLUSION:
  The explosion approach gives "infinite effective time" at a point.
  But for integral-type computations, the integrand varies with E(t),
  and the explosion doesn't help because E changes during integration.

  For fixed-point computations, the explosion converges to the right
  value at t=1, but the fixed point shifts when E changes for t>1.

  Both issues stem from the same root cause:
  E(t) = eᵗ passes through e at t=1 but doesn't freeze.

  The inability to freeze a transcendental value in polynomial ODEs
  appears to be the fundamental obstruction.

XIANG'S QUESTION TO INVESTIGATE:
  Can the DNA32 resolution technique do more than just remove the
  explosion? Can it ALSO freeze variables at their explosion-time
  values?

  The DNA32 paper uses time reparametrization (dτ/dt = 1/v^k).
  This stretches the explosion to τ → ∞ and makes the system bounded.
  But in τ-time, we're back to having ONE infinity, and the variable
  just converges as τ → ∞ (standard PIVP).

  Is there a DIFFERENT resolution that preserves the "finite time
  convergence" property while keeping the system bounded and polynomial?
""")
