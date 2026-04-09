"""
Test Xiang's idea: add a τe^{-τ} kick to dx/dτ to escape the fixed point.

The reparametrized system WITH kick:
  dy0/dτ = y1*(1-x)*x²*(4+x)
  dy1/dτ = y2*(1-x)*x²*(4+x)
  dy2/dτ = (1-x)*[1 - x*(10+3x)*y2 - (2+x)*y1]
  dx/dτ  = (1-x)*x²*(4+x) + p       <-- kick added here
  ds/dτ  = 1                          (auxiliary: τ variable)
  du/dτ  = -u                         (auxiliary: e^{-τ})
  dp/dτ  = u - p                      (auxiliary: p = τe^{-τ})

ICs: y0=0, y1=1/2, y2=-1/24, x=0, s=0, u=1, p=0  (all rational!)

Question: does y0 → (2/5)ζ(3)?
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import comb

ZETA3 = 1.2020569031595942
TARGET = (2.0/5) * ZETA3  # F(1) = 0.48082276...

# Reference: exact series values for comparison
def apery_series(x, N=300):
    s = 0.0
    for n in range(1, N+1):
        s += ((-1)**(n-1)) / (n**3 * comb(2*n, n, exact=True)) * x**n
    return s

def apery_deriv1(x, N=300):
    s = 0.0
    for n in range(1, N+1):
        s += ((-1)**(n-1)) / (n**2 * comb(2*n, n, exact=True)) * x**(n-1)
    return s

def apery_deriv2(x, N=300):
    s = 0.0
    for n in range(2, N+1):
        s += ((-1)**(n-1)) * (n-1) / (n**2 * comb(2*n, n, exact=True)) * x**(n-2)
    return s

# ============================================================
# System A: Kicked system (kick on x only, y equations unchanged)
# ============================================================
def rhs_kicked(tau, Y):
    y0, y1, y2, x, s, u, p = Y

    x2_4x = x**2 * (4 + x)
    one_mx = 1 - x

    dy0 = y1 * one_mx * x2_4x
    dy1 = y2 * one_mx * x2_4x
    dy2 = one_mx * (1 - x*(10 + 3*x)*y2 - (2 + x)*y1)
    dx  = one_mx * x2_4x + p   # <-- kick
    ds  = 1.0
    du  = -u
    dp  = u - p

    return [dy0, dy1, dy2, dx, ds, du, dp]

# ============================================================
# System B: Kicked system, y equations also use dx/dτ (consistent)
# But dy2 equation would need 1/[x²(4+x)] — let's test what happens
# if we just use the polynomial approximation dy2 = (1-x)*N + p*N_approx
# Actually, let's test the "inconsistent" version A first.
# ============================================================

# ============================================================
# System C: No kick, start from x=ε with exact series ICs (baseline)
# ============================================================
def rhs_reparam(tau, Y):
    y0, y1, y2, x = Y

    x2_4x = x**2 * (4 + x)
    one_mx = 1 - x

    dy0 = y1 * one_mx * x2_4x
    dy1 = y2 * one_mx * x2_4x
    dy2 = one_mx * (1 - x*(10 + 3*x)*y2 - (2 + x)*y1)
    dx  = one_mx * x2_4x

    return [dy0, dy1, dy2, dx]


print("=" * 60)
print("Target: (2/5)*ζ(3) = {:.15f}".format(TARGET))
print("=" * 60)

# --- Test A: Kicked system from x=0 ---
print("\n--- Test A: Kicked system (τe^{-τ} on dx/dτ), start x=0 ---")

Y0_kick = [0.0, 0.5, -1.0/24, 0.0, 0.0, 1.0, 0.0]

sol_A = solve_ivp(rhs_kicked, [0, 500], Y0_kick,
                  method='RK45', rtol=1e-12, atol=1e-15,
                  dense_output=True, max_step=0.5)

print(f"  Integration status: {sol_A.message}")
print(f"  τ_final = {sol_A.t[-1]:.1f}")
print(f"  x(τ_final) = {sol_A.y[3, -1]:.15f}  (should → 1)")
print(f"  y0(τ_final) = {sol_A.y[0, -1]:.15f}")
print(f"  Target       = {TARGET:.15f}")
print(f"  Error        = {abs(sol_A.y[0, -1] - TARGET):.6e}")
print(f"  ζ(3) from y0 = {sol_A.y[0, -1] * 5/2:.15f}")
print(f"  True ζ(3)    = {ZETA3:.15f}")

# Check trajectory
tau_samples = np.linspace(0, min(500, sol_A.t[-1]-1), 200)
Y_samples = sol_A.sol(tau_samples)
print(f"\n  Trajectory ranges:")
print(f"  y0: [{np.min(Y_samples[0]):.6f}, {np.max(Y_samples[0]):.6f}]")
print(f"  y1: [{np.min(Y_samples[1]):.6f}, {np.max(Y_samples[1]):.6f}]")
print(f"  y2: [{np.min(Y_samples[2]):.6f}, {np.max(Y_samples[2]):.6f}]")
print(f"  x:  [{np.min(Y_samples[3]):.6f}, {np.max(Y_samples[3]):.6f}]")
print(f"  p:  [{np.min(Y_samples[6]):.6f}, {np.max(Y_samples[6]):.6f}]")

# --- Test B: Baseline — reparametrized system from x=ε with exact ICs ---
print("\n--- Test B: Baseline (no kick), start x=0.01 with exact ICs ---")

eps = 0.01
Y0_base = [apery_series(eps), apery_deriv1(eps), apery_deriv2(eps), eps]

sol_B = solve_ivp(rhs_reparam, [0, 500], Y0_base,
                  method='RK45', rtol=1e-12, atol=1e-15,
                  dense_output=True, max_step=0.5)

print(f"  τ_final = {sol_B.t[-1]:.1f}")
print(f"  x(τ_final) = {sol_B.y[3, -1]:.15f}")
print(f"  y0(τ_final) = {sol_B.y[0, -1]:.15f}")
print(f"  Target       = {TARGET:.15f}")
print(f"  Error        = {abs(sol_B.y[0, -1] - TARGET):.6e}")

# --- Compare: how much does the kick distort? ---
print("\n--- Comparison ---")
err_A = abs(sol_A.y[0, -1] - TARGET)
err_B = abs(sol_B.y[0, -1] - TARGET)
print(f"  Kicked system error:   {err_A:.6e}")
print(f"  Baseline error:        {err_B:.6e}")
print(f"  Distortion from kick:  {abs(sol_A.y[0, -1] - sol_B.y[0, -1]):.6e}")

# --- Test D: Different kick magnitudes ---
print("\n--- Test D: Kicked system with scaled kick c*τe^{-τ} ---")
for c in [0.01, 0.1, 1.0, 10.0]:
    def rhs_scaled_kick(tau, Y, c=c):
        y0, y1, y2, x, s, u, p = Y
        x2_4x = x**2 * (4 + x)
        one_mx = 1 - x
        dy0 = y1 * one_mx * x2_4x
        dy1 = y2 * one_mx * x2_4x
        dy2 = one_mx * (1 - x*(10 + 3*x)*y2 - (2 + x)*y1)
        dx  = one_mx * x2_4x + c * p
        ds  = 1.0
        du  = -u
        dp  = u - p
        return [dy0, dy1, dy2, dx, ds, du, dp]

    sol = solve_ivp(rhs_scaled_kick, [0, 500], Y0_kick,
                    method='RK45', rtol=1e-12, atol=1e-15, max_step=0.5)
    err = abs(sol.y[0, -1] - TARGET)
    print(f"  c={c:6.2f}: y0 = {sol.y[0, -1]:.15f}, x = {sol.y[3, -1]:.10f}, err = {err:.6e}")
