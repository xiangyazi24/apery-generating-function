"""
Verify: the Apéry accelerated series generating function satisfies
  x²(4+x) F''' + x(10+3x) F'' + (2+x) F' = 1
with F(0)=0, F'(0)=1/2, F''(0)=-1/24, and F(1) = (2/5)ζ(3).
"""
import numpy as np
from scipy.special import comb
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

ZETA3 = 1.2020569031595942

# --- Part 1: Verify the series directly ---
def apery_series(x, N=200):
    """Compute F(x) = sum_{n=1}^N (-1)^{n-1} / (n^3 * C(2n,n)) * x^n"""
    s = 0.0
    for n in range(1, N+1):
        s += ((-1)**(n-1)) / (n**3 * comb(2*n, n, exact=True)) * x**n
    return s

def apery_series_deriv1(x, N=200):
    """F'(x) = sum_{n=1}^N (-1)^{n-1} n / (n^3 * C(2n,n)) * x^{n-1}
            = sum_{n=1}^N (-1)^{n-1} / (n^2 * C(2n,n)) * x^{n-1}"""
    s = 0.0
    for n in range(1, N+1):
        s += ((-1)**(n-1)) / (n**2 * comb(2*n, n, exact=True)) * x**(n-1)
    return s

def apery_series_deriv2(x, N=200):
    """F''(x) = sum_{n=2}^N (-1)^{n-1} (n-1) / (n^2 * C(2n,n)) * x^{n-2}"""
    s = 0.0
    for n in range(2, N+1):
        s += ((-1)**(n-1)) * (n-1) / (n**2 * comb(2*n, n, exact=True)) * x**(n-2)
    return s

print("=== Part 1: Series verification ===")
print(f"F(1)           = {apery_series(1.0):.15f}")
print(f"(2/5)*zeta(3)  = {(2/5)*ZETA3:.15f}")
print(f"F'(0)          = {apery_series_deriv1(0.0):.15f}  (should be 1/2 = 0.5)")
print(f"F''(0)         = {apery_series_deriv2(0.0):.15f}  (should be -1/24 = {-1/24:.15f})")

# Verify the ODE: x²(4+x)F''' + x(10+3x)F'' + (2+x)F' = 1
# at a test point x=0.5
x_test = 0.5
h = 1e-6
F = apery_series
Fp = lambda x: apery_series_deriv1(x)
Fpp = lambda x: apery_series_deriv2(x)
Fppp = lambda x: (Fpp(x+h) - Fpp(x-h)) / (2*h)

lhs = x_test**2 * (4+x_test) * Fppp(x_test) + x_test*(10+3*x_test)*Fpp(x_test) + (2+x_test)*Fp(x_test)
print(f"\nODE check at x={x_test}: LHS = {lhs:.10f}  (should be 1.0)")

# --- Part 2: Solve the ODE via x = 1 - e^{-t} ---
print("\n=== Part 2: ODE integration via x = 1 - e^{-t} ===")

def rhs_t_domain(t, y):
    """
    Variables: y = [F, F', F'', x]
    where x = 1 - e^{-t}, so x' = e^{-t} = 1 - x

    Chain rule: dF/dt = F' * x' = F' * (1-x)
    d(F')/dt = F'' * (1-x)
    d(F'')/dt = F''' * (1-x)

    From ODE: F''' = [1 - x(10+3x)F'' - (2+x)F'] / [x²(4+x)]
    """
    F_val, Fp_val, Fpp_val, x = y

    xdot = 1 - x  # = e^{-t}

    # Avoid singularity near x=0
    if abs(x) < 1e-12:
        # At x=0: use the series to get F'''(0)
        # From the series: F'''(0) = sum_{n=3} (-1)^{n-1}(n-1)(n-2)/(n^2 C(2n,n)) * 0
        # Actually F'''(0) can be computed from the recurrence
        # a_3 = (-1)^2 / (27 * C(6,3)) = 1/540
        # F'''(0) = 3! * a_3 / 3! ... let me use L'Hopital or series
        Fppp_val = 1/12  # placeholder, will be overridden quickly
    else:
        denom = x**2 * (4 + x)
        Fppp_val = (1 - x*(10 + 3*x)*Fpp_val - (2 + x)*Fp_val) / denom

    dF = Fp_val * xdot
    dFp = Fpp_val * xdot
    dFpp = Fppp_val * xdot
    dx = xdot

    return [dF, dFp, dFpp, dx]

# Initial conditions: F(0)=0, F'(0)=1/2, F''(0)=-1/24, x(0)=0
# Start slightly away from x=0 to avoid singularity
t_start = 0.01  # x(t_start) = 1 - e^{-0.01} ≈ 0.00995
x_start = 1 - np.exp(-t_start)
F_start = apery_series(x_start)
Fp_start = apery_series_deriv1(x_start)
Fpp_start = apery_series_deriv2(x_start)

y0 = [F_start, Fp_start, Fpp_start, x_start]

sol = solve_ivp(rhs_t_domain, [t_start, 30.0], y0,
                method='RK45', rtol=1e-12, atol=1e-14,
                dense_output=True)

t_final = sol.t[-1]
F_final = sol.y[0, -1]
x_final = sol.y[3, -1]

print(f"t_final = {t_final:.2f}, x_final = {x_final:.15f} (should → 1)")
print(f"F(x_final) = {F_final:.15f}")
print(f"(2/5)*ζ(3) = {(2/5)*ZETA3:.15f}")
print(f"ζ(3) from F = {F_final * 5/2:.15f}")
print(f"True ζ(3)   = {ZETA3:.15f}")
print(f"Error       = {abs(F_final * 5/2 - ZETA3):.2e}")

# Convergence rate
t_samples = np.linspace(t_start + 1, 25, 50)
y_samples = sol.sol(t_samples)
errors = np.abs(y_samples[0] * 5/2 - ZETA3)
# Fit log(error) vs t to get convergence rate
valid = errors > 1e-15
if np.sum(valid) > 2:
    coeffs = np.polyfit(t_samples[valid], np.log(errors[valid]), 1)
    alpha = -coeffs[0]
    print(f"\nConvergence rate α = {alpha:.4f} (first floor if ≈ 1)")

# Check boundedness
print(f"\n=== Boundedness check ===")
print(f"F  range: [{np.min(sol.y[0]):.6f}, {np.max(sol.y[0]):.6f}]")
print(f"F' range: [{np.min(sol.y[1]):.6f}, {np.max(sol.y[1]):.6f}]")
print(f"F'' range: [{np.min(sol.y[2]):.6f}, {np.max(sol.y[2]):.6f}]")
print(f"x  range: [{np.min(sol.y[3]):.6f}, {np.max(sol.y[3]):.6f}]")
