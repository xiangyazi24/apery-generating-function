# Computing ζ(3) via the Apéry Generating Function ODE

## Key Result

The Apéry accelerated series

$$F(x) = \sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{n^3 \binom{2n}{n}} x^n$$

satisfies the third-order linear ODE:

$$x^2(4+x) F''' + x(10+3x) F'' + (2+x) F' = 1$$

with **rational initial conditions**: F(0) = 0, F'(0) = 1/2, F''(0) = -1/24.

The value F(1) = (2/5)ζ(3), and x = 1 is within the radius of convergence (R = 4).

## Significance

This ODE has:
- Integer coefficients
- Rational initial values
- First-floor convergence rate (α ≈ 1.0001) under x = 1 - e^{-t}

## Open Problem

The ODE has a regular singular point at x = 0 (the coefficient of F''' vanishes as x²).
Can the singularity be regularized to yield a polynomial PIVP with rational ICs computing ζ(3)?

If yes, this would prove ζ(3) ∈ R_RTCRN (the bounded analog complexity class).

## Structure

- `notes/` — Working notes and derivations
- `code/` — Numerical verification scripts
