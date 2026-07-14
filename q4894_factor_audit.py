#!/usr/bin/env python3
"""Exact Ore-factor audit for Ramanujan Challenge Problem 2.5.

Reconstructs the scalar recurrence from the 3x3 CMF by Casorati minors,
removes the canonical Pochhammer twist, and runs Petkovsek's Hyper
algorithm (SymPy rsolve_hyper) on the operator and its adjoint.
"""
import sympy as sp
from sympy.solvers.recurr import rsolve_hyper

n = sp.symbols('n', integer=True)


def sh(e, k=1):
    return sp.expand(e.subs(n, n + k))


def shmat(A, k=1):
    return A.applyfunc(lambda e: sh(e, k))


# CMF matrix.
m11 = (-2*n-5)*(n+3)**2*(136*n**4+1424*n**3+5548*n**2+9551*n+6141)
m12 = 384*n**6+6384*n**5+44168*n**4+162698*n**3+336377*n**2+369933*n+169011
m13 = -480*n**4-4980*n**3-19210*n**2-32690*n-20730
m21 = (n+2)**2*(n+3)**2*(4*n+10)*(48*n**3+386*n**2+1017*n+879)
m22 = (n+2)**2*(-272*n**5-3848*n**4-21732*n**3-61184*n**2-85761*n-47808)
m23 = (n+2)**2*(320*n**3+2540*n**2+6610*n+5640)
m31 = (-4*n-10)*(n+2)**2*(n+3)**2*(32*n**4+302*n**3+1037*n**2+1530*n+813)
m32 = (n+2)**2*(192*n**6+2984*n**5+19116*n**4+64452*n**3+120256*n**2+117279*n+46476)
m33 = (n+2)**2*(-16*n**5-408*n**4-2912*n**3-8884*n**2-12254*n-6240)
M = sp.Matrix([[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]])

print('BEGIN_Q4894_AUDIT', flush=True)
print('detM=', sp.factor(M.det()), flush=True)

# Scalarization for any left-row solution, using the first output coordinate.
e = sp.Matrix([1, 0, 0])
v = [e,
     M*e,
     M*shmat(M, 1)*e,
     M*shmat(M, 1)*shmat(M, 2)*e]

c = []
for j in range(4):
    minor = sp.Matrix.hstack(*[v[k] for k in range(4) if k != j]).det()
    c.append(sp.expand((-1)**j * minor))

assert all(sp.expand(sum(c[j]*v[j][i] for j in range(4))) == 0 for i in range(3))
g = sp.gcd_list(c)
c = [sp.cancel(x/g) for x in c]
# Clear common rational content and normalize sign.
num_content = sp.gcd_list([sp.Poly(x, n, domain=sp.QQ).content() for x in c])
if num_content:
    c = [sp.cancel(x/num_content) for x in c]
if sp.LC(sp.Poly(c[3], n)) > 0:
    c = [-x for x in c]

print('raw_degrees=', [sp.degree(x, n) for x in c], flush=True)
print('raw_c0_factor=', sp.factor(c[0]), flush=True)
print('raw_c3_factor=', sp.factor(c[3]), flush=True)

# Q_n = H_n * qhat_n, with H_{n+1}/H_n = delta(n).
delta = -2*(n+2)**2*(n+3)**2*(2*n+5)*(2*n+7)**2
ell = []
prod = sp.Integer(1)
for j in range(4):
    if j:
        prod = sp.expand(prod * sh(delta, j-1))
    ell.append(sp.expand(c[j] * prod))

g = sp.gcd_list(ell)
ell = [sp.cancel(x/g) for x in ell]
content = sp.gcd_list([sp.Poly(x, n, domain=sp.QQ).content() for x in ell])
if content:
    ell = [sp.cancel(x/content) for x in ell]
if sp.LC(sp.Poly(ell[3], n)) < 0:
    ell = [-x for x in ell]

print('normalized_degrees=', [sp.degree(x, n) for x in ell], flush=True)
for j, x in enumerate(ell):
    print(f'ell{j}=', sp.expand(x), flush=True)
    print(f'ell{j}_factor=', sp.factor(x), flush=True)

# Leading Poincare polynomial.
lead = [sp.LC(sp.Poly(x, n)) for x in ell]
xi = sp.symbols('xi')
P = sp.expand(sum(lead[j]*xi**j for j in range(4)))
print('poincare=', sp.factor(P), flush=True)


def shifted(r, k):
    return sp.cancel(r.subs(n, n+k))


def riccati(coeffs, r):
    ans = sp.Integer(0)
    q = sp.Integer(1)
    for j in range(4):
        ans += coeffs[j] * q
        q = sp.cancel(q * shifted(r, j))
    return sp.cancel(ans)

# Useful explicit candidates.
candidates = {
    'one': sp.Integer(1),
    'pure_n_minus_3': (n+1)**3/(n+2)**3,
    'det_MH': (n+1)*(2*n+3)**2/((n+3)*(2*n+7)**2),
    'det_balanced': (n+2)**3*(2*n+3)**2/((n+1)**2*(n+3)*(2*n+7)**2),
}
for name, r in candidates.items():
    res = sp.together(riccati(ell, r))
    num = sp.factor(res.as_numer_denom()[0])
    print('candidate', name, 'zero=', num == 0, flush=True)
    if num != 0:
        print('candidate', name, 'residual_num_factor=', num, flush=True)

# S^3 L^*: coefficient of S^k is ell_{3-k}(n+k).
adj = [sp.expand(sh(ell[3-k], k)) for k in range(4)]
print('adjoint_degrees=', [sp.degree(x, n) for x in adj], flush=True)

for label, coeffs in [('RIGHT_FACTOR_TEST', ell),
                      ('LEFT_FACTOR_TEST_VIA_ADJOINT', adj)]:
    print('RSOLVE_HYPER_BEGIN', label, flush=True)
    try:
        sol = rsolve_hyper(coeffs, sp.Integer(0), n)
        print('RSOLVE_HYPER_RESULT', label, repr(sol), flush=True)
        if sol not in (None, 0, sp.Integer(0)):
            ratio = sp.factor(sp.cancel(sol.subs(n, n+1)/sol))
            print('RSOLVE_HYPER_RATIO', label, ratio, flush=True)
            print('RSOLVE_HYPER_RESIDUAL', label,
                  sp.factor(sp.together(riccati(coeffs, ratio)).as_numer_denom()[0]),
                  flush=True)
    except BaseException as ex:
        print('RSOLVE_HYPER_ERROR', label, type(ex).__name__, str(ex), flush=True)

print('END_Q4894_AUDIT', flush=True)
