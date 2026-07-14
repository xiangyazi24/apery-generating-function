#!/usr/bin/env python3
import sympy as sp
from sympy.solvers.recurr import rsolve_hyper

print('Q4859 SymPy exact audit starting', flush=True)
n = sp.symbols('n', integer=True)

def sh(e, k=1):
    return sp.expand(e.subs(n, n+k))

def shmat(A, k=1):
    return A.applyfunc(lambda e: sh(e,k))

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
print('matrix built', flush=True)
print('det factor', sp.factor(M.det()), flush=True)

e = sp.Matrix([1,0,0])
v0=e
v1=M*e
v2=M*shmat(M,1)*e
v3=M*shmat(M,1)*shmat(M,2)*e
vs=[v0,v1,v2,v3]
cs=[]
for k in range(4):
    C=sp.Matrix.hstack(*[vs[j] for j in range(4) if j != k])
    cs.append(sp.expand((-1)**k*C.det()))
    print('minor',k,'degree',sp.degree(cs[-1],n),flush=True)
assert all(sp.expand(sum(cs[k]*vs[k][i] for k in range(4))) == 0 for i in range(3))
g=sp.gcd_list(cs)
cs=[sp.cancel(c/g) for c in cs]
cg=sp.gcd_list(cs)
cs=[sp.cancel(c/cg) for c in cs]
if sp.LC(sp.Poly(cs[3],n))>0:
    cs=[-c for c in cs]
print('raw degrees',[sp.degree(c,n) for c in cs],flush=True)
print('raw c0 factor',sp.factor(cs[0]),flush=True)
print('raw c3 factor',sp.factor(cs[3]),flush=True)

delta=-2*(n+2)**2*(n+3)**2*(2*n+5)*(2*n+7)**2
nh=[]
prod=sp.Integer(1)
for j in range(4):
    if j:
        prod=sp.expand(prod*sh(delta,j-1))
    nh.append(sp.expand(cs[j]*prod))
gh=sp.gcd_list(nh)
nh=[sp.cancel(c/gh) for c in nh]
contents=[sp.Poly(c,n,domain=sp.QQ).content() for c in nh]
cont=sp.gcd_list(contents)
if cont:
    nh=[sp.cancel(c/cont) for c in nh]
if sp.LC(sp.Poly(nh[3],n))<0:
    nh=[-c for c in nh]
print('normalized degrees',[sp.degree(c,n) for c in nh],flush=True)
for j,c in enumerate(nh):
    print(f'ell{j} = {sp.expand(c)}',flush=True)
    print(f'ell{j} factor = {sp.factor(c)}',flush=True)
print('sum factor',sp.factor(sum(nh)),flush=True)

def shift_rat(r,k):
    return sp.cancel(r.subs(n,n+k))

def riccati(coeffs,r):
    ans=sp.Integer(0); q=sp.Integer(1)
    for j in range(4):
        ans += coeffs[j]*q
        q=sp.cancel(q*shift_rat(r,j))
    return sp.cancel(ans)

candidates={
'one':sp.Integer(1),
'nminus3':(n+1)**3/(n+2)**3,
'det_MH':(n+1)*(2*n+3)**2/((n+3)*(2*n+7)**2),
'det_balanced':(n+2)**3*(2*n+3)**2/((n+1)**2*(n+3)*(2*n+7)**2),
}
for name,r in candidates.items():
    rr=riccati(nh,r)
    print('candidate',name,'zero',rr==0,flush=True)
    if rr != 0:
        print('candidate',name,'resnum factor',sp.factor(sp.together(rr).as_numer_denom()[0]),flush=True)

adj=[sp.expand(sh(nh[3-k],k)) for k in range(4)]
print('adj degrees',[sp.degree(c,n) for c in adj],flush=True)
for label,coeffs in [('ORIGINAL_RIGHT',nh),('ADJOINT_RIGHT_EQ_LEFT_ORIGINAL',adj)]:
    print('RSOLVE_HYPER_CALL',label,flush=True)
    try:
        sol=rsolve_hyper(coeffs,sp.Integer(0),n)
        print('RSOLVE_HYPER_RESULT',label,repr(sol),flush=True)
        if sol not in (None,0,sp.Integer(0)):
            print('SOLUTION_RATIO',label,sp.factor(sp.cancel(sol.subs(n,n+1)/sol)),flush=True)
    except BaseException as ex:
        print('RSOLVE_HYPER_ERROR',label,type(ex).__name__,str(ex),flush=True)
print('Q4859 SymPy exact audit finished',flush=True)
