#!/usr/bin/env python3
"""Q5731: exact top-half target and Cartier first-digit audit."""
from __future__ import annotations
from math import comb, isqrt
LIMIT = 5000

def primes_upto(n: int) -> list[int]:
    mark = bytearray(b"\x01") * (n + 1)
    if n >= 0: mark[0] = 0
    if n >= 1: mark[1] = 0
    for p in range(2, isqrt(n) + 1):
        if mark[p]:
            mark[p*p:n+1:p] = b"\x00" * (((n-p*p)//p)+1)
    return [p for p in range(2,n+1) if mark[p]]

def apery_values(n: int) -> list[int]:
    if n == 0: return [1]
    b=[1,5]
    for m in range(1,n):
        num=(34*m**3+51*m**2+27*m+5)*b[m]-m**3*b[m-1]
        den=(m+1)**3
        q,rem=divmod(num,den); assert rem==0; b.append(q)
    return b[:n+1]

def C(n,k): return comb(n,k) if 0 <= k <= n else 0

def direct_shell_mod(M,d,modulus):
    a=M//d; out=0
    for t in range(M+1):
        X=sum(C(M,M-t+d*u) for u in range(-a,a+1))%modulus
        Z=sum(C(2*M-t,M-t+d*v) for v in range(-a,a+1))%modulus
        out=(out+(C(M,t)%modulus)*X*Z*Z)%modulus
    return out

def binomial_row_mod(n,modulus):
    row=[1]; value=1
    for k in range(n):
        value=value*(n-k)//(k+1); row.append(value%modulus)
    return row

def first_cell_shell_mod_p2(M,p):
    d=p-1; assert p <= M <= 2*p-3
    s=M-p; modulus=p*p; row=binomial_row_mod(M,modulus)
    z0=comb(2*M,M); zp=comb(2*M,M+d); zm=comb(2*M,s+1); out=0
    for t in range(M+1):
        X=row[t]
        if t>=d: X+=row[t-d]
        if t<=s+1: X+=row[s+1-t]
        X%=modulus; Z=(z0+zp+zm)%modulus
        out=(out+row[t]*X*Z*Z)%modulus
        if t==M: break
        N=2*M-t; k0=M-t; kp=M-t+d; km=s+1-t
        z0=z0*k0//N; zp=zp*kp//N; zm=0 if km<=0 else zm*km//N
    return out

def self_test_shell():
    checks=0
    for p in primes_upto(31):
        if p<5: continue
        for s in range(p-2):
            M=p+s
            assert first_cell_shell_mod_p2(M,p)==direct_shell_mod(M,p-1,p*p)
            checks+=1
    print('SHELL_SELF_TESTS',checks)

def top_half_targets(b):
    out=[]
    for p in primes_upto(LIMIT):
        for r in range(1,min(p-1,LIMIT-p)+1):
            if b[r]%p==0:
                assert r<=p-2
                out.append((p+r,p,r,r-1))
    return out

def main():
    self_test_shell(); b=apery_values(LIMIT-1); targets=top_half_targets(b)
    large=[x for x in targets if x[1]>5]; small=[x for x in targets if x[1]<=5]
    shell_zero=[]; ghost_zero=[]; ratio_zero=[]; cross_zero=[]; depth2=[]; records=[]
    for n,p,r,s in large:
        M=n-1; origin=b[M]%p
        assert origin==5*(b[s]%p)%p and origin!=0
        modulus=p*p; shell=first_cell_shell_mod_p2(M,p); assert shell%p==0
        shell_digit=shell//p; apery_digit=(b[r]//p)%p
        ghost=(shell_digit-apery_digit)%p
        origin_digit=((b[M]-5*b[s])//p)%p
        ratio_digit=shell_digit*pow(origin,-1,p)%p
        E_mod=(5*(b[s]%modulus)*shell-(b[r]%modulus)*(b[M]%modulus))%modulus
        assert E_mod%p==0
        cross_digit=E_mod//p
        assert cross_digit==5*(b[s]%p)*ghost%p
        rec=(n,p,r,shell_digit,apery_digit,ghost,origin,origin_digit,ratio_digit,cross_digit)
        records.append(rec)
        if shell_digit==0: shell_zero.append(rec)
        if ghost==0: ghost_zero.append(rec)
        if ratio_digit==0: ratio_zero.append(rec)
        if cross_digit==0: cross_zero.append(rec)
        if apery_digit==0: depth2.append(rec)
    counter=[x for x in records if x[0]==20 and x[1]==17]
    assert len(counter)==1 and counter[0][3]==7 and counter[0][4]==0 and counter[0][5]==7
    print('LIMIT',LIMIT)
    print('TOP_HALF_TARGETS_ALL',len(targets))
    print('TOP_HALF_TARGETS_P_GT_5',len(large))
    print('SMALL_TARGETS',small)
    print('FIRST_TARGETS',records[:20])
    print('SHELL_DIGIT_ZERO_COUNT',len(shell_zero)); print('SHELL_DIGIT_ZERO',shell_zero)
    print('GHOST_ZERO_COUNT',len(ghost_zero)); print('GHOST_ZERO',ghost_zero)
    print('RATIO_ZERO_COUNT',len(ratio_zero)); print('CROSS_ZERO_COUNT',len(cross_zero))
    print('APERY_DEPTH_AT_LEAST_TWO_COUNT',len(depth2)); print('APERY_DEPTH_AT_LEAST_TWO',depth2)
    print('COUNTEREXAMPLE_N20_P17',counter[0])
    print('PASS: every top-half target through n=5000 audited exactly')
if __name__=='__main__': main()
