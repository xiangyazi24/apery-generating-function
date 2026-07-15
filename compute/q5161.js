'use strict';

const fs = require('fs');
const OUT = 'compute/q5161-js-results.md';

function absBig(x){ return x < 0n ? -x : x; }
function gcdBig(a,b){ a=absBig(a); b=absBig(b); while(b){ const r=a%b; a=b; b=r; } return a; }
class Rat {
  constructor(n,d=1n){
    n=BigInt(n); d=BigInt(d);
    if(d===0n) throw new Error('zero denominator');
    if(d<0n){ n=-n; d=-d; }
    const g=gcdBig(n,d); this.n=n/g; this.d=d/g;
  }
  add(o){ o=toRat(o); return new Rat(this.n*o.d+o.n*this.d,this.d*o.d); }
  sub(o){ o=toRat(o); return new Rat(this.n*o.d-o.n*this.d,this.d*o.d); }
  mul(o){ o=toRat(o); return new Rat(this.n*o.n,this.d*o.d); }
  div(o){ o=toRat(o); return new Rat(this.n*o.d,this.d*o.n); }
  pow(k){ let r=new Rat(1n); for(let i=0;i<k;i++) r=r.mul(this); return r; }
  toString(){ return this.d===1n ? this.n.toString() : `${this.n}/${this.d}`; }
}
function toRat(x){ return x instanceof Rat ? x : new Rat(x); }

function modBig(x,p){ let r=Number(x % BigInt(p)); if(r<0) r+=p; return r; }
function egcd(a,b){ let x0=1,x1=0,y0=0,y1=1; while(b){ const q=Math.floor(a/b); [a,b]=[b,a-q*b]; [x0,x1]=[x1,x0-q*x1]; [y0,y1]=[y1,y0-q*y1]; } return [a,x0,y0]; }
function invMod(a,p){ a=((a%p)+p)%p; const [g,x]=egcd(a,p); if(g!==1) throw new Error(`nonunit ${a} mod ${p}`); return ((x%p)+p)%p; }
function ratMod(x,p){ return (modBig(x.n,p)*invMod(modBig(x.d,p),p))%p; }
function addm(a,b,p){ const x=a+b; return x>=p?x-p:x; }
function subm(a,b,p){ let x=a-b; if(x<0)x+=p; return x; }
function mulm(a,b,p){ return (a*b)%p; } // p near 10^6 => product safely below 2^53

function cooperTerms(N){
  const T=Array(N+1).fill(0n); T[0]=1n;
  for(let m=0;m<N;m++){
    const M=BigInt(m), tm1=m>=1?T[m-1]:0n, tm2=m>=2?T[m-2]:0n;
    const rhs=2n*(2n*M+1n)*(5n*M*M+5n*M+2n)*T[m]
      -8n*M*(7n*M*M+1n)*tm1
      +22n*M*(2n*M-1n)*(M-1n)*tm2;
    const den=(M+1n)**3n;
    if(rhs%den!==0n) throw new Error(`T nonintegral at ${m}`);
    T[m+1]=rhs/den;
  }
  return T;
}

function traceMoments(T,N,amax=3){
  const S=Array.from({length:amax+1},()=>Array(N+1).fill(0n));
  const M=Array.from({length:amax+1},()=>Array(N+1));
  for(let n=0;n<=N;n++){
    const totals=Array(amax+1).fill(0n);
    let choose=1n;
    for(let m=0;m<=2*n;m++){
      const base=choose*((-2n)**BigInt(2*n-m))*T[m];
      let mp=1n;
      for(let a=0;a<=amax;a++){ totals[a]+=mp*base; mp*=BigInt(m); }
      if(m<2*n) choose=choose*BigInt(2*n-m)/BigInt(m+1);
    }
    const den=256n**BigInt(n);
    for(let a=0;a<=amax;a++){ S[a][n]=totals[a]; M[a][n]=new Rat(totals[a],den); }
  }
  return {S,M};
}

function A27(n){ n=BigInt(n); return 1024n*(2n*n+5n)**4n*(2n*n+7n)**3n*(2n*n+9n)**3n*(946n*n*n+6407n*n+10860n); }
function B27(n){ n=BigInt(n); return 128n*(2n*n+7n)**3n*(2n*n+9n)**3n*(104060n*n**6n+1745370n*n**5n+12145238n*n**4n+44886481n*n**3n+92943995n*n**2n+102256019n*n+46709052n); }
function C27(n){ n=BigInt(n); return 16n*(n+3n)**4n*(2n*n+9n)**3n*(3784n*n**5n+57792n*n**4n+351019n*n**3n+1059230n*n**2n+1587211n*n+944620n); }
function D27(n){ n=BigInt(n); return (n+3n)**4n*(n+4n)**6n*(946n*n*n+4515n*n+5399n); }
function qTerms(N){
  const q=[new Rat(-215040420000n),new Rat(-167282265043404n,905n),new Rat(-964185327658080n,6071n)];
  for(let n=2;n<N;n++){
    q.push(new Rat(B27(n),A27(n)).mul(q[n])
      .sub(new Rat(C27(n-1),A27(n-1)).mul(q[n-1]))
      .add(new Rat(D27(n-2),A27(n-2)).mul(q[n-2])));
  }
  return q.slice(0,N+1);
}
function factorial(n){ let x=1n; for(let i=2n;i<=BigInt(n);i++)x*=i; return x; }
function gammaA(a,n){
  let num=1n, den=1n;
  const start=5-2*a;
  for(let i=0;i<n;i++){ num*=BigInt(start+2*i); den*=2n*BigInt(i+1); }
  return new Rat(num,den);
}

// Rank over F_p; matrix entries are Numbers in [0,p).
function rankMod(mat,p){
  if(mat.length===0) return 0;
  const A=mat.map(r=>r.slice()); const rows=A.length, cols=A[0].length;
  let rank=0;
  for(let col=0;col<cols && rank<rows;col++){
    let piv=rank; while(piv<rows && A[piv][col]===0)piv++;
    if(piv===rows)continue;
    [A[rank],A[piv]]=[A[piv],A[rank]];
    const inv=invMod(A[rank][col],p);
    for(let c=col;c<cols;c++)A[rank][c]=mulm(A[rank][c],inv,p);
    for(let r=rank+1;r<rows;r++){
      const f=A[r][col]; if(f===0)continue;
      for(let c=col;c<cols;c++)A[r][c]=subm(A[r][c],mulm(f,A[rank][c],p),p);
    }
    rank++;
  }
  return rank;
}

function fitRanks(M,q,{Amax=3,J=0,deg=3,pre=false,gammaShift=false,nrows=31},p){
  const rows=[], aug=[];
  const max=Math.min(nrows,q.length,M[0].length-J);
  for(let n=0;n<max;n++){
    const row=[];
    for(let a=0;a<=Amax;a++) for(let j=0;j<=J;j++) for(let d=0;d<=deg;d++){
      let x=M[a][n+j];
      if(pre)x=x.mul(gammaA(a,gammaShift?n+j:n));
      x=x.mul(BigInt(n)**BigInt(d));
      row.push(ratMod(x,p));
    }
    rows.push(row); aug.push(row.concat([ratMod(q[n],p)]));
  }
  return {rank:rankMod(rows,p), augrank:rankMod(aug,p), unknowns:rows[0].length, rows:max};
}

// RREF nullspace basis over F_p.
function nullspaceMod(mat,p){
  if(mat.length===0)return [];
  const A=mat.map(r=>r.slice()); const R=A.length,C=A[0].length; const piv=[];
  let rr=0;
  for(let c=0;c<C && rr<R;c++){
    let pr=rr; while(pr<R && A[pr][c]===0)pr++;
    if(pr===R)continue;
    [A[rr],A[pr]]=[A[pr],A[rr]];
    const iv=invMod(A[rr][c],p);
    for(let j=c;j<C;j++)A[rr][j]=mulm(A[rr][j],iv,p);
    for(let i=0;i<R;i++) if(i!==rr && A[i][c]!==0){ const f=A[i][c]; for(let j=c;j<C;j++)A[i][j]=subm(A[i][j],mulm(f,A[rr][j],p),p); }
    piv.push(c); rr++;
  }
  const pset=new Set(piv), free=[]; for(let c=0;c<C;c++)if(!pset.has(c))free.push(c);
  const basis=[];
  for(const f of free){ const v=Array(C).fill(0); v[f]=1; for(let i=piv.length-1;i>=0;i--){ const pc=piv[i]; v[pc]=A[i][f]===0?0:(p-A[i][f]); } basis.push(v); }
  return basis;
}
function recMatrix(seq,r,d,p,start,count){
  const vals=seq.map(x=>modBig(x,p)); const out=[];
  for(let n=start;n<start+count;n++){
    const powers=[1]; for(let k=1;k<=d;k++)powers.push(mulm(powers[k-1],n%p,p));
    const row=[]; for(let j=0;j<=r;j++)for(let k=0;k<=d;k++)row.push(mulm(vals[n+j],powers[k],p));
    out.push(row);
  }
  return out;
}
function matVec(row,v,p){ let s=0; for(let i=0;i<row.length;i++)s=addm(s,mulm(row[i],v[i],p),p); return s; }
function recurrenceExists(seq,r,d,p){
  const U=(r+1)*(d+1), avail=seq.length-r; if(avail<U+8)return false;
  const count=Math.min(avail,U+10); const first=recMatrix(seq,r,d,p,0,count); const basis=nullspaceMod(first,p); if(basis.length===0)return false;
  for(const v of basis){ let ok=true; for(let n=count;n<avail;n++){ const row=recMatrix(seq,r,d,p,n,1)[0]; if(matVec(row,v,p)!==0){ok=false;break;} } if(ok)return true; }
  // If the initial kernel had dimension >1, a held-out linear combination may survive.
  if(basis.length>1){
    const small=[]; for(let n=count;n<avail;n++){ const row=recMatrix(seq,r,d,p,n,1)[0]; small.push(basis.map(v=>matVec(row,v,p))); }
    return nullspaceMod(small,p).length>0;
  }
  return false;
}
function findRecurrence(seq,maxOrder,maxDegree,primes){
  for(let r=1;r<=maxOrder;r++){
    if(!primes.every(p=>recurrenceExists(seq,r,maxDegree,p)))continue;
    for(let d=0;d<=maxDegree;d++) if(primes.every(p=>recurrenceExists(seq,r,d,p)))return {order:r,degree:d};
  }
  return null;
}

const N=520;
const T=cooperTerms(2*N+2);
const {S,M}=traceMoments(T,N,3);
const q=qTerms(110);
const primes=[1000003,1000033];
const fits=[];
const specs=[
  ['direct A=3 J=0 deg=3',{Amax:3,J:0,deg:3,pre:false,nrows:31}],
  ['Gamma(n) A=3 J=0 deg=3',{Amax:3,J:0,deg:3,pre:true,nrows:31}],
  ['direct shifts A=3 J=1 deg=3',{Amax:3,J:1,deg:3,pre:false,nrows:70}],
  ['Gamma(n) shifts A=3 J=1 deg=3',{Amax:3,J:1,deg:3,pre:true,nrows:70}],
  ['Gamma(n+j) shifts A=3 J=1 deg=3',{Amax:3,J:1,deg:3,pre:true,gammaShift:true,nrows:70}],
  ['direct shifts A=3 J=2 deg=3',{Amax:3,J:2,deg:3,pre:false,nrows:100}],
  ['Gamma(n) shifts A=3 J=2 deg=3',{Amax:3,J:2,deg:3,pre:true,nrows:100}],
  ['Gamma(n+j) shifts A=3 J=2 deg=3',{Amax:3,J:2,deg:3,pre:true,gammaShift:true,nrows:100}],
];
for(const [name,spec] of specs)fits.push({name,byPrime:primes.map(p=>[p,fitRanks(M,q,spec,p)])});

const rec=[];
for(let a=0;a<4;a++)rec.push(findRecurrence(S[a],20,20,primes));

const lines=[];
lines.push('# Q5161 dependency-free exact computation','');
lines.push('Exact `BigInt` arithmetic was used for the table. Rank and recurrence searches were performed independently over `F_1000003` and `F_1000033`.','');
lines.push('## Exact values n=0,...,30','','```text','n | M0 | M1 | M2 | M3');
for(let n=0;n<=30;n++)lines.push(`${n} | ${M[0][n]} | ${M[1][n]} | ${M[2][n]} | ${M[3][n]}`);
lines.push('```','','## Cyclic-vector rank tests','');
for(const f of fits){ lines.push(`- **${f.name}**`); for(const [p,r] of f.byPrime)lines.push(`  - mod ${p}: rank=${r.rank}, augmented rank=${r.augrank}, unknowns=${r.unknowns}, rows=${r.rows}${r.augrank>r.rank?' — exact inconsistency certificate':''}`); }
lines.push('','## Minimal recurrences in the search box','');
lines.push('The search was exhaustive for order <=20 and polynomial degree <=20 on 521 exact terms, independently modulo both primes.');
for(let a=0;a<4;a++)lines.push(`- M${a}: ${rec[a]?`order **${rec[a].order}**, degree **${rec[a].degree}**`:'none found'}.`);
lines.push('','The geometric factor `256^n` used internally does not change recurrence order or degree.');
fs.writeFileSync(OUT,lines.join('\n')+'\n');
console.log(lines.join('\n'));
