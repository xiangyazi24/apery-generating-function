'use strict';

function abs(a){return a<0n?-a:a;}
function gcd(a,b){a=abs(a);b=abs(b);while(b){[a,b]=[b,a%b];}return a;}
class Q {
  constructor(n,d=1n){
    if(d===0n) throw new Error('zero denominator');
    if(d<0n){n=-n;d=-d;}
    const g=gcd(n,d); this.n=n/g; this.d=d/g;
  }
  add(o){return new Q(this.n*o.d+o.n*this.d,this.d*o.d);}
  sub(o){return new Q(this.n*o.d-o.n*this.d,this.d*o.d);}
  mul(o){return new Q(this.n*o.n,this.d*o.d);}
  div(o){return new Q(this.n*o.d,this.d*o.n);}
  neg(){return new Q(-this.n,this.d);}
  is0(){return this.n===0n;}
  eq(o){return this.n===o.n&&this.d===o.d;}
  toString(){return this.d===1n?this.n.toString():`${this.n}/${this.d}`;}
}
const q=(n,d=1n)=>new Q(BigInt(n),BigInt(d));

function solveLinear(A,b){
  const m=A.length,n=A[0].length;
  const M=A.map((row,i)=>row.map(x=>x).concat([b[i]]));
  let r=0; const piv=[];
  for(let c=0;c<n&&r<m;c++){
    let s=r; while(s<m&&M[s][c].is0())s++;
    if(s===m)continue;
    [M[r],M[s]]=[M[s],M[r]];
    const inv=M[r][c];
    for(let j=c;j<=n;j++)M[r][j]=M[r][j].div(inv);
    for(let i=0;i<m;i++)if(i!==r&&!M[i][c].is0()){
      const f=M[i][c];
      for(let j=c;j<=n;j++)M[i][j]=M[i][j].sub(f.mul(M[r][j]));
    }
    piv.push(c);r++;
  }
  for(let i=r;i<m;i++){
    let all=true;for(let c=0;c<n;c++)if(!M[i][c].is0())all=false;
    if(all&&!M[i][n].is0())return {status:'inconsistent'};
  }
  if(r<n)return {status:'underdetermined',rank:r};
  const x=Array(n).fill(null).map(()=>q(0));
  for(let i=0;i<r;i++)x[piv[i]]=M[i][n];
  return {status:'unique',x};
}

const b=[
'1','5','73','1445','33001','819005','21460825','584307365','16367912425','468690849005','13657436403073','403676083788125','12073365010564729'
].map(BigInt);
const P=['0','1','752/5','68166/5','1009792','65906575','3923917200','217706958252','11423506768896','573013050183045','27698545722940400','1298209110832084682','296397675127490541696/5'];
const R=['0','0','336/5','52227/5','1009792','76556650','4993349040','293892404281','16052592919552','828396488075940','40882931158213200','1946258512102316643','449715556094898411648/5'];
function parse(s){const z=s.split('/');return new Q(BigInt(z[0]),z[1]?BigInt(z[1]):1n);}
const seqs={P:P.map(parse),R:R.map(parse)};

function powBI(a,e){let z=1n;for(let i=0;i<e;i++)z*=a;return z;}

function fitSequence(name,S){
  console.log(`FIT_BEGIN ${name}`);
  for(let factor=0;factor<=7;factor++){
    for(let deg=0;deg<=5;deg++){
      const vars=2*(deg+1);
      if(vars>12)continue;
      const A=[],y=[];
      for(let m=1;m<=12;m++){
        const mb=BigInt(m), mf=powBI(mb,factor);
        const row=[];
        for(let j=0;j<=deg;j++)row.push(q(mf*powBI(mb,j)*b[m]));
        for(let j=0;j<=deg;j++)row.push(q(mf*powBI(mb,j)*b[m-1]));
        A.push(row); y.push(S[m]);
      }
      const sol=solveLinear(A,y);
      if(sol.status==='unique'){
        console.log(`FIT ${name} factor=m^${factor} deg=${deg} coeff=${sol.x.map(x=>x.toString()).join(',')}`);
      }
    }
  }
  console.log(`FIT_END ${name}`);
}
fitSequence('P',seqs.P);
fitSequence('R',seqs.R);

// Also test a common-coefficient reflection ansatz:
// P_m=m^a(A(m)b_m+B(m)b_{m-1}),
// R_m=m^a(B(m)b_m+A(m)b_{m-1}) or signed variants.
function fitJoint(mode,factor,deg){
  const vars=2*(deg+1), A=[],y=[];
  for(let m=1;m<=12;m++){
    const mb=BigInt(m),mf=powBI(mb,factor);
    const rowP=[],rowR=[];
    for(let j=0;j<=deg;j++){
      const t=mf*powBI(mb,j);
      rowP.push(q(t*b[m]));
    }
    for(let j=0;j<=deg;j++){
      const t=mf*powBI(mb,j);
      rowP.push(q(t*b[m-1]));
    }
    if(mode==='swap'){
      for(let j=0;j<=deg;j++){const t=mf*powBI(mb,j);rowR.push(q(t*b[m-1]));}
      for(let j=0;j<=deg;j++){const t=mf*powBI(mb,j);rowR.push(q(t*b[m]));}
    } else if(mode==='swap-negA'){
      for(let j=0;j<=deg;j++){const t=mf*powBI(mb,j);rowR.push(q(-t*b[m-1]));}
      for(let j=0;j<=deg;j++){const t=mf*powBI(mb,j);rowR.push(q(t*b[m]));}
    } else if(mode==='swap-negB'){
      for(let j=0;j<=deg;j++){const t=mf*powBI(mb,j);rowR.push(q(t*b[m-1]));}
      for(let j=0;j<=deg;j++){const t=mf*powBI(mb,j);rowR.push(q(-t*b[m]));}
    }
    A.push(rowP);y.push(seqs.P[m]);A.push(rowR);y.push(seqs.R[m]);
  }
  return solveLinear(A,y);
}
for(const mode of ['swap','swap-negA','swap-negB'])for(let factor=0;factor<=7;factor++)for(let deg=0;deg<=5;deg++){
  const sol=fitJoint(mode,factor,deg);
  if(sol.status==='unique')console.log(`JOINT mode=${mode} factor=m^${factor} deg=${deg} coeff=${sol.x.map(x=>x.toString()).join(',')}`);
}
