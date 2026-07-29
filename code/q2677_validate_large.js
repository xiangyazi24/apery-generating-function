'use strict';

// Independent large-range validation for Q2677.
// Computes exact integer Apéry numbers up to M*MAX_P and only then reduces
// endpoint values modulo p^6.  This avoids every modular-division issue at
// indices n+1 divisible by p.

const MAX_P = 4999;
const M = 20;

function modBig(a, m) { const r = a % m; return r < 0n ? r + m : r; }
function modInt(a, p) { a %= p; return a < 0 ? a + p : a; }
function egcd(a,b){let r0=a,r1=b,s0=1,s1=0;while(r1){const q=Math.floor(r0/r1);[r0,r1]=[r1,r0-q*r1];[s0,s1]=[s1,s0-q*s1];}return [r0,s0];}
function invInt(a,p){a=modInt(a,p);const [g,x]=egcd(a,p);if(g!==1)throw new Error(`nonunit ${a} mod ${p}`);return modInt(x,p);}
function powInt(a,e,p){let z=1;a=modInt(a,p);while(e){if(e&1)z=(z*a)%p;a=(a*a)%p;e=Math.floor(e/2);}return z;}
function primesUpTo(n){const s=new Uint8Array(n+1),out=[];for(let i=2;i<=n;i++)if(!s[i]){out.push(i);if(i*i<=n)for(let j=i*i;j<=n;j+=i)s[j]=1;}return out;}
function harmonicHalf(p,k){let z=0;for(let i=1;i<=(p-1)/2;i++)z=(z+powInt(invInt(i,p),k,p))%p;return z;}
function addTarget(map,n,p,key){if(!map.has(n))map.set(n,[]);map.get(n).push({p,key});}

const primes=primesUpTo(MAX_P).filter(p=>p>=7);
const recs=new Map(),targets=new Map();
let maxN=0;
for(const p of primes){
  const pb=BigInt(p),p6=pb**6n;
  recs.set(p,{p,p6,cap:{}});
  addTarget(targets,p-1,p,'delta');
  for(let m=1;m<=M;m++){
    addTarget(targets,m*p,p,`d${m}`);
    addTarget(targets,m*p-1,p,`f${m}`);
    maxN=Math.max(maxN,m*p);
  }
}
const bSmall=Array(M+1);
function capture(n,b){
  if(n<=M)bSmall[n]=b;
  const rows=targets.get(n);if(!rows)return;
  for(const {p,key} of rows){const r=recs.get(p);r.cap[key]=modBig(b,r.p6);}
}
let bm1=1n,bn=5n;capture(0,bm1);capture(1,bn);
for(let n=1;n<maxN;n++){
  const x=BigInt(n),x2=x*x,x3=x2*x;
  const A=34n*x3+51n*x2+27n*x+5n;
  const y=x+1n,den=y*y*y,num=A*bn-x3*bm1;
  if(num%den!==0n)throw new Error(`nonintegral recurrence at ${n}`);
  const bp1=num/den;bm1=bn;bn=bp1;capture(n+1,bn);
}

const E=Array(M+1),F=Array(M+1),P=Array(M+1),Q=Array(M+1);
for(let m=1;m<=M;m++){
  const x=BigInt(m),x2=x*x,x3=x2*x;
  const e=x3*(bSmall[m-1]-17n*bSmall[m]);
  const f=x3*(17n*bSmall[m-1]-bSmall[m]);
  const pp=x3*((68n+7n*x2)*bSmall[m]-(4n+11n*x2)*bSmall[m-1]);
  const qq=x3*((4n+11n*x2)*bSmall[m]-(68n+7n*x2)*bSmall[m-1]);
  if(e%12n||f%12n||pp%360n||qq%360n)throw new Error(`nonintegral carrier at m=${m}`);
  E[m]=e/12n;F[m]=f/12n;P[m]=pp/360n;Q[m]=qq/360n;
}

let divFail=0,thetaFail=0,rankFail=0,b5RecurrenceFail=0;
let tested=0;
const thetaZero=[];
const examples=[];

function bernoulliMod(n,p){
  const B=Array(n+1).fill(0);B[0]=1;
  for(let r=1;r<=n;r++){
    let sum=0,c=1;
    // B_r = -(1/(r+1))*sum_{k=0}^{r-1} C(r+1,k)B_k.
    for(let k=0;k<r;k++){
      if(k>0)c=modInt(c*(r+2-k)*invInt(k,p),p); // C(r+1,k)
      sum=modInt(sum+c*B[k],p);
    }
    B[r]=modInt(-sum*invInt(r+1,p),p);
  }
  return B[n];
}

for(const p of primes){
  const r=recs.get(p),pb=BigInt(p),p3=pb**3n,p5=pb**5n,p6=r.p6;
  const delta=modBig(r.cap.delta-1n,p6);
  if(delta%p3!==0n)throw new Error(`Delta valuation failure p=${p}`);
  const B5=modInt(-harmonicHalf(p,5)*invInt(6,p),p);
  if(p<=199){
    const B5b=bernoulliMod(p-5,p);
    if(B5!==B5b){b5RecurrenceFail++;console.log(`B5_REC_FAIL p=${p} half=${B5} rec=${B5b}`);}
  }
  let theta=null,allZero=true;
  for(let m=1;m<=M;m++){
    const dn=modBig(r.cap[`d${m}`]-bSmall[m]-E[m]*delta,p6);
    const fn=modBig(r.cap[`f${m}`]-bSmall[m-1]-F[m]*delta,p6);
    if(dn%p5!==0n||fn%p5!==0n){divFail++;if(divFail<10)console.log(`DIV_FAIL p=${p} m=${m}`);}
    const d=Number((dn/p5)%pb),f=Number((fn/p5)%pb);
    if(m===1)theta=d;
    const pm=Number(modBig(P[m],pb)),qm=Number(modBig(Q[m],pb));
    if(d!==modInt(pm*theta,p)||f!==modInt(qm*theta,p)){
      rankFail++;if(rankFail<10)console.log(`RANK_FAIL p=${p} m=${m} d=${d} f=${f} theta=${theta}`);
    }
    if(d||f)allZero=false;
    tested++;
  }
  if(theta!==modInt(-24*B5,p)){
    thetaFail++;if(thetaFail<10)console.log(`THETA_FAIL p=${p} theta=${theta} B5=${B5}`);
  }
  if(theta===0){thetaZero.push(p);if(!allZero){rankFail++;console.log(`ZERO_STRESS_FAIL p=${p}`);}}
  if([1009,2003,3001,4001,4999].includes(p))examples.push({p,theta,B5,d2:Number((modBig(r.cap.d2-bSmall[2]-E[2]*delta,p6)/p5)%pb),f2:Number((modBig(r.cap.f2-bSmall[1]-F[2]*delta,p6)/p5)%pb)});
}

console.log(`VALIDATION_SUMMARY primes=${primes.length} pRange=7..${MAX_P} M=${M} pairs=${tested} maxN=${maxN}`);
console.log(`FAILURES divisibility=${divFail} thetaB5=${thetaFail} rank1=${rankFail} B5HalfVsRecurrence=${b5RecurrenceFail}`);
console.log(`THETA_ZERO count=${thetaZero.length} primes=${thetaZero.join(',')}`);
console.log('LARGE_EXAMPLES=' + JSON.stringify(examples));
console.log('CARRIER_TABLE_HEADER=m,P_m,Q_m,PplusQ,PminusQ,det(EF;PQ)');
for(let m=1;m<=M;m++){
  const plus=P[m]+Q[m],minus=P[m]-Q[m],det=E[m]*Q[m]-F[m]*P[m];
  const detClosed=-(BigInt(m)**8n)*(bSmall[m]*bSmall[m]-bSmall[m-1]*bSmall[m-1])/24n;
  if(det!==detClosed)throw new Error(`det identity fail m=${m}`);
  console.log(`CARRIER=${m},${P[m]},${Q[m]},${plus},${minus},${det}`);
}
console.log('SAME_INDEX_TOP old=(-7,8) new=(1,336/5) determinant=-2392/5');
