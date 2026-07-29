'use strict';

// Q2677: exact Apéry endpoint computation modulo p^6.
// This script deliberately computes the integer Apéry recurrence exactly,
// then reduces only at the requested endpoint indices.  No modular division
// across n+1 divisible by p is used.

function modBig(a, m) {
  const r = a % m;
  return r < 0n ? r + m : r;
}

function absBig(a) { return a < 0n ? -a : a; }

function gcdBig(a, b) {
  a = absBig(a); b = absBig(b);
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
}

function isqrt(n) {
  if (n < 0n) throw new Error('sqrt of negative');
  if (n < 2n) return n;
  let x0 = 1n << (BigInt(n.toString(2).length) >> 1n);
  let x1 = (x0 + n / x0) >> 1n;
  while (x1 < x0) {
    x0 = x1;
    x1 = (x0 + n / x0) >> 1n;
  }
  return x0;
}

function egcdInt(a, b) {
  let oldR = a, r = b;
  let oldS = 1, s = 0;
  while (r !== 0) {
    const q = Math.floor(oldR / r);
    [oldR, r] = [r, oldR - q * r];
    [oldS, s] = [s, oldS - q * s];
  }
  return [oldR, oldS];
}

function modInt(a, p) {
  a %= p;
  return a < 0 ? a + p : a;
}

function invInt(a, p) {
  a = modInt(a, p);
  const [g, x] = egcdInt(a, p);
  if (g !== 1) throw new Error(`nonunit ${a} mod ${p}`);
  return modInt(x, p);
}

function powModInt(a, e, p) {
  let z = 1;
  a = modInt(a, p);
  while (e > 0) {
    if (e & 1) z = (z * a) % p;
    a = (a * a) % p;
    e = Math.floor(e / 2);
  }
  return z;
}

function powModBig(a, e, m) {
  let z = 1n;
  a = modBig(a, m);
  while (e > 0n) {
    if (e & 1n) z = (z * a) % m;
    a = (a * a) % m;
    e >>= 1n;
  }
  return z;
}

function primesUpTo(n) {
  const sieve = new Uint8Array(n + 1);
  const out = [];
  for (let i = 2; i <= n; i++) {
    if (!sieve[i]) {
      out.push(i);
      if (i * i <= n) for (let j = i * i; j <= n; j += i) sieve[j] = 1;
    }
  }
  return out;
}

function addTarget(targets, n, p, key) {
  if (!targets.has(n)) targets.set(n, []);
  targets.get(n).push({p, key});
}

function harmonicHalf(p, power) {
  let s = 0;
  for (let k = 1; k <= (p - 1) / 2; k++) {
    const inv = invInt(k, p);
    s = (s + powModInt(inv, power, p)) % p;
  }
  return s;
}

function crt(residues) {
  let R = 0n, N = 1n;
  for (const {r, p} of residues) {
    const pb = BigInt(p);
    const nmod = Number(N % pb);
    const diff = modInt(r - Number(R % pb), p);
    const t = modInt(diff * invInt(nmod, p), p);
    R += N * BigInt(t);
    N *= pb;
    R %= N;
  }
  return {R, N};
}

function rationalReconstruct(a, m) {
  a = modBig(a, m);
  const bound = isqrt(m / 2n);
  let r0 = m, r1 = a;
  let t0 = 0n, t1 = 1n;
  while (absBig(r1) > bound) {
    const q = r0 / r1;
    [r0, r1] = [r1, r0 - q * r1];
    [t0, t1] = [t1, t0 - q * t1];
  }
  if (t1 === 0n || absBig(t1) > bound) return null;
  let num = r1, den = t1;
  if (den < 0n) { num = -num; den = -den; }
  const g = gcdBig(num, den);
  num /= g; den /= g;
  if (modBig(num - a * den, m) !== 0n) return null;
  return {num, den};
}

function ratMod(q, p) {
  const n = Number(modBig(q.num, BigInt(p)));
  const d = Number(modBig(q.den, BigInt(p)));
  if (d === 0) return null;
  return modInt(n * invInt(d, p), p);
}

function fmtRat(q) {
  if (!q) return 'FAIL';
  return q.den === 1n ? q.num.toString() : `${q.num}/${q.den}`;
}

const MAX_P = 997;
const M = 12;
const TRAIN_MAX_P = 499;
const primes = primesUpTo(MAX_P).filter(p => p >= 7);
const records = new Map();
const targets = new Map();
let maxN = 0;

for (const p of primes) {
  const pb = BigInt(p);
  const p6 = pb ** 6n;
  const rec = {p, p6, cap: {}, d: Array(M + 1), f: Array(M + 1)};
  records.set(p, rec);
  addTarget(targets, p - 1, p, 'delta0');
  maxN = Math.max(maxN, p - 1);
  for (let m = 1; m <= M; m++) {
    addTarget(targets, m * p, p, `d${m}`);
    addTarget(targets, m * p - 1, p, `f${m}`);
    maxN = Math.max(maxN, m * p);
  }
}

const bSmall = Array(M + 1);
function capture(n, value) {
  if (n <= M) bSmall[n] = value;
  const list = targets.get(n);
  if (!list) return;
  for (const {p, key} of list) {
    const rec = records.get(p);
    rec.cap[key] = modBig(value, rec.p6);
  }
}

let bPrev = 1n;
let bCur = 5n;
capture(0, bPrev);
capture(1, bCur);
for (let n = 1; n < maxN; n++) {
  const nb = BigInt(n);
  const n2 = nb * nb, n3 = n2 * nb;
  const A = 34n * n3 + 51n * n2 + 27n * nb + 5n;
  const np1 = nb + 1n;
  const den = np1 * np1 * np1;
  const num = A * bCur - n3 * bPrev;
  if (num % den !== 0n) throw new Error(`nonintegral recurrence at n=${n}`);
  const bNext = num / den;
  bPrev = bCur;
  bCur = bNext;
  capture(n + 1, bCur);
}

const E = Array(M + 1), F = Array(M + 1);
for (let m = 1; m <= M; m++) {
  const mb = BigInt(m);
  const m3 = mb ** 3n;
  const eNum = m3 * (bSmall[m - 1] - 17n * bSmall[m]);
  const fNum = m3 * (17n * bSmall[m - 1] - bSmall[m]);
  if (eNum % 12n || fNum % 12n) throw new Error(`E/F nonintegral at m=${m}`);
  E[m] = eNum / 12n;
  F[m] = fNum / 12n;
}

let divisibilityFailures = 0;
for (const p of primes) {
  const rec = records.get(p);
  const pb = BigInt(p), p3 = pb ** 3n, p5 = pb ** 5n, p6 = rec.p6;
  rec.delta = modBig(rec.cap.delta0 - 1n, p6);
  if (rec.delta % p3 !== 0n) throw new Error(`Delta not p^3-divisible at p=${p}`);
  rec.deltaDigits = Number((rec.delta / p3) % pb);
  for (let m = 1; m <= M; m++) {
    const dn = modBig(rec.cap[`d${m}`] - bSmall[m] - E[m] * rec.delta, p6);
    const fn = modBig(rec.cap[`f${m}`] - bSmall[m - 1] - F[m] * rec.delta, p6);
    if (dn % p5 !== 0n || fn % p5 !== 0n) {
      divisibilityFailures++;
      console.log(`DIVFAIL p=${p} m=${m} dn/p5rem=${dn % p5} fn/p5rem=${fn % p5}`);
    }
    rec.d[m] = Number((dn / p5) % pb);
    rec.f[m] = Number((fn / p5) % pb);
  }
  const h3 = harmonicHalf(p, 3);
  const h5 = harmonicHalf(p, 5);
  rec.B3 = modInt(-h3 * invInt(2, p), p);
  rec.B5 = modInt(-h5 * invInt(6, p), p);
  rec.B3sq = (rec.B3 * rec.B3) % p;
  const p2 = pb * pb;
  rec.q2 = Number(((powModBig(2n, BigInt(p - 1), p2) - 1n) / pb) % pb);
  rec.q2B3 = (rec.q2 * rec.B3) % p;
  rec.theta = rec.d[1];
}

console.log(`SUMMARY primes=${primes.length} range=7..${MAX_P} M=${M} maxN=${maxN} divisibilityFailures=${divisibilityFailures}`);
console.log('SMALL_B=' + bSmall.slice(0, M + 1).map(x => x.toString()).join(','));
console.log('E=' + E.slice(1).map(x => x.toString()).join(','));
console.log('F=' + F.slice(1).map(x => x.toString()).join(','));

const selected = [7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97];
console.log('RAW_HEADER=p,Delta/p3,d1,B3,B5,B3sq,q2,q2B3,d2,d3,d4,f2,f3,f4');
for (const p of selected) {
  const r = records.get(p);
  console.log('RAW=' + [p,r.deltaDigits,r.theta,r.B3,r.B5,r.B3sq,r.q2,r.q2B3,r.d[2],r.d[3],r.d[4],r.f[2],r.f[3],r.f[4]].join(','));
}

function reconstructRatios(componentGetter, anchorGetter, label) {
  const training = [];
  for (const p of primes) {
    if (p > TRAIN_MAX_P) continue;
    const rec = records.get(p);
    const a = anchorGetter(rec);
    if (a === 0) continue;
    const ratio = modInt(componentGetter(rec) * invInt(a, p), p);
    training.push({r: ratio, p});
  }
  const {R, N} = crt(training);
  const q = rationalReconstruct(R, N);
  let holdoutTested = 0, holdoutFailures = 0;
  if (q) {
    for (const p of primes) {
      if (p <= TRAIN_MAX_P) continue;
      const rec = records.get(p);
      const a = anchorGetter(rec);
      const qm = ratMod(q, p);
      if (a === 0 || qm === null) continue;
      holdoutTested++;
      if (componentGetter(rec) !== (qm * a) % p) holdoutFailures++;
    }
  }
  console.log(`${label}=${fmtRat(q)} train=${training.length} holdout=${holdoutTested} failures=${holdoutFailures}`);
  return {q, training: training.length, holdoutTested, holdoutFailures};
}

console.log('RANK1_THETA_BEGIN');
let rank1All = true;
const dCoeff = Array(M + 1), fCoeff = Array(M + 1);
for (let m = 1; m <= M; m++) {
  const rd = reconstructRatios(r => r.d[m], r => r.theta, `DCOEFF_m${m}`);
  const rf = reconstructRatios(r => r.f[m], r => r.theta, `FCOEFF_m${m}`);
  dCoeff[m] = rd.q; fCoeff[m] = rf.q;
  if (!rd.q || rd.holdoutFailures || !rf.q || rf.holdoutFailures) rank1All = false;
}
console.log(`RANK1_THETA_ALL=${rank1All}`);
console.log('RANK1_THETA_END');

console.log('THETA_CANDIDATE_RATIOS_BEGIN');
for (const [name, getter] of [
  ['B5', r => r.B5],
  ['B3sq', r => r.B3sq],
  ['q2B3', r => r.q2B3],
  ['B3', r => r.B3],
  ['DeltaDigit', r => r.deltaDigits],
]) {
  reconstructRatios(r => r.theta, getter, `THETA_OVER_${name}`);
}
console.log('THETA_CANDIDATE_RATIOS_END');

// Independent direct/reflected endpoint determinant against the rank-one carrier.
// If rank one holds, each pair (d_m,f_m) is theta_p times a fixed vector.
// Print determinants D_m F_n-D_n F_m for small indices as an exact sequence diagnostic.
if (rank1All) {
  console.log('COEFF_TABLE_HEADER=m,D_m,F_m');
  for (let m = 1; m <= M; m++) console.log(`COEFF=${m},${fmtRat(dCoeff[m])},${fmtRat(fCoeff[m])}`);
}

// Full falsification: use reconstructed rank-one coefficients on every prime,
// including primes excluded because theta=0.  At theta=0 all residuals must vanish.
let fullRank1Failures = 0;
if (rank1All) {
  for (const p of primes) {
    const r = records.get(p);
    for (let m = 1; m <= M; m++) {
      const cd = ratMod(dCoeff[m], p), cf = ratMod(fCoeff[m], p);
      if (cd === null || cf === null || r.d[m] !== (cd * r.theta) % p || r.f[m] !== (cf * r.theta) % p) {
        fullRank1Failures++;
        if (fullRank1Failures <= 20) console.log(`RANK1_FULL_FAIL p=${p} m=${m}`);
      }
    }
  }
}
console.log(`RANK1_FULL_FAILURES=${fullRank1Failures}`);

// Theta-zero primes are especially strong falsification points.
const thetaZero = primes.filter(p => records.get(p).theta === 0);
console.log('THETA_ZERO_PRIMES=' + thetaZero.join(','));
for (const p of thetaZero) {
  const r = records.get(p);
  const nz = [];
  for (let m = 1; m <= M; m++) if (r.d[m] || r.f[m]) nz.push(`${m}:${r.d[m]}/${r.f[m]}`);
  console.log(`THETA_ZERO_CHECK p=${p} nonzero=${nz.join(';') || 'none'}`);
}
