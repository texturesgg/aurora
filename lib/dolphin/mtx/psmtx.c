// The SDK's paired-single matrix and vector routines (PSMTX*, PSVEC*), computing what the
// Gekko computes: the same operations in the same order, fused where the assembly fuses,
// each rounded as the Gekko rounds it (gekko.h). Each routine follows the SDK's assembly
// instruction by instruction, and its locals are named after the registers there.
//
// Every routine reads all of its inputs before it writes, as the assembly does, so the
// output may be an input.

#include <dolphin/mtx.h>

#include "gekko.h"

static const PS Unit01 = {0.0f, 1.0f};

void PSMTXIdentity(Mtx m) {
  const PS f0 = ps_fill(0.0f);
  PS f1 = ps_fill(1.0f);
  const PS f2 = ps_merge01(f0, f1);
  ps_store(&m[0][2], f0);
  ps_store(&m[1][2], f0);
  ps_store(&m[2][0], f0);
  ps_store(&m[1][0], f2);
  f1 = ps_merge10(f1, f0);
  ps_store(&m[0][0], f1);
  ps_store(&m[2][2], f1);
}

void PSMTXCopy(const Mtx src, Mtx dst) {
  for (int i = 0; i < 3; i++) {
    ps_store(&dst[i][0], ps_load(&src[i][0]));
    ps_store(&dst[i][2], ps_load(&src[i][2]));
  }
}

void PSMTXConcat(const Mtx a, const Mtx b, Mtx ab) {
  const PS f0 = ps_load(&a[0][0]);
  const PS f1 = ps_load(&a[0][2]);
  const PS f2 = ps_load(&a[1][0]);
  const PS f3 = ps_load(&a[1][2]);
  const PS f4 = ps_load(&a[2][0]);
  const PS f5 = ps_load(&a[2][2]);
  const PS f6 = ps_load(&b[0][0]);
  const PS f7 = ps_load(&b[0][2]);
  const PS f8 = ps_load(&b[1][0]);
  const PS f9 = ps_load(&b[1][2]);
  const PS f10 = ps_load(&b[2][0]);
  const PS f11 = ps_load(&b[2][2]);

  PS f12 = ps_muls0(f6, f0);
  PS f13 = ps_muls0(f7, f0);
  PS f14 = ps_muls0(f6, f2);
  PS f15 = ps_muls0(f7, f2);
  f12 = ps_madds1(f8, f0, f12);
  f14 = ps_madds1(f8, f2, f14);
  f13 = ps_madds1(f9, f0, f13);
  f15 = ps_madds1(f9, f2, f15);
  f12 = ps_madds0(f10, f1, f12);
  f13 = ps_madds0(f11, f1, f13);
  f14 = ps_madds0(f10, f3, f14);
  f15 = ps_madds0(f11, f3, f15);
  f13 = ps_madds1(Unit01, f1, f13);
  f15 = ps_madds1(Unit01, f3, f15);

  PS g2 = ps_muls0(f6, f4);
  PS g0 = ps_muls0(f7, f4);
  g2 = ps_madds1(f8, f4, g2);
  g0 = ps_madds1(f9, f4, g0);
  g2 = ps_madds0(f10, f5, g2);
  g0 = ps_madds0(f11, f5, g0);
  g0 = ps_madds1(Unit01, f5, g0);

  ps_store(&ab[0][0], f12);
  ps_store(&ab[0][2], f13);
  ps_store(&ab[1][0], f14);
  ps_store(&ab[1][2], f15);
  ps_store(&ab[2][0], g2);
  ps_store(&ab[2][2], g0);
}

void PSMTXTranspose(const Mtx src, Mtx xPose) {
  const PS f0 = ps_fill(0.0f);
  const PS f1 = ps_load(&src[0][0]);
  const PS f2 = ps_load(&src[1][0]);
  const PS f3 = ps_load1(&src[0][2]);
  const PS f2b = ps_load1(&src[1][2]);
  const PS f1b = ps_load(&src[2][0]);
  const f32 m22 = src[2][2];

  ps_store(&xPose[0][0], ps_merge00(f1, f2));
  ps_store(&xPose[1][0], ps_merge11(f1, f2));
  ps_store(&xPose[2][0], ps_merge00(f3, f2b));
  ps_store(&xPose[0][2], ps_merge00(f1b, f0));
  ps_store(&xPose[1][2], ps_merge10(f1b, f0));
  xPose[2][2] = m22;
  xPose[2][3] = 0.0f;
}

u32 PSMTXInverse(const Mtx src, Mtx inv) {
  const PS f0 = ps_load1(&src[0][0]);
  const PS f1 = ps_load(&src[0][1]);
  const PS f2 = ps_load1(&src[1][0]);
  PS f6 = ps_merge10(f1, f0);
  const PS f3 = ps_load(&src[1][1]);
  const PS f4 = ps_load1(&src[2][0]);
  PS f7 = ps_merge10(f3, f2);
  PS f5 = ps_load(&src[2][1]);
  PS f11 = ps_mul(f3, f6);
  PS f13 = ps_mul(f5, f7);
  PS f8 = ps_merge10(f5, f4);
  f11 = ps_msub(f1, f7, f11);
  PS f12 = ps_mul(f1, f8);
  f13 = ps_msub(f3, f8, f13);
  PS f10 = ps_mul(f3, f4);
  f12 = ps_msub(f5, f6, f12);
  PS f9 = ps_mul(f0, f5);
  f8 = ps_mul(f1, f2);
  f10 = ps_msub(f2, f5, f10);
  f7 = ps_mul(f0, f13);
  f9 = ps_msub(f1, f4, f9);
  f7 = ps_madd(f2, f12, f7);
  f8 = ps_msub(f0, f3, f8);
  f6 = ps_sub(f6, f6);
  f7 = ps_madd(f4, f11, f7);
  // ps_cmpo0: a NaN determinant goes on
  if (f7.ps0 == f6.ps0) {
    return 0;
  }

  PS r = ps_fill(gekko_fres(f7.ps0));
  f6 = ps_add(r, r);
  f5 = ps_mul(r, r);
  r = ps_nmsub(f7, f5, f6);
  const PS t1 = ps_fill(src[0][3]);
  const PS t2 = ps_fill(src[1][3]);
  const PS t3 = ps_fill(src[2][3]);
  f13 = ps_muls0(f13, r);
  f12 = ps_muls0(f12, r);
  f11 = ps_muls0(f11, r);
  f10 = ps_muls0(f10, r);
  f9 = ps_muls0(f9, r);
  f8 = ps_muls0(f8, r);
  PS x6 = ps_mul(f13, t1);
  x6 = ps_madd(f12, t2, x6);
  x6 = ps_nmadd(f11, t3, x6);
  PS x7 = ps_mul(f10, t1);
  x7 = ps_madd(f9, t2, x7);
  x7 = ps_nmadd(f8, t3, x7);

  ps_store(&inv[0][0], ps_merge00(f13, f12));
  ps_store(&inv[1][0], ps_merge11(f13, f12));
  ps_store1(&inv[2][0], f10);
  ps_store1(&inv[2][1], f9);
  ps_store1(&inv[2][2], f8);
  ps_store(&inv[0][2], ps_merge00(f11, x6));
  ps_store(&inv[1][2], ps_merge11(f11, x6));
  ps_store1(&inv[2][3], x7);
  return 1;
}

void PSMTXRotTrig(Mtx m, char axis, f32 sinA, f32 cosA) {
  const PS f0 = ps_fill(0.0f);
  const PS f3 = ps_fill(1.0f);
  const PS f1 = ps_fill(sinA);
  const PS f2 = ps_fill(cosA);
  const PS f4 = ps_neg(f1);

  switch (axis | 0x20) {
  case 'x':
    ps_store1(&m[0][0], f3);
    ps_store(&m[0][1], f0);
    ps_store(&m[0][3], f0);
    ps_store(&m[1][3], f0);
    ps_store1(&m[2][3], f0);
    ps_store(&m[2][1], ps_merge00(f1, f2));
    ps_store(&m[1][1], ps_merge00(f2, f4));
    break;
  case 'y':
    ps_store(&m[1][2], f0);
    ps_store(&m[0][0], ps_merge00(f2, f0));
    ps_store(&m[2][2], ps_merge00(f2, f0));
    ps_store(&m[1][0], ps_merge00(f0, f3));
    ps_store(&m[0][2], ps_merge00(f1, f0));
    ps_store(&m[2][0], ps_merge00(f4, f0));
    break;
  case 'z':
    ps_store(&m[0][2], f0);
    ps_store(&m[1][2], f0);
    ps_store(&m[2][0], f0);
    ps_store(&m[1][0], ps_merge00(f1, f2));
    ps_store(&m[0][0], ps_merge00(f2, f4));
    ps_store(&m[2][2], ps_merge00(f3, f0));
    break;
  default:
    break;
  }
}

void aurora_PSMTXRotAxisTrig(Mtx m, const Vec* axis, f32 sinA, f32 cosA) {
  Vec v;
  const PS f28 = ps_fill(0.0f);
  const PS f30 = ps_fill(sinA);
  const PS f31 = ps_fill(cosA);
  const PS f29 = ps_fill(1.0f - cosA);

  PSVECNormalize(axis, &v);
  PS f27 = ps_load(&v.x);
  const PS f1 = ps_fill(v.z);
  const PS f0 = ps_merge00(f31, f31);
  PS f4 = ps_muls0(f27, f29);
  PS f5 = ps_muls0(f1, f29);
  PS f3 = ps_muls1(f4, f27);
  PS f2 = ps_muls0(f4, f27);
  f27 = ps_muls0(f27, f30);
  f4 = ps_muls0(f4, f1);
  PS f6 = ps_fill(gekko_negate(fmaf(f1.ps0, f30.ps0, -f3.ps0)));  // fnmsubs
  const PS f7 = ps_fill(fmaf(f1.ps0, f30.ps0, f3.ps0));           // fmadds
  PS f9 = ps_neg(f27);
  const PS f8 = ps_sum0(f4, f28, f27);
  f2 = ps_sum0(f2, f6, f0);
  f3 = ps_sum1(f0, f7, f3);
  f6 = ps_sum0(f9, f28, f4);
  f9 = ps_sum0(f4, f4, f9);
  f5 = ps_muls0(f5, f1);
  f4 = ps_sum1(f27, f9, f4);
  f5 = ps_sum0(f5, f28, f0);

  ps_store(&m[0][2], f8);
  ps_store(&m[0][0], f2);
  ps_store(&m[1][0], f3);
  ps_store(&m[1][2], f6);
  ps_store(&m[2][0], f4);
  ps_store(&m[2][2], f5);
}

void PSMTXTrans(Mtx m, f32 xT, f32 yT, f32 zT) {
  m[0][0] = 1.0f;
  m[0][1] = 0.0f;
  m[0][2] = 0.0f;
  m[0][3] = xT;
  m[1][0] = 0.0f;
  m[1][1] = 1.0f;
  m[1][2] = 0.0f;
  m[1][3] = yT;
  m[2][0] = 0.0f;
  m[2][1] = 0.0f;
  m[2][2] = 1.0f;
  m[2][3] = zT;
}

void PSMTXScale(Mtx m, f32 xS, f32 yS, f32 zS) {
  m[0][0] = xS;
  m[0][1] = 0.0f;
  m[0][2] = 0.0f;
  m[0][3] = 0.0f;
  m[1][0] = 0.0f;
  m[1][1] = yS;
  m[1][2] = 0.0f;
  m[1][3] = 0.0f;
  m[2][0] = 0.0f;
  m[2][1] = 0.0f;
  m[2][2] = zS;
  m[2][3] = 0.0f;
}

void PSMTXQuat(Mtx m, const Quaternion* q) {
  const PS f1 = ps_fill(1.0f);
  const PS f4 = ps_load(&q->x);
  PS f5 = ps_load(&q->z);
  const PS f0 = ps_fill(1.0f - 1.0f);
  const PS f2 = ps_fill(1.0f + 1.0f);
  PS f6 = ps_mul(f4, f4);
  PS f9 = ps_merge10(f4, f4);
  PS f8 = ps_madd(f5, f5, f6);
  PS f7 = ps_mul(f5, f5);
  PS f3 = ps_sum0(f8, f8, f8);
  PS f10 = ps_muls1(f9, f5);
  PS f11 = ps_fill(gekko_fres(f3.ps0));
  f8 = ps_sum1(f7, f8, f6);
  f3 = ps_nmsub(f3, f11, f2);
  f7 = ps_muls1(f5, f5);
  f3 = ps_mul(f11, f3);
  f6 = ps_sum0(f6, f6, f6);
  f3 = ps_fill(f3.ps0 * f2.ps0);  // fmuls
  f11 = ps_madd(f4, f9, f7);
  f7 = ps_msub(f4, f9, f7);
  f6 = ps_nmsub(f6, f3, f1);
  f8 = ps_nmsub(f8, f3, f1);
  f11 = ps_mul(f11, f3);
  f7 = ps_mul(f7, f3);
  f9 = ps_madds0(f4, f5, f10);
  f5 = ps_merge00(f11, f8);
  f10 = ps_nmsub(f10, f2, f9);
  const PS g4 = ps_merge10(f8, f7);
  f9 = ps_mul(f9, f3);
  f10 = ps_mul(f10, f3);

  ps_store1(&m[0][3], f0);
  ps_store1(&m[2][3], f0);
  ps_store1(&m[2][2], f6);
  ps_store(&m[1][0], f5);
  ps_store(&m[0][0], g4);
  ps_store1(&m[0][2], f9);
  ps_store(&m[1][2], ps_merge10(f10, f0));
  ps_store(&m[2][0], ps_merge01(f10, f9));
}

void PSMTXMultVec(const Mtx m, const Vec* src, Vec* dst) {
  const PS f0 = ps_load(&src->x);
  const PS f1 = ps_load1(&src->z);
  const PS f5 = ps_madd(ps_load(&m[0][2]), f1, ps_mul(ps_load(&m[0][0]), f0));
  const PS f11 = ps_madd(ps_load(&m[1][2]), f1, ps_mul(ps_load(&m[1][0]), f0));
  const PS f5b = ps_madd(ps_load(&m[2][2]), f1, ps_mul(ps_load(&m[2][0]), f0));
  dst->x = ps_sum0(f5, f5, f5).ps0;
  dst->y = ps_sum0(f11, f11, f11).ps0;
  dst->z = ps_sum0(f5b, f5b, f5b).ps0;
}

void PSMTXMultVecSR(const Mtx m, const Vec* src, Vec* dst) {
  const PS f6 = ps_load(&src->x);
  const PS f7 = ps_load1(&src->z);
  PS f8 = ps_mul(ps_load(&m[0][0]), f6);
  PS f10 = ps_mul(ps_load(&m[1][0]), f6);
  PS f12 = ps_mul(ps_load(&m[2][0]), f6);
  f8 = ps_sum0(f8, f8, f8);
  f10 = ps_sum0(f10, f10, f10);
  f12 = ps_sum0(f12, f12, f12);
  dst->x = ps_madd(ps_load(&m[0][2]), f7, f8).ps0;
  dst->y = ps_madd(ps_load(&m[1][2]), f7, f10).ps0;
  dst->z = ps_madd(ps_load(&m[2][2]), f7, f12).ps0;
}

void PSVECAdd(const Vec* a, const Vec* b, Vec* ab) {
  const PS f6 = ps_add(ps_load(&a->x), ps_load(&b->x));
  const PS f7 = ps_add(ps_load1(&a->z), ps_load1(&b->z));
  ps_store(&ab->x, f6);
  ps_store1(&ab->z, f7);
}

void PSVECSubtract(const Vec* a, const Vec* b, Vec* a_b) {
  const PS f6 = ps_sub(ps_load(&a->x), ps_load(&b->x));
  const PS f7 = ps_sub(ps_load1(&a->z), ps_load1(&b->z));
  ps_store(&a_b->x, f6);
  ps_store1(&a_b->z, f7);
}

void PSVECScale(const Vec* src, Vec* dst, f32 scale) {
  const PS f1 = ps_fill(scale);
  const PS f0 = ps_muls0(ps_load(&src->x), f1);
  const PS f2 = ps_muls0(ps_load1(&src->z), f1);
  ps_store(&dst->x, f0);
  ps_store1(&dst->z, f2);
}

// The squared length as PSVECNormalize and PSVECMag sum it: (z*z + x*x) fused, then + y*y
static inline f32 gekko_vec_square_mag(const Vec* v) {
  const PS f2 = ps_load(&v->x);
  const PS f5 = ps_mul(f2, f2);
  const PS f3 = ps_load1(&v->z);
  const PS f4 = ps_madd(f3, f3, f5);
  return ps_sum0(f4, f3, f5).ps0;
}

// 1/sqrt(s): frsqrte's estimate and one Newton step, (0.5 * r) * (3 - s * (r * r))
static inline f32 gekko_rsqrt_step(f32 s) {
  const double r = frsqrte(s);
  const f32 rr = gekko_fmuls(r, r);
  const f32 half_r = gekko_fmuls(r, 0.5);
  const f32 step = gekko_negate(fmaf(rr, s, -3.0f));  // fnmsubs
  return step * half_r;
}

void PSVECNormalize(const Vec* src, Vec* unit) {
  const f32 f5 = gekko_rsqrt_step(gekko_vec_square_mag(src));
  const PS s = ps_fill(f5);
  const PS f2 = ps_muls0(ps_load(&src->x), s);
  const PS f3 = ps_muls0(ps_load1(&src->z), s);
  ps_store(&unit->x, f2);
  ps_store1(&unit->z, f3);
}

f32 PSVECMag(const Vec* v) {
  const f32 s = gekko_vec_square_mag(v);
  const f32 r = gekko_rsqrt_step(s);
  // fsel: a zero length gives r = NaN, and 0 * 0
  return s * (r >= 0.0f ? r : s);
}

f32 PSVECDotProduct(const Vec* a, const Vec* b) {
  const PS f2 = ps_mul(ps_load(&a->y), ps_load(&b->y));
  const PS f3 = ps_madd(ps_load(&a->x), ps_load(&b->x), f2);
  return ps_sum0(f3, f2, f2).ps0;
}

void PSVECCrossProduct(const Vec* a, const Vec* b, Vec* axb) {
  const PS f1 = ps_load(&b->x);
  const PS f2 = ps_fill(a->z);
  const PS f0 = ps_load(&a->x);
  const PS f6 = ps_merge10(f1, f1);
  const PS f3 = ps_fill(b->z);
  const PS f4 = ps_mul(f1, f2);
  const PS f7 = ps_muls0(f1, f0);
  const PS f5 = ps_msub(f0, f3, f4);
  const PS f8 = ps_msub(f0, f6, f7);
  const PS f9 = ps_merge11(f5, f5);
  const PS f10 = ps_neg(ps_merge01(f5, f8));
  ps_store1(&axb->x, f9);
  ps_store(&axb->y, f10);
}
