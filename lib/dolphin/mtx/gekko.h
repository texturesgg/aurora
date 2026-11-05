#ifndef AURORA_MTX_GEKKO_H
#define AURORA_MTX_GEKKO_H

// The Gekko's floating-point unit, as the SDK's paired-single routines use it (psmtx.c).
//
// A paired-single register holds two lanes, and each lane of each instruction is one
// single-precision operation, rounded once: ps_madd is fmaf in each lane, ps_sum0 is one
// addition. A lane here is a float. The two estimate instructions, frsqrte and fres, read
// tables (frsqrte is ppc_math.h's). Single-precision multiplies round their second
// operand (frC) to 26 significant bits first, which only changes a value that isn't a
// float already: frsqrte's estimate, the only one of those these routines multiply.
//
// The routines must be built without contraction (-ffp-contract=off): every fused
// operation here is an explicit fmaf.

#include <dolphin/ppc_math.h>
#include <dolphin/types.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  f32 ps0;
  f32 ps1;
} PS;

static inline uint64_t gekko_bits(double d) {
  uint64_t u;
  memcpy(&u, &d, sizeof(u));
  return u;
}

static inline double gekko_double(uint64_t u) {
  double d;
  memcpy(&d, &u, sizeof(d));
  return d;
}

// psq_l with W=0 loads two floats, with W=1 one float and 1.0
static inline PS ps_load(const f32* p) { return (PS){p[0], p[1]}; }
static inline PS ps_load1(const f32* p) { return (PS){p[0], 1.0f}; }
// lfs, and every scalar single-precision result, fills both lanes
static inline PS ps_fill(f32 v) { return (PS){v, v}; }
static inline void ps_store(f32* p, PS a) {
  p[0] = a.ps0;
  p[1] = a.ps1;
}
static inline void ps_store1(f32* p, PS a) { p[0] = a.ps0; }

// fnmadd and fnmsub negate their result unless it is a NaN
static inline f32 gekko_negate(f32 r) { return isnan(r) ? r : -r; }

static inline PS ps_add(PS a, PS b) { return (PS){a.ps0 + b.ps0, a.ps1 + b.ps1}; }
static inline PS ps_sub(PS a, PS b) { return (PS){a.ps0 - b.ps0, a.ps1 - b.ps1}; }
static inline PS ps_neg(PS a) { return (PS){-a.ps0, -a.ps1}; }
// a * c
static inline PS ps_mul(PS a, PS c) { return (PS){a.ps0 * c.ps0, a.ps1 * c.ps1}; }
static inline PS ps_muls0(PS a, PS c) { return (PS){a.ps0 * c.ps0, a.ps1 * c.ps0}; }
static inline PS ps_muls1(PS a, PS c) { return (PS){a.ps0 * c.ps1, a.ps1 * c.ps1}; }
// a * c + b
static inline PS ps_madd(PS a, PS c, PS b) { return (PS){fmaf(a.ps0, c.ps0, b.ps0), fmaf(a.ps1, c.ps1, b.ps1)}; }
static inline PS ps_madds0(PS a, PS c, PS b) { return (PS){fmaf(a.ps0, c.ps0, b.ps0), fmaf(a.ps1, c.ps0, b.ps1)}; }
static inline PS ps_madds1(PS a, PS c, PS b) { return (PS){fmaf(a.ps0, c.ps1, b.ps0), fmaf(a.ps1, c.ps1, b.ps1)}; }
// a * c - b
static inline PS ps_msub(PS a, PS c, PS b) { return (PS){fmaf(a.ps0, c.ps0, -b.ps0), fmaf(a.ps1, c.ps1, -b.ps1)}; }
// -(a * c - b)
static inline PS ps_nmsub(PS a, PS c, PS b) {
  PS r = ps_msub(a, c, b);
  return (PS){gekko_negate(r.ps0), gekko_negate(r.ps1)};
}
// -(a * c + b)
static inline PS ps_nmadd(PS a, PS c, PS b) {
  PS r = ps_madd(a, c, b);
  return (PS){gekko_negate(r.ps0), gekko_negate(r.ps1)};
}
// ps_sum0 d, a, c, b: (a.ps0 + b.ps1, c.ps1); ps_sum1: (c.ps0, a.ps0 + b.ps1)
static inline PS ps_sum0(PS a, PS c, PS b) { return (PS){a.ps0 + b.ps1, c.ps1}; }
static inline PS ps_sum1(PS a, PS c, PS b) { return (PS){c.ps0, a.ps0 + b.ps1}; }
static inline PS ps_merge00(PS a, PS b) { return (PS){a.ps0, b.ps0}; }
static inline PS ps_merge01(PS a, PS b) { return (PS){a.ps0, b.ps1}; }
static inline PS ps_merge10(PS a, PS b) { return (PS){a.ps1, b.ps0}; }
static inline PS ps_merge11(PS a, PS b) { return (PS){a.ps1, b.ps1}; }

// A single-precision multiply's frC, rounded to 26 significant bits (half away from zero),
// as the Gekko does before it multiplies. A subnormal double rounds at the same place
// relative to its leading bit.
static inline double gekko_round_frc(double c) {
  uint64_t bits = gekko_bits(c);
  const uint64_t exponent = bits & 0x7ff0000000000000ULL;
  const uint64_t fraction = bits & 0x000fffffffffffffULL;
  int64_t keep = (int64_t)0xfffffffff8000000ULL;
  uint64_t round = 0x8000000;
  if (exponent == 0 && fraction != 0) {
    const uint32_t shift = ppc_clz64(fraction) - 11;
    keep >>= shift;
    round >>= shift;
  }
  bits = (bits & (uint64_t)keep) + (bits & round);
  return gekko_double(bits);
}

// fmuls with double operands: the product of a and the rounded c is exact in double, then
// rounded once to single
static inline f32 gekko_fmuls(double a, double c) { return (f32)(a * gekko_round_frc(c)); }

// fres: the reciprocal estimate, a float. Its table is the Gekko's, as measured by
// https://achurch.org/cpu-tests/ppc750cl.s.
static inline f32 gekko_fres(f32 x) {
  static const struct {
    int32_t base;
    int32_t dec;
  } table[32] = {
      {0x7ff800, 0x3e1}, {0x783800, 0x3a7}, {0x70ea00, 0x371}, {0x6a0800, 0x340}, {0x638800, 0x313},
      {0x5d6200, 0x2ea}, {0x579000, 0x2c4}, {0x520800, 0x2a0}, {0x4cc800, 0x27f}, {0x47ca00, 0x261},
      {0x430800, 0x245}, {0x3e8000, 0x22a}, {0x3a2c00, 0x212}, {0x360800, 0x1fb}, {0x321400, 0x1e5},
      {0x2e4a00, 0x1d1}, {0x2aa800, 0x1be}, {0x272c00, 0x1ac}, {0x23d600, 0x19b}, {0x209e00, 0x18b},
      {0x1d8800, 0x17c}, {0x1a9000, 0x16e}, {0x17ae00, 0x15b}, {0x14f800, 0x15b}, {0x124400, 0x143},
      {0x0fbe00, 0x143}, {0x0d3800, 0x12d}, {0x0ade00, 0x12d}, {0x088400, 0x11a}, {0x065000, 0x11a},
      {0x041c00, 0x108}, {0x020c00, 0x106},
  };
  const uint64_t bits = gekko_bits((double)x);
  const uint64_t sign = bits & 0x8000000000000000ULL;
  const uint64_t exponent = bits & 0x7ff0000000000000ULL;
  const uint64_t mantissa = bits & 0x000fffffffffffffULL;

  if (exponent == 0 && mantissa == 0) {
    return copysignf(INFINITY, x);
  }
  if (exponent == 0x7ff0000000000000ULL) {
    return mantissa == 0 ? copysignf(0.0f, x) : x;
  }
  // too small for a float reciprocal, subnormals included
  if (exponent < (895ULL << 52)) {
    return copysignf(FLT_MAX, x);
  }
  if (exponent >= (1149ULL << 52)) {
    return copysignf(0.0f, x);
  }
  const uint32_t key = (uint32_t)(mantissa >> 37);
  const int32_t base = table[key >> 10].base;
  const int32_t dec = table[key >> 10].dec;
  const uint64_t fraction = (uint64_t)(base - (dec * (int32_t)(key & 0x3ff) + 1) / 2) << 29;
  return (f32)gekko_double(sign | ((0x7fdULL << 52) - exponent) | fraction);
}

#endif
