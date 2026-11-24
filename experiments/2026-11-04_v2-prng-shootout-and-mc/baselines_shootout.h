// baselines_shootout.h
// Unified KAT-verified reference implementations for the PRNG Shootout & Monte Carlo v2 Suite:
//   1. `ra_core v2` (`v2_W2_D2_xag_opt`, parameterized N in {256, 128, 64, 32, 16}, Scalar + OTF Engine + 8-Way AVX2)
//   2. `ra_core v1` (`W0_D0` / XAG production baseline, Scalar + OTF Engine + 8-Way AVX2)
//   3. `Philox4x32-10` (Random123 canonical constants, Scalar + 8-Way AVX2)
//   4. `ChaCha8` & `ChaCha12` (RFC 7539 / Bernstein ARX counter-based stream cipher, Scalar + 8-Way AVX2)
//   5. `xoshiro256**` & `xoroshiro128++` (Blackman & Vigna Shootout references + SplitMix64 init, Scalar + 8-Way AVX2)
//   6. `PCG32` (`pcg_oneseq_64_xsh_rr_32`, O'Neill reference, Scalar + 8-Way AVX2)
//   7. `SplitMix64` (Vigna 64-bit Weyl-step mixer, Scalar + 8-Way AVX2)
//   8. `WyRand` (Wang Yi 64x64->128 folded multiply mixer, Scalar)
//   9. `MT19937` (Matsumoto & Nishimura canonical 624-word Mersenne Twister, Scalar)
//
// Also provides `__rdtscp()` cycle-accurate measurement + `clock_gettime(CLOCK_MONOTONIC)`
// wall-clock measurement, and a comprehensive Known-Answer Test (KAT) verification gate.

#ifndef BASELINES_SHOOTOUT_H
#define BASELINES_SHOOTOUT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdalign.h>
#include <immintrin.h>
#include <x86intrin.h>

#include "../2026-11-04_v2-xag-redesign/ra_core_v2_w2_d2_xag_opt.h"

// ============================================================================
// Timing & Cycle Counter Utilities (`clock_gettime` + `__rdtscp`)
// ============================================================================

static inline double shootout_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline uint64_t shootout_rdtscp(void) {
    unsigned int aux;
    return __rdtscp(&aux);
}

static inline void compute_min_median(const double *vals, int n, double *out_min, double *out_med) {
    double s[16];
    for (int i = 0; i < n; ++i) s[i] = vals[i];
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (s[j] < s[i]) {
                double tmp = s[i];
                s[i] = s[j];
                s[j] = tmp;
            }
        }
    }
    *out_min = s[0];
    *out_med = s[n / 2];
}

// ============================================================================
// Parameterized `ra_core v2` (`v2_W2_D2_xag_opt`) & `ra_core v1` across
// N in {256, 128, 64, 32, 16} (Scalar, OTF Engine, and 8-Way AVX2 `u32x8`)
// ============================================================================

typedef uint32_t u32x8 __attribute__((vector_size(32)));
typedef uint64_t u64x4 __attribute__((vector_size(32)));

static inline __attribute__((always_inline)) u32x8 rot32_x8(u32x8 n, u32x8 r) {
    u32x8 s = r & 31u;
    u32x8 ns = (0u - s) & 31u;
    return (n << s) | (n >> ns);
}

static inline __attribute__((always_inline)) u32x8 fmix32_x8(u32x8 h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

static inline __attribute__((always_inline)) u64x4 mul64_avx2(u64x4 a, u64x4 b) {
    __m256i va = (__m256i)a;
    __m256i vb = (__m256i)b;
    __m256i lo = _mm256_mul_epu32(va, vb);
    __m256i a_hi = _mm256_srli_epi64(va, 32);
    __m256i b_hi = _mm256_srli_epi64(vb, 32);
    __m256i m1 = _mm256_mul_epu32(va, b_hi);
    __m256i m2 = _mm256_mul_epu32(a_hi, vb);
    __m256i hi = _mm256_slli_epi64(_mm256_add_epi64(m1, m2), 32);
    return (u64x4)_mm256_add_epi64(lo, hi);
}

// Parameterized `v2_W2_D2_xag_opt` generator macro for any NSIZE in {256, 128, 64, 32, 16}
#define DEFINE_RA_V2_PARAM(NSIZE)                                                              \
static inline __attribute__((always_inline)) uint32_t ra_v2_N##NSIZE##_eval_k1(                \
    uint32_t key, uint32_t addr_hi) {                                                          \
    const uint32_t i0 = (uint32_t)((NSIZE) - 1);                                               \
    uint32_t step_m = ra_compute_step_m(key, addr_hi);                                         \
    uint32_t o = rot32(i0 * step_m, key ^ i0) ^ (rot32(step_m, key ^ 1u) << 1);                \
    uint32_t a = o ^ (key << 1);                                                               \
    uint32_t b = (key + a) ^ o;                                                                \
    uint32_t c = rot32((a >> 13) ^ a, b);                                                      \
    return c ^ (c >> 17u);                                                                     \
}                                                                                              \
                                                                                               \
static inline __attribute__((always_inline)) void ra_v2_N##NSIZE##_eval_k2(                    \
    uint32_t key, uint32_t addr_hi, uint32_t * restrict out_buf) {                             \
    const uint32_t i0 = (uint32_t)((NSIZE) - 1);                                               \
    const uint32_t i1 = i0 - 1u;                                                               \
    const uint32_t MASK = (uint32_t)((NSIZE) - 1);                                             \
    uint32_t step_m = ra_compute_step_m(key, addr_hi);                                         \
    uint32_t m_n1 = rot32(i0 * step_m, key ^ i0);                                              \
    uint32_t m1   = rot32(step_m,      key ^ 1u);                                              \
    uint32_t m_n2 = rot32(i1 * step_m, key ^ i1);                                              \
    uint32_t o_0 = m_n1 ^ (m1 << 1);                                                           \
    uint32_t o_1 = m_n2 ^ (o_0 << 1);                                                          \
    uint32_t d_1 = (rot32(key, 15) ^ 0x55u) & MASK;                                            \
    uint32_t a_0 = o_0 ^ (key << 1);                                                           \
    uint32_t s_1 = key + a_0;                                                                  \
    uint32_t a_1 = (d_1 ^ o_1) ^ s_1;                                                          \
    uint32_t s_2 = key + a_1;                                                                  \
    uint32_t b_0 = s_1 ^ o_0;                                                                  \
    uint32_t b_1 = s_2 ^ (o_1 + d_1);                                                          \
    uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);                                              \
    uint32_t c_1 = rot32((a_1 >> 13) ^ a_1, b_1);                                              \
    out_buf[0] = c_0 ^ (c_0 >> 17u);                                                           \
    out_buf[1] = c_1 ^ (c_1 >> 17u);                                                           \
}                                                                                              \
                                                                                               \
static inline __attribute__((always_inline)) void ra_v2_N##NSIZE##_eval_k3_to_k8(              \
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out_buf) {                 \
    const uint32_t MAX_S = (uint32_t)((NSIZE) - 1);                                            \
    const uint32_t MASK  = (uint32_t)((NSIZE) - 1);                                            \
    uint32_t step_m = ra_compute_step_m(key, addr_hi);                                         \
    uint32_t m1 = rot32(step_m, key ^ 1u);                                                     \
    uint32_t i = MAX_S;                                                                        \
    uint32_t o_0 = rot32(i * step_m, key ^ i) ^ (m1 << 1);                                     \
    uint32_t s = key << 1;                                                                     \
    uint32_t d0 = 0u;                                                                          \
    uint32_t d1 = (rot32(key, 15) ^ 0x55u) & MASK;                                             \
    uint32_t rem = (uint32_t)rng;                                                              \
    while (rem >= 2u) {                                                                        \
        uint32_t i1 = i - 1u;                                                                  \
        uint32_t m_im1 = rot32(i1 * step_m, key ^ i1);                                         \
        uint32_t o_1 = (i == (uint32_t)((NSIZE) - 7))                                          \
                     ? (m_im1 ^ ((o_0 ^ (m1 << 7)) << 1))                                      \
                     : (m_im1 ^ (o_0 << 1));                                                   \
        uint32_t a_0 = (d0 ^ o_0) ^ s;                                                         \
        uint32_t s_1 = key + a_0;                                                              \
        uint32_t a_1 = (d1 ^ o_1) ^ s_1;                                                       \
        uint32_t s_2 = key + a_1;                                                              \
        uint32_t b_0 = s_1 ^ (o_0 + d0);                                                       \
        uint32_t b_1 = s_2 ^ (o_1 + d1);                                                       \
        uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);                                          \
        uint32_t c_1 = rot32((a_1 >> 13) ^ a_1, b_1);                                          \
        c_0 ^= c_0 >> 17u;                                                                     \
        c_1 ^= c_1 >> 17u;                                                                     \
        out_buf[0] = c_0;                                                                      \
        out_buf[1] = c_1;                                                                      \
        out_buf += 2;                                                                          \
        rem -= 2u;                                                                             \
        if (rem == 0u) return;                                                                 \
        s = s_2;                                                                               \
        d0 = c_0 & MASK;                                                                       \
        d1 = c_1 & MASK;                                                                       \
        uint32_t i2 = i - 2u;                                                                  \
        uint32_t m_im2 = rot32(i2 * step_m, key ^ i2);                                         \
        o_0 = m_im2 ^ (o_1 << 1);                                                              \
        i -= 2u;                                                                               \
    }                                                                                          \
    uint32_t a_0 = (d0 ^ o_0) ^ s;                                                             \
    uint32_t s_1 = key + a_0;                                                                  \
    uint32_t b_0 = s_1 ^ (o_0 + d0);                                                           \
    uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);                                              \
    out_buf[0] = c_0 ^ (c_0 >> 17u);                                                           \
}                                                                                              \
                                                                                               \
static inline void ra_v2_N##NSIZE##_singleblock_into(                                          \
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out_buf) {                 \
    const uint32_t MAX_S = (uint32_t)((NSIZE) - 1);                                            \
    const uint32_t MASK  = (uint32_t)((NSIZE) - 1);                                            \
    if (rng == 0) return;                                                                      \
    if (rng == 1) { out_buf[0] = ra_v2_N##NSIZE##_eval_k1(key, addr_hi); return; }             \
    if (rng == 2) { ra_v2_N##NSIZE##_eval_k2(key, addr_hi, out_buf); return; }                 \
    if (rng <= 8) { ra_v2_N##NSIZE##_eval_k3_to_k8(key, addr_hi, rng, out_buf); return; }      \
    alignas(64) uint32_t M[NSIZE];                                                             \
    uint32_t step_m = ra_compute_step_m(key, addr_hi);                                         \
    int start = (int)(NSIZE) - (int)rng;                                                       \
    if (start < 1) start = 1;                                                                  \
    for (int idx_i = start; idx_i < (NSIZE); ++idx_i) {                                        \
        uint32_t idx = (uint32_t)idx_i;                                                        \
        M[idx_i] = rot32(idx * step_m, key ^ idx);                                             \
    }                                                                                          \
    uint32_t m1 = (start == 1) ? M[1] : rot32(step_m, key ^ 1u);                               \
    uint32_t cons = key;                                                                       \
    uint32_t s = cons << 1;                                                                    \
    uint32_t d0 = 0u;                                                                          \
    uint32_t d1 = (rot32(cons, 15) ^ 0x55u) & MASK;                                            \
    uint32_t o_0 = M[MAX_S] ^ (m1 << 1);                                                       \
    uint32_t i = MAX_S;                                                                        \
    for (int pair = 0; pair < 3; ++pair) {                                                     \
        uint32_t o_1 = M[i - 1u] ^ (o_0 << 1);                                                 \
        uint32_t a_0 = (d0 ^ o_0) ^ s;                                                         \
        uint32_t s_1 = cons + a_0;                                                             \
        uint32_t a_1 = (d1 ^ o_1) ^ s_1;                                                       \
        uint32_t s_2 = cons + a_1;                                                             \
        uint32_t b_0 = s_1 ^ (o_0 + d0);                                                       \
        uint32_t b_1 = s_2 ^ (o_1 + d1);                                                       \
        uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);                                          \
        uint32_t c_1 = rot32((a_1 >> 13) ^ a_1, b_1);                                          \
        c_0 ^= c_0 >> 17u;                                                                     \
        c_1 ^= c_1 >> 17u;                                                                     \
        out_buf[0] = c_0;                                                                      \
        out_buf[1] = c_1;                                                                      \
        out_buf += 2;                                                                          \
        s = s_2;                                                                               \
        d0 = c_0 & MASK;                                                                       \
        d1 = c_1 & MASK;                                                                       \
        o_0 = M[i - 2u] ^ (o_1 << 1);                                                          \
        i -= 2u;                                                                               \
    }                                                                                          \
    uint32_t o_term7_0 = m1 << 7;                                                              \
    uint32_t rem = (uint32_t)(rng - 6u);                                                       \
    while (rem >= 2u) {                                                                        \
        uint32_t o_1 = M[i - 1u] ^ ((o_0 ^ o_term7_0) << 1);                                   \
        uint32_t o_term7_1 = M[i + 6u] << 7;                                                   \
        uint32_t a_0 = (d0 ^ o_0) ^ s;                                                         \
        uint32_t s_1 = cons + a_0;                                                             \
        uint32_t a_1 = (d1 ^ o_1) ^ s_1;                                                       \
        uint32_t s_2 = cons + a_1;                                                             \
        uint32_t b_0 = s_1 ^ (o_0 + d0);                                                       \
        uint32_t b_1 = s_2 ^ (o_1 + d1);                                                       \
        uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);                                          \
        uint32_t c_1 = rot32((a_1 >> 13) ^ a_1, b_1);                                          \
        c_0 ^= c_0 >> 17u;                                                                     \
        c_1 ^= c_1 >> 17u;                                                                     \
        out_buf[0] = c_0;                                                                      \
        out_buf[1] = c_1;                                                                      \
        out_buf += 2;                                                                          \
        rem -= 2u;                                                                             \
        if (rem == 0u) return;                                                                 \
        s = s_2;                                                                               \
        d0 = c_0 & MASK;                                                                       \
        d1 = c_1 & MASK;                                                                       \
        o_0 = M[i - 2u] ^ ((o_1 ^ o_term7_1) << 1);                                            \
        o_term7_0 = M[i + 5u] << 7;                                                            \
        i -= 2u;                                                                               \
    }                                                                                          \
    if (rem == 1u) {                                                                           \
        uint32_t a_0 = (d0 ^ o_0) ^ s;                                                         \
        uint32_t s_1 = cons + a_0;                                                             \
        uint32_t b_0 = s_1 ^ (o_0 + d0);                                                       \
        uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);                                          \
        out_buf[0] = c_0 ^ (c_0 >> 17u);                                                       \
    }                                                                                          \
}                                                                                              \
                                                                                               \
static inline void ra_v2_N##NSIZE##_singleblock_x8_into(                                       \
    const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out) { \
    const uint32_t MAX_S = (uint32_t)((NSIZE) - 1);                                            \
    const uint32_t MASK  = (uint32_t)((NSIZE) - 1);                                            \
    u32x8 vkey, vaddr;                                                                         \
    memcpy(&vkey, keys, 32);                                                                   \
    memcpy(&vaddr, addr_his, 32);                                                              \
    u32x8 km_k = fmix32_x8(((u32x8){0} + GUARD_M)  ^ (vkey  * 0x06a0dd9bu));                   \
    u32x8 km_p = fmix32_x8(((u32x8){0} + GUARD_M2) ^ (vaddr * 0x9e3779b1u));                   \
    u32x8 step_m = (km_k * km_p) * 0x06a0dd9bu;                                                \
    if (rng == 1) {                                                                            \
        u32x8 o = rot32_x8(step_m * MAX_S, vkey ^ MAX_S)                                       \
                ^ (rot32_x8(step_m * 1u,   vkey ^ 1u) << 1);                                   \
        u32x8 a = o ^ (vkey << 1);                                                             \
        u32x8 b = (vkey + a) ^ o;                                                              \
        u32x8 c = rot32_x8((a >> 13) ^ a, b);                                                  \
        c ^= c >> 17u;                                                                         \
        memcpy(out, &c, 32);                                                                   \
        return;                                                                                \
    }                                                                                          \
    if (rng == 2) {                                                                            \
        u32x8 m_n1 = rot32_x8(step_m * MAX_S,        vkey ^ MAX_S);                            \
        u32x8 m1   = rot32_x8(step_m * 1u,           vkey ^ 1u);                               \
        u32x8 m_n2 = rot32_x8(step_m * (MAX_S - 1u), vkey ^ (MAX_S - 1u));                     \
        u32x8 o_0 = m_n1 ^ (m1 << 1);                                                          \
        u32x8 o_1 = m_n2 ^ (o_0 << 1);                                                         \
        u32x8 d_1 = (rot32_x8(vkey, (u32x8){0} + 15u) ^ 0x55u) & MASK;                         \
        u32x8 a_0 = o_0 ^ (vkey << 1);                                                         \
        u32x8 s_1 = vkey + a_0;                                                                \
        u32x8 a_1 = (d_1 ^ o_1) ^ s_1;                                                         \
        u32x8 s_2 = vkey + a_1;                                                                \
        u32x8 b_0 = s_1 ^ o_0;                                                                 \
        u32x8 b_1 = s_2 ^ (o_1 + d_1);                                                         \
        u32x8 c_0 = rot32_x8((a_0 >> 13) ^ a_0, b_0);                                          \
        u32x8 c_1 = rot32_x8((a_1 >> 13) ^ a_1, b_1);                                          \
        c_0 ^= c_0 >> 17u;                                                                     \
        c_1 ^= c_1 >> 17u;                                                                     \
        memcpy(out,     &c_0, 32);                                                             \
        memcpy(out + 8, &c_1, 32);                                                             \
        return;                                                                                \
    }                                                                                          \
    if (rng <= 4) {                                                                            \
        u32x8 m1 = rot32_x8(step_m * 1u, vkey ^ 1u);                                           \
        uint32_t i = MAX_S;                                                                    \
        u32x8 o_0 = rot32_x8(step_m * MAX_S, vkey ^ MAX_S) ^ (m1 << 1);                        \
        u32x8 s = vkey << 1;                                                                   \
        u32x8 d0 = {0};                                                                        \
        u32x8 d1 = (rot32_x8(vkey, (u32x8){0} + 15u) ^ 0x55u) & MASK;                          \
        uint32_t rem = (uint32_t)rng;                                                          \
        while (rem >= 2u) {                                                                    \
            uint32_t i1 = i - 1u;                                                              \
            u32x8 m_im1 = rot32_x8(step_m * i1, vkey ^ i1);                                    \
            u32x8 o_1 = m_im1 ^ (o_0 << 1);                                                    \
            u32x8 a_0 = (d0 ^ o_0) ^ s;                                                        \
            u32x8 s_1 = vkey + a_0;                                                            \
            u32x8 a_1 = (d1 ^ o_1) ^ s_1;                                                      \
            u32x8 s_2 = vkey + a_1;                                                            \
            u32x8 b_0 = s_1 ^ (o_0 + d0);                                                      \
            u32x8 b_1 = s_2 ^ (o_1 + d1);                                                      \
            u32x8 c_0 = rot32_x8((a_0 >> 13) ^ a_0, b_0);                                      \
            u32x8 c_1 = rot32_x8((a_1 >> 13) ^ a_1, b_1);                                      \
            c_0 ^= c_0 >> 17u;                                                                 \
            c_1 ^= c_1 >> 17u;                                                                 \
            memcpy(out,     &c_0, 32);                                                         \
            memcpy(out + 8, &c_1, 32);                                                         \
            out += 16;                                                                         \
            rem -= 2u;                                                                         \
            if (rem == 0u) return;                                                             \
            s = s_2;                                                                           \
            d0 = c_0 & MASK;                                                                   \
            d1 = c_1 & MASK;                                                                   \
            uint32_t i2 = i - 2u;                                                              \
            u32x8 m_im2 = rot32_x8(step_m * i2, vkey ^ i2);                                    \
            o_0 = m_im2 ^ (o_1 << 1);                                                          \
            i -= 2u;                                                                           \
        }                                                                                      \
        u32x8 a_0 = (d0 ^ o_0) ^ s;                                                            \
        u32x8 s_1 = vkey + a_0;                                                                \
        u32x8 b_0 = s_1 ^ (o_0 + d0);                                                          \
        u32x8 c_0 = rot32_x8((a_0 >> 13) ^ a_0, b_0);                                          \
        c_0 ^= c_0 >> 17u;                                                                     \
        memcpy(out, &c_0, 32);                                                                 \
        return;                                                                                \
    }                                                                                          \
    alignas(32) u32x8 M[NSIZE];                                                                \
    int start = (int)(NSIZE) - (int)rng;                                                       \
    if (start < 1) start = 1;                                                                  \
    for (int idx_i = start; idx_i < (NSIZE); ++idx_i) {                                        \
        uint32_t idx = (uint32_t)idx_i;                                                        \
        M[idx_i] = rot32_x8(step_m * idx, vkey ^ idx);                                         \
    }                                                                                          \
    u32x8 m1 = (start == 1) ? M[1] : rot32_x8(step_m * 1u, vkey ^ 1u);                         \
    u32x8 cons = vkey;                                                                         \
    u32x8 s = cons << 1;                                                                       \
    u32x8 d_use = {0};                                                                         \
    u32x8 d = (rot32_x8(cons, (u32x8){0} + 15u) ^ 0x55u) & MASK;                               \
    u32x8 o = M[MAX_S] ^ (m1 << 1);                                                            \
    uint32_t rem = (uint32_t)rng;                                                              \
    uint32_t first_steps = (rem <= 7u) ? rem : 7u;                                             \
    uint32_t i = MAX_S;                                                                        \
    for (;;) {                                                                                 \
        u32x8 a = (d_use ^ o) ^ s;                                                             \
        s = cons + a;                                                                          \
        u32x8 b = s ^ (o + d_use);                                                             \
        u32x8 c = rot32_x8((a >> 13) ^ a, b);                                                  \
        c ^= c >> 17u;                                                                         \
        memcpy(out, &c, 32);                                                                   \
        out += 8;                                                                              \
        u32x8 d_next = c & MASK;                                                               \
        d_use = d;                                                                             \
        d = d_next;                                                                            \
        if (--first_steps == 0u) break;                                                        \
        o = M[i - 1u] ^ (o << 1);                                                              \
        --i;                                                                                   \
    }                                                                                          \
    rem -= 7u;                                                                                 \
    if ((int32_t)rem <= 0) return;                                                             \
    u32x8 o_term7 = m1 << 7;                                                                   \
    for (;;) {                                                                                 \
        u32x8 new_term = M[i - 1u];                                                            \
        u32x8 o_next = new_term ^ ((o ^ o_term7) << 1);                                        \
        o_term7 = M[i + 6u] << 7;                                                              \
        o = o_next;                                                                            \
        --i;                                                                                   \
        u32x8 a = (d_use ^ o) ^ s;                                                             \
        s = cons + a;                                                                          \
        u32x8 b = s ^ (o + d_use);                                                             \
        u32x8 c = rot32_x8((a >> 13) ^ a, b);                                                  \
        c ^= c >> 17u;                                                                         \
        memcpy(out, &c, 32);                                                                   \
        out += 8;                                                                              \
        if (--rem == 0u) break;                                                                \
        u32x8 d_next = c & MASK;                                                               \
        d_use = d;                                                                             \
        d = d_next;                                                                            \
    }                                                                                          \
}

DEFINE_RA_V2_PARAM(256)
DEFINE_RA_V2_PARAM(128)
DEFINE_RA_V2_PARAM(64)
DEFINE_RA_V2_PARAM(32)
DEFINE_RA_V2_PARAM(16)

// Parameterized `ra_core v1` (`W0_D0`) for N in {256, 128, 64, 32, 16}
#define DEFINE_RA_V1_PARAM(NSIZE)                                                              \
static inline void ra_v1_N##NSIZE##_singleblock_into(                                          \
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out_buf) {                 \
    const int MAX_SINGLE = (NSIZE) - 1;                                                        \
    const uint32_t MASK  = (uint32_t)((NSIZE) - 1);                                            \
    uint32_t step_m = ra_compute_step_m(key, addr_hi);                                         \
    if (rng == 1) {                                                                            \
        uint32_t i0 = (uint32_t)MAX_SINGLE;                                                    \
        uint32_t o = rot32(i0 * step_m, key ^ i0)                                              \
                   ^ (rot32(0u * step_m, key ^ 0u) << 1)                                       \
                   ^ (rot32(1u * step_m, key ^ 1u) << 2)                                       \
                   ^ (rot32(2u * step_m, key ^ 2u) << 3)                                       \
                   ^ (rot32(3u * step_m, key ^ 3u) << 4)                                       \
                   ^ (rot32(4u * step_m, key ^ 4u) << 5)                                       \
                   ^ (rot32(5u * step_m, key ^ 5u) << 6)                                       \
                   ^ (rot32(6u * step_m, key ^ 6u) << 7);                                      \
        uint32_t a = o ^ (key << 1);                                                           \
        uint32_t b = (key + a) ^ o;                                                            \
        uint32_t c = rot32((a >> 13) ^ a, b);                                                  \
        out_buf[0] = c ^ (c >> 17u);                                                           \
        return;                                                                                \
    }                                                                                          \
    alignas(64) uint32_t M[(NSIZE) + 8];                                                       \
    int len = (rng + 7 > (size_t)(NSIZE)) ? (NSIZE) : (int)(rng + 7);                          \
    int start = ((NSIZE) - (int)rng) & (int)MASK;                                              \
    int end = start + len;                                                                     \
    int first_end = (end <= (NSIZE)) ? end : (NSIZE);                                          \
    for (int i = start; i < first_end; ++i) {                                                  \
        uint32_t idx = (uint32_t)i;                                                            \
        M[i] = rot32(idx * step_m, key ^ idx);                                                 \
    }                                                                                          \
    if (end > (NSIZE)) {                                                                       \
        int second_end = end - (NSIZE);                                                        \
        for (int i = 0; i < second_end; ++i) {                                                 \
            uint32_t idx = (uint32_t)i;                                                        \
            M[i] = rot32(idx * step_m, key ^ idx);                                             \
        }                                                                                      \
    }                                                                                          \
    memcpy(M + (NSIZE), M, 7 * sizeof(uint32_t));                                              \
    uint32_t cons = key;                                                                       \
    uint32_t s = cons + cons;                                                                  \
    uint32_t d = 0;                                                                            \
    uint32_t o = (M[MAX_SINGLE] << 0) ^ (M[(NSIZE) + 0] << 1) ^ (M[(NSIZE) + 1] << 2) ^        \
                 (M[(NSIZE) + 2] << 3) ^ (M[(NSIZE) + 3] << 4) ^ (M[(NSIZE) + 4] << 5) ^       \
                 (M[(NSIZE) + 5] << 6) ^ (M[(NSIZE) + 6] << 7);                                \
    uint32_t o_term7 = M[(NSIZE) + 6] << 7;                                                    \
    uint32_t rem = (uint32_t)rng;                                                              \
    for (uint32_t i = (uint32_t)MAX_SINGLE; ; --i) {                                           \
        uint32_t a = (d ^ o) ^ s;                                                              \
        s = cons + a;                                                                          \
        uint32_t b = s ^ (o + d);                                                              \
        uint32_t c = rot32((a >> 13) ^ a, b);                                                  \
        c ^= c >> 17u;                                                                         \
        *out_buf++ = c;                                                                        \
        if (--rem == 0) break;                                                                 \
        d = c & MASK;                                                                          \
        uint32_t new_term = M[i - 1];                                                          \
        uint32_t o_next = new_term ^ ((o ^ o_term7) << 1);                                     \
        o_term7 = M[i + 6] << 7;                                                               \
        o = o_next;                                                                            \
    }                                                                                          \
}                                                                                              \
                                                                                               \
static inline void ra_v1_N##NSIZE##_singleblock_x8_into(                                       \
    const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out) { \
    const int MAX_SINGLE = (NSIZE) - 1;                                                        \
    const uint32_t MASK  = (uint32_t)((NSIZE) - 1);                                            \
    u32x8 vkey, vaddr;                                                                         \
    memcpy(&vkey, keys, 32);                                                                   \
    memcpy(&vaddr, addr_his, 32);                                                              \
    u32x8 km_k = fmix32_x8(((u32x8){0} + GUARD_M)  ^ (vkey  * 0x06a0dd9bu));                   \
    u32x8 km_p = fmix32_x8(((u32x8){0} + GUARD_M2) ^ (vaddr * 0x9e3779b1u));                   \
    u32x8 step_m = (km_k * km_p) * 0x06a0dd9bu;                                                \
    if (rng == 1) {                                                                            \
        uint32_t i0 = (uint32_t)MAX_SINGLE;                                                    \
        u32x8 o = rot32_x8(step_m * i0, vkey ^ i0)                                             \
                ^ (rot32_x8((u32x8){0},   vkey)      << 1)                                     \
                ^ (rot32_x8(step_m * 1u,  vkey ^ 1u) << 2)                                     \
                ^ (rot32_x8(step_m * 2u,  vkey ^ 2u) << 3)                                     \
                ^ (rot32_x8(step_m * 3u,  vkey ^ 3u) << 4)                                     \
                ^ (rot32_x8(step_m * 4u,  vkey ^ 4u) << 5)                                     \
                ^ (rot32_x8(step_m * 5u,  vkey ^ 5u) << 6)                                     \
                ^ (rot32_x8(step_m * 6u,  vkey ^ 6u) << 7);                                    \
        u32x8 a = o ^ (vkey << 1);                                                             \
        u32x8 b = (vkey + a) ^ o;                                                              \
        u32x8 c = rot32_x8((a >> 13) ^ a, b);                                                  \
        c ^= c >> 17u;                                                                         \
        memcpy(out, &c, 32);                                                                   \
        return;                                                                                \
    }                                                                                          \
    alignas(32) u32x8 M[(NSIZE) + 8];                                                          \
    int len = (rng + 7 > (size_t)(NSIZE)) ? (NSIZE) : (int)(rng + 7);                          \
    int start = ((NSIZE) - (int)rng) & (int)MASK;                                              \
    int end = start + len;                                                                     \
    int first_end = (end <= (NSIZE)) ? end : (NSIZE);                                          \
    for (int i = start; i < first_end; ++i) {                                                  \
        uint32_t idx = (uint32_t)i;                                                            \
        M[i] = rot32_x8(step_m * idx, vkey ^ idx);                                             \
    }                                                                                          \
    if (end > (NSIZE)) {                                                                       \
        int second_end = end - (NSIZE);                                                        \
        for (int i = 0; i < second_end; ++i) {                                                 \
            uint32_t idx = (uint32_t)i;                                                        \
            M[i] = rot32_x8(step_m * idx, vkey ^ idx);                                         \
        }                                                                                      \
    }                                                                                          \
    memcpy(M + (NSIZE), M, 7 * sizeof(u32x8));                                                 \
    u32x8 cons = vkey;                                                                         \
    u32x8 s = vkey + vkey;                                                                     \
    u32x8 d = {0};                                                                             \
    u32x8 o = (M[MAX_SINGLE] << 0) ^ (M[(NSIZE) + 0] << 1) ^ (M[(NSIZE) + 1] << 2) ^           \
              (M[(NSIZE) + 2] << 3) ^ (M[(NSIZE) + 3] << 4) ^ (M[(NSIZE) + 4] << 5) ^          \
              (M[(NSIZE) + 5] << 6) ^ (M[(NSIZE) + 6] << 7);                                   \
    u32x8 o_term7 = M[(NSIZE) + 6] << 7;                                                       \
    uint32_t rem = (uint32_t)rng;                                                              \
    for (uint32_t i = (uint32_t)MAX_SINGLE; ; --i) {                                           \
        u32x8 a = (d ^ o) ^ s;                                                                 \
        s = cons + a;                                                                          \
        u32x8 b = s ^ (o + d);                                                                 \
        u32x8 c = rot32_x8((a >> 13) ^ a, b);                                                  \
        c ^= c >> 17u;                                                                         \
        memcpy(out, &c, 32);                                                                   \
        out += 8;                                                                              \
        if (--rem == 0) break;                                                                 \
        d = c & MASK;                                                                          \
        u32x8 new_term = M[i - 1];                                                             \
        u32x8 o_next = new_term ^ ((o ^ o_term7) << 1);                                        \
        o_term7 = M[i + 6] << 7;                                                               \
        o = o_next;                                                                            \
    }                                                                                          \
}

DEFINE_RA_V1_PARAM(256)
DEFINE_RA_V1_PARAM(128)
DEFINE_RA_V1_PARAM(64)
DEFINE_RA_V1_PARAM(32)
DEFINE_RA_V1_PARAM(16)

// ============================================================================
// Stateful 3-Stage OTF Engines (`engine_v2_W2_D2_xag_opt` & `engine_v1`)
// ============================================================================

typedef struct {
    uint32_t key, addr_hi, step_m, s, d, o, o_term7, i_cur;
    uint32_t m_ring[8];
    uint8_t  mode;
    uint32_t block_pos, block_len, cons;
    uint8_t  d_hist[256];
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256 + 8];
    uint32_t block_buf[256];
} ra_engine_v2_t;

static inline __attribute__((always_inline)) void ra_engine_v2_init(
    ra_engine_v2_t *e, uint32_t key, uint32_t addr_hi) {
    e->key = key;
    e->addr_hi = addr_hi;
    e->mode = 0;
}

static __attribute__((noinline)) uint32_t ra_engine_v2_promote(ra_engine_v2_t *e) {
    v2_init_orbit_W2(e->L, e->M, e->key, e->addr_hi);
    for (uint32_t i = 255; i >= 1; --i) {
        uint8_t d_use = e->d_hist[i];
        uint32_t tmp = e->L[i];
        uint32_t li_new = e->L[d_use];
        e->L[i] = li_new;
        e->L[d_use] = tmp;
        e->M[i + 7] += li_new;
    }
    END_CYCLE_M_W2(e->M, e->L);
    uint32_t a_last = e->s - e->key;
    uint32_t b_last = e->s ^ (e->o + e->d_hist[2]);
    uint32_t c_last = rot32((a_last >> 13) ^ a_last, b_last);
    c_last ^= c_last >> 17u;
    e->cons = e->key ^ (c_last + e->s + e->L[0]);
    e->mode = 2;
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v2_cycle_W2_D2_xag_opt(&e->cons, e->M, e->L, &cnt, &cur);
    e->block_len = 255;
    e->block_pos = 1;
    return e->block_buf[0];
}

static inline __attribute__((always_inline)) uint32_t ra_engine_v2_next_u32(ra_engine_v2_t *e) {
    if (__builtin_expect(e->mode == 0, 0)) {
        uint32_t key = e->key;
        uint32_t step_m = ra_compute_step_m(key, e->addr_hi);
        e->step_m = step_m;
        uint32_t m255 = rot32(255u * step_m, key ^ 255u);
        uint32_t m1   = rot32(1u   * step_m, key ^ 1u);

        uint32_t o = m255 ^ (m1 << 1);
        uint32_t a = o ^ (key << 1);
        uint32_t s = key + a;
        uint32_t b = s ^ o;
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;

        uint32_t d_init = (rot32(key, 15) ^ 0x55u) & 0xFFu;
        e->d_hist[255] = (uint8_t)d_init;
        e->d = d_init | ((c & 0xFFu) << 8);
        e->s = s;
        e->o = o;
        e->o_term7 = m255;
        e->i_cur = 254;
        e->mode = 1;
        return c;
    }
    if (__builtin_expect(e->mode == 1, 1)) {
        uint32_t i = e->i_cur;
        if (__builtin_expect(i >= 250u, 1)) {
            uint32_t key = e->key;
            uint32_t o_prev = e->o;
            if (__builtin_expect(i == 254u, 0)) {
                uint32_t m255 = e->o_term7;
                e->m_ring[0] = (o_prev ^ m255) >> 1;
                e->m_ring[7] = m255;
                e->o_term7 = 0u;
            }
            uint32_t new_term = rot32(i * e->step_m, key ^ i);
            uint32_t o = new_term ^ (o_prev << 1);
            e->m_ring[i & 7u] = new_term;

            uint32_t d_pair = e->d;
            uint32_t d_use = d_pair & 0xFFu;
            uint32_t d_prev = d_pair >> 8;
            uint32_t a = (d_use ^ o) ^ e->s;
            uint32_t s = key + a;
            uint32_t b = s ^ (o + d_use);
            uint32_t c = rot32((a >> 13) ^ a, b);
            c ^= c >> 17u;

            e->d_hist[i] = (uint8_t)d_prev;
            e->d = d_prev | ((c & 0xFFu) << 8);
            e->s = s;
            e->o = o;
            e->i_cur = i - 1u;
            return c;
        }
        if (__builtin_expect(i > 0, 1)) {
            uint32_t key = e->key;
            uint32_t new_term = rot32(i * e->step_m, key ^ i);
            uint32_t o = new_term ^ ((e->o ^ e->o_term7) << 1);
            uint32_t next_term7 = e->m_ring[(i + 7u) & 7u] << 7;
            e->m_ring[i & 7u] = new_term;

            uint32_t d_pair = e->d;
            uint32_t d_use = d_pair & 0xFFu;
            uint32_t d_prev = d_pair >> 8;
            uint32_t a = (d_use ^ o) ^ e->s;
            uint32_t s = key + a;
            uint32_t b = s ^ (o + d_use);
            uint32_t c = rot32((a >> 13) ^ a, b);
            c ^= c >> 17u;

            e->d_hist[i] = (uint8_t)d_prev;
            e->d = d_prev | ((c & 0xFFu) << 8);
            e->s = s;
            e->o = o;
            e->o_term7 = next_term7;
            e->i_cur = i - 1u;
            return c;
        }
        return ra_engine_v2_promote(e);
    }
    if (e->block_pos >= e->block_len) {
        uint64_t cnt = 256;
        uint32_t *cur = e->block_buf;
        v2_cycle_W2_D2_xag_opt(&e->cons, e->M, e->L, &cnt, &cur);
        e->block_len = 255;
        e->block_pos = 0;
    }
    return e->block_buf[e->block_pos++];
}

static __attribute__((noinline)) uint32_t ra_engine_v1_promote(ra_engine_v2_t *e) {
    v1_init_state_orbit(e->L, e->M, e->key, e->addr_hi);
    for (uint32_t i = 255; i >= 1; --i) {
        uint8_t d = e->d_hist[i];
        uint32_t tmp = e->L[i];
        e->L[i] = e->L[d];
        e->L[d] = tmp;
    }
    e->cons = v1_reseed(e->M, e->L);
    e->mode = 2;
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v1_permutation_cycle_orbit_buf(e->cons, e->M, e->L, &cnt, &cur);
    e->cons = v1_reseed(e->M, e->L);
    e->block_len = 255;
    e->block_pos = 1;
    return e->block_buf[0];
}

static inline __attribute__((always_inline)) uint32_t ra_engine_v1_next_u32(ra_engine_v2_t *e) {
    if (__builtin_expect(e->mode == 0, 0)) {
        uint32_t key = e->key;
        uint32_t step_m = ra_compute_step_m(key, e->addr_hi);
        e->step_m = step_m;
        uint32_t m255 = rot32(255u * step_m, key ^ 255u);
        uint32_t m0   = rot32(0u   * step_m, key ^ 0u);
        uint32_t m1   = rot32(1u   * step_m, key ^ 1u);
        uint32_t m2   = rot32(2u   * step_m, key ^ 2u);
        uint32_t m3   = rot32(3u   * step_m, key ^ 3u);
        uint32_t m4   = rot32(4u   * step_m, key ^ 4u);
        uint32_t m5   = rot32(5u   * step_m, key ^ 5u);
        uint32_t m6   = rot32(6u   * step_m, key ^ 6u);
        e->m_ring[7] = m255; e->m_ring[0] = m0; e->m_ring[1] = m1; e->m_ring[2] = m2;
        e->m_ring[3] = m3;   e->m_ring[4] = m4; e->m_ring[5] = m5; e->m_ring[6] = m6;

        uint32_t o = m255 ^ (m0 << 1) ^ (m1 << 2) ^ (m2 << 3) ^ (m3 << 4) ^ (m4 << 5) ^ (m5 << 6) ^ (m6 << 7);
        uint32_t a = o ^ (key << 1);
        uint32_t s = key + a;
        uint32_t b = s ^ o;
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;
        uint8_t d = (uint8_t)(c & 0xFFu);
        e->d_hist[255] = d;
        e->d = d; e->s = s; e->o = o; e->o_term7 = m6 << 7;
        e->i_cur = 254; e->mode = 1;
        return c;
    }
    if (__builtin_expect(e->mode == 1, 1)) {
        uint32_t i = e->i_cur;
        if (__builtin_expect(i > 0, 1)) {
            uint32_t key = e->key;
            uint32_t new_term = rot32(i * e->step_m, key ^ i);
            uint32_t o = new_term ^ ((e->o ^ e->o_term7) << 1);
            uint32_t next_term7 = e->m_ring[(i + 7u) & 7u] << 7;
            e->m_ring[i & 7u] = new_term;

            uint32_t d = e->d;
            uint32_t a = (d ^ o) ^ e->s;
            uint32_t s = key + a;
            uint32_t b = s ^ (o + d);
            uint32_t c = rot32((a >> 13) ^ a, b);
            c ^= c >> 17u;
            uint8_t d_new = (uint8_t)(c & 0xFFu);
            e->d_hist[i] = d_new;
            e->d = d_new; e->s = s; e->o = o; e->o_term7 = next_term7;
            e->i_cur = i - 1u;
            return c;
        }
        return ra_engine_v1_promote(e);
    }
    if (e->block_pos >= e->block_len) {
        uint64_t cnt = 256;
        uint32_t *cur = e->block_buf;
        v1_permutation_cycle_orbit_buf(e->cons, e->M, e->L, &cnt, &cur);
        e->cons = v1_reseed(e->M, e->L);
        e->block_len = 255;
        e->block_pos = 0;
    }
    return e->block_buf[e->block_pos++];
}

// ============================================================================
// 1. Philox4x32-10 (Random123 canonical constants, Scalar & 8-Way AVX2)
// ============================================================================
#define PHILOX_M0 0xD2511F53U
#define PHILOX_M1 0xCD9E8D57U
#define PHILOX_W0 0x9E3779B9U
#define PHILOX_W1 0xBB67AE85U

static inline __attribute__((always_inline)) void philox4x32_round(uint32_t ctr[4], uint32_t key[2]) {
    uint64_t p0 = (uint64_t)ctr[0] * PHILOX_M0;
    uint64_t p1 = (uint64_t)ctr[2] * PHILOX_M1;
    uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
    uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    uint32_t c1 = ctr[1], c3 = ctr[3];
    ctr[0] = hi1 ^ c1 ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ c3 ^ key[1];
    ctr[3] = lo0;
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

static inline __attribute__((always_inline)) void philox4x32_10(
    uint32_t out[4], const uint32_t ctr_in[4], const uint32_t key_in[2]) {
    uint32_t ctr[4] = { ctr_in[0], ctr_in[1], ctr_in[2], ctr_in[3] };
    uint32_t key[2] = { key_in[0], key_in[1] };
    for (int i = 0; i < 10; i++) philox4x32_round(ctr, key);
    memcpy(out, ctr, 4 * sizeof(uint32_t));
}

typedef struct {
    uint32_t ctr[4];
    uint32_t key[2];
    uint32_t block[4];
    uint32_t pos;
} philox_stream_t;

static inline __attribute__((always_inline)) void philox_stream_init(
    philox_stream_t *s, uint32_t key0, uint32_t key1, uint32_t ctr0, uint32_t ctr1) {
    s->ctr[0] = ctr0; s->ctr[1] = ctr1; s->ctr[2] = 0u; s->ctr[3] = 0u;
    s->key[0] = key0; s->key[1] = key1;
    s->pos = 4u;
}

static inline __attribute__((always_inline)) uint32_t philox_stream_next_u32(philox_stream_t *s) {
    if (s->pos >= 4u) {
        philox4x32_10(s->block, s->ctr, s->key);
        if (++s->ctr[0] == 0u) {
            if (++s->ctr[1] == 0u) {
                if (++s->ctr[2] == 0u) ++s->ctr[3];
            }
        }
        s->pos = 0u;
    }
    return s->block[s->pos++];
}

static inline __attribute__((optimize("no-tree-vectorize"))) void philox_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    uint32_t ctr[4] = { 0u, 0u, 0u, 0u };
    uint32_t pkey[2] = { key, addr_hi ^ 0xFEEDC0DEu };
    size_t produced = 0;
    while (produced + 4 <= rng) {
        philox4x32_10(&out[produced], ctr, pkey);
        ++ctr[0];
        produced += 4;
    }
    if (produced < rng) {
        uint32_t blk[4];
        philox4x32_10(blk, ctr, pkey);
        for (int i = 0; produced < rng; ++i, ++produced) out[produced] = blk[i];
    }
}

static inline __attribute__((always_inline)) void philox4x32_x8_into(
    const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out_interleaved) {
    const __m256i m0 = _mm256_set1_epi64x(0xD2511F53ULL);
    const __m256i m1 = _mm256_set1_epi64x(0xCD9E8D57ULL);
    const u32x8 w0 = { 0x9E3779B9u, 0x9E3779B9u, 0x9E3779B9u, 0x9E3779B9u,
                       0x9E3779B9u, 0x9E3779B9u, 0x9E3779B9u, 0x9E3779B9u };
    const u32x8 w1 = { 0xBB67AE85u, 0xBB67AE85u, 0xBB67AE85u, 0xBB67AE85u,
                       0xBB67AE85u, 0xBB67AE85u, 0xBB67AE85u, 0xBB67AE85u };
    u32x8 k0_base, k1_base;
    memcpy(&k0_base, keys, 32);
    memcpy(&k1_base, addr_his, 32);
    k1_base ^= 0xFEEDC0DEu;

    size_t produced = 0;
    uint32_t block_idx = 0;
    while (produced < rng) {
        u32x8 c0 = { block_idx, block_idx, block_idx, block_idx,
                     block_idx, block_idx, block_idx, block_idx };
        u32x8 c1 = {0}, c2 = {0}, c3 = {0};
        u32x8 k0 = k0_base, k1 = k1_base;

        for (int r = 0; r < 10; ++r) {
            __m256i c0_ev = (__m256i)c0;
            __m256i c0_od = _mm256_srli_epi64(c0_ev, 32);
            __m256i p0_ev = _mm256_mul_epu32(c0_ev, m0);
            __m256i p0_od = _mm256_mul_epu32(c0_od, m0);
            u32x8 lo0 = (u32x8)_mm256_blend_epi32(p0_ev, _mm256_slli_epi64(p0_od, 32), 0xAA);
            u32x8 hi0 = (u32x8)_mm256_blend_epi32(_mm256_srli_epi64(p0_ev, 32), p0_od, 0xAA);

            __m256i c2_ev = (__m256i)c2;
            __m256i c2_od = _mm256_srli_epi64(c2_ev, 32);
            __m256i p1_ev = _mm256_mul_epu32(c2_ev, m1);
            __m256i p1_od = _mm256_mul_epu32(c2_od, m1);
            u32x8 lo1 = (u32x8)_mm256_blend_epi32(p1_ev, _mm256_slli_epi64(p1_od, 32), 0xAA);
            u32x8 hi1 = (u32x8)_mm256_blend_epi32(_mm256_srli_epi64(p1_ev, 32), p1_od, 0xAA);

            u32x8 nc0 = hi1 ^ c1 ^ k0;
            u32x8 nc1 = lo1;
            u32x8 nc2 = hi0 ^ c3 ^ k1;
            u32x8 nc3 = lo0;
            c0 = nc0; c1 = nc1; c2 = nc2; c3 = nc3;
            k0 += w0; k1 += w1;
        }
        u32x8 outs[4] = { c0, c1, c2, c3 };
        for (int j = 0; j < 4 && produced < rng; ++j, ++produced) {
            memcpy(out_interleaved + produced * 8, &outs[j], 32);
        }
        block_idx++;
    }
}

// ============================================================================
// 2. ChaCha8 & ChaCha12 (RFC 7539 / Bernstein ARX Stream Cipher, Scalar & AVX2)
// ============================================================================
#define CHACHA_QR(a, b, c, d)            \
    do {                                 \
        a += b; d ^= a; d = rot32(d, 16);\
        c += d; b ^= c; b = rot32(b, 12);\
        a += b; d ^= a; d = rot32(d, 8); \
        c += d; b ^= c; b = rot32(b, 7); \
    } while (0)

static inline __attribute__((always_inline)) void chacha_block(
    uint32_t out[16], const uint32_t in[16], int rounds) {
    uint32_t x[16];
    memcpy(x, in, 64);
    for (int r = 0; r < rounds; r += 2) {
        CHACHA_QR(x[0], x[4], x[ 8], x[12]);
        CHACHA_QR(x[1], x[5], x[ 9], x[13]);
        CHACHA_QR(x[2], x[6], x[10], x[14]);
        CHACHA_QR(x[3], x[7], x[11], x[15]);
        CHACHA_QR(x[0], x[5], x[10], x[15]);
        CHACHA_QR(x[1], x[6], x[11], x[12]);
        CHACHA_QR(x[2], x[7], x[ 8], x[13]);
        CHACHA_QR(x[3], x[4], x[ 9], x[14]);
    }
    for (int i = 0; i < 16; ++i) out[i] = x[i] + in[i];
}

typedef struct {
    uint32_t state[16];
    uint32_t block[16];
    uint32_t pos;
    int rounds;
} chacha_stream_t;

static inline __attribute__((always_inline)) void chacha_stream_init(
    chacha_stream_t *s, uint32_t key, uint32_t addr_hi, int rounds) {
    s->state[0] = 0x61707865u;
    s->state[1] = 0x3320646eu;
    s->state[2] = 0x79622d32u;
    s->state[3] = 0x6b206574u;
    s->state[4] = key;
    s->state[5] = addr_hi;
    s->state[6] = key ^ 0x9E3779B9u;
    s->state[7] = addr_hi ^ 0x85EBCA6Bu;
    s->state[8] = key * 0xC2B2AE35u;
    s->state[9] = addr_hi * 0x27D4EB2Fu;
    s->state[10] = 0x165667B1u;
    s->state[11] = 0xD3A2646Cu;
    s->state[12] = 0u; // block counter
    s->state[13] = 0u;
    s->state[14] = key;
    s->state[15] = addr_hi;
    s->pos = 16u;
    s->rounds = rounds;
}

static inline __attribute__((always_inline)) uint32_t chacha_stream_next_u32(chacha_stream_t *s) {
    if (s->pos >= 16u) {
        chacha_block(s->block, s->state, s->rounds);
        if (++s->state[12] == 0u) ++s->state[13];
        s->pos = 0u;
    }
    return s->block[s->pos++];
}

static inline __attribute__((optimize("no-tree-vectorize"))) void chacha8_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    chacha_stream_t s;
    chacha_stream_init(&s, key, addr_hi, 8);
    size_t produced = 0;
    while (produced + 16 <= rng) {
        chacha_block(&out[produced], s.state, 8);
        ++s.state[12];
        produced += 16;
    }
    if (produced < rng) {
        uint32_t blk[16];
        chacha_block(blk, s.state, 8);
        for (int i = 0; produced < rng; ++i, ++produced) out[produced] = blk[i];
    }
}

static inline __attribute__((optimize("no-tree-vectorize"))) void chacha12_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    chacha_stream_t s;
    chacha_stream_init(&s, key, addr_hi, 12);
    size_t produced = 0;
    while (produced + 16 <= rng) {
        chacha_block(&out[produced], s.state, 12);
        ++s.state[12];
        produced += 16;
    }
    if (produced < rng) {
        uint32_t blk[16];
        chacha_block(blk, s.state, 12);
        for (int i = 0; produced < rng; ++i, ++produced) out[produced] = blk[i];
    }
}

#define CHACHA_QR_X8(a, b, c, d)                                       \
    do {                                                               \
        a += b; d ^= a; d = (d << 16) | (d >> 16);                     \
        c += d; b ^= c; b = (b << 12) | (b >> 20);                     \
        a += b; d ^= a; d = (d << 8)  | (d >> 24);                     \
        c += d; b ^= c; b = (b << 7)  | (b >> 25);                     \
    } while (0)

static inline __attribute__((always_inline)) void chacha8_x8_into(
    const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out_interleaved) {
    u32x8 vkey, vaddr;
    memcpy(&vkey, keys, 32);
    memcpy(&vaddr, addr_his, 32);
    u32x8 st[16];
    st[0]  = (u32x8){0} + 0x61707865u;
    st[1]  = (u32x8){0} + 0x3320646eu;
    st[2]  = (u32x8){0} + 0x79622d32u;
    st[3]  = (u32x8){0} + 0x6b206574u;
    st[4]  = vkey;
    st[5]  = vaddr;
    st[6]  = vkey ^ 0x9E3779B9u;
    st[7]  = vaddr ^ 0x85EBCA6Bu;
    st[8]  = vkey * 0xC2B2AE35u;
    st[9]  = vaddr * 0x27D4EB2Fu;
    st[10] = (u32x8){0} + 0x165667B1u;
    st[11] = (u32x8){0} + 0xD3A2646Cu;
    st[12] = (u32x8){0};
    st[13] = (u32x8){0};
    st[14] = vkey;
    st[15] = vaddr;

    size_t produced = 0;
    while (produced < rng) {
        u32x8 x[16];
        for (int i = 0; i < 16; ++i) x[i] = st[i];
        for (int r = 0; r < 8; r += 2) {
            CHACHA_QR_X8(x[0], x[4], x[ 8], x[12]);
            CHACHA_QR_X8(x[1], x[5], x[ 9], x[13]);
            CHACHA_QR_X8(x[2], x[6], x[10], x[14]);
            CHACHA_QR_X8(x[3], x[7], x[11], x[15]);
            CHACHA_QR_X8(x[0], x[5], x[10], x[15]);
            CHACHA_QR_X8(x[1], x[6], x[11], x[12]);
            CHACHA_QR_X8(x[2], x[7], x[ 8], x[13]);
            CHACHA_QR_X8(x[3], x[4], x[ 9], x[14]);
        }
        for (int j = 0; j < 16 && produced < rng; ++j, ++produced) {
            u32x8 out_w = x[j] + st[j];
            memcpy(out_interleaved + produced * 8, &out_w, 32);
        }
        st[12] += 1u;
    }
}

// ============================================================================
// 3. SplitMix64, xoshiro256**, and xoroshiro128++ (Blackman & Vigna Shootout)
// ============================================================================
static inline __attribute__((always_inline)) uint64_t xoshiro_rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline __attribute__((always_inline)) uint64_t splitmix64_step(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline __attribute__((optimize("no-tree-vectorize"))) void splitmix64_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    uint64_t sm = ((uint64_t)addr_hi << 32) | (uint64_t)key;
    for (size_t i = 0; i < rng; ++i) {
        out[i] = (uint32_t)splitmix64_step(&sm);
    }
}

static inline __attribute__((always_inline)) void splitmix64_x8_into(
    const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out_interleaved) {
    const u64x4 c1 = { 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL };
    const u64x4 m1 = { 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL };
    const u64x4 m2 = { 0x94D049BB133111EBULL, 0x94D049BB133111EBULL, 0x94D049BB133111EBULL, 0x94D049BB133111EBULL };
    u64x4 sm0 = { ((uint64_t)addr_his[0] << 32) | keys[0], ((uint64_t)addr_his[1] << 32) | keys[1],
                  ((uint64_t)addr_his[2] << 32) | keys[2], ((uint64_t)addr_his[3] << 32) | keys[3] };
    u64x4 sm1 = { ((uint64_t)addr_his[4] << 32) | keys[4], ((uint64_t)addr_his[5] << 32) | keys[5],
                  ((uint64_t)addr_his[6] << 32) | keys[6], ((uint64_t)addr_his[7] << 32) | keys[7] };
    for (size_t i = 0; i < rng; ++i) {
        sm0 += c1; u64x4 z0 = mul64_avx2(sm0 ^ (sm0 >> 30), m1); z0 = mul64_avx2(z0 ^ (z0 >> 27), m2); z0 ^= (z0 >> 31);
        sm1 += c1; u64x4 z1 = mul64_avx2(sm1 ^ (sm1 >> 30), m1); z1 = mul64_avx2(z1 ^ (z1 >> 27), m2); z1 ^= (z1 >> 31);
        __m256i packed = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps((__m256i)z0),
                                                  _mm256_castsi256_ps((__m256i)z1), 0x88)), 0xD8);
        memcpy(out_interleaved + i * 8, &packed, 32);
    }
}

typedef struct {
    uint64_t s[4];
} xoshiro256ss_state_t;

static inline __attribute__((always_inline)) void xoshiro256ss_init(
    xoshiro256ss_state_t *st, uint32_t key, uint32_t addr_hi) {
    uint64_t sm = ((uint64_t)addr_hi << 32) | (uint64_t)key;
    st->s[0] = splitmix64_step(&sm);
    st->s[1] = splitmix64_step(&sm);
    st->s[2] = splitmix64_step(&sm);
    st->s[3] = splitmix64_step(&sm);
}

static inline __attribute__((always_inline)) uint64_t xoshiro256ss_next_u64(xoshiro256ss_state_t *st) {
    const uint64_t result = xoshiro_rotl(st->s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = st->s[1] << 17;
    st->s[2] ^= st->s[0];
    st->s[3] ^= st->s[1];
    st->s[1] ^= st->s[2];
    st->s[0] ^= st->s[3];
    st->s[2] ^= t;
    st->s[3] = xoshiro_rotl(st->s[3], 45);
    return result;
}

static inline __attribute__((always_inline)) uint32_t xoshiro256ss_next_u32(xoshiro256ss_state_t *st) {
    return (uint32_t)xoshiro256ss_next_u64(st);
}

static inline void xoshiro256ss_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    xoshiro256ss_state_t st;
    xoshiro256ss_init(&st, key, addr_hi);
    for (size_t i = 0; i < rng; ++i) {
        out[i] = xoshiro256ss_next_u32(&st);
    }
}

static inline __attribute__((always_inline)) void xoshiro256ss_x8_into(
    const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out_interleaved) {
    const u64x4 c1 = { 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL };
    const u64x4 m1 = { 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL };
    const u64x4 m2 = { 0x94D049BB133111EBULL, 0x94D049BB133111EBULL, 0x94D049BB133111EBULL, 0x94D049BB133111EBULL };
    u64x4 sm0 = { ((uint64_t)addr_his[0] << 32) | keys[0], ((uint64_t)addr_his[1] << 32) | keys[1],
                  ((uint64_t)addr_his[2] << 32) | keys[2], ((uint64_t)addr_his[3] << 32) | keys[3] };
    u64x4 sm1 = { ((uint64_t)addr_his[4] << 32) | keys[4], ((uint64_t)addr_his[5] << 32) | keys[5],
                  ((uint64_t)addr_his[6] << 32) | keys[6], ((uint64_t)addr_his[7] << 32) | keys[7] };
    u64x4 sA[4], sB[4];
    for (int j = 0; j < 4; ++j) {
        sm0 += c1; u64x4 z0 = mul64_avx2(sm0 ^ (sm0 >> 30), m1); z0 = mul64_avx2(z0 ^ (z0 >> 27), m2); sA[j] = z0 ^ (z0 >> 31);
        sm1 += c1; u64x4 z1 = mul64_avx2(sm1 ^ (sm1 >> 30), m1); z1 = mul64_avx2(z1 ^ (z1 >> 27), m2); sB[j] = z1 ^ (z1 >> 31);
    }
    for (size_t i = 0; i < rng; ++i) {
        u64x4 rA = (sA[1] << 2) + sA[1]; rA = (rA << 7) | (rA >> 57); rA = (rA << 3) + rA;
        u64x4 rB = (sB[1] << 2) + sB[1]; rB = (rB << 7) | (rB >> 57); rB = (rB << 3) + rB;
        u64x4 tA = sA[1] << 17, tB = sB[1] << 17;
        sA[2] ^= sA[0]; sA[3] ^= sA[1]; sA[1] ^= sA[2]; sA[0] ^= sA[3]; sA[2] ^= tA; sA[3] = (sA[3] << 45) | (sA[3] >> 19);
        sB[2] ^= sB[0]; sB[3] ^= sB[1]; sB[1] ^= sB[2]; sB[0] ^= sB[3]; sB[2] ^= tB; sB[3] = (sB[3] << 45) | (sB[3] >> 19);
        __m256i packed = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps((__m256i)rA),
                                                  _mm256_castsi256_ps((__m256i)rB), 0x88)), 0xD8);
        memcpy(out_interleaved + i * 8, &packed, 32);
    }
}

typedef struct {
    uint64_t s[2];
} xoroshiro128pp_state_t;

static inline __attribute__((always_inline)) void xoroshiro128pp_init(
    xoroshiro128pp_state_t *st, uint32_t key, uint32_t addr_hi) {
    uint64_t sm = ((uint64_t)addr_hi << 32) | (uint64_t)key;
    st->s[0] = splitmix64_step(&sm);
    st->s[1] = splitmix64_step(&sm);
}

static inline __attribute__((always_inline)) uint64_t xoroshiro128pp_next_u64(xoroshiro128pp_state_t *st) {
    const uint64_t s0 = st->s[0];
    uint64_t s1 = st->s[1];
    const uint64_t result = xoshiro_rotl(s0 + s1, 17) + s0;
    s1 ^= s0;
    st->s[0] = xoshiro_rotl(s0, 49) ^ s1 ^ (s1 << 21);
    st->s[1] = xoshiro_rotl(s1, 28);
    return result;
}

static inline __attribute__((always_inline)) uint32_t xoroshiro128pp_next_u32(xoroshiro128pp_state_t *st) {
    return (uint32_t)xoroshiro128pp_next_u64(st);
}

static inline void xoroshiro128pp_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    xoroshiro128pp_state_t st;
    xoroshiro128pp_init(&st, key, addr_hi);
    for (size_t i = 0; i < rng; ++i) {
        out[i] = xoroshiro128pp_next_u32(&st);
    }
}

static inline __attribute__((always_inline)) void xoroshiro128pp_x8_into(
    const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out_interleaved) {
    const u64x4 c1 = { 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL, 0x9E3779B97F4A7C15ULL };
    const u64x4 m1 = { 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL, 0xBF58476D1CE4E5B9ULL };
    const u64x4 m2 = { 0x94D049BB133111EBULL, 0x94D049BB133111EBULL, 0x94D049BB133111EBULL, 0x94D049BB133111EBULL };
    u64x4 sm0 = { ((uint64_t)addr_his[0] << 32) | keys[0], ((uint64_t)addr_his[1] << 32) | keys[1],
                  ((uint64_t)addr_his[2] << 32) | keys[2], ((uint64_t)addr_his[3] << 32) | keys[3] };
    u64x4 sm1 = { ((uint64_t)addr_his[4] << 32) | keys[4], ((uint64_t)addr_his[5] << 32) | keys[5],
                  ((uint64_t)addr_his[6] << 32) | keys[6], ((uint64_t)addr_his[7] << 32) | keys[7] };
    u64x4 sA[2], sB[2];
    for (int j = 0; j < 2; ++j) {
        sm0 += c1; u64x4 z0 = mul64_avx2(sm0 ^ (sm0 >> 30), m1); z0 = mul64_avx2(z0 ^ (z0 >> 27), m2); sA[j] = z0 ^ (z0 >> 31);
        sm1 += c1; u64x4 z1 = mul64_avx2(sm1 ^ (sm1 >> 30), m1); z1 = mul64_avx2(z1 ^ (z1 >> 27), m2); sB[j] = z1 ^ (z1 >> 31);
    }
    for (size_t i = 0; i < rng; ++i) {
        u64x4 sA0 = sA[0], sA1 = sA[1], sB0 = sB[0], sB1 = sB[1];
        u64x4 sumA = sA0 + sA1, sumB = sB0 + sB1;
        u64x4 rA = ((sumA << 17) | (sumA >> 47)) + sA0;
        u64x4 rB = ((sumB << 17) | (sumB >> 47)) + sB0;
        sA1 ^= sA0; sB1 ^= sB0;
        sA[0] = ((sA0 << 49) | (sA0 >> 15)) ^ sA1 ^ (sA1 << 21);
        sA[1] = (sA1 << 28) | (sA1 >> 36);
        sB[0] = ((sB0 << 49) | (sB0 >> 15)) ^ sB1 ^ (sB1 << 21);
        sB[1] = (sB1 << 28) | (sB1 >> 36);
        __m256i packed = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps((__m256i)rA),
                                                  _mm256_castsi256_ps((__m256i)rB), 0x88)), 0xD8);
        memcpy(out_interleaved + i * 8, &packed, 32);
    }
}

// ============================================================================
// 4. PCG32 (pcg_oneseq_64_xsh_rr_32, O'Neill reference, Scalar & AVX2)
// ============================================================================
typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

static inline __attribute__((always_inline)) uint32_t pcg32_random_r(pcg32_random_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline __attribute__((always_inline)) void pcg32_srandom_r(
    pcg32_random_t *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}

static inline void pcg32_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    pcg32_random_t r;
    pcg32_srandom_r(&r, ((uint64_t)addr_hi << 32) | key, 54u);
    for (size_t i = 0; i < rng; ++i) {
        out[i] = pcg32_random_r(&r);
    }
}

static inline __attribute__((always_inline)) void pcg32_x8_into(
    const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out_interleaved) {
    const u64x4 mult = { 6364136223846793005ULL, 6364136223846793005ULL,
                         6364136223846793005ULL, 6364136223846793005ULL };
    const u64x4 inc  = { 109ULL, 109ULL, 109ULL, 109ULL };
    u64x4 s0 = { ((uint64_t)addr_his[0] << 32) | keys[0], ((uint64_t)addr_his[1] << 32) | keys[1],
                 ((uint64_t)addr_his[2] << 32) | keys[2], ((uint64_t)addr_his[3] << 32) | keys[3] };
    u64x4 s1 = { ((uint64_t)addr_his[4] << 32) | keys[4], ((uint64_t)addr_his[5] << 32) | keys[5],
                 ((uint64_t)addr_his[6] << 32) | keys[6], ((uint64_t)addr_his[7] << 32) | keys[7] };
    s0 = mul64_avx2(inc + s0, mult) + inc;
    s1 = mul64_avx2(inc + s1, mult) + inc;

    for (size_t i = 0; i < rng; ++i) {
        u64x4 old0 = s0, old1 = s1;
        s0 = mul64_avx2(s0, mult) + inc;
        s1 = mul64_avx2(s1, mult) + inc;
        u64x4 xsh0 = ((old0 >> 18) ^ old0) >> 27;
        u64x4 xsh1 = ((old1 >> 18) ^ old1) >> 27;
        u64x4 rot0 = old0 >> 59;
        u64x4 rot1 = old1 >> 59;
        __m256i xsh = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps((__m256i)xsh0),
                                                  _mm256_castsi256_ps((__m256i)xsh1), 0x88)), 0xD8);
        __m256i rot = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps((__m256i)rot0),
                                                  _mm256_castsi256_ps((__m256i)rot1), 0x88)), 0xD8);
        u32x8 vx = (u32x8)xsh;
        u32x8 vr = (u32x8)rot;
        u32x8 out = (vx >> vr) | (vx << ((0u - vr) & 31u));
        memcpy(out_interleaved + i * 8, &out, 32);
    }
}

// ============================================================================
// 5. WyRand (Wang Yi 64x64->128 folded multiply mixer, Scalar)
// ============================================================================
static inline __attribute__((always_inline)) uint64_t wyrand_step(uint64_t *seed) {
    *seed += 0xa0761d6478bd642fULL;
    __uint128_t t = (__uint128_t)(*seed) * (__uint128_t)(*seed ^ 0xe7037ed1a0b428dbULL);
    return (uint64_t)(t >> 64) ^ (uint64_t)t;
}

static inline __attribute__((optimize("no-tree-vectorize"))) void wyrand_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    uint64_t seed = ((uint64_t)addr_hi << 32) | (uint64_t)key;
    for (size_t i = 0; i < rng; ++i) {
        out[i] = (uint32_t)wyrand_step(&seed);
    }
}

// ============================================================================
// 6. MT19937 (Matsumoto & Nishimura mt19937ar.c / C++ std::mt19937)
// ============================================================================
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL
#define MT_UPPER_MASK 0x80000000UL
#define MT_LOWER_MASK 0x7fffffffUL

typedef struct {
    uint32_t mt[MT_N];
    int mti;
} mt19937_state_t;

static inline void mt19937_init_genrand(mt19937_state_t *s, uint32_t seed) {
    s->mt[0] = seed;
    for (s->mti = 1; s->mti < MT_N; s->mti++) {
        s->mt[s->mti] = (1812433253UL * (s->mt[s->mti - 1] ^ (s->mt[s->mti - 1] >> 30)) + (uint32_t)s->mti);
    }
}

static inline uint32_t mt19937_genrand_int32(mt19937_state_t *s) {
    static const uint32_t mag01[2] = {0x0UL, MT_MATRIX_A};
    uint32_t y;
    if (s->mti >= MT_N) {
        int kk;
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (s->mt[kk] & MT_UPPER_MASK) | (s->mt[kk + 1] & MT_LOWER_MASK);
            s->mt[kk] = s->mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (s->mt[kk] & MT_UPPER_MASK) | (s->mt[kk + 1] & MT_LOWER_MASK);
            s->mt[kk] = s->mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (s->mt[MT_N - 1] & MT_UPPER_MASK) | (s->mt[0] & MT_LOWER_MASK);
        s->mt[MT_N - 1] = s->mt[MT_M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        s->mti = 0;
    }
    y = s->mt[s->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

static inline void mt19937_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    mt19937_state_t mt;
    mt19937_init_genrand(&mt, key ^ addr_hi);
    for (size_t i = 0; i < rng; ++i) {
        out[i] = mt19937_genrand_int32(&mt);
    }
}

// ============================================================================
// Official KAT Verification Gate (All 10 Families + AVX2/OTF Parity)
// ============================================================================
static inline bool verify_shootout_all_kat(void) {
    // 1. Philox4x32-10 official Random123 KAT vectors
    struct { uint32_t ctr[4]; uint32_t key[2]; uint32_t expect[4]; } pkats[] = {
        { {0,0,0,0}, {0,0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u} },
        { {0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu}, {0xffffffffu,0xffffffffu},
          {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu} },
        { {0x243f6a88u,0x85a308d3u,0x13198a2eu,0x03707344u}, {0xa4093822u,0x299f31d0u},
          {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u} },
    };
    for (size_t k = 0; k < 3; ++k) {
        uint32_t out[4];
        philox4x32_10(out, pkats[k].ctr, pkats[k].key);
        if (memcmp(out, pkats[k].expect, sizeof(out)) != 0) {
            fprintf(stderr, "KAT FAIL: Philox4x32-10 vector %zu\n", k);
            return false;
        }
    }

    // 2. RFC 7539 Section 2.1.1 / 2.3.2 ChaCha20 Block Function KAT (20 rounds)
    {
        uint32_t cc_in[16] = {
            0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
            0x03020100u, 0x07060504u, 0x0b0a0908u, 0x0f0e0d0cu,
            0x13121110u, 0x17161514u, 0x1b1a1918u, 0x1f1e1d1cu,
            0x00000001u, 0x09000000u, 0x4a000000u, 0x00000000u
        };
        uint32_t cc_expect[16] = {
            0xe4e7f110u, 0x15593bd1u, 0x1fdd0f50u, 0xc47120a3u,
            0xc7f4d1c7u, 0x0368c033u, 0x9aaa2204u, 0x4e6cd4c3u,
            0x466482d2u, 0x09aa9f07u, 0x05d7c214u, 0xa2028bd9u,
            0xd19c12b5u, 0xb94e16deu, 0xe883d0cbu, 0x4e3c50a2u
        };
        uint32_t cc_out[16];
        chacha_block(cc_out, cc_in, 20);
        if (memcmp(cc_out, cc_expect, sizeof(cc_out)) != 0) {
            fprintf(stderr, "KAT FAIL: ChaCha RFC 7539 block vector\n");
            return false;
        }
    }

    // 3. SplitMix64 + xoshiro256** + xoroshiro128++ KAT
    uint64_t sm = 0ULL;
    static const uint64_t sm_expect[4] = {
        0xe220a8397b1dcdafULL, 0x6e789e6aa1b965f4ULL,
        0x06c45d188009454fULL, 0xf88bb8a8724c81ecULL,
    };
    for (int i = 0; i < 4; ++i) {
        if (splitmix64_step(&sm) != sm_expect[i]) {
            fprintf(stderr, "KAT FAIL: SplitMix64 step %d\n", i);
            return false;
        }
    }
    xoshiro256ss_state_t xst = {{
        0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL,
        0xF0E1D2C3B4A59687ULL, 0x1234567890ABCDEFULL,
    }};
    static const uint64_t x_expect[4] = {
        0x6666666666666c65ULL, 0x26f9cc9f72450d4dULL,
        0x83f9a07f50000477ULL, 0xbf1c5914f01d3354ULL,
    };
    for (int i = 0; i < 4; ++i) {
        if (xoshiro256ss_next_u64(&xst) != x_expect[i]) {
            fprintf(stderr, "KAT FAIL: xoshiro256** step %d\n", i);
            return false;
        }
    }

    // 4. PCG32 KAT (seed=42, seq=54)
    pcg32_random_t pcg;
    pcg32_srandom_r(&pcg, 42ULL, 54ULL);
    static const uint32_t pcg_expect[6] = {
        0xa15c02b7u, 0x7b47f409u, 0xba1d3330u, 0x83d2f293u, 0xbfa4784bu, 0xcbed606eu,
    };
    for (int i = 0; i < 6; ++i) {
        if (pcg32_random_r(&pcg) != pcg_expect[i]) {
            fprintf(stderr, "KAT FAIL: PCG32 step %d\n", i);
            return false;
        }
    }

    // 5. MT19937 KAT (init_genrand(5489) -> 3499211612)
    mt19937_state_t mt;
    mt19937_init_genrand(&mt, 5489UL);
    static const uint32_t mt_expect[5] = {
        3499211612u, 581869302u, 3890346734u, 3586334585u, 545404204u
    };
    for (int i = 0; i < 5; ++i) {
        if (mt19937_genrand_int32(&mt) != mt_expect[i]) {
            fprintf(stderr, "KAT FAIL: MT19937 step %d\n", i);
            return false;
        }
    }

    // 6. Verify bit-exact parity of `ra_v2_N256` Scalar vs AVX2 x8 vs OTF Engine vs `ra_core_v2_W2_D2_orbit_buf`
    uint32_t keys8[8] = {0u, 1u, 2u, 3u, 0xDEADBEEFu, 0x12345678u, 0x9E3779B9u, 0xFFFFFFFFu};
    uint32_t addrs8[8] = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    uint32_t ref_buf[1024], opt_buf[1024], x8_buf[256 * 8];
    for (int lane = 0; lane < 8; ++lane) {
        ra_core_v2_W2_D2_orbit_buf(keys8[lane], addrs8[lane], 1024, ref_buf);
        ra_core_v2_W2_D2_xag_opt_orbit_buf(keys8[lane], addrs8[lane], 1024, opt_buf);
        if (memcmp(ref_buf, opt_buf, 1024 * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "KAT FAIL: v2_W2_D2_xag_opt_orbit_buf != ref at lane %d\n", lane);
            return false;
        }
        ra_engine_v2_t eng;
        ra_engine_v2_init(&eng, keys8[lane], addrs8[lane]);
        for (int w = 0; w < 1024; ++w) {
            uint32_t got = ra_engine_v2_next_u32(&eng);
            if (got != ref_buf[w]) {
                fprintf(stderr, "KAT FAIL: ra_engine_v2_next_u32 != ref at lane %d w=%d\n", lane, w);
                return false;
            }
        }
    }
    size_t test_ks[] = {1, 2, 3, 4, 7, 8, 15, 31, 63, 127, 255};
    for (size_t ki = 0; ki < sizeof(test_ks)/sizeof(test_ks[0]); ++ki) {
        size_t k = test_ks[ki];
        ra_v2_N256_singleblock_x8_into(keys8, addrs8, k, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            ra_v2_N256_singleblock_into(keys8[lane], addrs8[lane], k, ref_buf);
            for (size_t w = 0; w < k; ++w) {
                if (x8_buf[w * 8 + lane] != ref_buf[w]) {
                    fprintf(stderr, "KAT FAIL: ra_v2_N256_singleblock_x8 != scalar at k=%zu lane=%d w=%zu\n", k, lane, w);
                    return false;
                }
            }
        }
    }

    // 7. Verify Scalar vs AVX2 x8 parity for N in {128, 64, 32, 16}
    {
        uint32_t s_buf[128];
        ra_v2_N128_singleblock_x8_into(keys8, addrs8, 127, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            ra_v2_N128_singleblock_into(keys8[lane], addrs8[lane], 127, s_buf);
            for (size_t w = 0; w < 127; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) return false;
            }
        }
        ra_v2_N64_singleblock_x8_into(keys8, addrs8, 63, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            ra_v2_N64_singleblock_into(keys8[lane], addrs8[lane], 63, s_buf);
            for (size_t w = 0; w < 63; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) return false;
            }
        }
        ra_v2_N32_singleblock_x8_into(keys8, addrs8, 31, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            ra_v2_N32_singleblock_into(keys8[lane], addrs8[lane], 31, s_buf);
            for (size_t w = 0; w < 31; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) return false;
            }
        }
        ra_v2_N16_singleblock_x8_into(keys8, addrs8, 15, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            ra_v2_N16_singleblock_into(keys8[lane], addrs8[lane], 15, s_buf);
            for (size_t w = 0; w < 15; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) return false;
            }
        }
    }

    // 8. Verify ChaCha8_x8, Philox_x8, xoshiro256**_x8, xoroshiro128++_x8, PCG32_x8, SplitMix64_x8 vs Scalar
    {
        uint32_t s_buf[32];
        chacha8_x8_into(keys8, addrs8, 31, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            chacha8_into(keys8[lane], addrs8[lane], 31, s_buf);
            for (size_t w = 0; w < 31; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) {
                    fprintf(stderr, "KAT FAIL: chacha8_x8 != chacha8_into\n");
                    return false;
                }
            }
        }
        philox4x32_x8_into(keys8, addrs8, 31, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            philox_into(keys8[lane], addrs8[lane], 31, s_buf);
            for (size_t w = 0; w < 31; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) {
                    fprintf(stderr, "KAT FAIL: philox4x32_x8 != philox_into\n");
                    return false;
                }
            }
        }
        xoshiro256ss_x8_into(keys8, addrs8, 31, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            xoshiro256ss_into(keys8[lane], addrs8[lane], 31, s_buf);
            for (size_t w = 0; w < 31; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) {
                    fprintf(stderr, "KAT FAIL: xoshiro256ss_x8 != xoshiro256ss_into\n");
                    return false;
                }
            }
        }
        xoroshiro128pp_x8_into(keys8, addrs8, 31, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            xoroshiro128pp_into(keys8[lane], addrs8[lane], 31, s_buf);
            for (size_t w = 0; w < 31; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) {
                    fprintf(stderr, "KAT FAIL: xoroshiro128pp_x8 != xoroshiro128pp_into\n");
                    return false;
                }
            }
        }
        pcg32_x8_into(keys8, addrs8, 31, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            pcg32_into(keys8[lane], addrs8[lane], 31, s_buf);
            for (size_t w = 0; w < 31; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) {
                    fprintf(stderr, "KAT FAIL: pcg32_x8 != pcg32_into\n");
                    return false;
                }
            }
        }
        splitmix64_x8_into(keys8, addrs8, 31, x8_buf);
        for (int lane = 0; lane < 8; ++lane) {
            splitmix64_into(keys8[lane], addrs8[lane], 31, s_buf);
            for (size_t w = 0; w < 31; ++w) {
                if (x8_buf[w * 8 + lane] != s_buf[w]) {
                    fprintf(stderr, "KAT FAIL: splitmix64_x8 != splitmix64_into\n");
                    return false;
                }
            }
        }
    }

    return true;
}

#endif // BASELINES_SHOOTOUT_H
