// ra_core.h
// Public header and inline core primitives for ra_core v2 Final
// (W2_D2_xag_opt + Opsi_A3 2D Cross-Pre-Mix Key-Mixing).
//
// Copyright (c) 2025-2026 Hamas A. Rahman
// Licensed under CC BY-NC-SA 4.0.

#ifndef RA_CORE_H
#define RA_CORE_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdalign.h>

#ifndef RA_N
#define RA_N 256
#endif

#define RA_MASK       ((uint32_t)(RA_N - 1))
#define RA_MAX_SINGLE (RA_N - 1)
#define RA_IDX_OFF    1u

// Weak-key and 64-bit address-widening guard constants (popcount == 16 each)
#define GUARD_L  0x38916df4u
#define GUARD_M  0x6c26fc92u
#define GUARD_L2 0x3eb13b90u
#define GUARD_M2 0x392456deu

static inline __attribute__((always_inline)) uint32_t rot32(uint32_t n, uint32_t r) {
    uint32_t s = r & 31u;
    return (n << s) | (n >> ((-s) & 31u));
}

static inline __attribute__((always_inline)) uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// ----------------------------------------------------------------------
// Opsi_A3 Key-Mixing & Step Derivation Primitives (2026-11-04 Stage 1 Winner)
// Cross-mixes (key, addr_hi) into both u and v BEFORE the two fmix32 calls,
// derives k_mix = km_k ^ rot32(km_p, 13), and forces odd bijective step_m/step_l
// with +1 index offset (`(idx + 1u) * step_m`). Zero extra instructions in the
// generation loop.
// ----------------------------------------------------------------------

static inline __attribute__((always_inline)) void ra_derive_m_A3(
    uint32_t key, uint32_t addr_hi, uint32_t *step_m_out, uint32_t *k_mix_out) {
    uint32_t p_addr = 0x9e3779b1u * addr_hi;
    uint32_t u_m = (0x06a0dd9bu * key) + p_addr;
    uint32_t v_m = p_addr ^ rot32(key, 17);
    uint32_t km_k = fmix32(GUARD_M  ^ u_m);
    uint32_t km_p = fmix32(GUARD_M2 ^ v_m);
    *k_mix_out  = km_k ^ rot32(km_p, 13);
    *step_m_out = (0x06a0dd9bu * (km_k + km_p)) | 1u;
}

static inline __attribute__((always_inline)) uint32_t ra_derive_l_A3(
    uint32_t key, uint32_t addr_hi) {
    uint32_t p_addr = 0x85ebca77u * addr_hi;
    uint32_t u_l = (0x9e3779b7u * key) + p_addr;
    uint32_t v_l = p_addr ^ rot32(key, 15);
    uint32_t kl_k = fmix32(GUARD_L  ^ u_l);
    uint32_t kl_p = fmix32(GUARD_L2 ^ v_l);
    return (0x9e3779b7u * (kl_k + kl_p)) | 1u;
}

// End-of-cycle O(1) guard fold for W2_D2 in-place L->M coupling
#define END_CYCLE_M_W2(M, L)                                                         \
    do {                                                                             \
        M[0] += M[RA_N] - M[1];                                                      \
        for (int g = 1; g < 7; ++g) {                                                \
            M[g] += M[RA_N + g];                                                     \
            M[RA_N + g] = 0u;                                                        \
        }                                                                            \
        M[7] += L[0];                                                                \
        M[RA_N] = M[1];                                                              \
    } while (0)

// ----------------------------------------------------------------------
// Register-only algebraic fast-paths for K = 1, K = 2, and K = 3..8
// ----------------------------------------------------------------------

static inline __attribute__((always_inline)) uint32_t ra_eval_k1_word(
    uint32_t key, uint32_t addr_hi) {
    uint32_t step_m, k_mix;
    ra_derive_m_A3(key, addr_hi, &step_m, &k_mix);
    uint32_t i0 = (uint32_t)RA_MAX_SINGLE;
    uint32_t o = rot32((i0 + RA_IDX_OFF) * step_m, k_mix ^ i0)
               ^ (rot32((1u + RA_IDX_OFF) * step_m, k_mix ^ 1u) << 1);
    uint32_t a = o ^ (k_mix << 1);
    uint32_t b = (k_mix + a) ^ o;
    uint32_t c = rot32((a >> 13) ^ a, b);
    return c ^ (c >> 17u);
}

static inline __attribute__((always_inline)) void ra_eval_k2_words(
    uint32_t key, uint32_t addr_hi, uint32_t * restrict out_buf) {
    uint32_t step_m, k_mix;
    ra_derive_m_A3(key, addr_hi, &step_m, &k_mix);
    uint32_t i0 = (uint32_t)RA_MAX_SINGLE;
    uint32_t i1 = i0 - 1u;

    uint32_t m_n1 = rot32((i0 + RA_IDX_OFF) * step_m, k_mix ^ i0);
    uint32_t m1   = rot32((1u + RA_IDX_OFF) * step_m, k_mix ^ 1u);
    uint32_t m_n2 = rot32((i1 + RA_IDX_OFF) * step_m, k_mix ^ i1);

    uint32_t o_0 = m_n1 ^ (m1 << 1);
    uint32_t o_1 = m_n2 ^ (o_0 << 1);
    uint32_t d_1 = (rot32(k_mix, 15) ^ 0x55u) & RA_MASK;

    uint32_t a_0 = o_0 ^ (k_mix << 1);
    uint32_t s_1 = k_mix + a_0;
    uint32_t a_1 = (d_1 ^ o_1) ^ s_1;
    uint32_t s_2 = k_mix + a_1;

    uint32_t b_0 = s_1 ^ o_0;
    uint32_t b_1 = s_2 ^ (o_1 + d_1);
    uint32_t u_0 = (a_0 >> 13) ^ a_0;
    uint32_t u_1 = (a_1 >> 13) ^ a_1;
    uint32_t c_0 = rot32(u_0, b_0);
    uint32_t c_1 = rot32(u_1, b_1);
    out_buf[0] = c_0 ^ (c_0 >> 17u);
    out_buf[1] = c_1 ^ (c_1 >> 17u);
}

static inline __attribute__((always_inline)) void ra_eval_k3_to_k8_words(
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out_buf) {
    uint32_t step_m, k_mix;
    ra_derive_m_A3(key, addr_hi, &step_m, &k_mix);
    uint32_t m1 = rot32((1u + RA_IDX_OFF) * step_m, k_mix ^ 1u);
    uint32_t i = (uint32_t)RA_MAX_SINGLE;

    uint32_t o_0 = rot32((i + RA_IDX_OFF) * step_m, k_mix ^ i) ^ (m1 << 1);
    uint32_t s = k_mix << 1;
    uint32_t d0 = 0u;
    uint32_t d1 = (rot32(k_mix, 15) ^ 0x55u) & RA_MASK;
    uint32_t rem = (uint32_t)rng;

    while (rem >= 2u) {
        uint32_t i1 = i - 1u;
        uint32_t m_im1 = rot32((i1 + RA_IDX_OFF) * step_m, k_mix ^ i1);
        uint32_t o_1 = (i == (uint32_t)(RA_N - 7))
                     ? (m_im1 ^ ((o_0 ^ (m1 << 7)) << 1))
                     : (m_im1 ^ (o_0 << 1));

        uint32_t a_0 = (d0 ^ o_0) ^ s;
        uint32_t s_1 = k_mix + a_0;
        uint32_t a_1 = (d1 ^ o_1) ^ s_1;
        uint32_t s_2 = k_mix + a_1;

        uint32_t b_0 = s_1 ^ (o_0 + d0);
        uint32_t b_1 = s_2 ^ (o_1 + d1);
        uint32_t u_0 = (a_0 >> 13) ^ a_0;
        uint32_t u_1 = (a_1 >> 13) ^ a_1;
        uint32_t c_0 = rot32(u_0, b_0);
        uint32_t c_1 = rot32(u_1, b_1);
        c_0 ^= c_0 >> 17u;
        c_1 ^= c_1 >> 17u;

        out_buf[0] = c_0;
        out_buf[1] = c_1;
        out_buf += 2;

        rem -= 2u;
        if (rem == 0u) return;

        s = s_2;
        d0 = c_0 & RA_MASK;
        d1 = c_1 & RA_MASK;
        uint32_t i2 = i - 2u;
        uint32_t m_im2 = rot32((i2 + RA_IDX_OFF) * step_m, k_mix ^ i2);
        o_0 = m_im2 ^ (o_1 << 1);
        i -= 2u;
    }

    uint32_t a_0 = (d0 ^ o_0) ^ s;
    uint32_t s_1 = k_mix + a_0;
    uint32_t b_0 = s_1 ^ (o_0 + d0);
    uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);
    out_buf[0] = c_0 ^ (c_0 >> 17u);
}

// ----------------------------------------------------------------------
// 2-Step Unrolled W2_D2_xag_opt Cycle Functions (Singleblock & Multi-Cycle Orbit)
// ----------------------------------------------------------------------

static inline __attribute__((always_inline)) void ra_singleblock_cycle_v2_buf(
    uint32_t cons, const uint32_t * restrict M, uint32_t m1, size_t rng, uint32_t * restrict out_buf) {
    uint32_t s = cons << 1;
    uint32_t d0 = 0u;
    uint32_t d1 = (rot32(cons, 15) ^ 0x55u) & RA_MASK;
    uint32_t o_0 = M[RA_MAX_SINGLE] ^ (m1 << 1);
    uint32_t i = (uint32_t)RA_MAX_SINGLE;

    // First 3 pairs (6 words, i = N-1 down to N-6): o_term7 == 0 algebraically!
    for (int pair = 0; pair < 3; ++pair) {
        uint32_t o_1 = M[i - 1u] ^ (o_0 << 1);

        uint32_t a_0 = (d0 ^ o_0) ^ s;
        uint32_t s_1 = cons + a_0;
        uint32_t a_1 = (d1 ^ o_1) ^ s_1;
        uint32_t s_2 = cons + a_1;

        uint32_t b_0 = s_1 ^ (o_0 + d0);
        uint32_t b_1 = s_2 ^ (o_1 + d1);
        uint32_t u_0 = (a_0 >> 13) ^ a_0;
        uint32_t u_1 = (a_1 >> 13) ^ a_1;
        uint32_t c_0 = rot32(u_0, b_0);
        uint32_t c_1 = rot32(u_1, b_1);
        c_0 ^= c_0 >> 17u;
        c_1 ^= c_1 >> 17u;

        out_buf[0] = c_0;
        out_buf[1] = c_1;
        out_buf += 2;

        s = s_2;
        d0 = c_0 & RA_MASK;
        d1 = c_1 & RA_MASK;
        o_0 = M[i - 2u] ^ (o_1 << 1);
        i -= 2u;
    }

    // Now i == RA_N - 7, o_term7_0 == m1 << 7
    uint32_t o_term7_0 = m1 << 7;
    uint32_t rem = (uint32_t)(rng - 6u);

    while (rem >= 2u) {
        uint32_t o_1 = M[i - 1u] ^ ((o_0 ^ o_term7_0) << 1);
        uint32_t o_term7_1 = M[i + 6u] << 7;

        uint32_t a_0 = (d0 ^ o_0) ^ s;
        uint32_t s_1 = cons + a_0;
        uint32_t a_1 = (d1 ^ o_1) ^ s_1;
        uint32_t s_2 = cons + a_1;

        uint32_t b_0 = s_1 ^ (o_0 + d0);
        uint32_t b_1 = s_2 ^ (o_1 + d1);
        uint32_t u_0 = (a_0 >> 13) ^ a_0;
        uint32_t u_1 = (a_1 >> 13) ^ a_1;
        uint32_t c_0 = rot32(u_0, b_0);
        uint32_t c_1 = rot32(u_1, b_1);
        c_0 ^= c_0 >> 17u;
        c_1 ^= c_1 >> 17u;

        out_buf[0] = c_0;
        out_buf[1] = c_1;
        out_buf += 2;

        rem -= 2u;
        if (rem == 0u) return;

        s = s_2;
        d0 = c_0 & RA_MASK;
        d1 = c_1 & RA_MASK;
        o_0 = M[i - 2u] ^ ((o_1 ^ o_term7_1) << 1);
        o_term7_0 = M[i + 5u] << 7;
        i -= 2u;
    }

    if (rem == 1u) {
        uint32_t a_0 = (d0 ^ o_0) ^ s;
        uint32_t s_1 = cons + a_0;
        uint32_t b_0 = s_1 ^ (o_0 + d0);
        uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);
        out_buf[0] = c_0 ^ (c_0 >> 17u);
    }
}

static inline bool ra_orbit_cycle_v2_buf(
    uint32_t * restrict cons_io, uint32_t * restrict M, uint32_t * restrict L,
    uint64_t *count, uint32_t **out_cursor) {
    uint32_t cons = *cons_io;
    uint32_t s = cons << 1;
    uint32_t d0 = 0u;
    uint32_t d1 = (rot32(cons, 15) ^ 0x55u) & RA_MASK;
    uint32_t o_0 = M[RA_MAX_SINGLE] ^ (M[RA_N] << 1);
    uint32_t o_term7_0 = 0u;
    uint32_t * restrict out = *out_cursor;

    if (*count > (uint64_t)RA_MAX_SINGLE) {
        *count -= (uint64_t)RA_MAX_SINGLE;
        uint32_t i = (uint32_t)RA_MAX_SINGLE;

        for (; i >= 3u; i -= 2u) {
            uint32_t o_1 = M[i - 1u] ^ ((o_0 ^ o_term7_0) << 1);
            uint32_t o_term7_1 = M[i + 6u] << 7;

            uint32_t a_0 = (d0 ^ o_0) ^ s;
            uint32_t s_1 = cons + a_0;
            uint32_t a_1 = (d1 ^ o_1) ^ s_1;
            uint32_t s_2 = cons + a_1;

            uint32_t b_0 = s_1 ^ (o_0 + d0);
            uint32_t b_1 = s_2 ^ (o_1 + d1);
            uint32_t u_0 = (a_0 >> 13) ^ a_0;
            uint32_t u_1 = (a_1 >> 13) ^ a_1;
            uint32_t c_0 = rot32(u_0, b_0);
            uint32_t c_1 = rot32(u_1, b_1);
            c_0 ^= c_0 >> 17u;
            c_1 ^= c_1 >> 17u;

            out[0] = c_0;
            out[1] = c_1;
            out += 2;

            o_0 = M[i - 2u] ^ ((o_1 ^ o_term7_1) << 1);
            o_term7_0 = M[i + 5u] << 7;

            uint32_t d2 = c_0 & RA_MASK;
            uint32_t d3 = c_1 & RA_MASK;

            uint32_t tmp0 = L[i];
            uint32_t li0 = L[d1];
            L[i] = li0;
            L[d1] = tmp0;
            M[i + 7u] += li0;

            uint32_t tmp1 = L[i - 1u];
            uint32_t li1 = L[d2];
            L[i - 1u] = li1;
            L[d2] = tmp1;
            M[i + 6u] += li1;

            s = s_2;
            d0 = d2;
            d1 = d3;
        }

        uint32_t a_0 = (d0 ^ o_0) ^ s;
        s = cons + a_0;
        uint32_t b_0 = s ^ (o_0 + d0);
        uint32_t c = rot32((a_0 >> 13) ^ a_0, b_0);
        c ^= c >> 17u;
        *out++ = c;

        uint32_t tmp0 = L[1];
        uint32_t li0 = L[d1];
        L[1] = li0;
        L[d1] = tmp0;
        M[8] += li0;

        END_CYCLE_M_W2(M, L);
        *cons_io = cons ^ (c + s + L[0]);
        *out_cursor = out;
        return false;
    } else {
        uint32_t rem = (uint32_t)(*count);
        *count = 1;
        uint32_t i = (uint32_t)RA_MAX_SINGLE;

        while (rem >= 2u) {
            uint32_t o_1 = M[i - 1u] ^ ((o_0 ^ o_term7_0) << 1);
            uint32_t o_term7_1 = M[i + 6u] << 7;

            uint32_t a_0 = (d0 ^ o_0) ^ s;
            uint32_t s_1 = cons + a_0;
            uint32_t a_1 = (d1 ^ o_1) ^ s_1;
            uint32_t s_2 = cons + a_1;

            uint32_t b_0 = s_1 ^ (o_0 + d0);
            uint32_t b_1 = s_2 ^ (o_1 + d1);
            uint32_t u_0 = (a_0 >> 13) ^ a_0;
            uint32_t u_1 = (a_1 >> 13) ^ a_1;
            uint32_t c_0 = rot32(u_0, b_0);
            uint32_t c_1 = rot32(u_1, b_1);
            c_0 ^= c_0 >> 17u;
            c_1 ^= c_1 >> 17u;

            out[0] = c_0;
            out[1] = c_1;
            out += 2;

            rem -= 2u;
            if (rem == 0u) {
                *out_cursor = out;
                return true;
            }

            s = s_2;
            d0 = c_0 & RA_MASK;
            d1 = c_1 & RA_MASK;
            o_0 = M[i - 2u] ^ ((o_1 ^ o_term7_1) << 1);
            o_term7_0 = M[i + 5u] << 7;
            i -= 2u;
        }

        if (rem == 1u) {
            uint32_t a_0 = (d0 ^ o_0) ^ s;
            uint32_t s_1 = cons + a_0;
            uint32_t b_0 = s_1 ^ (o_0 + d0);
            uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);
            *out++ = c_0 ^ (c_0 >> 17u);
        }
        *out_cursor = out;
        return true;
    }
}

static inline void ra_init_orbit_v2(
    uint32_t * restrict L, uint32_t * restrict M,
    uint32_t key, uint32_t addr_hi, uint32_t * restrict k_mix_out) {
    uint32_t step_m, k_mix;
    ra_derive_m_A3(key, addr_hi, &step_m, &k_mix);
    uint32_t step_l = ra_derive_l_A3(key, addr_hi);
    uint32_t acc_m = RA_IDX_OFF * step_m;
    uint32_t acc_l = RA_IDX_OFF * step_l;
    for (int i = 0; i < RA_N; ++i) {
        uint32_t idx = (uint32_t)i;
        uint32_t r = k_mix ^ idx;
        L[i] = rot32(acc_l, r);
        M[i] = rot32(acc_m, r);
        acc_l += step_l;
        acc_m += step_m;
    }
    M[RA_N] = M[1];
    memset(M + RA_N + 1, 0, 6 * sizeof(uint32_t));
    *k_mix_out = k_mix;
}

static inline uint32_t ra_singleblock_v2_buf(
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out_buf) {
    uint32_t step_m, k_mix;
    ra_derive_m_A3(key, addr_hi, &step_m, &k_mix);
    if (rng == 0) return k_mix;
    if (rng == 1) {
        out_buf[0] = ra_eval_k1_word(key, addr_hi);
        return k_mix;
    }
    if (rng == 2) {
        ra_eval_k2_words(key, addr_hi, out_buf);
        return k_mix;
    }
    if (rng <= 8) {
        ra_eval_k3_to_k8_words(key, addr_hi, rng, out_buf);
        return k_mix;
    }
    alignas(64) uint32_t M[RA_N];
    int start = (int)RA_N - (int)rng;
    if (start < 1) start = 1;
    uint32_t acc_m = ((uint32_t)start + RA_IDX_OFF) * step_m;
    for (int i = start; i < RA_N; ++i) {
        uint32_t idx = (uint32_t)i;
        M[i] = rot32(acc_m, k_mix ^ idx);
        acc_m += step_m;
    }
    uint32_t m1 = (start == 1) ? M[1] : rot32((1u + RA_IDX_OFF) * step_m, k_mix ^ 1u);
    ra_singleblock_cycle_v2_buf(k_mix, M, m1, rng, out_buf);
    return k_mix;
}

static inline uint32_t ra_orbit_v2_buf(
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out_buf) {
    if (rng == 0) return key;
    if (rng <= (size_t)RA_MAX_SINGLE) {
        return ra_singleblock_v2_buf(key, addr_hi, rng, out_buf);
    }
    alignas(64) uint32_t L[RA_N];
    alignas(64) uint32_t M[RA_N + 8];
    uint64_t count = rng;
    uint64_t iteration = rng / (size_t)RA_MAX_SINGLE + 1u;
    uint32_t cons = 0u;
    uint32_t *cursor = out_buf;
    ra_init_orbit_v2(L, M, key, addr_hi, &cons);
    for (size_t it = 0; it < iteration; ++it) {
        bool done = ra_orbit_cycle_v2_buf(&cons, M, L, &count, &cursor);
        if (done) return cons;
    }
    return cons;
}

// Public entry points (implemented in ra_core.c / ra_core_engine.c)
uint32_t ra_core_orbit(uint32_t key, uint32_t addr_hi, size_t rng, FILE *raw_stream);
uint32_t ra_core_singleblock(uint32_t key, uint32_t addr_hi, size_t rng, FILE *raw_stream);

#endif // RA_CORE_H
