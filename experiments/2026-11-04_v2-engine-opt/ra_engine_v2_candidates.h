// ra_engine_v2_candidates.h
// Stage 2 Engine Optimization Candidates on top of Official Winner `Opsi_A3`:
//   0. `eng_v1_baseline`: Production `ra_engine_v1` (3-Stage OTF W0_D0) from Shootout
//   1. `eng_v2_old`:      Previous `ra_engine_v2` on `Opsi_A3` (with duplicated i>=250 branch)
//   2. `eng_v2_E1`:       Clean Single-Step + Zero Inner Branch + Running Accumulator `acc_m`
//   3. `eng_v2_E2`:       2-Step Coupled OTF Pair Evaluation (Lag-2 `D2` Unrolled Stage 1)
//   4. `eng_v2_E3`:       Unified Fast-Buffer Front-End + Exponential Doubling Micro-Burst Stage 1

#ifndef RA_ENGINE_V2_CANDIDATES_H
#define RA_ENGINE_V2_CANDIDATES_H

#include "../2026-11-04_v2-prng-shootout-and-mc/baselines_shootout.h"

// ============================================================================
// Official Stage 1 Winner `Opsi_A3` Primitives & Orbit Reference
// (From `../2026-11-04_v2-keymix-optA/ra_core_v2_keymix_candidates.h`)
// ============================================================================

#define IDX_OFF_A3 1u

static inline __attribute__((always_inline)) void derive_m_A3(
    uint32_t key, uint32_t addr_hi, uint32_t *step_m_out, uint32_t *k_mix_out) {
    uint32_t p_addr = 0x9e3779b1u * addr_hi;
    uint32_t u_m = (0x06a0dd9bu * key) + p_addr;
    uint32_t v_m = p_addr ^ rot32(key, 17);
    uint32_t km_k = fmix32(GUARD_M  ^ u_m);
    uint32_t km_p = fmix32(GUARD_M2 ^ v_m);
    *k_mix_out  = km_k ^ rot32(km_p, 13);
    *step_m_out = (0x06a0dd9bu * (km_k + km_p)) | 1u;
}

static inline __attribute__((always_inline)) uint32_t derive_l_A3(
    uint32_t key, uint32_t addr_hi) {
    uint32_t p_addr = 0x85ebca77u * addr_hi;
    uint32_t u_l = (0x9e3779b7u * key) + p_addr;
    uint32_t v_l = p_addr ^ rot32(key, 15);
    uint32_t kl_k = fmix32(GUARD_L  ^ u_l);
    uint32_t kl_p = fmix32(GUARD_L2 ^ v_l);
    return (0x9e3779b7u * (kl_k + kl_p)) | 1u;
}

static inline __attribute__((always_inline)) void singleblock_cycle_W2_D2_xag_opt_common(
    uint32_t cons, const uint32_t * restrict M, uint32_t m1, size_t rng, uint32_t * restrict out_buf) {
    uint32_t s = cons << 1;
    uint32_t d0 = 0u;
    uint32_t d1 = (rot32(cons, 15) ^ 0x55u) & 0xFFu;
    uint32_t o_0 = M[255] ^ (m1 << 1);
    uint32_t i = 255u;

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
        d0 = c_0 & 0xFFu;
        d1 = c_1 & 0xFFu;
        o_0 = M[i - 2u] ^ (o_1 << 1);
        i -= 2u;
    }

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
        d0 = c_0 & 0xFFu;
        d1 = c_1 & 0xFFu;
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

static inline bool v2_cycle_W2_D2_xag_opt_common(
    uint32_t * restrict cons_io, uint32_t * restrict M, uint32_t * restrict L,
    uint64_t *count, uint32_t **out_cursor) {
    uint32_t cons = *cons_io;
    uint32_t s = cons << 1;
    uint32_t d0 = 0u;
    uint32_t d1 = (rot32(cons, 15) ^ 0x55u) & 0xFFu;
    uint32_t o_0 = M[255] ^ (M[256] << 1);
    uint32_t o_term7_0 = 0u;
    uint32_t * restrict out = *out_cursor;

    if (*count > 255ULL) {
        *count -= 255ULL;
        uint32_t i = 255u;

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

            uint32_t d2 = c_0 & 0xFFu;
            uint32_t d3 = c_1 & 0xFFu;

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
        uint32_t i = 255u;

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
            d0 = c_0 & 0xFFu;
            d1 = c_1 & 0xFFu;
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

static inline void v2_init_orbit_A3(
    uint32_t * restrict L, uint32_t * restrict M,
    uint32_t key, uint32_t addr_hi, uint32_t * restrict k_mix_out) {
    uint32_t step_m, k_mix;
    derive_m_A3(key, addr_hi, &step_m, &k_mix);
    uint32_t step_l = derive_l_A3(key, addr_hi);
    uint32_t acc_m = IDX_OFF_A3 * step_m;
    uint32_t acc_l = IDX_OFF_A3 * step_l;
    for (int i = 0; i < 256; ++i) {
        uint32_t idx = (uint32_t)i;
        uint32_t r = k_mix ^ idx;
        L[i] = rot32(acc_l, r);
        M[i] = rot32(acc_m, r);
        acc_l += step_l;
        acc_m += step_m;
    }
    M[256] = M[1];
    memset(M + 257, 0, 6 * sizeof(uint32_t));
    *k_mix_out = k_mix;
}

static inline uint32_t ra_singleblock_opt_A3(
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out_buf) {
    uint32_t step_m, k_mix;
    derive_m_A3(key, addr_hi, &step_m, &k_mix);
    if (rng == 0) return k_mix;
    if (rng == 1) {
        uint32_t o = rot32(256u * step_m, k_mix ^ 255u) ^ (rot32(2u * step_m, k_mix ^ 1u) << 1);
        uint32_t a = o ^ (k_mix << 1);
        uint32_t b = (k_mix + a) ^ o;
        uint32_t c = rot32((a >> 13) ^ a, b);
        out_buf[0] = c ^ (c >> 17u);
        return key;
    }
    if (rng <= 8) {
        uint32_t m1 = rot32(2u * step_m, k_mix ^ 1u);
        uint32_t i = 255u;
        uint32_t o_0 = rot32(256u * step_m, k_mix ^ 255u) ^ (m1 << 1);
        uint32_t s = k_mix << 1;
        uint32_t d0 = 0u;
        uint32_t d1 = (rot32(k_mix, 15) ^ 0x55u) & 0xFFu;
        uint32_t rem = (uint32_t)rng;
        while (rem >= 2u) {
            uint32_t i1 = i - 1u;
            uint32_t m_im1 = rot32((i1 + 1u) * step_m, k_mix ^ i1);
            uint32_t o_1 = (i == 249u) ? (m_im1 ^ ((o_0 ^ (m1 << 7)) << 1)) : (m_im1 ^ (o_0 << 1));
            uint32_t a_0 = (d0 ^ o_0) ^ s;
            uint32_t s_1 = k_mix + a_0;
            uint32_t a_1 = (d1 ^ o_1) ^ s_1;
            uint32_t s_2 = k_mix + a_1;
            uint32_t b_0 = s_1 ^ (o_0 + d0);
            uint32_t b_1 = s_2 ^ (o_1 + d1);
            uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);
            uint32_t c_1 = rot32((a_1 >> 13) ^ a_1, b_1);
            c_0 ^= c_0 >> 17u;
            c_1 ^= c_1 >> 17u;
            out_buf[0] = c_0;
            out_buf[1] = c_1;
            out_buf += 2;
            rem -= 2u;
            if (rem == 0u) return key;
            s = s_2;
            d0 = c_0 & 0xFFu;
            d1 = c_1 & 0xFFu;
            uint32_t i2 = i - 2u;
            uint32_t m_im2 = rot32((i2 + 1u) * step_m, k_mix ^ i2);
            o_0 = m_im2 ^ (o_1 << 1);
            i -= 2u;
        }
        uint32_t a_0 = (d0 ^ o_0) ^ s;
        uint32_t s_1 = k_mix + a_0;
        uint32_t b_0 = s_1 ^ (o_0 + d0);
        uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);
        out_buf[0] = c_0 ^ (c_0 >> 17u);
        return key;
    }
    alignas(64) uint32_t M[256];
    int start = 256 - (int)rng;
    if (start < 1) start = 1;
    uint32_t acc_m = ((uint32_t)start + 1u) * step_m;
    for (int i = start; i < 256; ++i) {
        uint32_t idx = (uint32_t)i;
        M[i] = rot32(acc_m, k_mix ^ idx);
        acc_m += step_m;
    }
    uint32_t m1 = (start == 1) ? M[1] : rot32(2u * step_m, k_mix ^ 1u);
    singleblock_cycle_W2_D2_xag_opt_common(k_mix, M, m1, rng, out_buf);
    return k_mix;
}

static __attribute__((noinline, noclone)) uint32_t ra_orbit_opt_A3(
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out_buf) {
    if (rng == 0) return key;
    if (rng <= 255u) {
        return ra_singleblock_opt_A3(key, addr_hi, rng, out_buf);
    }
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256 + 8];
    uint64_t count = rng;
    uint64_t iteration = rng / 255u + 1u;
    uint32_t cons = 0u;
    uint32_t *cursor = out_buf;
    v2_init_orbit_A3(L, M, key, addr_hi, &cons);
    for (size_t it = 0; it < iteration; ++it) {
        bool done = v2_cycle_W2_D2_xag_opt_common(&cons, M, L, &count, &cursor);
        if (done) return cons;
    }
    return cons;
}

typedef struct {
    uint32_t key, addr_hi, step_m, s, d, o, o_term7, i_cur;
    uint32_t m_ring[8];
    uint8_t  mode;
    uint32_t block_pos, block_len, cons;
    uint8_t  d_hist[256];
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256 + 8];
    uint32_t block_buf[256];
} engine_v2_keymix_t;

static __attribute__((noinline)) uint32_t engine_promote_A3(engine_v2_keymix_t *e) {
    uint32_t k_mix = 0u;
    v2_init_orbit_A3(e->L, e->M, e->key, e->addr_hi, &k_mix);
    for (uint32_t i = 255u; i >= 1u; --i) {
        uint8_t d_use = e->d_hist[i];
        uint32_t tmp = e->L[i];
        uint32_t li_new = e->L[d_use];
        e->L[i] = li_new;
        e->L[d_use] = tmp;
        e->M[i + 7u] += li_new;
    }
    END_CYCLE_M_W2(e->M, e->L);
    uint32_t a_last = e->s - k_mix;
    uint32_t b_last = e->s ^ (e->o + e->d_hist[2]);
    uint32_t c_last = rot32((a_last >> 13) ^ a_last, b_last);
    c_last ^= c_last >> 17u;
    e->cons = k_mix ^ (c_last + e->s + e->L[0]);
    e->mode = 2;
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &cnt, &cur);
    e->block_len = 255;
    e->block_pos = 1;
    return e->block_buf[0];
}

// ============================================================================
// 1. `eng_v2_old`: Control (`ra_engine_v2` with old branching structure on `Opsi_A3`)
// ============================================================================

typedef engine_v2_keymix_t ra_engine_v2_old_t;

static inline __attribute__((always_inline)) void ra_engine_v2_old_init(
    ra_engine_v2_old_t *e, uint32_t key, uint32_t addr_hi) {
    e->key = key;
    e->addr_hi = addr_hi;
    e->mode = 0;
}

static inline __attribute__((always_inline)) uint32_t ra_engine_v2_old_next_u32(
    ra_engine_v2_old_t *e) {
    if (__builtin_expect(e->mode == 0, 0)) {
        uint32_t step_m, k_mix;
        derive_m_A3(e->key, e->addr_hi, &step_m, &k_mix);
        uint32_t i0 = 255u;
        e->step_m = step_m;
        e->cons   = k_mix;
        uint32_t m_n1 = rot32((i0 + IDX_OFF_A3) * step_m, k_mix ^ i0);
        uint32_t m1   = rot32((1u + IDX_OFF_A3) * step_m, k_mix ^ 1u);
        uint32_t o = m_n1 ^ (m1 << 1);
        uint32_t a = o ^ (k_mix << 1);
        uint32_t s = k_mix + a;
        uint32_t b = s ^ o;
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;
        uint32_t d_init = (rot32(k_mix, 15) ^ 0x55u) & 0xFFu;
        e->d_hist[i0] = (uint8_t)d_init;
        e->d = d_init | ((c & 0xFFu) << 8);
        e->s = s;
        e->o = o;
        e->o_term7 = m_n1;
        e->i_cur = 254u;
        e->mode = 1;
        return c;
    }
    if (__builtin_expect(e->mode == 1, 1)) {
        uint32_t i = e->i_cur;
        if (__builtin_expect(i >= 250u, 1)) {
            uint32_t k_mix = e->cons;
            uint32_t o_prev = e->o;
            if (__builtin_expect(i == 254u, 0)) {
                uint32_t m_n1 = e->o_term7;
                e->m_ring[0] = (o_prev ^ m_n1) >> 1;
                e->m_ring[7] = m_n1;
                e->o_term7 = 0u;
            }
            uint32_t new_term = rot32((i + IDX_OFF_A3) * e->step_m, k_mix ^ i);
            uint32_t o = new_term ^ (o_prev << 1);
            e->m_ring[i & 7u] = new_term;
            uint32_t d_pair = e->d;
            uint32_t d_use = d_pair & 0xFFu;
            uint32_t d_prev = d_pair >> 8;
            uint32_t a = (d_use ^ o) ^ e->s;
            uint32_t s = k_mix + a;
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
        if (__builtin_expect(i > 0u, 1)) {
            uint32_t k_mix = e->cons;
            uint32_t new_term = rot32((i + IDX_OFF_A3) * e->step_m, k_mix ^ i);
            uint32_t o = new_term ^ ((e->o ^ e->o_term7) << 1);
            uint32_t next_term7 = e->m_ring[(i + 7u) & 7u] << 7;
            e->m_ring[i & 7u] = new_term;
            uint32_t d_pair = e->d;
            uint32_t d_use = d_pair & 0xFFu;
            uint32_t d_prev = d_pair >> 8;
            uint32_t a = (d_use ^ o) ^ e->s;
            uint32_t s = k_mix + a;
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
        return engine_promote_A3(e);
    }
    if (e->block_pos >= e->block_len) {
        uint64_t cnt = 256;
        uint32_t *cur = e->block_buf;
        v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &cnt, &cur);
        e->block_len = 255;
        e->block_pos = 0;
    }
    return e->block_buf[e->block_pos++];
}

// ============================================================================
// 2. `eng_v2_E1`: Clean Single-Step + Zero Inner Branch + Running `acc_m`
// ============================================================================

typedef struct {
    uint32_t key, addr_hi, step_m, acc_m, s, d, o, o_term7, i_cur, cons;
    uint32_t m_ring[8];
    uint8_t  mode;
    uint32_t block_pos, block_len;
    uint8_t  d_hist[256];
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256 + 8];
    uint32_t block_buf[256];
} ra_engine_v2_E1_t;

static inline __attribute__((always_inline)) void ra_engine_v2_E1_init(
    ra_engine_v2_E1_t *e, uint32_t key, uint32_t addr_hi) {
    e->key = key;
    e->addr_hi = addr_hi;
    e->mode = 0;
}

static __attribute__((noinline)) uint32_t ra_engine_v2_E1_promote(ra_engine_v2_E1_t *e) {
    uint32_t k_mix = 0u;
    v2_init_orbit_A3(e->L, e->M, e->key, e->addr_hi, &k_mix);
    for (uint32_t i = 255u; i >= 1u; --i) {
        uint8_t d_use = e->d_hist[i];
        uint32_t tmp = e->L[i];
        uint32_t li_new = e->L[d_use];
        e->L[i] = li_new;
        e->L[d_use] = tmp;
        e->M[i + 7u] += li_new;
    }
    END_CYCLE_M_W2(e->M, e->L);
    uint32_t a_last = e->s - k_mix;
    uint32_t b_last = e->s ^ (e->o + e->d_hist[2]);
    uint32_t c_last = rot32((a_last >> 13) ^ a_last, b_last);
    c_last ^= c_last >> 17u;
    e->cons = k_mix ^ (c_last + e->s + e->L[0]);
    e->mode = 3;
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &cnt, &cur);
    e->block_len = 255;
    e->block_pos = 1;
    return e->block_buf[0];
}

static __attribute__((noinline)) uint32_t ra_engine_v2_E1_stage2_refill(ra_engine_v2_E1_t *e) {
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &cnt, &cur);
    e->block_len = 255;
    e->block_pos = 1;
    return e->block_buf[0];
}

static inline __attribute__((always_inline)) uint32_t ra_engine_v2_E1_next_u32(
    ra_engine_v2_E1_t *e) {
    if (__builtin_expect(e->mode == 0, 0)) {
        uint32_t step_m, k_mix;
        derive_m_A3(e->key, e->addr_hi, &step_m, &k_mix);
        uint32_t m255 = rot32(256u * step_m, k_mix ^ 255u);
        uint32_t m1   = rot32(2u   * step_m, k_mix ^ 1u);
        uint32_t o = m255 ^ (m1 << 1);
        uint32_t a = o ^ (k_mix << 1);
        uint32_t s = k_mix + a;
        uint32_t b = s ^ o;
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;
        e->step_m  = step_m;
        e->cons    = k_mix;
        e->s       = s;
        e->o       = o;
        e->o_term7 = m255;
        e->d       = c & 0xFFu;
        e->mode    = 1;
        return c;
    }
    if (__builtin_expect(e->mode == 2, 1)) {
        uint32_t i = e->i_cur;
        if (__builtin_expect(i > 0u, 1)) {
            uint32_t k_mix = e->cons;
            uint32_t acc_m = e->acc_m;
            uint32_t new_term = rot32(acc_m, k_mix ^ i);
            uint32_t o = new_term ^ ((e->o ^ e->o_term7) << 1);
            uint32_t next_term7 = e->m_ring[(i + 7u) & 7u] << 7;
            e->m_ring[i & 7u] = new_term;

            uint32_t d_pair = e->d;
            uint32_t d_use  = d_pair & 0xFFu;
            uint32_t d_prev = d_pair >> 8;
            uint32_t a = (d_use ^ o) ^ e->s;
            uint32_t s = k_mix + a;
            uint32_t b = s ^ (o + d_use);
            uint32_t c = rot32((a >> 13) ^ a, b);
            c ^= c >> 17u;

            e->d_hist[i] = (uint8_t)d_prev;
            e->d = d_prev | ((c & 0xFFu) << 8);
            e->s = s;
            e->o = o;
            e->o_term7 = next_term7;
            e->acc_m = acc_m - e->step_m;
            e->i_cur = i - 1u;
            return c;
        }
        return ra_engine_v2_E1_promote(e);
    }
    if (__builtin_expect(e->mode == 1, 1)) {
        uint32_t k_mix  = e->cons;
        uint32_t step_m = e->step_m;
        uint32_t o_prev = e->o;
        uint32_t m255   = e->o_term7;
        uint32_t m1     = (o_prev ^ m255) >> 1;

        e->m_ring[0] = m1;
        memset(&e->m_ring[1], 0, 5 * sizeof(uint32_t));
        e->m_ring[7] = m255;

        uint32_t acc_m = 255u * step_m;
        uint32_t new_term = rot32(acc_m, k_mix ^ 254u);
        uint32_t o = new_term ^ (o_prev << 1);
        e->m_ring[6] = new_term;

        uint32_t d_use  = (rot32(k_mix, 15) ^ 0x55u) & 0xFFu;
        uint32_t d_prev = e->d;
        e->d_hist[255] = (uint8_t)d_use;
        e->d_hist[254] = (uint8_t)d_prev;

        uint32_t a = (d_use ^ o) ^ e->s;
        uint32_t s = k_mix + a;
        uint32_t b = s ^ (o + d_use);
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;

        e->d = d_prev | ((c & 0xFFu) << 8);
        e->s = s;
        e->o = o;
        e->o_term7 = 0u;
        e->acc_m = acc_m - step_m;
        e->i_cur = 253u;
        e->mode  = 2;
        return c;
    }
    if (__builtin_expect(e->block_pos >= e->block_len, 0)) {
        return ra_engine_v2_E1_stage2_refill(e);
    }
    return e->block_buf[e->block_pos++];
}

// ============================================================================
// 3. `eng_v2_E2`: Even-K-Aligned 2-Step Coupled OTF Pair Evaluation (Lag-2 `D2`)
// ============================================================================
// Schedule:
//   - K = 1   (i = 255, mode 0): 1 word  (#1) -> 0 wasted words at K=1
//   - K = 2   (i = 254, mode 1): 1 word  (#2) -> 0 wasted words at K=2
//   - K = 3..254 (i = 253..2, mode 2/3): 2-Step Coupled Pair (i, i-1) -> 0 wasted words at K=4,8,16,64!
//   - K = 255 (i = 1, mode 3):   1 word  (#255) -> 0 wasted words at K=255
//   - K >= 256 (i = 0, mode 4):  Stage 2 full-orbit block_buf

typedef struct {
    uint32_t key, addr_hi, step_m, acc_m, s, d, o, o_term7, i_cur, cons, cached_word;
    uint32_t m_ring[8];
    uint8_t  mode;
    uint32_t block_pos, block_len;
    uint8_t  d_hist[256];
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256 + 8];
    uint32_t block_buf[256];
} ra_engine_v2_E2_t;

static inline __attribute__((always_inline)) void ra_engine_v2_E2_init(
    ra_engine_v2_E2_t *e, uint32_t key, uint32_t addr_hi) {
    e->key = key;
    e->addr_hi = addr_hi;
    e->mode = 0;
}

static __attribute__((noinline)) uint32_t ra_engine_v2_E2_promote(ra_engine_v2_E2_t *e) {
    uint32_t k_mix = 0u;
    v2_init_orbit_A3(e->L, e->M, e->key, e->addr_hi, &k_mix);
    for (uint32_t i = 255u; i >= 1u; --i) {
        uint8_t d_use = e->d_hist[i];
        uint32_t tmp = e->L[i];
        uint32_t li_new = e->L[d_use];
        e->L[i] = li_new;
        e->L[d_use] = tmp;
        e->M[i + 7u] += li_new;
    }
    END_CYCLE_M_W2(e->M, e->L);
    e->cons = k_mix ^ (e->cached_word + e->s + e->L[0]);
    e->mode = 4;
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &cnt, &cur);
    e->block_len = 255;
    e->block_pos = 1;
    return e->block_buf[0];
}

static __attribute__((noinline)) uint32_t ra_engine_v2_E2_stage2_refill(ra_engine_v2_E2_t *e) {
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &cnt, &cur);
    e->block_len = 255;
    e->block_pos = 1;
    return e->block_buf[0];
}

static inline __attribute__((always_inline)) uint32_t ra_engine_v2_E2_next_u32(
    ra_engine_v2_E2_t *e) {
    if (__builtin_expect(e->mode == 0, 0)) {
        uint32_t step_m, k_mix;
        derive_m_A3(e->key, e->addr_hi, &step_m, &k_mix);
        uint32_t m255 = rot32(256u * step_m, k_mix ^ 255u);
        uint32_t m1   = rot32(2u   * step_m, k_mix ^ 1u);
        uint32_t o = m255 ^ (m1 << 1);
        uint32_t a = o ^ (k_mix << 1);
        uint32_t s = k_mix + a;
        uint32_t b = s ^ o;
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;
        e->step_m  = step_m;
        e->cons    = k_mix;
        e->s       = s;
        e->o       = o;
        e->o_term7 = m255;
        e->d       = c & 0xFFu;
        e->mode    = 1;
        return c;
    }
    if (__builtin_expect(e->mode == 2, 1)) {
        e->mode = 3;
        return e->cached_word;
    }
    if (__builtin_expect(e->mode == 3, 1)) {
        uint32_t i = e->i_cur;
        if (__builtin_expect(i >= 3u, 1)) {
            uint32_t k_mix  = e->cons;
            uint32_t step_m = e->step_m;
            uint32_t acc_m  = e->acc_m;
            uint32_t i1     = i - 1u;

            uint32_t m_0 = rot32(acc_m,          k_mix ^ i);
            uint32_t m_1 = rot32(acc_m - step_m, k_mix ^ i1);

            uint32_t t7_0 = e->o_term7;
            uint32_t t7_1 = e->m_ring[i1 & 7u] << 7;
            uint32_t t7_2 = e->m_ring[(i - 2u) & 7u] << 7;

            uint32_t o_0 = m_0 ^ ((e->o ^ t7_0) << 1);
            uint32_t o_1 = m_1 ^ ((o_0  ^ t7_1) << 1);

            e->m_ring[i  & 7u] = m_0;
            e->m_ring[i1 & 7u] = m_1;

            uint32_t d_pair = e->d;
            uint32_t d0 = d_pair & 0xFFu;
            uint32_t d1 = d_pair >> 8;

            uint32_t a_0 = (d0 ^ o_0) ^ e->s;
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

            e->d_hist[i]  = (uint8_t)d1;
            e->d_hist[i1] = (uint8_t)(c_0 & 0xFFu);
            e->d = (c_0 & 0xFFu) | ((c_1 & 0xFFu) << 8);
            e->s = s_2;
            e->o = o_1;
            e->o_term7 = t7_2;
            e->acc_m = acc_m - (step_m << 1);
            e->i_cur = i - 2u;
            e->cached_word = c_1;
            e->mode = 2;
            return c_0;
        }
        if (i == 1u) {
            uint32_t k_mix = e->cons;
            uint32_t m_0   = rot32(e->acc_m, k_mix ^ 1u);
            uint32_t o_0   = m_0 ^ ((e->o ^ e->o_term7) << 1);
            uint32_t d_pair = e->d;
            uint32_t d0 = d_pair & 0xFFu;
            uint32_t d1 = d_pair >> 8;
            uint32_t a_0 = (d0 ^ o_0) ^ e->s;
            uint32_t s_1 = k_mix + a_0;
            uint32_t b_0 = s_1 ^ (o_0 + d0);
            uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);
            c_0 ^= c_0 >> 17u;
            e->d_hist[1] = (uint8_t)d1;
            e->s = s_1;
            e->o = o_0;
            e->cached_word = c_0;
            e->i_cur = 0u;
            return c_0;
        }
        return ra_engine_v2_E2_promote(e);
    }
    if (__builtin_expect(e->mode == 1, 1)) {
        // K = 2 (Word #2, i = 254): single-word evaluation, 0 wasted words!
        uint32_t k_mix  = e->cons;
        uint32_t step_m = e->step_m;
        uint32_t o_prev = e->o;
        uint32_t m255   = e->o_term7;
        uint32_t m1     = (o_prev ^ m255) >> 1;

        e->m_ring[0] = m1;
        e->m_ring[1] = 0u;
        e->m_ring[2] = 0u;
        e->m_ring[3] = 0u;
        e->m_ring[4] = 0u;
        e->m_ring[7] = m255;

        uint32_t acc_m = 255u * step_m;
        uint32_t m254  = rot32(acc_m, k_mix ^ 254u);
        uint32_t o     = m254 ^ (o_prev << 1);
        e->m_ring[6]   = m254;

        uint32_t d_use  = (rot32(k_mix, 15) ^ 0x55u) & 0xFFu;
        uint32_t d_prev = e->d;
        e->d_hist[255] = (uint8_t)d_use;
        e->d_hist[254] = (uint8_t)d_prev;

        uint32_t a = (d_use ^ o) ^ e->s;
        uint32_t s = k_mix + a;
        uint32_t b = s ^ (o + d_use);
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;

        e->d = d_prev | ((c & 0xFFu) << 8);
        e->s = s;
        e->o = o;
        e->o_term7 = 0u;
        e->acc_m = acc_m - step_m; // 254 * step_m, ready for i = 253
        e->i_cur = 253u;
        e->mode  = 3;
        return c;
    }
    if (__builtin_expect(e->block_pos >= e->block_len, 0)) {
        return ra_engine_v2_E2_stage2_refill(e);
    }
    return e->block_buf[e->block_pos++];
}

// ============================================================================
// 4. `eng_v2_E3`: Even-K-Aligned Fast-Dispatch + 4-Pair Micro-Burst (K >= 9)
// ============================================================================
// Combines the exact 3-store init & scalar `cached_word` fast-path of `E2` for K <= 8
// with register-resident 4-Pair (8-word) Micro-Bursts for K >= 9 and Stage 2 block buffer!

typedef struct {
    uint32_t key, addr_hi, step_m, acc_m, s, d, o, o_term7, i_cur, cons, cached_word;
    uint32_t m_ring[8];
    uint8_t  mode;
    uint32_t block_pos, block_len;
    uint8_t  d_hist[256];
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256 + 8];
    uint32_t block_buf[256];
} ra_engine_v2_E3_t;

static inline __attribute__((always_inline)) void ra_engine_v2_E3_init(
    ra_engine_v2_E3_t *e, uint32_t key, uint32_t addr_hi) {
    e->key = key;
    e->addr_hi = addr_hi;
    e->mode = 0u;
}

static __attribute__((noinline)) uint32_t ra_engine_v2_E3_promote(ra_engine_v2_E3_t *e) {
    uint32_t k_mix = 0u;
    v2_init_orbit_A3(e->L, e->M, e->key, e->addr_hi, &k_mix);
    for (uint32_t i = 255u; i >= 1u; --i) {
        uint8_t d_use = e->d_hist[i];
        uint32_t tmp = e->L[i];
        uint32_t li_new = e->L[d_use];
        e->L[i] = li_new;
        e->L[d_use] = tmp;
        e->M[i + 7u] += li_new;
    }
    END_CYCLE_M_W2(e->M, e->L);
    uint32_t c_last = e->block_buf[e->block_len - 1u];
    e->cons = k_mix ^ (c_last + e->s + e->L[0]);
    e->mode = 5u;
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &cnt, &cur);
    e->block_len = 255u;
    e->block_pos = 1u;
    return e->block_buf[0];
}

static __attribute__((noinline)) uint32_t ra_engine_v2_E3_burst_refill(ra_engine_v2_E3_t *e) {
    if (__builtin_expect(e->mode == 4u, 1)) {
        uint32_t i = e->i_cur;
        if (__builtin_expect(i >= 8u, 1)) {
            uint32_t k_mix  = e->cons;
            uint32_t step_m = e->step_m;
            uint32_t step2  = step_m << 1;
            uint32_t acc_m  = e->acc_m;
            uint32_t s      = e->s;
            uint32_t o_prev = e->o;
            uint32_t t7_0   = e->o_term7;
            uint32_t d_pair = e->d;
            uint32_t d0     = d_pair & 0xFFu;
            uint32_t d1     = d_pair >> 8;
            uint32_t * restrict out = e->block_buf;

            for (int p = 0; p < 4; ++p) {
                uint32_t i1  = i - 1u;
                uint32_t m_0 = rot32(acc_m,          k_mix ^ i);
                uint32_t m_1 = rot32(acc_m - step_m, k_mix ^ i1);

                uint32_t t7_1 = e->m_ring[i1 & 7u] << 7;
                uint32_t t7_2 = e->m_ring[(i - 2u) & 7u] << 7;

                uint32_t o_0 = m_0 ^ ((o_prev ^ t7_0) << 1);
                uint32_t o_1 = m_1 ^ ((o_0    ^ t7_1) << 1);

                e->m_ring[i  & 7u] = m_0;
                e->m_ring[i1 & 7u] = m_1;

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

                e->d_hist[i]  = (uint8_t)d1;
                e->d_hist[i1] = (uint8_t)(c_0 & 0xFFu);

                out[0] = c_0;
                out[1] = c_1;
                out += 2;

                s      = s_2;
                o_prev = o_1;
                t7_0   = t7_2;
                d0     = c_0 & 0xFFu;
                d1     = c_1 & 0xFFu;
                acc_m -= step2;
                i     -= 2u;
            }

            e->s         = s;
            e->o         = o_prev;
            e->o_term7   = t7_0;
            e->d         = d0 | (d1 << 8);
            e->acc_m     = acc_m;
            e->i_cur     = i;
            e->block_pos = 1u;
            e->block_len = 8u;
            return e->block_buf[0];
        }
        if (i == 7u) {
            uint32_t k_mix  = e->cons;
            uint32_t step_m = e->step_m;
            uint32_t step2  = step_m << 1;
            uint32_t acc_m  = e->acc_m;
            uint32_t s      = e->s;
            uint32_t o_prev = e->o;
            uint32_t t7_0   = e->o_term7;
            uint32_t d_pair = e->d;
            uint32_t d0     = d_pair & 0xFFu;
            uint32_t d1     = d_pair >> 8;
            uint32_t * restrict out = e->block_buf;

            for (int p = 0; p < 3; ++p) {
                uint32_t i1  = i - 1u;
                uint32_t m_0 = rot32(acc_m,          k_mix ^ i);
                uint32_t m_1 = rot32(acc_m - step_m, k_mix ^ i1);

                uint32_t t7_1 = e->m_ring[i1 & 7u] << 7;
                uint32_t t7_2 = e->m_ring[(i - 2u) & 7u] << 7;

                uint32_t o_0 = m_0 ^ ((o_prev ^ t7_0) << 1);
                uint32_t o_1 = m_1 ^ ((o_0    ^ t7_1) << 1);

                e->m_ring[i  & 7u] = m_0;
                e->m_ring[i1 & 7u] = m_1;

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

                e->d_hist[i]  = (uint8_t)d1;
                e->d_hist[i1] = (uint8_t)(c_0 & 0xFFu);

                out[0] = c_0;
                out[1] = c_1;
                out += 2;

                s      = s_2;
                o_prev = o_1;
                t7_0   = t7_2;
                d0     = c_0 & 0xFFu;
                d1     = c_1 & 0xFFu;
                acc_m -= step2;
                i     -= 2u;
            }
            uint32_t m_0 = rot32(acc_m, k_mix ^ 1u);
            uint32_t o_0 = m_0 ^ ((o_prev ^ t7_0) << 1);
            uint32_t a_0 = (d0 ^ o_0) ^ s;
            uint32_t s_1 = k_mix + a_0;
            uint32_t b_0 = s_1 ^ (o_0 + d0);
            uint32_t c_0 = rot32((a_0 >> 13) ^ a_0, b_0);
            c_0 ^= c_0 >> 17u;
            e->d_hist[1] = (uint8_t)d1;
            out[0]       = c_0;

            e->s         = s_1;
            e->o         = o_0;
            e->i_cur     = 0u;
            e->block_pos = 1u;
            e->block_len = 7u;
            return e->block_buf[0];
        }
        return ra_engine_v2_E3_promote(e);
    }
    uint64_t cnt = 256;
    uint32_t *cur = e->block_buf;
    v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &cnt, &cur);
    e->block_len = 255u;
    e->block_pos = 1u;
    return e->block_buf[0];
}

static inline __attribute__((always_inline)) uint32_t ra_engine_v2_E3_next_u32(
    ra_engine_v2_E3_t *e) {
    uint8_t mode = e->mode;
    if (__builtin_expect(mode == 0u, 0)) {
        uint32_t step_m, k_mix;
        derive_m_A3(e->key, e->addr_hi, &step_m, &k_mix);
        uint32_t m255 = rot32(256u * step_m, k_mix ^ 255u);
        uint32_t m1   = rot32(2u   * step_m, k_mix ^ 1u);
        uint32_t o = m255 ^ (m1 << 1);
        uint32_t a = o ^ (k_mix << 1);
        uint32_t s = k_mix + a;
        uint32_t b = s ^ o;
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;
        e->step_m  = step_m;
        e->cons    = k_mix;
        e->s       = s;
        e->o       = o;
        e->o_term7 = m255;
        e->d       = c & 0xFFu;
        e->mode    = 1u;
        return c;
    }
    if (__builtin_expect(mode == 2u, 1)) {
        e->mode = (e->i_cur == 247u) ? 4u : 3u;
        return e->cached_word;
    }
    if (__builtin_expect(mode == 3u, 1)) {
        // Inlined 1-pair evaluation for i in {253, 251, 249} (K = 3..8, zero memset!)
        uint32_t i      = e->i_cur;
        uint32_t k_mix  = e->cons;
        uint32_t step_m = e->step_m;
        uint32_t acc_m  = e->acc_m;
        uint32_t i1     = i - 1u;

        uint32_t m_0 = rot32(acc_m,          k_mix ^ i);
        uint32_t m_1 = rot32(acc_m - step_m, k_mix ^ i1);

        uint32_t o_0 = m_0 ^ (e->o << 1);
        uint32_t t7_1 = (i == 249u) ? (e->m_ring[0] << 7) : 0u;
        uint32_t o_1 = m_1 ^ ((o_0 ^ t7_1) << 1);

        e->m_ring[i  & 7u] = m_0;
        e->m_ring[i1 & 7u] = m_1;

        uint32_t d_pair = e->d;
        uint32_t d0 = d_pair & 0xFFu;
        uint32_t d1 = d_pair >> 8;

        uint32_t a_0 = (d0 ^ o_0) ^ e->s;
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

        e->d_hist[i]  = (uint8_t)d1;
        e->d_hist[i1] = (uint8_t)(c_0 & 0xFFu);
        e->d = (c_0 & 0xFFu) | ((c_1 & 0xFFu) << 8);
        e->s = s_2;
        e->o = o_1;
        e->acc_m = acc_m - (step_m << 1);
        e->i_cur = i - 2u;
        e->cached_word = c_1;
        e->mode = 2u;
        if (i == 249u) {
            e->o_term7   = e->m_ring[7] << 7;
            e->block_pos = 0u;
            e->block_len = 0u;
        }
        return c_0;
    }
    if (__builtin_expect(mode == 1u, 1)) {
        // K = 2 (Word #2, i = 254): single-word evaluation, 0 wasted words, 0 memset!
        uint32_t k_mix  = e->cons;
        uint32_t step_m = e->step_m;
        uint32_t o_prev = e->o;
        uint32_t m255   = e->o_term7;
        uint32_t m1     = (o_prev ^ m255) >> 1;

        e->m_ring[0] = m1;
        e->m_ring[7] = m255;

        uint32_t acc_m = 255u * step_m;
        uint32_t m254  = rot32(acc_m, k_mix ^ 254u);
        uint32_t o     = m254 ^ (o_prev << 1);
        e->m_ring[6]   = m254;

        uint32_t d_use  = (rot32(k_mix, 15) ^ 0x55u) & 0xFFu;
        uint32_t d_prev = e->d;
        e->d_hist[255] = (uint8_t)d_use;
        e->d_hist[254] = (uint8_t)d_prev;

        uint32_t a = (d_use ^ o) ^ e->s;
        uint32_t s = k_mix + a;
        uint32_t b = s ^ (o + d_use);
        uint32_t c = rot32((a >> 13) ^ a, b);
        c ^= c >> 17u;

        e->d = d_prev | ((c & 0xFFu) << 8);
        e->s = s;
        e->o = o;
        e->acc_m = acc_m - step_m; // 254 * step_m, ready for i = 253
        e->i_cur = 253u;
        e->mode  = 3u;
        return c;
    }
    uint32_t pos = e->block_pos;
    if (__builtin_expect(pos < e->block_len, 1)) {
        e->block_pos = pos + 1u;
        return e->block_buf[pos];
    }
    return ra_engine_v2_E3_burst_refill(e);
}

#endif // RA_ENGINE_V2_CANDIDATES_H
