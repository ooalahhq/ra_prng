// ra_core_engine.c
// Production library & stateful engine layer for ra_core v2 Final:
//   - Key-Mixing & Cycle Core: Opsi_A3 + W2_D2_xag_opt (from ra_core.h / ra_core.c)
//   - Stateful Streaming Engine (`ra_core_engine_t`): `eng_v2_E2` (2026-11-04 Stage 2 Winner:
//     Even-K-Aligned 2-Step Coupled OTF Pair Evaluation with 0 wasted words at K=1, 2,
//     even K <= 254, and K=255, plus automatic Stage-2 promotion at K >= 256).
//   - 8-Way AVX2 SIMD Batch Generator (`ra_core_singleblock_x8_into` / `ra_core_x8`):
//     100% bit-exact lockstep 8-lane vector evaluation (`u32x8`) on Opsi_A3.

#include <stdalign.h>

#define main ra_core_unused_main
#include "ra_core.c"
#undef main

#include "ra_core_engine.h"

// ----------------------------------------------------------------------
// Stateless direct-buffer wrappers & 8-way lockstep AVX2 (`u32x8`)
// ----------------------------------------------------------------------

uint32_t ra_core_eval_k1(uint32_t key, uint32_t addr_hi) {
    return ra_eval_k1_word(key, addr_hi);
}

void ra_core_singleblock_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t *out) {
    if (rng == 0) return;
    if (rng > (size_t)RA_MAX_SINGLE) {
        fprintf(stderr, "ra_core_singleblock_into: rng=%zu > %d out of scope\n", rng, RA_MAX_SINGLE);
        abort();
    }
    (void)ra_singleblock_v2_buf(key, addr_hi, rng, out);
}

void ra_core_orbit_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t *out) {
    if (rng == 0) return;
    (void)ra_orbit_v2_buf(key, addr_hi, rng, out);
}

void ra_core_default_into(uint32_t key, uint32_t addr_hi, uint32_t *out, size_t rng) {
    if (rng == 0) return;
    (void)ra_orbit_v2_buf(key, addr_hi, rng, out);
}

typedef uint32_t u32x8 __attribute__((vector_size(32)));

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

static inline __attribute__((always_inline)) void ra_derive_m_x8_A3(
    u32x8 vkey, u32x8 vaddr, u32x8 *step_m_out, u32x8 *k_mix_out) {
    u32x8 p_addr = vaddr * 0x9e3779b1u;
    u32x8 u_m = (vkey * 0x06a0dd9bu) + p_addr;
    u32x8 v_m = p_addr ^ rot32_x8(vkey, (u32x8){0} + 17u);
    u32x8 km_k = fmix32_x8(((u32x8){0} + GUARD_M)  ^ u_m);
    u32x8 km_p = fmix32_x8(((u32x8){0} + GUARD_M2) ^ v_m);
    *k_mix_out  = km_k ^ rot32_x8(km_p, (u32x8){0} + 13u);
    *step_m_out = ((km_k + km_p) * 0x06a0dd9bu) | 1u;
}

void ra_core_singleblock_x8_into(const uint32_t keys[8], const uint32_t addr_his[8],
                                 size_t rng, uint32_t *out_interleaved) {
    if (rng == 0) return;
    if (rng > (size_t)RA_MAX_SINGLE) abort();

    u32x8 vkey, vaddr;
    memcpy(&vkey, keys, 32);
    memcpy(&vaddr, addr_his, 32);

    u32x8 step_m, vk_mix;
    ra_derive_m_x8_A3(vkey, vaddr, &step_m, &vk_mix);
    uint32_t i0 = (uint32_t)RA_MAX_SINGLE;

    if (rng == 1) {
        u32x8 o = rot32_x8(step_m * (i0 + RA_IDX_OFF), vk_mix ^ i0)
                ^ (rot32_x8(step_m * (1u + RA_IDX_OFF), vk_mix ^ 1u) << 1);
        u32x8 a = o ^ (vk_mix << 1);
        u32x8 b = (vk_mix + a) ^ o;
        u32x8 c = rot32_x8((a >> 13) ^ a, b);
        c ^= c >> 17u;
        memcpy(out_interleaved, &c, 32);
        return;
    }

    if (rng == 2) {
        uint32_t i1 = i0 - 1u;
        u32x8 m_n1 = rot32_x8(step_m * (i0 + RA_IDX_OFF), vk_mix ^ i0);
        u32x8 m1   = rot32_x8(step_m * (1u + RA_IDX_OFF), vk_mix ^ 1u);
        u32x8 m_n2 = rot32_x8(step_m * (i1 + RA_IDX_OFF), vk_mix ^ i1);
        u32x8 o_0 = m_n1 ^ (m1 << 1);
        u32x8 o_1 = m_n2 ^ (o_0 << 1);
        u32x8 d_1 = (rot32_x8(vk_mix, (u32x8){0} + 15u) ^ 0x55u) & (uint32_t)RA_MASK;
        u32x8 a_0 = o_0 ^ (vk_mix << 1);
        u32x8 s_1 = vk_mix + a_0;
        u32x8 a_1 = (d_1 ^ o_1) ^ s_1;
        u32x8 s_2 = vk_mix + a_1;
        u32x8 b_0 = s_1 ^ o_0;
        u32x8 b_1 = s_2 ^ (o_1 + d_1);
        u32x8 c_0 = rot32_x8((a_0 >> 13) ^ a_0, b_0);
        u32x8 c_1 = rot32_x8((a_1 >> 13) ^ a_1, b_1);
        c_0 ^= c_0 >> 17u;
        c_1 ^= c_1 >> 17u;
        memcpy(out_interleaved,     &c_0, 32);
        memcpy(out_interleaved + 8, &c_1, 32);
        return;
    }

    if (rng <= 4) {
        u32x8 m1 = rot32_x8(step_m * (1u + RA_IDX_OFF), vk_mix ^ 1u);
        uint32_t i = i0;
        u32x8 o_0 = rot32_x8(step_m * (i0 + RA_IDX_OFF), vk_mix ^ i0) ^ (m1 << 1);
        u32x8 s = vk_mix << 1;
        u32x8 d0 = {0};
        u32x8 d1 = (rot32_x8(vk_mix, (u32x8){0} + 15u) ^ 0x55u) & (uint32_t)RA_MASK;
        uint32_t rem = (uint32_t)rng;
        uint32_t *out = out_interleaved;
        while (rem >= 2u) {
            uint32_t i1 = i - 1u;
            u32x8 m_im1 = rot32_x8(step_m * (i1 + RA_IDX_OFF), vk_mix ^ i1);
            u32x8 o_1 = m_im1 ^ (o_0 << 1);
            u32x8 a_0 = (d0 ^ o_0) ^ s;
            u32x8 s_1 = vk_mix + a_0;
            u32x8 a_1 = (d1 ^ o_1) ^ s_1;
            u32x8 s_2 = vk_mix + a_1;
            u32x8 b_0 = s_1 ^ (o_0 + d0);
            u32x8 b_1 = s_2 ^ (o_1 + d1);
            u32x8 c_0 = rot32_x8((a_0 >> 13) ^ a_0, b_0);
            u32x8 c_1 = rot32_x8((a_1 >> 13) ^ a_1, b_1);
            c_0 ^= c_0 >> 17u;
            c_1 ^= c_1 >> 17u;
            memcpy(out,     &c_0, 32);
            memcpy(out + 8, &c_1, 32);
            out += 16;
            rem -= 2u;
            if (rem == 0u) return;
            s = s_2;
            d0 = c_0 & (uint32_t)RA_MASK;
            d1 = c_1 & (uint32_t)RA_MASK;
            uint32_t i2 = i - 2u;
            u32x8 m_im2 = rot32_x8(step_m * (i2 + RA_IDX_OFF), vk_mix ^ i2);
            o_0 = m_im2 ^ (o_1 << 1);
            i -= 2u;
        }
        u32x8 a_0 = (d0 ^ o_0) ^ s;
        u32x8 s_1 = vk_mix + a_0;
        u32x8 b_0 = s_1 ^ (o_0 + d0);
        u32x8 c_0 = rot32_x8((a_0 >> 13) ^ a_0, b_0);
        c_0 ^= c_0 >> 17u;
        memcpy(out, &c_0, 32);
        return;
    }

    alignas(32) u32x8 M[RA_N];
    int start = (int)RA_N - (int)rng;
    if (start < 1) start = 1;
    u32x8 acc_m = step_m * ((uint32_t)start + RA_IDX_OFF);
    for (int i = start; i < RA_N; ++i) {
        uint32_t idx = (uint32_t)i;
        M[i] = rot32_x8(acc_m, vk_mix ^ idx);
        acc_m += step_m;
    }
    u32x8 m1 = (start == 1) ? M[1] : rot32_x8(step_m * (1u + RA_IDX_OFF), vk_mix ^ 1u);
    u32x8 cons = vk_mix;
    u32x8 s = cons << 1;
    u32x8 d_use = {0};
    u32x8 d = (rot32_x8(cons, (u32x8){0} + 15u) ^ 0x55u) & (uint32_t)RA_MASK;
    u32x8 o = M[RA_MAX_SINGLE] ^ (m1 << 1);
    uint32_t rem = (uint32_t)rng;
    uint32_t first_steps = (rem <= 7u) ? rem : 7u;
    uint32_t i = i0;
    uint32_t *out = out_interleaved;

    for (;;) {
        u32x8 a = (d_use ^ o) ^ s;
        s = cons + a;
        u32x8 b = s ^ (o + d_use);
        u32x8 c = rot32_x8((a >> 13) ^ a, b);
        c ^= c >> 17u;
        memcpy(out, &c, 32);
        out += 8;
        u32x8 d_next = c & (uint32_t)RA_MASK;
        d_use = d;
        d = d_next;
        if (--first_steps == 0u) break;
        o = M[i - 1u] ^ (o << 1);
        --i;
    }

    rem -= 7u;
    if ((int32_t)rem <= 0) return;

    u32x8 o_term7 = m1 << 7;
    for (;;) {
        u32x8 new_term = M[i - 1u];
        u32x8 o_next = new_term ^ ((o ^ o_term7) << 1);
        o_term7 = M[i + 6u] << 7;
        o = o_next;
        --i;
        u32x8 a = (d_use ^ o) ^ s;
        s = cons + a;
        u32x8 b = s ^ (o + d_use);
        u32x8 c = rot32_x8((a >> 13) ^ a, b);
        c ^= c >> 17u;
        memcpy(out, &c, 32);
        out += 8;
        if (--rem == 0u) break;
        u32x8 d_next = c & (uint32_t)RA_MASK;
        d_use = d;
        d = d_next;
    }
}

void ra_core_x8(const uint32_t keys[8], const uint32_t addr_his[8],
                size_t rng, uint32_t *out_interleaved) {
    ra_core_singleblock_x8_into(keys, addr_his, rng, out_interleaved);
}

// ----------------------------------------------------------------------
// Stateful Orbit Streaming Engine (`eng_v2_E2`: Even-K-Aligned 2-Step Coupled OTF)
// Schedule:
//   - mode 0 (seeded, K=1): Word #1 (`i = 255`), 2 taps (`m255`, `m1`), 0 wasted words
//   - mode 1 (after K=1, K=2): Word #2 (`i = 254`), sets up `m_ring`, 0 wasted words
//   - mode 3 / mode 2 (K=3..254, `i = 253..2`): 2-Step Coupled Pair `(i, i-1)` with
//     running accumulator `acc_m`, returning `c_0` and caching `c_1` in `cached_word`
//     (0 wasted words at all even K: K=4, 8, 16, 32, 64, 128, 254)
//   - mode 3 (`i = 1`, K=255): Word #255 (`i = 1`), 0 wasted words
//   - mode 3 (`i = 0`, K=256) -> mode 4: Stage 1 -> Stage 2 promotion (materializes
//     `L[256]` and `M[256+8]`, replays `d_hist[255..1]` swaps + `M[i+7] += L[d_use]`,
//     runs `END_CYCLE_M_W2`, and fills 255-word `block_buf`)
// ----------------------------------------------------------------------

struct ra_core_engine_s {
    uint32_t key;
    uint32_t addr_hi;
    uint32_t step_m;
    uint32_t acc_m;
    uint32_t s;
    uint32_t d;
    uint32_t o;
    uint32_t o_term7;
    uint32_t i_cur;
    uint32_t cons;
    uint32_t cached_word;
    uint32_t m_ring[8];
    uint8_t  mode;
    uint32_t block_pos;
    uint32_t block_len;
    uint64_t it;
    uint8_t  d_hist[RA_N];
    alignas(64) uint32_t L[RA_N];
    alignas(64) uint32_t M[RA_N + 8];
    uint32_t block_buf[RA_N];
};

size_t ra_core_engine_struct_size(void) { return sizeof(ra_core_engine_t); }

static void ra_core_engine_fill_block(ra_core_engine_t *e) {
    uint64_t cnt = (uint64_t)RA_N;
    uint32_t *cur = e->block_buf;
    (void)ra_orbit_cycle_v2_buf(&e->cons, e->M, e->L, &cnt, &cur);
    ++e->it;
    e->block_len = (uint32_t)RA_MAX_SINGLE;
    e->block_pos = 0u;
}

static __attribute__((noinline)) uint32_t ra_core_engine_promote(ra_core_engine_t *e) {
    uint32_t k_mix = 0u;
    ra_init_orbit_v2(e->L, e->M, e->key, e->addr_hi, &k_mix);
    for (uint32_t i = (uint32_t)RA_MAX_SINGLE; i >= 1u; --i) {
        uint8_t d_use = e->d_hist[i];
        uint32_t tmp = e->L[i];
        uint32_t li_new = e->L[d_use];
        e->L[i] = li_new;
        e->L[d_use] = tmp;
        e->M[i + 7u] += li_new;
    }
    END_CYCLE_M_W2(e->M, e->L);
    e->cons = k_mix ^ (e->cached_word + e->s + e->L[0]);
    e->it = 1ULL;
    e->mode = 4u;
    ra_core_engine_fill_block(e);
    e->block_pos = 1u;
    return e->block_buf[0];
}

static __attribute__((noinline)) uint32_t ra_core_engine_stage2_refill(ra_core_engine_t *e) {
    ra_core_engine_fill_block(e);
    e->block_pos = 1u;
    return e->block_buf[0];
}

void ra_core_engine_init(ra_core_engine_t *e, uint32_t key, uint32_t addr_hi) {
    e->key = key;
    e->addr_hi = addr_hi;
    e->mode = 0u;
}

uint32_t ra_core_engine_next_u32(ra_core_engine_t *e) {
    if (__builtin_expect(e->mode == 0u, 0)) {
        uint32_t step_m, k_mix;
        ra_derive_m_A3(e->key, e->addr_hi, &step_m, &k_mix);
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
    if (__builtin_expect(e->mode == 2u, 1)) {
        e->mode = 3u;
        return e->cached_word;
    }
    if (__builtin_expect(e->mode == 3u, 1)) {
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
            e->mode = 2u;
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
        return ra_core_engine_promote(e);
    }
    if (__builtin_expect(e->mode == 1u, 1)) {
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
        e->acc_m = acc_m - step_m;
        e->i_cur = 253u;
        e->mode  = 3u;
        return c;
    }
    if (__builtin_expect(e->block_pos >= e->block_len, 0)) {
        return ra_core_engine_stage2_refill(e);
    }
    return e->block_buf[e->block_pos++];
}

uint64_t ra_core_engine_next_u64(ra_core_engine_t *e) {
    uint32_t lo = ra_core_engine_next_u32(e);
    uint32_t hi = ra_core_engine_next_u32(e);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

double ra_core_engine_next_double(ra_core_engine_t *e) {
    return (ra_core_engine_next_u64(e) >> 11) * (1.0 / 9007199254740992.0);
}

void ra_core_engine_next(ra_core_engine_t *e, size_t n, uint32_t *out) {
    if (n == 0) return;
    if (e->mode == 0u && n > (size_t)RA_MAX_SINGLE) {
        ra_init_orbit_v2(e->L, e->M, e->key, e->addr_hi, &e->cons);
        e->it = 0ULL;
        e->block_len = 0u;
        e->block_pos = 0u;
        e->mode = 4u;
    }
    size_t written = 0;
    while (e->mode != 4u && written < n) {
        out[written++] = ra_core_engine_next_u32(e);
    }
    while (written < n) {
        if (e->block_pos >= e->block_len) ra_core_engine_fill_block(e);
        size_t avail = e->block_len - e->block_pos;
        size_t take = (n - written < avail) ? (n - written) : avail;
        memcpy(out + written, e->block_buf + e->block_pos, take * sizeof(uint32_t));
        e->block_pos += (uint32_t)take;
        written += take;
    }
}

void ra_core_engine_discard(ra_core_engine_t *e, uint64_t z) {
    if (z == 0ULL) return;
    if (e->mode == 0u && z > (uint64_t)RA_MAX_SINGLE) {
        ra_init_orbit_v2(e->L, e->M, e->key, e->addr_hi, &e->cons);
        e->it = 0ULL;
        e->block_len = 0u;
        e->block_pos = 0u;
        e->mode = 4u;
    }
    while (e->mode != 4u && z > 0ULL) {
        (void)ra_core_engine_next_u32(e);
        --z;
    }
    while (z > 0ULL) {
        if (e->block_pos >= e->block_len) ra_core_engine_fill_block(e);
        uint64_t avail = (uint64_t)(e->block_len - e->block_pos);
        uint64_t take = (z < avail) ? z : avail;
        e->block_pos += (uint32_t)take;
        z -= take;
    }
}

uint64_t ra_core_engine_position(const ra_core_engine_t *e) {
    if (e->mode == 0u) return 0ULL;
    if (e->mode == 1u) return 1ULL;
    if (e->mode == 2u) return (uint64_t)(254u - e->i_cur);
    if (e->mode == 3u) return (uint64_t)(255u - e->i_cur);
    return e->it * 255ULL - (uint64_t)(e->block_len - e->block_pos);
}

uint32_t ra_core_engine_key(const ra_core_engine_t *e) { return e->key; }
uint32_t ra_core_engine_addr_hi(const ra_core_engine_t *e) { return e->addr_hi; }

void ra_core_engine_fork(const ra_core_engine_t *src, ra_core_engine_t *child) {
    *child = *src;
    if (child->mode == 0u) {
        ra_init_orbit_v2(child->L, child->M, child->key, child->addr_hi, &child->cons);
        child->it = 0ULL;
    } else if (child->mode < 4u) {
        while (!(child->mode == 3u && child->i_cur == 0u)) {
            (void)ra_core_engine_next_u32(child);
        }
        uint32_t k_mix = 0u;
        ra_init_orbit_v2(child->L, child->M, child->key, child->addr_hi, &k_mix);
        for (uint32_t i = (uint32_t)RA_MAX_SINGLE; i >= 1u; --i) {
            uint8_t d_use = child->d_hist[i];
            uint32_t tmp = child->L[i];
            uint32_t li_new = child->L[d_use];
            child->L[i] = li_new;
            child->L[d_use] = tmp;
            child->M[i + 7u] += li_new;
        }
        END_CYCLE_M_W2(child->M, child->L);
        child->cons = k_mix ^ (child->cached_word + child->s + child->L[0]);
        child->it = 1ULL;
    }
    END_CYCLE_M_W2(child->M, child->L);
    child->cons = fmix32(child->cons ^ child->L[0] ^ child->M[0] ^ 0x9E3779B9u);
    child->block_len = 0u;
    child->block_pos = 0u;
    child->mode = 4u;
}

void ra_core_engine_save(const ra_core_engine_t *e, uint8_t *out) {
    memcpy(out, e, sizeof(ra_core_engine_t));
}

void ra_core_engine_load(ra_core_engine_t *e, const uint8_t *in) {
    memcpy(e, in, sizeof(ra_core_engine_t));
}

// Convenience short-name aliases (`ra_engine_*`)
void ra_engine_seed(ra_core_engine_t *e, uint32_t key) {
    ra_core_engine_init(e, key, 0u);
}

void ra_engine_seed_ex(ra_core_engine_t *e, uint32_t key, uint32_t addr_hi) {
    ra_core_engine_init(e, key, addr_hi);
}

uint32_t ra_engine_next_u32(ra_core_engine_t *e) {
    return ra_core_engine_next_u32(e);
}

uint64_t ra_engine_next_u64(ra_core_engine_t *e) {
    return ra_core_engine_next_u64(e);
}

double ra_engine_next_f64(ra_core_engine_t *e) {
    return ra_core_engine_next_double(e);
}

void ra_engine_fill_u32(ra_core_engine_t *e, uint32_t *out, size_t n) {
    ra_core_engine_next(e, n, out);
}

void ra_engine_discard(ra_core_engine_t *e, uint64_t z) {
    ra_core_engine_discard(e, z);
}

// ----------------------------------------------------------------------
// Deterministic Fisher-Yates shuffle (permutes indices [0, n) in place).
// ----------------------------------------------------------------------

#define RA_CORE_SHUFFLE_BLOCK_K 255u

void ra_core_shuffle_indices(uint32_t key, size_t n, uint32_t *out_indices) {
    for (size_t i = 0; i < n; ++i) out_indices[i] = (uint32_t)i;
    if (n < 2) return;

    uint32_t block[RA_CORE_SHUFFLE_BLOCK_K];
    uint32_t block_len = 0, block_pos = 0;
    uint64_t block_idx = 0;

    for (size_t i = n - 1; i > 0; --i) {
        if (block_pos >= block_len) {
            size_t remaining_steps = i;
            block_len = (uint32_t)((remaining_steps < RA_CORE_SHUFFLE_BLOCK_K)
                                        ? remaining_steps
                                        : RA_CORE_SHUFFLE_BLOCK_K);
            uint32_t block_key = key + (uint32_t)(block_idx * 0x9E3779B9u);
            ra_core_singleblock_into(block_key, /*addr_hi=*/0u, block_len, block);
            block_pos = 0;
            ++block_idx;
        }
        uint32_t word = block[block_pos++];
        size_t range = i + 1;
        size_t j = (size_t)(((uint64_t)word * (uint64_t)range) >> 32);
        uint32_t tmp = out_indices[i];
        out_indices[i] = out_indices[j];
        out_indices[j] = tmp;
    }
}
