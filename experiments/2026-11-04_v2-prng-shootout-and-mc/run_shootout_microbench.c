// run_shootout_microbench.c
// Unified PRNG Shootout Microbenchmark Suite (`ra_core v2` vs `ra_core v1` vs All Competitors)
//
// Measures five standard metrics simultaneously via `clock_gettime(CLOCK_MONOTONIC)` + `__rdtscp()`:
//   1. `cpb` (`cycles/byte`)
//   2. `cycles/word` (CPU cycles per 32-bit word)
//   3. `ns/word` (nanoseconds per 32-bit word)
//   4. `ns/64-bit` (nanoseconds per 64-bit output, Vigna Shootout table convention)
//   5. `MiB/s` (2^20 B/s) & `GB/s` (10^9 B/s)
//
// Across FOUR benchmark suites:
//   Suite A — Vigna Shootout Inlined Register-Accumulator Tight Loop (Continuous Steady-State vs Per-Seed K=1 & K=255)
//   Suite B — Materialized Direct-Buffer Scalar Throughput across K in {1, 2, 4, 8, 15, 31, 63, 127, 255, 256, 510, 10000, stream}
//   Suite C — Apples-to-Apples 8-Way AVX2 SIMD (`u32x8`) Throughput across K in {1, 2, 4, 8, 15, 31, 63, 127, 255} and N in {256..16}
//   Suite D — Stateful 3-Stage OTF Engine (`next_u32()` API) across K in {1, 2, 4, 8, 16, 64, 255, 256, 1000}
//
// Outputs both formatted Markdown tables to stdout and structured JSON to `data/shootout_microbench.json`.

#include "baselines_shootout.h"
#include <sys/stat.h>

typedef struct {
    double mibs;
    double gbs;
    double ns_word;
    double ns_64b;
    double cyc_word;
    double cpb;
} metric_t;

static inline metric_t make_metric(double sec, double cyc, double total_words) {
    metric_t m;
    double total_bytes = total_words * 4.0;
    m.mibs     = (total_bytes / (1024.0 * 1024.0)) / sec;
    m.gbs      = (total_bytes / 1e9) / sec;
    m.ns_word  = (sec * 1e9) / total_words;
    m.ns_64b   = m.ns_word * 2.0;
    m.cyc_word = cyc / total_words;
    m.cpb      = cyc / total_bytes;
    return m;
}

static void record_sample(FILE *out, const char *suite, const char *regime,
                          const char *generator, int round, double sec,
                          double cycles, double bytes) {
    fprintf(out, "%s,%s,%s,%d,%.12g,%.12g,%.0f\n",
            suite, regime, generator, round, sec, cycles, bytes);
}

// ============================================================================
// SUITE A: Vigna PRNG Shootout Tight Inlined Register-Accumulator Mode
// ============================================================================
// Measures pure ALU/register recurrence speed (`acc ^= next()`) without array store overhead,
// both in continuous steady-state (10,000,000 words) and per-seed addressed K=1 (2,000,000 draws).

typedef struct {
    const char *name;
    const char *category;
    int state_bytes;
    metric_t inlined_stream;
    metric_t inlined_k1;
    metric_t buffer_stream;
    metric_t buffer_k1;
} suite_a_row_t;

static __attribute__((noinline, noclone)) void bench_inlined_stream_ra_v2(double *sec_out, double *cyc_out, size_t n_words) {
    alignas(64) uint32_t L[256], M[256 + 8];
    v2_init_orbit_W2(L, M, 0xC0FFEEu, 0x12345678u);
    uint32_t cons = 0xC0FFEEu;
    uint32_t acc = 0;
    size_t rem_total = n_words;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    while (rem_total > 0) {
        uint32_t s = cons << 1;
        uint32_t d0 = 0u;
        uint32_t d1 = (rot32(cons, 15) ^ 0x55u) & 0xFFu;
        uint32_t o_0 = M[255] ^ (M[256] << 1);
        uint32_t o_term7_0 = 0u;
        for (uint32_t i = 255u; i >= 3u; i -= 2u) {
            uint32_t o_1 = M[i - 1u] ^ ((o_0 ^ o_term7_0) << 1);
            uint32_t o_term7_1 = M[i + 6u] << 7;
            uint32_t a_0 = (d0 ^ o_0) ^ s;
            uint32_t s_1 = cons + a_0;
            uint32_t a_1 = (d1 ^ o_1) ^ s_1;
            uint32_t s_2 = cons + a_1;
            uint32_t b_0 = s_1 ^ (o_0 + d0);
            uint32_t b_1 = s_2 ^ (o_1 + d1);
            uint32_t w0 = rot32((a_0 >> 13) ^ a_0, b_0);
            uint32_t w1 = rot32((a_1 >> 13) ^ a_1, b_1);
            w0 ^= w0 >> 17u;
            w1 ^= w1 >> 17u;
            acc ^= w0 ^ w1;
            o_0 = M[i - 2u] ^ ((o_1 ^ o_term7_1) << 1);
            o_term7_0 = M[i + 5u] << 7;
            uint32_t d2 = w0 & 0xFFu;
            uint32_t d3 = w1 & 0xFFu;
            uint32_t tmp0 = L[i], li0 = L[d1]; L[i] = li0; L[d1] = tmp0; M[i + 7u] += li0;
            uint32_t tmp1 = L[i - 1u], li1 = L[d2]; L[i - 1u] = li1; L[d2] = tmp1; M[i + 6u] += li1;
            s = s_2; d0 = d2; d1 = d3;
        }
        uint32_t a_0 = (d0 ^ o_0) ^ s;
        s = cons + a_0;
        uint32_t b_0 = s ^ (o_0 + d0);
        uint32_t w_last = rot32((a_0 >> 13) ^ a_0, b_0);
        w_last ^= w_last >> 17u;
        acc ^= w_last;
        uint32_t tmp0 = L[1], li0 = L[d1]; L[1] = li0; L[d1] = tmp0; M[8] += li0;
        END_CYCLE_M_W2(M, L);
        cons = cons ^ (w_last + s + L[0]);
        rem_total = (rem_total >= 255) ? (rem_total - 255) : 0;
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint32_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone)) void bench_inlined_stream_ra_v1(double *sec_out, double *cyc_out, size_t n_words) {
    alignas(64) uint32_t L[256], M[256];
    v1_init_state_orbit(L, M, 0xC0FFEEu, 0x12345678u);
    uint32_t cons = 0xC0FFEEu;
    uint32_t acc = 0;
    size_t rem_total = n_words;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    while (rem_total > 0) {
        uint32_t s = cons + cons;
        uint32_t d = 0;
        uint32_t o = (M[255] << 0) ^ (M[0] << 1) ^ (M[1] << 2) ^ (M[2] << 3) ^
                     (M[3] << 4)   ^ (M[4] << 5) ^ (M[5] << 6) ^ (M[6] << 7);
        uint32_t o_term7 = M[6] << 7;
        for (uint32_t i = 255; i > 0; --i) {
            uint32_t a = (d ^ o) ^ s;
            s = cons + a;
            uint32_t b = s ^ (o + d);
            uint32_t c = rot32((a >> 13) ^ a, b);
            c ^= c >> 17u;
            acc ^= c;
            d = c & 0xFFu;
            uint32_t tmp = L[i]; L[i] = L[d]; L[d] = tmp;
            if (i > 1) {
                uint32_t new_term = M[(i - 1) & 0xFFu];
                uint32_t o_next = new_term ^ ((o ^ o_term7) << 1);
                o_term7 = M[(i + 6) & 0xFFu] << 7;
                o = o_next;
            }
        }
        cons = v1_reseed(M, L);
        rem_total = (rem_total >= 255) ? (rem_total - 255) : 0;
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint32_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_philox(double *sec_out, double *cyc_out, size_t n_words) {
    uint32_t ctr[4] = {0, 0, 0, 0}, key[2] = {0xC0FFEEu, 0xFEEDC0DEu}, out[4];
    uint32_t acc = 0;
    size_t blocks = n_words / 4;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t b = 0; b < blocks; ++b) {
        philox4x32_10(out, ctr, key);
        ++ctr[0];
        asm volatile("" : "+m"(ctr[0]));
        acc ^= out[0] ^ out[1] ^ out[2] ^ out[3];
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint32_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_chacha8(double *sec_out, double *cyc_out, size_t n_words) {
    chacha_stream_t s;
    chacha_stream_init(&s, 0xC0FFEEu, 0x12345678u, 8);
    uint32_t out[16];
    uint32_t acc = 0;
    size_t blocks = n_words / 16;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t b = 0; b < blocks; ++b) {
        chacha_block(out, s.state, 8);
        ++s.state[12];
        asm volatile("" : "+m"(s.state[12]));
        acc ^= out[0]  ^ out[1]  ^ out[2]  ^ out[3]
             ^ out[4]  ^ out[5]  ^ out[6]  ^ out[7]
             ^ out[8]  ^ out[9]  ^ out[10] ^ out[11]
             ^ out[12] ^ out[13] ^ out[14] ^ out[15];
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint32_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_chacha12(double *sec_out, double *cyc_out, size_t n_words) {
    chacha_stream_t s;
    chacha_stream_init(&s, 0xC0FFEEu, 0x12345678u, 12);
    uint32_t out[16];
    uint32_t acc = 0;
    size_t blocks = n_words / 16;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t b = 0; b < blocks; ++b) {
        chacha_block(out, s.state, 12);
        ++s.state[12];
        asm volatile("" : "+m"(s.state[12]));
        acc ^= out[0]  ^ out[1]  ^ out[2]  ^ out[3]
             ^ out[4]  ^ out[5]  ^ out[6]  ^ out[7]
             ^ out[8]  ^ out[9]  ^ out[10] ^ out[11]
             ^ out[12] ^ out[13] ^ out[14] ^ out[15];
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint32_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_xoshiro256ss(double *sec_out, double *cyc_out, size_t n_words) {
    xoshiro256ss_state_t st;
    xoshiro256ss_init(&st, 0xC0FFEEu, 0x12345678u);
    uint64_t acc = 0;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t i = 0; i < n_words; ++i) {
        acc ^= xoshiro256ss_next_u64(&st);
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint64_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_xoroshiro128pp(double *sec_out, double *cyc_out, size_t n_words) {
    xoroshiro128pp_state_t st;
    xoroshiro128pp_init(&st, 0xC0FFEEu, 0x12345678u);
    uint64_t acc = 0;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t i = 0; i < n_words; ++i) {
        acc ^= xoroshiro128pp_next_u64(&st);
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint64_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_pcg32(double *sec_out, double *cyc_out, size_t n_words) {
    pcg32_random_t r;
    pcg32_srandom_r(&r, 0x12345678C0FFEEULL, 54u);
    uint32_t acc = 0;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t i = 0; i < n_words; ++i) {
        acc ^= pcg32_random_r(&r);
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint32_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_splitmix64(double *sec_out, double *cyc_out, size_t n_words) {
    uint64_t sm = 0x12345678C0FFEEULL;
    uint64_t acc = 0;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t i = 0; i < n_words; ++i) {
        asm volatile("" : "+r"(sm));
        acc ^= splitmix64_step(&sm);
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint64_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_wyrand(double *sec_out, double *cyc_out, size_t n_words) {
    uint64_t seed = 0x12345678C0FFEEULL;
    uint64_t acc = 0;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t i = 0; i < n_words; ++i) {
        asm volatile("" : "+r"(seed));
        acc ^= wyrand_step(&seed);
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint64_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void bench_inlined_stream_mt19937(double *sec_out, double *cyc_out, size_t n_words) {
    mt19937_state_t mt;
    mt19937_init_genrand(&mt, 5489UL);
    uint32_t acc = 0;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (size_t i = 0; i < n_words; ++i) {
        acc ^= mt19937_genrand_int32(&mt);
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint32_t sink = acc; (void)sink;
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
}

// Per-seed Inlined K=1 (Random-Access / Per-Particle 1st Draw)
#define DEFINE_INLINED_K1_BENCH(FNAME, EXPR)                                                 \
static __attribute__((noinline, noclone, optimize("no-tree-vectorize"))) void FNAME(double *sec_out, double *cyc_out, size_t calls) { \
    uint32_t acc = 0;                                                                        \
    double t0 = shootout_now_sec();                                                          \
    uint64_t c0 = shootout_rdtscp();                                                         \
    for (size_t c = 0; c < calls; ++c) {                                                     \
        uint32_t key = (uint32_t)(c * 2654435761u + 1u);                                     \
        uint32_t addr_hi = 0x20261104u;                                                      \
        asm volatile("" : "+r"(key), "+r"(addr_hi));                                         \
        acc ^= (uint32_t)(EXPR);                                                             \
    }                                                                                        \
    uint64_t c1 = shootout_rdtscp();                                                         \
    double t1 = shootout_now_sec();                                                          \
    volatile uint32_t sink = acc; (void)sink;                                                \
    *sec_out = t1 - t0;                                                                      \
    *cyc_out = (double)(c1 - c0);                                                            \
}

DEFINE_INLINED_K1_BENCH(bench_inlined_k1_ra_v2, ra_eval_k1_W2(key, addr_hi))
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_ra_v1, ra_eval_k1_W0(key, addr_hi))

static inline __attribute__((always_inline)) uint32_t philox_k1_inline(uint32_t key, uint32_t addr_hi) {
    uint32_t ctr[4] = {0, 0, 0, 0}, pkey[2] = {key, addr_hi ^ 0xFEEDC0DEu}, out[4];
    philox4x32_10(out, ctr, pkey);
    return out[0];
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_philox, philox_k1_inline(key, addr_hi))

static inline __attribute__((always_inline)) uint32_t chacha8_k1_inline(uint32_t key, uint32_t addr_hi) {
    chacha_stream_t s;
    chacha_stream_init(&s, key, addr_hi, 8);
    return chacha_stream_next_u32(&s);
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_chacha8, chacha8_k1_inline(key, addr_hi))

static inline __attribute__((always_inline)) uint32_t chacha12_k1_inline(uint32_t key, uint32_t addr_hi) {
    chacha_stream_t s;
    chacha_stream_init(&s, key, addr_hi, 12);
    return chacha_stream_next_u32(&s);
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_chacha12, chacha12_k1_inline(key, addr_hi))

static inline __attribute__((always_inline)) uint32_t xoshiro256ss_k1_inline(uint32_t key, uint32_t addr_hi) {
    xoshiro256ss_state_t st;
    xoshiro256ss_init(&st, key, addr_hi);
    return xoshiro256ss_next_u32(&st);
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_xoshiro256ss, xoshiro256ss_k1_inline(key, addr_hi))

static inline __attribute__((always_inline)) uint32_t xoroshiro128pp_k1_inline(uint32_t key, uint32_t addr_hi) {
    xoroshiro128pp_state_t st;
    xoroshiro128pp_init(&st, key, addr_hi);
    return xoroshiro128pp_next_u32(&st);
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_xoroshiro128pp, xoroshiro128pp_k1_inline(key, addr_hi))

static inline __attribute__((always_inline)) uint32_t pcg32_k1_inline(uint32_t key, uint32_t addr_hi) {
    pcg32_random_t r;
    pcg32_srandom_r(&r, ((uint64_t)addr_hi << 32) | key, 54u);
    return pcg32_random_r(&r);
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_pcg32, pcg32_k1_inline(key, addr_hi))

static inline __attribute__((always_inline)) uint32_t splitmix64_k1_inline(uint32_t key, uint32_t addr_hi) {
    uint64_t sm = ((uint64_t)addr_hi << 32) | key;
    return (uint32_t)splitmix64_step(&sm);
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_splitmix64, splitmix64_k1_inline(key, addr_hi))

static inline __attribute__((always_inline)) uint32_t wyrand_k1_inline(uint32_t key, uint32_t addr_hi) {
    uint64_t seed = ((uint64_t)addr_hi << 32) | key;
    return (uint32_t)wyrand_step(&seed);
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_wyrand, wyrand_k1_inline(key, addr_hi))

static inline uint32_t mt19937_k1_inline(uint32_t key, uint32_t addr_hi) {
    mt19937_state_t mt;
    mt19937_init_genrand(&mt, key ^ addr_hi);
    return mt19937_genrand_int32(&mt);
}
DEFINE_INLINED_K1_BENCH(bench_inlined_k1_mt19937, mt19937_k1_inline(key, addr_hi))

// ============================================================================
// SUITE B: Materialized Direct-Buffer Scalar Throughput across K and N
// ============================================================================
typedef void (*buf_gen_fn_t)(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out);

static void wrap_ra_v2_N256_orbit(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    ra_core_v2_W2_D2_xag_opt_orbit_buf(k, a, rng, out);
}
static void wrap_ra_v1_N256_orbit(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    ra_core_v1_orbit_buf(k, a, rng, out);
}
static void wrap_ra_v2_N128_sb(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    ra_v2_N128_singleblock_into(k, a, rng, out);
}
static void wrap_ra_v2_N64_sb(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    ra_v2_N64_singleblock_into(k, a, rng, out);
}
static void wrap_ra_v2_N32_sb(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    ra_v2_N32_singleblock_into(k, a, rng, out);
}
static void wrap_ra_v2_N16_sb(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    ra_v2_N16_singleblock_into(k, a, rng, out);
}
static void wrap_philox_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    philox_into(k, a, rng, out);
}
static void wrap_chacha8_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    chacha8_into(k, a, rng, out);
}
static void wrap_chacha12_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    chacha12_into(k, a, rng, out);
}
static void wrap_xoshiro256ss_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    xoshiro256ss_into(k, a, rng, out);
}
static void wrap_xoroshiro128pp_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    xoroshiro128pp_into(k, a, rng, out);
}
static void wrap_pcg32_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    pcg32_into(k, a, rng, out);
}
static void wrap_splitmix64_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    splitmix64_into(k, a, rng, out);
}
static void wrap_wyrand_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    wyrand_into(k, a, rng, out);
}
static void wrap_mt19937_buf(uint32_t k, uint32_t a, size_t rng, uint32_t * restrict out) {
    mt19937_into(k, a, rng, out);
}

typedef struct {
    const char *name;
    const char *family;
    int max_k_single; // 0 if unlimited
    int state_bytes;
    buf_gen_fn_t fn;
} scalar_gen_desc_t;

#define N_SCALAR_GENS 15
static const scalar_gen_desc_t SCALAR_GENS[N_SCALAR_GENS] = {
    { "ra_core_v2 (N=256)", "ra_core_v2", 0,   2048, wrap_ra_v2_N256_orbit    },
    { "ra_core_v2 (N=128)", "ra_core_v2", 127, 1024, wrap_ra_v2_N128_sb       },
    { "ra_core_v2 (N=64)",  "ra_core_v2", 63,  512,  wrap_ra_v2_N64_sb        },
    { "ra_core_v2 (N=32)",  "ra_core_v2", 31,  256,  wrap_ra_v2_N32_sb        },
    { "ra_core_v2 (N=16)",  "ra_core_v2", 15,  128,  wrap_ra_v2_N16_sb        },
    { "ra_core_v1 (N=256)", "ra_core_v1", 0,   2048, wrap_ra_v1_N256_orbit    },
    { "Philox4x32-10",      "CBRNG",      0,   24,   wrap_philox_buf          },
    { "ChaCha8",            "CBRNG/ARX",  0,   64,   wrap_chacha8_buf         },
    { "ChaCha12",           "CBRNG/ARX",  0,   64,   wrap_chacha12_buf        },
    { "xoshiro256**",       "Linear",     0,   32,   wrap_xoshiro256ss_buf    },
    { "xoroshiro128++",     "Linear",     0,   16,   wrap_xoroshiro128pp_buf  },
    { "PCG32",              "LCG/Perm",   0,   16,   wrap_pcg32_buf           },
    { "SplitMix64",         "Weyl/Mix",   0,   8,    wrap_splitmix64_buf      },
    { "WyRand",             "Weyl/MUL128",0,   8,    wrap_wyrand_buf          },
    { "MT19937",            "Twister",    0,   2500, wrap_mt19937_buf         },
};

#define N_SCALAR_REGIMES 13
static const size_t SCALAR_K_VALS[N_SCALAR_REGIMES] = {
    1, 2, 4, 8, 15, 31, 63, 127, 255, 256, 510, 10000, 10000000
};
static const char *SCALAR_K_LABELS[N_SCALAR_REGIMES] = {
    "K=1", "K=2", "K=4", "K=8", "K=15", "K=31", "K=63", "K=127", "K=255", "K=256", "K=510", "K=10000", "stream"
};
static const long long SCALAR_CALLS[N_SCALAR_REGIMES] = {
    3000000LL, 2500000LL, 2000000LL, 1500000LL, 1000000LL, 700000LL,
    450000LL, 300000LL, 200000LL, 160000LL, 120000LL, 800LL, 1LL
};

// ============================================================================
// SUITE C: 8-Way AVX2 SIMD (`u32x8`) Throughput across K and N
// ============================================================================
typedef void (*avx2_gen_fn_t)(const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out);

typedef struct {
    const char *name;
    const char *family;
    int max_k_single;
    avx2_gen_fn_t fn;
} avx2_gen_desc_t;

#define N_AVX2_GENS 12
static const avx2_gen_desc_t AVX2_GENS[N_AVX2_GENS] = {
    { "ra_core_v2_x8 (N=256)", "ra_core_v2", 255, ra_v2_N256_singleblock_x8_into },
    { "ra_core_v2_x8 (N=128)", "ra_core_v2", 127, ra_v2_N128_singleblock_x8_into },
    { "ra_core_v2_x8 (N=64)",  "ra_core_v2", 63,  ra_v2_N64_singleblock_x8_into  },
    { "ra_core_v2_x8 (N=32)",  "ra_core_v2", 31,  ra_v2_N32_singleblock_x8_into  },
    { "ra_core_v2_x8 (N=16)",  "ra_core_v2", 15,  ra_v2_N16_singleblock_x8_into  },
    { "ra_core_v1_x8 (N=256)", "ra_core_v1", 255, ra_v1_N256_singleblock_x8_into },
    { "Philox4x32-10_x8",      "CBRNG",      0,   philox4x32_x8_into             },
    { "ChaCha8_x8",            "CBRNG/ARX",  0,   chacha8_x8_into                },
    { "xoshiro256**_x8",       "Linear",     0,   xoshiro256ss_x8_into           },
    { "xoroshiro128++_x8",     "Linear",     0,   xoroshiro128pp_x8_into         },
    { "PCG32_x8",              "LCG/Perm",   0,   pcg32_x8_into                  },
    { "SplitMix64_x8",         "Weyl/Mix",   0,   splitmix64_x8_into             },
};

#define N_AVX2_REGIMES 9
static const size_t AVX2_K_VALS[N_AVX2_REGIMES] = { 1, 2, 4, 8, 15, 31, 63, 127, 255 };
static const char *AVX2_K_LABELS[N_AVX2_REGIMES] = {
    "K=1", "K=2", "K=4", "K=8", "K=15", "K=31", "K=63", "K=127", "K=255"
};
static const long long AVX2_CALLS[N_AVX2_REGIMES] = {
    720000LL, 600000LL, 480000LL, 360000LL, 240000LL,
    160000LL, 100000LL, 72000LL, 48000LL
};

int main(void) {
    if (!verify_shootout_all_kat()) {
        fprintf(stderr, "FATAL: KAT verification failed!\n");
        return 1;
    }
    printf("[KAT GATE] All 10 PRNG families + AVX2 x8 + 3-Stage OTF Engine bit-exact parity: PASS\n\n");

    const int ROUNDS = 7;
    FILE *samples = fopen("data/shootout_samples.csv", "w");
    if (!samples) {
        perror("data/shootout_samples.csv");
        return 1;
    }
    fprintf(samples, "suite,regime,generator,round,elapsed_s,cycles,output_bytes\n");

    // ------------------------------------------------------------------------
    // Run Suite B first so we also populate Buffer K=1 & Stream for Suite A
    // ------------------------------------------------------------------------
    metric_t scalar_mat[N_SCALAR_GENS][N_SCALAR_REGIMES];
    bool scalar_valid[N_SCALAR_GENS][N_SCALAR_REGIMES];
    memset(scalar_mat, 0, sizeof(scalar_mat));
    memset(scalar_valid, 0, sizeof(scalar_valid));

    for (int ri = 0; ri < N_SCALAR_REGIMES; ++ri) {
        size_t k = SCALAR_K_VALS[ri];
        long long calls = SCALAR_CALLS[ri];
        uint32_t *buf = (uint32_t *)aligned_alloc(64, (k + 16) * sizeof(uint32_t));
        double best_sec[N_SCALAR_GENS];
        double best_cyc[N_SCALAR_GENS];
        for (int g = 0; g < N_SCALAR_GENS; ++g) { best_sec[g] = 1e18; best_cyc[g] = 1e18; }

        volatile uint32_t sink = 0;
        for (int r = 0; r <= ROUNDS; ++r) {
            for (int idx = 0; idx < N_SCALAR_GENS; ++idx) {
                int g = (r + idx) % N_SCALAR_GENS;
                if (SCALAR_GENS[g].max_k_single > 0 && k > (size_t)SCALAR_GENS[g].max_k_single) {
                    continue;
                }
                // For MT19937 at small K with many calls, reduce calls to keep runtime reasonable while accurate
                long long eff_calls = calls;
                if (g == 14 && k <= 63 && eff_calls > 15000LL) eff_calls = 15000LL;

                buf_gen_fn_t fn = SCALAR_GENS[g].fn;
                uint32_t acc = 0;
                double t0 = shootout_now_sec();
                uint64_t c0 = shootout_rdtscp();
                if (calls == 1LL) {
                    fn(0xC0FFEEu + (uint32_t)r, 0x12345678u, k, buf);
                    asm volatile("" : : "g"(buf) : "memory");
                    acc ^= buf[0] ^ buf[k / 2] ^ buf[k - 1];
                } else {
                    for (long long c = 0; c < eff_calls; ++c) {
                        uint32_t key = (uint32_t)(c * 2654435761u + 1u);
                        fn(key, 0u, k, buf);
                        asm volatile("" : : "g"(buf) : "memory");
                        acc += buf[0] ^ (buf[k - 1] << 1);
                    }
                }
                uint64_t c1 = shootout_rdtscp();
                double t1 = shootout_now_sec();
                sink ^= acc;
                if (r == 0) continue; // untimed-for-summary warmup, same work as a measured round
                record_sample(samples, "B", SCALAR_K_LABELS[ri], SCALAR_GENS[g].name,
                              r, t1 - t0, (double)(c1 - c0),
                              (double)eff_calls * (double)k * 4.0);

                double scale = (double)calls / (double)eff_calls;
                double dt = (t1 - t0) * scale;
                double dc = (double)(c1 - c0) * scale;
                if (dt < best_sec[g]) {
                    best_sec[g] = dt;
                    best_cyc[g] = dc;
                }
            }
        }
        (void)sink;
        free(buf);

        double total_words = (double)calls * (double)k;
        for (int g = 0; g < N_SCALAR_GENS; ++g) {
            if (SCALAR_GENS[g].max_k_single > 0 && k > (size_t)SCALAR_GENS[g].max_k_single) {
                scalar_valid[g][ri] = false;
            } else {
                scalar_valid[g][ri] = true;
                scalar_mat[g][ri] = make_metric(best_sec[g], best_cyc[g], total_words);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Run Suite A: Inlined Tight-Loop Shootout Mode vs Buffer Mode
    // ------------------------------------------------------------------------
    typedef void (*inlined_fn_t)(double *sec_out, double *cyc_out, size_t n);
    struct {
        const char *name;
        const char *category;
        int state_bytes;
        int scalar_idx;
        int stream_output_bits;
        inlined_fn_t fn_stream;
        inlined_fn_t fn_k1;
    } suite_a_defs[11] = {
        { "ra_core_v2 (N=256)", "Bipartite Perm+Weyl", 2048, 0,  32, bench_inlined_stream_ra_v2,          bench_inlined_k1_ra_v2          },
        { "ra_core_v1 (N=256)", "Bipartite Perm+Weyl", 2048, 5,  32, bench_inlined_stream_ra_v1,          bench_inlined_k1_ra_v1          },
        { "Philox4x32-10",      "Counter-Based (MUL)", 24,   6,  32, bench_inlined_stream_philox,         bench_inlined_k1_philox         },
        { "ChaCha8",            "Counter-Based (ARX)", 64,   7,  32, bench_inlined_stream_chacha8,        bench_inlined_k1_chacha8        },
        { "ChaCha12",           "Counter-Based (ARX)", 64,   8,  32, bench_inlined_stream_chacha12,       bench_inlined_k1_chacha12       },
        { "xoshiro256**",       "Scrambled Linear",    32,   9,  64, bench_inlined_stream_xoshiro256ss,   bench_inlined_k1_xoshiro256ss   },
        { "xoroshiro128++",     "Scrambled Linear",    16,   10, 64, bench_inlined_stream_xoroshiro128pp, bench_inlined_k1_xoroshiro128pp },
        { "PCG32",              "Permuted LCG",        16,   11, 32, bench_inlined_stream_pcg32,          bench_inlined_k1_pcg32          },
        { "SplitMix64",         "64-bit Weyl Mixer",   8,    12, 64, bench_inlined_stream_splitmix64,     bench_inlined_k1_splitmix64     },
        { "WyRand",             "64x64->128 Folded",   8,    13, 64, bench_inlined_stream_wyrand,         bench_inlined_k1_wyrand         },
        { "MT19937",            "GF(2) Twisted Reg",   2500, 14, 32, bench_inlined_stream_mt19937,        bench_inlined_k1_mt19937        },
    };

    suite_a_row_t suite_a_rows[11];
    const size_t INLINED_STREAM_WORDS = 102000000ULL; // exact multiple of 255 and 16
    const size_t INLINED_K1_CALLS     = 20000000ULL;

    for (int i = 0; i < 11; ++i) {
        double best_s_sec = 1e18, best_s_cyc = 1e18;
        double best_k1_sec = 1e18, best_k1_cyc = 1e18;
        size_t eff_k1 = (i == 10) ? 20000ULL : INLINED_K1_CALLS;
        for (int r = 0; r <= ROUNDS; ++r) {
            double s_sec, s_cyc, k1_sec, k1_cyc;
            suite_a_defs[i].fn_stream(&s_sec, &s_cyc, INLINED_STREAM_WORDS);
            if (r > 0) {
                record_sample(samples, "A", "inlined_stream", suite_a_defs[i].name, r,
                              s_sec, s_cyc,
                              (double)INLINED_STREAM_WORDS * suite_a_defs[i].stream_output_bits / 8.0);
            }
            if (r > 0 && s_sec < best_s_sec) { best_s_sec = s_sec; best_s_cyc = s_cyc; }
            suite_a_defs[i].fn_k1(&k1_sec, &k1_cyc, eff_k1);
            if (r > 0) {
                record_sample(samples, "A", "inlined_k1", suite_a_defs[i].name, r,
                              k1_sec, k1_cyc, (double)eff_k1 * 4.0);
            }
            if (r > 0 && k1_sec < best_k1_sec) { best_k1_sec = k1_sec; best_k1_cyc = k1_cyc; }
        }
        suite_a_rows[i].name           = suite_a_defs[i].name;
        suite_a_rows[i].category       = suite_a_defs[i].category;
        suite_a_rows[i].state_bytes    = suite_a_defs[i].state_bytes;
        suite_a_rows[i].inlined_stream = make_metric(best_s_sec, best_s_cyc,
            (double)INLINED_STREAM_WORDS * suite_a_defs[i].stream_output_bits / 32.0);
        suite_a_rows[i].inlined_k1     = make_metric(best_k1_sec, best_k1_cyc, (double)eff_k1);
        suite_a_rows[i].buffer_stream  = scalar_mat[suite_a_defs[i].scalar_idx][12];
        suite_a_rows[i].buffer_k1      = scalar_mat[suite_a_defs[i].scalar_idx][0];
    }

    // ------------------------------------------------------------------------
    // Run Suite C: 8-Way AVX2 SIMD (`u32x8`) Throughput across K
    // ------------------------------------------------------------------------
    metric_t avx2_mat[N_AVX2_GENS][N_AVX2_REGIMES];
    bool avx2_valid[N_AVX2_GENS][N_AVX2_REGIMES];
    memset(avx2_mat, 0, sizeof(avx2_mat));
    memset(avx2_valid, 0, sizeof(avx2_valid));

    uint32_t x8_keys[8] = {0, 1, 2, 3, 0xDEADBEEFu, 0x12345678u, 0x9E3779B9u, 0xFFFFFFFFu};
    uint32_t x8_addrs[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    alignas(64) uint32_t x8_buf[256 * 8];

    for (int ri = 0; ri < N_AVX2_REGIMES; ++ri) {
        size_t k = AVX2_K_VALS[ri];
        long long calls = AVX2_CALLS[ri];
        double best_sec[N_AVX2_GENS], best_cyc[N_AVX2_GENS];
        for (int g = 0; g < N_AVX2_GENS; ++g) { best_sec[g] = 1e18; best_cyc[g] = 1e18; }

        volatile uint32_t sink = 0;
        for (int r = 0; r <= ROUNDS; ++r) {
            for (int idx = 0; idx < N_AVX2_GENS; ++idx) {
                int g = (r + idx) % N_AVX2_GENS;
                if (AVX2_GENS[g].max_k_single > 0 && k > (size_t)AVX2_GENS[g].max_k_single) {
                    continue;
                }
                avx2_gen_fn_t fn = AVX2_GENS[g].fn;
                uint32_t acc = 0;
                double t0 = shootout_now_sec();
                uint64_t c0 = shootout_rdtscp();
                for (long long c = 0; c < calls; ++c) {
                    for (int lane = 0; lane < 8; ++lane) {
                        x8_keys[lane] = ((uint32_t)(c * 8 + lane) * 2654435761u) + 1u;
                    }
                    fn(x8_keys, x8_addrs, k, x8_buf);
                    asm volatile("" : : "g"(x8_buf) : "memory");
                    acc += x8_buf[0] ^ x8_buf[(k - 1) * 8];
                }
                uint64_t c1 = shootout_rdtscp();
                double t1 = shootout_now_sec();
                sink ^= acc;
                double dt = t1 - t0;
                double dc = (double)(c1 - c0);
                if (r == 0) continue;
                record_sample(samples, "C", AVX2_K_LABELS[ri], AVX2_GENS[g].name,
                              r, dt, dc, (double)calls * (double)k * 8.0 * 4.0);
                if (dt < best_sec[g]) {
                    best_sec[g] = dt;
                    best_cyc[g] = dc;
                }
            }
        }
        (void)sink;
        double total_words = (double)calls * (double)k * 8.0;
        for (int g = 0; g < N_AVX2_GENS; ++g) {
            if (AVX2_GENS[g].max_k_single > 0 && k > (size_t)AVX2_GENS[g].max_k_single) {
                avx2_valid[g][ri] = false;
            } else {
                avx2_valid[g][ri] = true;
                avx2_mat[g][ri] = make_metric(best_sec[g], best_cyc[g], total_words);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Run Suite D: Stateful Streaming API (`next_u32()` per draw across K)
    // ------------------------------------------------------------------------
    #define N_STREAM_ENGINES 6
    #define N_STREAM_K 9
    static const size_t STREAM_K_VALS[N_STREAM_K] = { 1, 2, 4, 8, 16, 64, 255, 256, 1000 };
    static const char *STREAM_K_LABELS[N_STREAM_K] = {
        "K=1", "K=2", "K=4", "K=8", "K=16", "K=64", "K=255", "K=256", "K=1000"
    };
    static const long long STREAM_K_CALLS[N_STREAM_K] = {
        4200000LL, 3360000LL, 2400000LL, 1800000LL, 1200000LL,
        480000LL, 180000LL, 144000LL, 48000LL
    };
    const char *stream_eng_names[N_STREAM_ENGINES] = {
        "ra_engine_v2 (3-Stage OTF)",
        "ra_engine_v1 (3-Stage OTF)",
        "Philox4x32-10 (stream)",
        "ChaCha8 (stream)",
        "xoshiro256** (stream)",
        "PCG32 (stream)"
    };
    metric_t stream_eng_mat[N_STREAM_ENGINES][N_STREAM_K];

    for (int ki = 0; ki < N_STREAM_K; ++ki) {
        size_t k = STREAM_K_VALS[ki];
        long long calls = STREAM_K_CALLS[ki];
        double best_sec[N_STREAM_ENGINES], best_cyc[N_STREAM_ENGINES];
        for (int e = 0; e < N_STREAM_ENGINES; ++e) { best_sec[e] = 1e18; best_cyc[e] = 1e18; }
        volatile uint32_t sink = 0;

#define RECORD_ENGINE_SAMPLE(E) do { \
    if (r > 0) { \
        double cycles = (double)(c1 - c0); \
        record_sample(samples, "D", STREAM_K_LABELS[ki], stream_eng_names[E], \
                      r, dt, cycles, (double)calls * (double)k * 4.0); \
        if (dt < best_sec[E]) { best_sec[E] = dt; best_cyc[E] = cycles; } \
    } \
} while (0)

        for (int r = 0; r <= ROUNDS; ++r) {
            // 0: ra_engine_v2
            {
                ra_engine_v2_t eng; uint32_t acc = 0;
                double t0 = shootout_now_sec(); uint64_t c0 = shootout_rdtscp();
                for (long long c = 0; c < calls; ++c) {
                    ra_engine_v2_init(&eng, (uint32_t)(c * 2654435761u + 1u), 0u);
                    for (size_t w = 0; w < k; ++w) acc ^= ra_engine_v2_next_u32(&eng);
                }
                uint64_t c1 = shootout_rdtscp(); double dt = shootout_now_sec() - t0;
                sink ^= acc;
                RECORD_ENGINE_SAMPLE(0);
            }
            // 1: ra_engine_v1
            {
                ra_engine_v2_t eng; uint32_t acc = 0;
                double t0 = shootout_now_sec(); uint64_t c0 = shootout_rdtscp();
                for (long long c = 0; c < calls; ++c) {
                    ra_engine_v2_init(&eng, (uint32_t)(c * 2654435761u + 1u), 0u);
                    for (size_t w = 0; w < k; ++w) acc ^= ra_engine_v1_next_u32(&eng);
                }
                uint64_t c1 = shootout_rdtscp(); double dt = shootout_now_sec() - t0;
                sink ^= acc;
                RECORD_ENGINE_SAMPLE(1);
            }
            // 2: Philox stream
            {
                philox_stream_t st; uint32_t acc = 0;
                double t0 = shootout_now_sec(); uint64_t c0 = shootout_rdtscp();
                for (long long c = 0; c < calls; ++c) {
                    philox_stream_init(&st, (uint32_t)(c * 2654435761u + 1u), 0x20261104u, 0u, 0u);
                    for (size_t w = 0; w < k; ++w) acc ^= philox_stream_next_u32(&st);
                }
                uint64_t c1 = shootout_rdtscp(); double dt = shootout_now_sec() - t0;
                sink ^= acc;
                RECORD_ENGINE_SAMPLE(2);
            }
            // 3: ChaCha8 stream
            {
                chacha_stream_t st; uint32_t acc = 0;
                double t0 = shootout_now_sec(); uint64_t c0 = shootout_rdtscp();
                for (long long c = 0; c < calls; ++c) {
                    chacha_stream_init(&st, (uint32_t)(c * 2654435761u + 1u), 0x20261104u, 8);
                    for (size_t w = 0; w < k; ++w) acc ^= chacha_stream_next_u32(&st);
                }
                uint64_t c1 = shootout_rdtscp(); double dt = shootout_now_sec() - t0;
                sink ^= acc;
                RECORD_ENGINE_SAMPLE(3);
            }
            // 4: xoshiro256** stream
            {
                xoshiro256ss_state_t st; uint32_t acc = 0;
                double t0 = shootout_now_sec(); uint64_t c0 = shootout_rdtscp();
                for (long long c = 0; c < calls; ++c) {
                    xoshiro256ss_init(&st, (uint32_t)(c * 2654435761u + 1u), 0u);
                    for (size_t w = 0; w < k; ++w) acc ^= xoshiro256ss_next_u32(&st);
                }
                uint64_t c1 = shootout_rdtscp(); double dt = shootout_now_sec() - t0;
                sink ^= acc;
                RECORD_ENGINE_SAMPLE(4);
            }
            // 5: PCG32 stream
            {
                pcg32_random_t st; uint32_t acc = 0;
                double t0 = shootout_now_sec(); uint64_t c0 = shootout_rdtscp();
                for (long long c = 0; c < calls; ++c) {
                    pcg32_srandom_r(&st, (uint64_t)(uint32_t)(c * 2654435761u + 1u), 54u);
                    for (size_t w = 0; w < k; ++w) acc ^= pcg32_random_r(&st);
                }
                uint64_t c1 = shootout_rdtscp(); double dt = shootout_now_sec() - t0;
                sink ^= acc;
                RECORD_ENGINE_SAMPLE(5);
            }
        }
        (void)sink;
        double total_words = (double)calls * (double)k;
        for (int e = 0; e < N_STREAM_ENGINES; ++e) {
            stream_eng_mat[e][ki] = make_metric(best_sec[e], best_cyc[e], total_words);
        }
#undef RECORD_ENGINE_SAMPLE
    }

    // ------------------------------------------------------------------------
    // Print Concise Summary Tables to Stdout
    // ------------------------------------------------------------------------
    printf("=== SUITE A: Shootout-Inspired (Inlined Register-Accumulator vs Materialized Buffer) ===\n");
    printf("| Generator | State | Inlined Stream (cpb / ns/64b / MiB/s) | Buffer Stream (cpb / ns/64b / MiB/s) | Inlined K=1 (cyc/w / ns/w / MiB/s) | Buffer K=1 (cyc/w / ns/w / MiB/s) |\n");
    printf("|---|---:|---:|---:|---:|---:|\n");
    for (int i = 0; i < 11; ++i) {
        suite_a_row_t *r = &suite_a_rows[i];
        printf("| **`%s`** | %d B | %.2f cpb / %.2f ns / %.0f | %.2f cpb / %.2f ns / %.0f | %.1f cyc / %.2f ns / %.1f | %.1f cyc / %.2f ns / %.1f |\n",
               r->name, r->state_bytes,
               r->inlined_stream.cpb, r->inlined_stream.ns_64b, r->inlined_stream.mibs,
               r->buffer_stream.cpb,  r->buffer_stream.ns_64b,  r->buffer_stream.mibs,
               r->inlined_k1.cyc_word, r->inlined_k1.ns_word, r->inlined_k1.mibs,
               r->buffer_k1.cyc_word,  r->buffer_k1.ns_word,  r->buffer_k1.mibs);
    }

    printf("\n=== SUITE B: Materialized Direct-Buffer Scalar Throughput (MiB/s) across K ===\n");
    printf("| Generator | K=1 | K=2 | K=4 | K=8 | K=15 | K=31 | K=63 | K=127 | K=255 | K=256 | K=510 | K=10000 | stream |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int g = 0; g < N_SCALAR_GENS; ++g) {
        printf("| **`%s`** |", SCALAR_GENS[g].name);
        for (int ri = 0; ri < N_SCALAR_REGIMES; ++ri) {
            if (!scalar_valid[g][ri]) printf(" — |");
            else printf(" %.1f |", scalar_mat[g][ri].mibs);
        }
        printf("\n");
    }

    printf("\n=== SUITE C: Apples-to-Apples 8-Way AVX2 SIMD (`u32x8`) Throughput (MiB/s) across K ===\n");
    printf("| Generator | K=1 | K=2 | K=4 | K=8 | K=15 | K=31 | K=63 | K=127 | K=255 |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int g = 0; g < N_AVX2_GENS; ++g) {
        printf("| **`%s`** |", AVX2_GENS[g].name);
        for (int ri = 0; ri < N_AVX2_REGIMES; ++ri) {
            if (!avx2_valid[g][ri]) printf(" — |");
            else printf(" %.1f |", avx2_mat[g][ri].mibs);
        }
        printf("\n");
    }

    printf("\n=== SUITE D: Stateful Streaming API (`next_u32()` per draw) Throughput (MiB/s) ===\n");
    printf("| Engine | K=1 | K=2 | K=4 | K=8 | K=16 | K=64 | K=255 | K=256 | K=1000 |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int e = 0; e < N_STREAM_ENGINES; ++e) {
        printf("| **`%s`** |", stream_eng_names[e]);
        for (int ki = 0; ki < N_STREAM_K; ++ki) {
            printf(" %.1f |", stream_eng_mat[e][ki].mibs);
        }
        printf("\n");
    }

    // ------------------------------------------------------------------------
    // Write Full JSON Output (`data/shootout_microbench.json`)
    // ------------------------------------------------------------------------
    mkdir("data", 0755);
    FILE *fp = fopen("data/shootout_microbench.json", "w");
    if (!fp) {
        perror("fopen data/shootout_microbench.json");
        return 1;
    }

    fprintf(fp, "{\n");
    // Suite A
    fprintf(fp, "  \"suite_a_vigna_shootout\": [\n");
    for (int i = 0; i < 11; ++i) {
        suite_a_row_t *r = &suite_a_rows[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": \"%s\", \"category\": \"%s\", \"state_bytes\": %d,\n",
                r->name, r->category, r->state_bytes);
        fprintf(fp, "      \"inlined_stream\": {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f},\n",
                r->inlined_stream.mibs, r->inlined_stream.gbs, r->inlined_stream.ns_word, r->inlined_stream.ns_64b, r->inlined_stream.cyc_word, r->inlined_stream.cpb);
        fprintf(fp, "      \"buffer_stream\":  {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f},\n",
                r->buffer_stream.mibs, r->buffer_stream.gbs, r->buffer_stream.ns_word, r->buffer_stream.ns_64b, r->buffer_stream.cyc_word, r->buffer_stream.cpb);
        fprintf(fp, "      \"inlined_k1\":     {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f},\n",
                r->inlined_k1.mibs, r->inlined_k1.gbs, r->inlined_k1.ns_word, r->inlined_k1.ns_64b, r->inlined_k1.cyc_word, r->inlined_k1.cpb);
        fprintf(fp, "      \"buffer_k1\":      {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f}\n",
                r->buffer_k1.mibs, r->buffer_k1.gbs, r->buffer_k1.ns_word, r->buffer_k1.ns_64b, r->buffer_k1.cyc_word, r->buffer_k1.cpb);
        fprintf(fp, "    }%s\n", (i + 1 < 11) ? "," : "");
    }
    fprintf(fp, "  ],\n");

    // Suite B
    fprintf(fp, "  \"suite_b_scalar_buffer\": {\n");
    fprintf(fp, "    \"k_labels\": [");
    for (int ri = 0; ri < N_SCALAR_REGIMES; ++ri) {
        fprintf(fp, "\"%s\"%s", SCALAR_K_LABELS[ri], (ri + 1 < N_SCALAR_REGIMES) ? ", " : "");
    }
    fprintf(fp, "],\n    \"generators\": [\n");
    for (int g = 0; g < N_SCALAR_GENS; ++g) {
        fprintf(fp, "      {\n        \"name\": \"%s\", \"family\": \"%s\", \"state_bytes\": %d, \"max_k_single\": %d,\n        \"regimes\": {\n",
                SCALAR_GENS[g].name, SCALAR_GENS[g].family, SCALAR_GENS[g].state_bytes, SCALAR_GENS[g].max_k_single);
        for (int ri = 0; ri < N_SCALAR_REGIMES; ++ri) {
            if (!scalar_valid[g][ri]) {
                fprintf(fp, "          \"%s\": null%s\n", SCALAR_K_LABELS[ri], (ri + 1 < N_SCALAR_REGIMES) ? "," : "");
            } else {
                metric_t *m = &scalar_mat[g][ri];
                fprintf(fp, "          \"%s\": {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f}%s\n",
                        SCALAR_K_LABELS[ri], m->mibs, m->gbs, m->ns_word, m->ns_64b, m->cyc_word, m->cpb,
                        (ri + 1 < N_SCALAR_REGIMES) ? "," : "");
            }
        }
        fprintf(fp, "        }\n      }%s\n", (g + 1 < N_SCALAR_GENS) ? "," : "");
    }
    fprintf(fp, "    ]\n  },\n");

    // Suite C
    fprintf(fp, "  \"suite_c_avx2_x8\": {\n");
    fprintf(fp, "    \"k_labels\": [");
    for (int ri = 0; ri < N_AVX2_REGIMES; ++ri) {
        fprintf(fp, "\"%s\"%s", AVX2_K_LABELS[ri], (ri + 1 < N_AVX2_REGIMES) ? ", " : "");
    }
    fprintf(fp, "],\n    \"generators\": [\n");
    for (int g = 0; g < N_AVX2_GENS; ++g) {
        fprintf(fp, "      {\n        \"name\": \"%s\", \"family\": \"%s\", \"max_k_single\": %d,\n        \"regimes\": {\n",
                AVX2_GENS[g].name, AVX2_GENS[g].family, AVX2_GENS[g].max_k_single);
        for (int ri = 0; ri < N_AVX2_REGIMES; ++ri) {
            if (!avx2_valid[g][ri]) {
                fprintf(fp, "          \"%s\": null%s\n", AVX2_K_LABELS[ri], (ri + 1 < N_AVX2_REGIMES) ? "," : "");
            } else {
                metric_t *m = &avx2_mat[g][ri];
                fprintf(fp, "          \"%s\": {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f}%s\n",
                        AVX2_K_LABELS[ri], m->mibs, m->gbs, m->ns_word, m->ns_64b, m->cyc_word, m->cpb,
                        (ri + 1 < N_AVX2_REGIMES) ? "," : "");
            }
        }
        fprintf(fp, "        }\n      }%s\n", (g + 1 < N_AVX2_GENS) ? "," : "");
    }
    fprintf(fp, "    ]\n  },\n");

    // Suite D
    fprintf(fp, "  \"suite_d_stateful_engine\": {\n");
    fprintf(fp, "    \"k_values\": [1, 2, 4, 8, 16, 64, 255, 256, 1000],\n");
    fprintf(fp, "    \"engines\": [\n");
    for (int e = 0; e < N_STREAM_ENGINES; ++e) {
        fprintf(fp, "      {\n        \"name\": \"%s\",\n        \"regimes\": {\n", stream_eng_names[e]);
        for (int ki = 0; ki < N_STREAM_K; ++ki) {
            metric_t *m = &stream_eng_mat[e][ki];
            fprintf(fp, "          \"K=%zu\": {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f}%s\n",
                    STREAM_K_VALS[ki], m->mibs, m->gbs, m->ns_word, m->ns_64b, m->cyc_word, m->cpb,
                    (ki + 1 < N_STREAM_K) ? "," : "");
        }
        fprintf(fp, "        }\n      }%s\n", (e + 1 < N_STREAM_ENGINES) ? "," : "");
    }
    fprintf(fp, "    ]\n  }\n");
    fprintf(fp, "}\n");
    fclose(fp);
    if (fclose(samples) != 0) {
        perror("data/shootout_samples.csv");
        return 1;
    }
    printf("\nSaved full JSON microbenchmark dataset to data/shootout_microbench.json\n");
    return 0;
}
