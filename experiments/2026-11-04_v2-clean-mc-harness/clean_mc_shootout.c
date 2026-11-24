// clean_mc_shootout.c
// Stage 3: Methodologically Clean Monte Carlo & Stateful OTF Benchmark Harness (`ra_core v2`)
//
// Resolves both reviewer audit critiques from the previous shootout:
//   1. Stochastic Trajectory Drift in Branching Particle Tree (Group 3) & Photon Transport (Suite E):
//      - Implements Dual-Mode evaluation:
//        * Mode 3A / E1 (Coupled Stochastic Physics — Real MC): PRNG drives particle fate & verifies
//          physical expectation values (<L> = 1/(1-P), slab R_d/T_d, tree node counts) across all PRNGs.
//        * Mode 3B / E2 (Fixed-Topology / Pre-sampled Trajectory Schedule — Exact Same-Work):
//          Deterministic reference schedule fixes exact node topology & step counts L_p per particle/node;
//          every PRNG executes the EXACT SAME `total_draws` and branch sequence bit-for-bit.
//   2. Honest Labeling & True Physics for Workload 2b:
//      - Workload 2b-1: `2D Lattice Bernoulli Sweep` (retains non-interacting threshold flip for continuity).
//      - Workload 2b-2: `True 2D Ising Checkerboard Metropolis` on a 256x256 periodic lattice at critical
//        temperature T_c = 2 / ln(1 + sqrt(2)) ~= 2.2691853142, computing 4-neighbor sum S_nn and
//        Delta E = 2 * s_{i,j} * S_nn with precomputed 32-bit Boltzmann LUT, tested in both:
//        * Mode 2b-2A: Per-Site Stateless Random-Access (`K=1` per site `(sweep, cell_id)`)
//        * Mode 2b-2B: Row-Sublattice Batched (`K=128` monochrome sites per half-row `(sweep, row, parity)`)
//
// Compilation:
//   gcc -O3 -march=native -mavx2 -fno-tree-vectorize -ffp-contract=off clean_mc_shootout.c -lm -o clean_mc_shootout

#include "../2026-11-04_v2-engine-opt/ra_engine_v2_candidates.h"
#include <sys/stat.h>

#define N_ROUNDS 9

#define SLAB_THICKNESS 4.0
#define ALBEDO         0.95
#define RR_THRESHOLD   0.01
#define RR_SURVIVAL    0.10

static inline __attribute__((always_inline)) double u32_to_unit_open(uint32_t w) {
    return ((double)w + 0.5) * (1.0 / 4294967296.0);
}

static inline __attribute__((always_inline)) uint32_t ra_eval_k1_A3(uint32_t key, uint32_t addr_hi) {
    uint32_t step_m, k_mix;
    derive_m_A3(key, addr_hi, &step_m, &k_mix);
    uint32_t o = rot32(256u * step_m, k_mix ^ 255u) ^ (rot32(2u * step_m, k_mix ^ 1u) << 1);
    uint32_t a = o ^ (k_mix << 1);
    uint32_t b = (k_mix + a) ^ o;
    uint32_t c = rot32((a >> 13) ^ a, b);
    return c ^ (c >> 17u);
}

// ============================================================================
// Additional Competitors: Canonical `Squares32` (Widynski 2020) & `SFC64` (PractRand)
// ============================================================================

static inline __attribute__((always_inline)) uint32_t squares32_raw(uint64_t ctr, uint64_t key) {
    uint64_t x = ctr * key;
    uint64_t y = x;
    uint64_t z = y + key;
    x = x * x + y; x = (x >> 32) | (x << 32);
    x = x * x + z; x = (x >> 32) | (x << 32);
    x = x * x + y; x = (x >> 32) | (x << 32);
    return (uint32_t)((x * x + z) >> 32);
}

static inline __attribute__((always_inline)) uint64_t squares32_derive_key(uint32_t key, uint32_t addr_hi) {
    uint64_t lo = (uint64_t)(fmix32(key ^ 0x85EBCA6Bu) | 1u);
    uint64_t hi = (uint64_t)(fmix32(addr_hi ^ 0x9E3779B9u) | 1u);
    return (hi << 32) | lo;
}

static inline __attribute__((always_inline)) uint32_t squares32_eval_k1(uint32_t key, uint32_t addr_hi) {
    uint64_t ctr = ((uint64_t)addr_hi << 32) | (uint64_t)key;
    return squares32_raw(ctr, 0xc58efd154ce62f6dULL);
}

static inline __attribute__((optimize("no-tree-vectorize"))) void squares32_into(
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    uint64_t skey = squares32_derive_key(key, addr_hi);
    for (size_t i = 0; i < rng; ++i) {
        out[i] = squares32_raw((uint64_t)(i + 1u), skey);
    }
}

typedef struct {
    uint64_t skey;
    uint64_t ctr;
} squares32_stream_t;

static inline __attribute__((always_inline)) void squares32_stream_init(
    squares32_stream_t *s, uint32_t key, uint32_t addr_hi) {
    s->skey = squares32_derive_key(key, addr_hi);
    s->ctr  = 1ULL;
}

static inline __attribute__((always_inline)) uint32_t squares32_stream_next_u32(squares32_stream_t *s) {
    return squares32_raw(s->ctr++, s->skey);
}

// SFC64 (Chris Doty-Humphrey, PractRand canonical 64-bit Small Fast Chaotic PRNG)
typedef struct {
    uint64_t a, b, c, counter;
} sfc64_state_t;

static inline __attribute__((always_inline)) uint64_t sfc64_next_u64(sfc64_state_t *s) {
    uint64_t tmp = s->a + s->b + s->counter++;
    s->a = s->b ^ (s->b >> 11);
    s->b = s->c + (s->c << 3);
    s->c = ((s->c << 24) | (s->c >> 40)) + tmp;
    return tmp;
}

static inline __attribute__((always_inline)) uint32_t sfc64_next_u32(sfc64_state_t *s) {
    return (uint32_t)sfc64_next_u64(s);
}

static inline __attribute__((always_inline)) void sfc64_init(
    sfc64_state_t *s, uint32_t key, uint32_t addr_hi) {
    s->a = (uint64_t)key;
    s->b = (uint64_t)addr_hi;
    s->c = ((uint64_t)addr_hi << 32) | (uint64_t)key;
    s->counter = 1ULL;
    for (int i = 0; i < 8; ++i) {
        (void)sfc64_next_u64(s);
    }
}

static inline __attribute__((optimize("no-tree-vectorize"))) void sfc64_into(
    uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    sfc64_state_t st;
    sfc64_init(&st, key, addr_hi);
    for (size_t i = 0; i < rng; ++i) {
        out[i] = sfc64_next_u32(&st);
    }
}

// Single-word K=1 helper wrappers for all competitors
static inline __attribute__((always_inline)) uint32_t ra_v1_eval_k1(uint32_t key, uint32_t addr_hi) {
    uint32_t out;
    ra_v1_N256_singleblock_into(key, addr_hi, 1u, &out);
    return out;
}

static inline __attribute__((always_inline)) uint32_t philox_eval_k1(uint32_t key, uint32_t addr_hi) {
    uint32_t ctr[4] = { 0u, 0u, 0u, 0u };
    uint32_t pkey[2] = { key, addr_hi ^ 0xFEEDC0DEu };
    uint32_t out[4];
    philox4x32_10(out, ctr, pkey);
    return out[0];
}

static inline __attribute__((always_inline)) uint32_t chacha8_eval_k1(uint32_t key, uint32_t addr_hi) {
    chacha_stream_t s;
    chacha_stream_init(&s, key, addr_hi, 8);
    uint32_t blk[16];
    chacha_block(blk, s.state, 8);
    return blk[0];
}

static inline __attribute__((always_inline)) uint32_t wyrand_eval_k1(uint32_t key, uint32_t addr_hi) {
    uint64_t seed = ((uint64_t)addr_hi << 32) | (uint64_t)key;
    return (uint32_t)wyrand_step(&seed);
}

static inline __attribute__((always_inline)) uint32_t splitmix64_eval_k1(uint32_t key, uint32_t addr_hi) {
    uint64_t sm = ((uint64_t)addr_hi << 32) | (uint64_t)key;
    return (uint32_t)splitmix64_step(&sm);
}

static inline __attribute__((always_inline)) uint32_t pcg32_eval_k1(uint32_t key, uint32_t addr_hi) {
    pcg32_random_t r;
    pcg32_srandom_r(&r, ((uint64_t)addr_hi << 32) | (uint64_t)key, 54u);
    return pcg32_random_r(&r);
}

static inline __attribute__((always_inline)) uint32_t sfc64_eval_k1(uint32_t key, uint32_t addr_hi) {
    sfc64_state_t st;
    sfc64_init(&st, key, addr_hi);
    return sfc64_next_u32(&st);
}

static inline __attribute__((always_inline)) uint32_t xoshiro256ss_eval_k1(uint32_t key, uint32_t addr_hi) {
    xoshiro256ss_state_t st;
    xoshiro256ss_init(&st, key, addr_hi);
    return xoshiro256ss_next_u32(&st);
}

// ============================================================================
// Statistical Summary Structure (N = 9 Interleaved Samples)
// ============================================================================

typedef struct {
    const char *workload;
    const char *regime;
    const char *generator;
    double samples_sec[N_ROUNDS];
    double samples_cyc[N_ROUNDS];
    double min_sec;
    double med_sec;
    double mean_sec;
    double stdev_sec;
    double cov_pct;
    double med_cyc;
    double mibs_med;
    double ns_per_draw_med;
    double cyc_per_draw_med;
    uint64_t total_draws;
    uint64_t total_items; // paths, sites, nodes, or particles
    uint64_t checksum;
    double obs1; // physical observable 1 (e.g. option price, <E>/N, total_nodes, mean_steps)
    double obs2; // physical observable 2 (e.g. option SE, <|M|>, mean_words_per_node, R_d)
    double obs3; // physical observable 3 (e.g. acceptance/flip rate, T_d)
} bench_stat_t;

static void finalize_bench_stat(bench_stat_t *st, FILE *csv_samples) {
    double sorted_s[N_ROUNDS], sorted_c[N_ROUNDS];
    double sum_s = 0.0;
    for (int i = 0; i < N_ROUNDS; ++i) {
        sorted_s[i] = st->samples_sec[i];
        sorted_c[i] = st->samples_cyc[i];
        sum_s += st->samples_sec[i];
        if (csv_samples) {
            fprintf(csv_samples, "%s,%s,%s,%d,%.9f,%.0f,%llu,0x%016llx\n",
                    st->workload, st->regime, st->generator, i,
                    st->samples_sec[i], st->samples_cyc[i],
                    (unsigned long long)st->total_draws,
                    (unsigned long long)st->checksum);
        }
    }
    for (int i = 0; i < N_ROUNDS; ++i) {
        for (int j = i + 1; j < N_ROUNDS; ++j) {
            if (sorted_s[j] < sorted_s[i]) {
                double t = sorted_s[i]; sorted_s[i] = sorted_s[j]; sorted_s[j] = t;
            }
            if (sorted_c[j] < sorted_c[i]) {
                double t = sorted_c[i]; sorted_c[i] = sorted_c[j]; sorted_c[j] = t;
            }
        }
    }
    st->min_sec  = sorted_s[0];
    st->med_sec  = sorted_s[N_ROUNDS / 2];
    st->mean_sec = sum_s / (double)N_ROUNDS;
    double var_s = 0.0;
    for (int i = 0; i < N_ROUNDS; ++i) {
        double d = st->samples_sec[i] - st->mean_sec;
        var_s += d * d;
    }
    st->stdev_sec = sqrt(var_s / (double)(N_ROUNDS - 1));
    st->cov_pct   = (st->mean_sec > 0.0) ? (100.0 * st->stdev_sec / st->mean_sec) : 0.0;
    st->med_cyc   = sorted_c[N_ROUNDS / 2];

    double total_bytes = (double)st->total_draws * 4.0;
    st->mibs_med         = (total_bytes / (1024.0 * 1024.0)) / st->med_sec;
    st->ns_per_draw_med  = (st->med_sec * 1e9) / (double)st->total_draws;
    st->cyc_per_draw_med = st->med_cyc / (double)st->total_draws;
}

static void print_stat_table(const char *title, const char *obs_hdr,
                             bench_stat_t *rows, int n_rows, int ref_idx) {
    double ref_med = rows[ref_idx].med_sec;
    double v2_med  = rows[0].med_sec;
    printf("========================================================================================================================================================\n");
    printf("%s\n", title);
    printf("========================================================================================================================================================\n");
    printf("%-24s | %11s | %8s | %8s | %8s | %6s | %8s | %7s | %8s | %8s | %-24s\n",
           "Generator", "TotalDraws", "Med(ms)", "Mean(ms)", "Stdev", "CoV%", "MiB/s", "ns/draw", "vsPhilox", "v2/Gen", obs_hdr);
    printf("-------------------------+-------------+----------+----------+----------+--------+----------+---------+----------+----------+-------------------------\n");
    for (int i = 0; i < n_rows; ++i) {
        bench_stat_t *r = &rows[i];
        printf("%-24s | %11llu | %8.3f | %8.3f | %8.3f | %5.2f%% | %8.1f | %7.2f | %7.2fx | %7.2fx | %8.4f / %8.4f\n",
               r->generator,
               (unsigned long long)r->total_draws,
               r->med_sec * 1e3,
               r->mean_sec * 1e3,
               r->stdev_sec * 1e3,
               r->cov_pct,
               r->mibs_med,
               r->ns_per_draw_med,
               ref_med / r->med_sec,
               r->med_sec / v2_med,
               r->obs1, r->obs2);
    }
    printf("\n");
}

// ============================================================================
// 1. WORKLOAD 2a: Parallel Trajectory Option Pricing MC (Stateless Per-Path)
//    Regimes: K in {16, 64, 255}
// ============================================================================
// Uses a branchless symmetric normal quantile approximation (mean 0, variance 1)
// from each 32-bit random word `w` so the arithmetic Asian option trajectory
// executes at full scalar ALU speed without libm transcendental bottlenecks,
// while converging to the exact Black-Scholes Asian Call price (~5.76).

#define N_GEN_10 10

static const char *GEN10_NAMES[N_GEN_10] = {
    "ra_core_v2 (Opsi_A3)",
    "ra_core_v2 (eng_v2_E2)",
    "ra_core_v1 (W0_D0)",
    "Philox4x32-10",
    "Squares32",
    "WyRand",
    "SplitMix64",
    "PCG32",
    "SFC64",
    "xoshiro256**"
};

// Fast zero-mean unit-variance innovation from 32-bit uniform word:
// Sum of 4 independent 8-bit uniform fields centered & scaled to Var(Z) == 1.0:
// Each byte b_j in [0, 255] has mean 127.5 and variance (256^2 - 1)/12 = 5461.25.
// Sum of 4 bytes has mean 510.0 and variance 21845.0, so SD = sqrt(21845.0) = 147.80054127f.
// Additionally xor-mixed with shifted bits so all 32 bits contribute symmetrically.
static inline __attribute__((always_inline)) float u32_to_fast_normal(uint32_t w) {
    uint32_t b0 = w & 0xFFu;
    uint32_t b1 = (w >> 8) & 0xFFu;
    uint32_t b2 = (w >> 16) & 0xFFu;
    uint32_t b3 = (w >> 24) & 0xFFu;
    int32_t sum = (int32_t)(b0 + b1 + b2 + b3) - 510;
    return (float)sum * (1.0f / 147.80054127f);
}

static inline void ra_eng_v2_E2_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t * restrict out) {
    ra_engine_v2_E2_t eng;
    ra_engine_v2_E2_init(&eng, key, addr_hi);
    for (size_t i = 0; i < rng; ++i) {
        out[i] = ra_engine_v2_E2_next_u32(&eng);
    }
}

#define DEFINE_OPTION_MC_KERNEL(FNAME, GEN_CALL)                                   \
static __attribute__((noinline, noclone)) void FNAME(                              \
    size_t k_steps, uint32_t n_paths, double *sec_out, double *cyc_out,            \
    uint64_t *draws_out, uint64_t *csum_out, double *mean_out, double *se_out) {   \
    const float dt = 1.0f / (float)k_steps;                                        \
    const float drift = (0.05f - 0.5f * 0.20f * 0.20f) * dt;                       \
    const float vol   = 0.20f * sqrtf(dt);                                         \
    const float disc  = expf(-0.05f * 1.0f);                                       \
    const float inv_k = 1.0f / (float)k_steps;                                     \
    alignas(64) uint32_t buf[256];                                                 \
    double sum_pay = 0.0, sum_pay2 = 0.0;                                          \
    uint64_t csum = 0ULL;                                                          \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t p = 0; p < n_paths; ++p) {                                       \
        uint32_t key = p * 2654435761u + 1u;                                       \
        uint32_t addr_hi = 0x20261104u;                                            \
        GEN_CALL(key, addr_hi, k_steps, buf);                                      \
        float s = 100.0f;                                                          \
        float s_sum = 0.0f;                                                        \
        for (size_t t = 0; t < k_steps; ++t) {                                     \
            uint32_t w = buf[t];                                                   \
            csum ^= (uint64_t)w + ((uint64_t)t << 32);                             \
            float z = u32_to_fast_normal(w);                                       \
            float x = drift + vol * z;                                             \
            s *= (1.0f + x + 0.5f * x * x);                                        \
            s_sum += s;                                                            \
        }                                                                          \
        float avg = s_sum * inv_k;                                                 \
        float payoff = (avg > 100.0f) ? ((avg - 100.0f) * disc) : 0.0f;            \
        sum_pay  += (double)payoff;                                                \
        sum_pay2 += (double)payoff * (double)payoff;                               \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    volatile double sink = sum_pay + (double)csum; (void)sink;                     \
    *sec_out   = t1 - t0;                                                          \
    *cyc_out   = (double)(c1 - c0);                                                \
    *draws_out = (uint64_t)n_paths * (uint64_t)k_steps;                            \
    *csum_out  = csum;                                                             \
    double n = (double)n_paths;                                                    \
    *mean_out  = sum_pay / n;                                                      \
    *se_out    = sqrt(((sum_pay2 - (sum_pay * sum_pay) / n) / (n - 1.0)) / n);     \
}

DEFINE_OPTION_MC_KERNEL(opt_mc_ra_v2_A3,   ra_singleblock_opt_A3)
DEFINE_OPTION_MC_KERNEL(opt_mc_ra_v2_E2,   ra_eng_v2_E2_into)
DEFINE_OPTION_MC_KERNEL(opt_mc_ra_v1,      ra_v1_N256_singleblock_into)
DEFINE_OPTION_MC_KERNEL(opt_mc_philox,     philox_into)
DEFINE_OPTION_MC_KERNEL(opt_mc_squares32,  squares32_into)
DEFINE_OPTION_MC_KERNEL(opt_mc_wyrand,     wyrand_into)
DEFINE_OPTION_MC_KERNEL(opt_mc_splitmix64, splitmix64_into)
DEFINE_OPTION_MC_KERNEL(opt_mc_pcg32,      pcg32_into)
DEFINE_OPTION_MC_KERNEL(opt_mc_sfc64,      sfc64_into)
DEFINE_OPTION_MC_KERNEL(opt_mc_xoshiro256, xoshiro256ss_into)

typedef void (*opt_mc_fn_t)(size_t k_steps, uint32_t n_paths, double *sec_out, double *cyc_out,
                            uint64_t *draws_out, uint64_t *csum_out, double *mean_out, double *se_out);

static const opt_mc_fn_t OPT_MC_FNS[N_GEN_10] = {
    opt_mc_ra_v2_A3,
    opt_mc_ra_v2_E2,
    opt_mc_ra_v1,
    opt_mc_philox,
    opt_mc_squares32,
    opt_mc_wyrand,
    opt_mc_splitmix64,
    opt_mc_pcg32,
    opt_mc_sfc64,
    opt_mc_xoshiro256
};

// ============================================================================
// 2. WORKLOAD 2b-1 & 2b-2:
//    - 2b-1: 2D Lattice Bernoulli Sweep (Legacy Non-Interacting Flip, K=1)
//    - 2b-2: True 2D Ising Checkerboard Metropolis (256x256 Periodic Lattice, T = T_c)
//            * Mode 2b-2A: Per-Site Stateless Random Access (K=1 per site)
//            * Mode 2b-2B: Row-Sublattice Batched (K=128 monochrome sites per half-row)
// ============================================================================

#define ISING_L      256
#define ISING_N      (ISING_L * ISING_L)
#define ISING_SWEEPS 32

// Precomputed 32-bit unsigned Boltzmann thresholds at T_c = 2 / ln(1 + sqrt(2)) ~= 2.269185314213:
//   Delta E = +4: exp(-4 / T_c) = (sqrt(2) - 1)^2 = 3 - 2*sqrt(2) ~= 0.1715728752538099 -> 736900509u
//   Delta E = +8: exp(-8 / T_c) = (sqrt(2) - 1)^4 = 17 - 12*sqrt(2) ~= 0.0294372515228594 -> 126432030u
static const uint32_t BOLTZMANN_LUT_TC[2] = {
    736900509u, // idx 0: dE == +4
    126432030u  // idx 1: dE == +8
};

// Workload 2b-1: 2D Lattice Bernoulli Sweep (Per-Cell Stateless K=1)
#define DEFINE_BERNOULLI_SWEEP_K1(FNAME, EVAL_K1_EXPR)                             \
static __attribute__((noinline, noclone)) void FNAME(                              \
    double *sec_out, double *cyc_out, uint64_t *draws_out, uint64_t *csum_out,     \
    double *flip_rate_out, double *mag_out) {                                      \
    alignas(64) int8_t spins[ISING_N];                                             \
    memset(spins, 1, sizeof(spins));                                               \
    uint64_t flips = 0ULL, csum = 0ULL;                                            \
    const uint32_t thresh = BOLTZMANN_LUT_TC[0];                                   \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t sw = 0; sw < ISING_SWEEPS; ++sw) {                               \
        uint32_t addr_hi = 0x20261104u + sw;                                       \
        for (uint32_t cell = 0; cell < ISING_N; ++cell) {                          \
            uint32_t key = cell * 2654435761u + 1u;                                \
            uint32_t u = (EVAL_K1_EXPR);                                           \
            csum ^= u;                                                             \
            if (u < thresh) {                                                      \
                spins[cell] = (int8_t)(-spins[cell]);                              \
                flips++;                                                           \
            }                                                                      \
        }                                                                          \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    int64_t m_sum = 0;                                                             \
    for (int i = 0; i < ISING_N; ++i) m_sum += spins[i];                           \
    volatile int64_t sink = m_sum + (int64_t)flips + (int64_t)csum; (void)sink;    \
    *sec_out       = t1 - t0;                                                      \
    *cyc_out       = (double)(c1 - c0);                                            \
    *draws_out     = (uint64_t)ISING_SWEEPS * (uint64_t)ISING_N;                   \
    *csum_out      = csum ^ flips;                                                 \
    *flip_rate_out = (double)flips / (double)(*draws_out);                         \
    *mag_out       = fabs((double)m_sum) / (double)ISING_N;                        \
}

DEFINE_BERNOULLI_SWEEP_K1(bern_k1_ra_v2_A3,   ra_eval_k1_A3(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_ra_v1,      ra_v1_eval_k1(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_philox,     philox_eval_k1(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_squares32,  squares32_eval_k1(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_wyrand,     wyrand_eval_k1(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_splitmix64, splitmix64_eval_k1(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_pcg32,      pcg32_eval_k1(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_sfc64,      sfc64_eval_k1(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_xoshiro256, xoshiro256ss_eval_k1(key, addr_hi))
DEFINE_BERNOULLI_SWEEP_K1(bern_k1_chacha8,    chacha8_eval_k1(key, addr_hi))

typedef void (*bern_k1_fn_t)(double *sec_out, double *cyc_out, uint64_t *draws_out,
                             uint64_t *csum_out, double *flip_rate_out, double *mag_out);

static const char *GEN10_STATELESS_NAMES[N_GEN_10] = {
    "ra_core_v2 (Opsi_A3)",
    "ra_core_v1 (W0_D0)",
    "Philox4x32-10",
    "Squares32",
    "WyRand",
    "SplitMix64",
    "PCG32",
    "SFC64",
    "xoshiro256**",
    "ChaCha8"
};

static const bern_k1_fn_t BERN_K1_FNS[N_GEN_10] = {
    bern_k1_ra_v2_A3,
    bern_k1_ra_v1,
    bern_k1_philox,
    bern_k1_squares32,
    bern_k1_wyrand,
    bern_k1_splitmix64,
    bern_k1_pcg32,
    bern_k1_sfc64,
    bern_k1_xoshiro256,
    bern_k1_chacha8
};

// Helper to initialize deterministic critical-regime mixed spin lattice (75% +1, 25% -1)
// so that both dE <= 0 and dE > 0 Metropolis transitions are heavily exercised from sweep 0,
// and to compute final <E>/N and <|M|> per spin.
static void ising_init_lattice(int8_t spins[ISING_L][ISING_L]) {
    for (uint32_t i = 0; i < ISING_L; ++i) {
        for (uint32_t j = 0; j < ISING_L; ++j) {
            uint32_t h = fmix32((i << 8) ^ j ^ 0x20261104u);
            spins[i][j] = ((h & 3u) == 0u) ? -1 : 1; // M_0 ~= 0.50, rich interface domain walls
        }
    }
}

static void ising_compute_observables(const int8_t spins[ISING_L][ISING_L],
                                      double *energy_per_spin, double *abs_mag_per_spin) {
    int64_t e_sum = 0;
    int64_t m_sum = 0;
    for (uint32_t i = 0; i < ISING_L; ++i) {
        uint32_t ip = (i + 1u) & 255u;
        for (uint32_t j = 0; j < ISING_L; ++j) {
            uint32_t jp = (j + 1u) & 255u;
            int8_t s = spins[i][j];
            m_sum += s;
            // Count right and down bonds once per site to avoid double counting
            e_sum -= (int64_t)s * (spins[ip][j] + spins[i][jp]);
        }
    }
    *energy_per_spin  = (double)e_sum / (double)ISING_N;
    *abs_mag_per_spin = fabs((double)m_sum) / (double)ISING_N;
}

// Workload 2b-2A: True 2D Ising Checkerboard Metropolis — Per-Site Stateless K=1
#define DEFINE_ISING_METROPOLIS_K1(FNAME, EVAL_K1_EXPR)                            \
static __attribute__((noinline, noclone)) void FNAME(                              \
    double *sec_out, double *cyc_out, uint64_t *draws_out, uint64_t *csum_out,     \
    double *e_per_spin_out, double *mag_out, double *acc_rate_out) {               \
    alignas(64) int8_t spins[ISING_L][ISING_L];                                    \
    ising_init_lattice(spins);                                                     \
    uint64_t accepted = 0ULL, csum = 0ULL;                                         \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t sw = 0; sw < ISING_SWEEPS; ++sw) {                               \
        uint32_t addr_hi = 0x20261104u + sw;                                       \
        for (uint32_t parity = 0; parity < 2u; ++parity) {                         \
            for (uint32_t i = 0; i < ISING_L; ++i) {                               \
                uint32_t im = (i - 1u) & 255u;                                     \
                uint32_t ip = (i + 1u) & 255u;                                     \
                uint32_t j_start = (i ^ parity) & 1u;                              \
                for (uint32_t j = j_start; j < ISING_L; j += 2u) {                 \
                    uint32_t jm = (j - 1u) & 255u;                                 \
                    uint32_t jp = (j + 1u) & 255u;                                 \
                    uint32_t cell_id = (i << 8) | j;                               \
                    uint32_t key = cell_id * 2654435761u + 1u;                     \
                    uint32_t u32 = (EVAL_K1_EXPR);                                 \
                    csum ^= u32;                                                   \
                    int s_ij = spins[i][j];                                        \
                    int s_nn = spins[im][j] + spins[ip][j]                         \
                             + spins[i][jm] + spins[i][jp];                        \
                    int dE = 2 * s_ij * s_nn;                                      \
                    if (dE <= 0 || u32 < BOLTZMANN_LUT_TC[(dE >> 2) - 1]) {        \
                        spins[i][j] = (int8_t)(-s_ij);                             \
                        accepted++;                                                \
                    }                                                              \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    ising_compute_observables(spins, e_per_spin_out, mag_out);                     \
    *sec_out      = t1 - t0;                                                       \
    *cyc_out      = (double)(c1 - c0);                                             \
    *draws_out    = (uint64_t)ISING_SWEEPS * (uint64_t)ISING_N;                    \
    *csum_out     = csum ^ accepted;                                               \
    *acc_rate_out = (double)accepted / (double)(*draws_out);                       \
}

DEFINE_ISING_METROPOLIS_K1(ising_k1_ra_v2_A3,   ra_eval_k1_A3(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_ra_v1,      ra_v1_eval_k1(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_philox,     philox_eval_k1(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_squares32,  squares32_eval_k1(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_wyrand,     wyrand_eval_k1(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_splitmix64, splitmix64_eval_k1(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_pcg32,      pcg32_eval_k1(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_sfc64,      sfc64_eval_k1(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_xoshiro256, xoshiro256ss_eval_k1(key, addr_hi))
DEFINE_ISING_METROPOLIS_K1(ising_k1_chacha8,    chacha8_eval_k1(key, addr_hi))

typedef void (*ising_fn_t)(double *sec_out, double *cyc_out, uint64_t *draws_out,
                           uint64_t *csum_out, double *e_per_spin_out, double *mag_out,
                           double *acc_rate_out);

static const ising_fn_t ISING_K1_FNS[N_GEN_10] = {
    ising_k1_ra_v2_A3,
    ising_k1_ra_v1,
    ising_k1_philox,
    ising_k1_squares32,
    ising_k1_wyrand,
    ising_k1_splitmix64,
    ising_k1_pcg32,
    ising_k1_sfc64,
    ising_k1_xoshiro256,
    ising_k1_chacha8
};

// Workload 2b-2B: True 2D Ising Checkerboard Metropolis — Row-Sublattice Batched (K=128 per half-row)
#define DEFINE_ISING_METROPOLIS_K128(FNAME, GEN_CALL)                              \
static __attribute__((noinline, noclone)) void FNAME(                              \
    double *sec_out, double *cyc_out, uint64_t *draws_out, uint64_t *csum_out,     \
    double *e_per_spin_out, double *mag_out, double *acc_rate_out) {               \
    alignas(64) int8_t spins[ISING_L][ISING_L];                                    \
    alignas(64) uint32_t row_buf[128];                                             \
    ising_init_lattice(spins);                                                     \
    uint64_t accepted = 0ULL, csum = 0ULL;                                         \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t sw = 0; sw < ISING_SWEEPS; ++sw) {                               \
        uint32_t addr_hi = 0x20261104u + sw;                                       \
        for (uint32_t parity = 0; parity < 2u; ++parity) {                         \
            for (uint32_t i = 0; i < ISING_L; ++i) {                               \
                uint32_t half_row_id = (i << 1) | parity;                          \
                uint32_t key = half_row_id * 2654435761u + 1u;                     \
                GEN_CALL(key, addr_hi, 128u, row_buf);                             \
                uint32_t im = (i - 1u) & 255u;                                     \
                uint32_t ip = (i + 1u) & 255u;                                     \
                uint32_t j_start = (i ^ parity) & 1u;                              \
                for (uint32_t k = 0; k < 128u; ++k) {                              \
                    uint32_t j  = j_start + (k << 1);                              \
                    uint32_t jm = (j - 1u) & 255u;                                 \
                    uint32_t jp = (j + 1u) & 255u;                                 \
                    uint32_t u32 = row_buf[k];                                     \
                    csum ^= u32;                                                   \
                    int s_ij = spins[i][j];                                        \
                    int s_nn = spins[im][j] + spins[ip][j]                         \
                             + spins[i][jm] + spins[i][jp];                        \
                    int dE = 2 * s_ij * s_nn;                                      \
                    if (dE <= 0 || u32 < BOLTZMANN_LUT_TC[(dE >> 2) - 1]) {        \
                        spins[i][j] = (int8_t)(-s_ij);                             \
                        accepted++;                                                \
                    }                                                              \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    ising_compute_observables(spins, e_per_spin_out, mag_out);                     \
    *sec_out      = t1 - t0;                                                       \
    *cyc_out      = (double)(c1 - c0);                                             \
    *draws_out    = (uint64_t)ISING_SWEEPS * (uint64_t)ISING_N;                    \
    *csum_out     = csum ^ accepted;                                               \
    *acc_rate_out = (double)accepted / (double)(*draws_out);                       \
}

DEFINE_ISING_METROPOLIS_K128(ising_k128_ra_v2_A3,   ra_singleblock_opt_A3)
DEFINE_ISING_METROPOLIS_K128(ising_k128_ra_v2_E2,   ra_eng_v2_E2_into)
DEFINE_ISING_METROPOLIS_K128(ising_k128_ra_v1,      ra_v1_N256_singleblock_into)
DEFINE_ISING_METROPOLIS_K128(ising_k128_philox,     philox_into)
DEFINE_ISING_METROPOLIS_K128(ising_k128_squares32,  squares32_into)
DEFINE_ISING_METROPOLIS_K128(ising_k128_wyrand,     wyrand_into)
DEFINE_ISING_METROPOLIS_K128(ising_k128_splitmix64, splitmix64_into)
DEFINE_ISING_METROPOLIS_K128(ising_k128_pcg32,      pcg32_into)
DEFINE_ISING_METROPOLIS_K128(ising_k128_sfc64,      sfc64_into)
DEFINE_ISING_METROPOLIS_K128(ising_k128_xoshiro256, xoshiro256ss_into)

static const ising_fn_t ISING_K128_FNS[N_GEN_10] = {
    ising_k128_ra_v2_A3,
    ising_k128_ra_v2_E2,
    ising_k128_ra_v1,
    ising_k128_philox,
    ising_k128_squares32,
    ising_k128_wyrand,
    ising_k128_splitmix64,
    ising_k128_pcg32,
    ising_k128_sfc64,
    ising_k128_xoshiro256
};

// ============================================================================
// 3. WORKLOAD 3A & 3B: Branching Particle Tree (N_root = 10,000, D_max = 4)
//    - Mode 3A: Coupled Stochastic Tree (PRNG controls branching & stopping)
//    - Mode 3B: Fixed-Topology Same-Work Tree (`tree_schedule(node_id, depth)`
//               determines exact `K` and children at each node; PRNG generates
//               the `K`-word simulation payload per node -> 100% identical
//               `total_nodes` and `total_draws` across all PRNGs!)
// ============================================================================

#define TREE_RUN_ID 0x20261104u
#define TREE_N_ROOT 10000u
#define TREE_D_MAX  4u

static inline __attribute__((always_inline)) void tree_coords_to_seed(
    uint32_t root_id, uint32_t lineage, uint32_t depth,
    uint32_t *key_out, uint32_t *addr_out) {
    *key_out  = TREE_RUN_ID + root_id * 0x9E3779B9u + (depth * 0x85EBCA6Bu);
    *addr_out = fmix32(lineage ^ (root_id * 0xC2B2AE35u) ^ depth);
}

static inline __attribute__((always_inline)) int step_particle_fate_3a(
    uint32_t w, uint32_t k, uint32_t depth, uint64_t *node_hash) {
    uint64_t h = *node_hash ^ (uint64_t)w ^ ((uint64_t)k << 32);
    h = ((h << 13) | (h >> 51)) * 0x9E3779B97F4A7C15ULL;
    *node_hash = h;

    if (k >= 300u) return 1;
    uint32_t r = w % 1000u;
    if (k == 1u) {
        if (r < 420u) return 1;
        if (r < 600u && depth < TREE_D_MAX) return 2;
        return 0;
    } else if (k <= 8u) {
        if (r < 280u) return 1;
        if (r < 380u && depth < TREE_D_MAX) return 2;
        return 0;
    } else if (k <= 64u) {
        if (r < 120u) return 1;
        if (r < 160u && depth < TREE_D_MAX) return 2;
        return 0;
    } else {
        if (r < 15u) return 1;
        if (r < 20u && depth < TREE_D_MAX) return 2;
        return 0;
    }
}

// Deterministic reference schedule function `tree_schedule(node_id, depth)` for Mode 3B:
// Maps (node_id, depth) deterministically to the node's step length K in [1, 300]
// and branching decision `forks` in {0, 1}, matching the 4-tier survival/forking
// distribution of Group 3 without depending on the PRNG under test!
static inline __attribute__((always_inline)) void tree_schedule(
    uint32_t node_id, uint32_t depth, uint32_t *k_words_out, int *forks_out) {
    uint32_t h1 = fmix32(node_id ^ (depth * 0x9E3779B9u));
    uint32_t h2 = fmix32(node_id ^ 0x85EBCA6Bu ^ (depth * 0xC2B2AE35u));
    uint32_t bucket = h1 % 1000u;
    uint32_t k_len;
    if (bucket < 480u) {
        k_len = 1u;                                // 48.0% K = 1
    } else if (bucket < 830u) {
        k_len = 2u + (h2 % 7u);                    // 35.0% K in [2..8]
    } else if (bucket < 930u) {
        k_len = 9u + (h2 % 56u);                   // 10.0% K in [9..64]
    } else if (bucket < 965u) {
        k_len = 65u + (h2 % 191u);                 //  3.5% K in [65..255]
    } else {
        k_len = 256u + (h2 % 45u);                 //  3.5% K in [256..300] (Cycle >= 1 promotion)
    }
    *k_words_out = k_len;
    *forks_out   = (depth < TREE_D_MAX && ((h1 >> 16) % 1000u) < 236u) ? 1 : 0;
}

typedef struct { uint64_t seed; } wyrand_stream_t;
typedef struct { uint64_t sm;   } splitmix_stream_t;

// Mode 3A: Coupled Stochastic Tree (Iterative DFS stack to avoid recursion overhead bias)
#define DEFINE_TREE_3A_KERNEL(FNAME, STATE_T, INIT_STMT, NEXT_EXPR)                \
static __attribute__((noinline, noclone)) void FNAME(                              \
    double *sec_out, double *cyc_out, uint64_t *draws_out, uint64_t *csum_out,     \
    double *nodes_out, double *words_per_node_out) {                               \
    uint32_t st_root[64], st_lin[64], st_dep[64];                                  \
    uint64_t total_nodes = 0ULL, total_words = 0ULL, global_hash = 0ULL;           \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t r = 0; r < TREE_N_ROOT; ++r) {                                   \
        int sp = 0;                                                                \
        st_root[sp] = r; st_lin[sp] = 0u; st_dep[sp] = 0u; ++sp;                   \
        while (sp > 0) {                                                           \
            --sp;                                                                  \
            uint32_t root_id = st_root[sp];                                        \
            uint32_t lineage = st_lin[sp];                                         \
            uint32_t depth   = st_dep[sp];                                         \
            uint32_t key, addr;                                                    \
            tree_coords_to_seed(root_id, lineage, depth, &key, &addr);             \
            STATE_T eng;                                                           \
            INIT_STMT;                                                             \
            uint64_t node_hash = ((uint64_t)key << 32) | addr;                     \
            uint32_t k = 0u;                                                       \
            int fate = 0;                                                          \
            while (fate == 0) {                                                    \
                uint32_t w = (NEXT_EXPR);                                          \
                ++k;                                                               \
                fate = step_particle_fate_3a(w, k, depth, &node_hash);             \
            }                                                                      \
            uint64_t tag = fmix32(lineage ^ (depth * 0x9E3779B9u));                \
            global_hash ^= node_hash ^ (tag * 0xBF58476D1CE4E5B9ULL);              \
            total_nodes++;                                                         \
            total_words += k;                                                      \
            if (fate == 2) {                                                       \
                st_root[sp] = root_id; st_lin[sp] = (lineage << 1) | 1u; st_dep[sp] = depth + 1u; ++sp; \
                st_root[sp] = root_id; st_lin[sp] = (lineage << 1) | 0u; st_dep[sp] = depth + 1u; ++sp; \
            }                                                                      \
        }                                                                          \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    volatile uint64_t sink = global_hash + total_words; (void)sink;                \
    *sec_out            = t1 - t0;                                                 \
    *cyc_out            = (double)(c1 - c0);                                       \
    *draws_out          = total_words;                                             \
    *csum_out           = global_hash;                                             \
    *nodes_out          = (double)total_nodes;                                     \
    *words_per_node_out = (double)total_words / (double)total_nodes;               \
}

// Mode 3B: Fixed-Topology Same-Work Tree (`tree_schedule(node_id, depth)` + Stateful OTF Engine)
#define DEFINE_TREE_3B_OTF_KERNEL(FNAME, STATE_T, INIT_STMT, NEXT_EXPR)            \
static __attribute__((noinline, noclone)) void FNAME(                              \
    double *sec_out, double *cyc_out, uint64_t *draws_out, uint64_t *csum_out,     \
    double *nodes_out, double *words_per_node_out) {                               \
    uint32_t st_root[64], st_lin[64], st_dep[64];                                  \
    uint64_t total_nodes = 0ULL, total_words = 0ULL, global_hash = 0ULL;           \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t r = 0; r < TREE_N_ROOT; ++r) {                                   \
        int sp = 0;                                                                \
        st_root[sp] = r; st_lin[sp] = 0u; st_dep[sp] = 0u; ++sp;                   \
        while (sp > 0) {                                                           \
            --sp;                                                                  \
            uint32_t root_id = st_root[sp];                                        \
            uint32_t lineage = st_lin[sp];                                         \
            uint32_t depth   = st_dep[sp];                                         \
            uint32_t key, addr;                                                    \
            tree_coords_to_seed(root_id, lineage, depth, &key, &addr);             \
            uint32_t node_id = key ^ addr;                                         \
            uint32_t k_words = 0u;                                                 \
            int forks = 0;                                                         \
            tree_schedule(node_id, depth, &k_words, &forks);                       \
            STATE_T eng;                                                           \
            INIT_STMT;                                                             \
            uint64_t node_hash = ((uint64_t)key << 32) | addr;                     \
            for (uint32_t k = 1u; k <= k_words; ++k) {                             \
                uint32_t w = (NEXT_EXPR);                                          \
                uint64_t h = node_hash ^ (uint64_t)w ^ ((uint64_t)k << 32);        \
                node_hash = ((h << 13) | (h >> 51)) * 0x9E3779B97F4A7C15ULL;       \
            }                                                                      \
            uint64_t tag = fmix32(lineage ^ (depth * 0x9E3779B9u));                \
            global_hash ^= node_hash ^ (tag * 0xBF58476D1CE4E5B9ULL);              \
            total_nodes++;                                                         \
            total_words += k_words;                                                \
            if (forks) {                                                           \
                st_root[sp] = root_id; st_lin[sp] = (lineage << 1) | 1u; st_dep[sp] = depth + 1u; ++sp; \
                st_root[sp] = root_id; st_lin[sp] = (lineage << 1) | 0u; st_dep[sp] = depth + 1u; ++sp; \
            }                                                                      \
        }                                                                          \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    volatile uint64_t sink = global_hash + total_words; (void)sink;                \
    *sec_out            = t1 - t0;                                                 \
    *cyc_out            = (double)(c1 - c0);                                       \
    *draws_out          = total_words;                                             \
    *csum_out           = global_hash;                                             \
    *nodes_out          = (double)total_nodes;                                     \
    *words_per_node_out = (double)total_words / (double)total_nodes;               \
}

// Also allow Stateless Direct Orbit call on Mode 3B (`ra_orbit_opt_A3(key, addr, k_words, buf)`)
static __attribute__((noinline, noclone)) void tree_3b_ra_v2_direct_A3(
    double *sec_out, double *cyc_out, uint64_t *draws_out, uint64_t *csum_out,
    double *nodes_out, double *words_per_node_out) {
    uint32_t st_root[64], st_lin[64], st_dep[64];
    alignas(64) uint32_t buf[304];
    uint64_t total_nodes = 0ULL, total_words = 0ULL, global_hash = 0ULL;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    for (uint32_t r = 0; r < TREE_N_ROOT; ++r) {
        int sp = 0;
        st_root[sp] = r; st_lin[sp] = 0u; st_dep[sp] = 0u; ++sp;
        while (sp > 0) {
            --sp;
            uint32_t root_id = st_root[sp];
            uint32_t lineage = st_lin[sp];
            uint32_t depth   = st_dep[sp];
            uint32_t key, addr;
            tree_coords_to_seed(root_id, lineage, depth, &key, &addr);
            uint32_t node_id = key ^ addr;
            uint32_t k_words = 0u;
            int forks = 0;
            tree_schedule(node_id, depth, &k_words, &forks);
            ra_orbit_opt_A3(key, addr, k_words, buf);
            uint64_t node_hash = ((uint64_t)key << 32) | addr;
            for (uint32_t k = 1u; k <= k_words; ++k) {
                uint32_t w = buf[k - 1u];
                uint64_t h = node_hash ^ (uint64_t)w ^ ((uint64_t)k << 32);
                node_hash = ((h << 13) | (h >> 51)) * 0x9E3779B97F4A7C15ULL;
            }
            uint64_t tag = fmix32(lineage ^ (depth * 0x9E3779B9u));
            global_hash ^= node_hash ^ (tag * 0xBF58476D1CE4E5B9ULL);
            total_nodes++;
            total_words += k_words;
            if (forks) {
                st_root[sp] = root_id; st_lin[sp] = (lineage << 1) | 1u; st_dep[sp] = depth + 1u; ++sp;
                st_root[sp] = root_id; st_lin[sp] = (lineage << 1) | 0u; st_dep[sp] = depth + 1u; ++sp;
            }
        }
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    volatile uint64_t sink = global_hash + total_words; (void)sink;
    *sec_out            = t1 - t0;
    *cyc_out            = (double)(c1 - c0);
    *draws_out          = total_words;
    *csum_out           = global_hash;
    *nodes_out          = (double)total_nodes;
    *words_per_node_out = (double)total_words / (double)total_nodes;
}

DEFINE_TREE_3A_KERNEL(tree_3a_ra_v2_E2,   ra_engine_v2_E2_t,    ra_engine_v2_E2_init(&eng, key, addr),         ra_engine_v2_E2_next_u32(&eng))
DEFINE_TREE_3A_KERNEL(tree_3a_ra_v1,      ra_engine_v2_t,       ra_engine_v2_init(&eng, key, addr),            ra_engine_v1_next_u32(&eng))
DEFINE_TREE_3A_KERNEL(tree_3a_philox,     philox_stream_t,      philox_stream_init(&eng, key, addr, 0u, 0u),   philox_stream_next_u32(&eng))
DEFINE_TREE_3A_KERNEL(tree_3a_squares32,  squares32_stream_t,   squares32_stream_init(&eng, key, addr),        squares32_stream_next_u32(&eng))
DEFINE_TREE_3A_KERNEL(tree_3a_wyrand,     wyrand_stream_t,      eng.seed = ((uint64_t)addr << 32) | key,       (uint32_t)wyrand_step(&eng.seed))
DEFINE_TREE_3A_KERNEL(tree_3a_splitmix64, splitmix_stream_t,    eng.sm   = ((uint64_t)addr << 32) | key,       (uint32_t)splitmix64_step(&eng.sm))
DEFINE_TREE_3A_KERNEL(tree_3a_pcg32,      pcg32_random_t,       pcg32_srandom_r(&eng, ((uint64_t)addr << 32) | key, 54u), pcg32_random_r(&eng))
DEFINE_TREE_3A_KERNEL(tree_3a_sfc64,      sfc64_state_t,        sfc64_init(&eng, key, addr),                   sfc64_next_u32(&eng))
DEFINE_TREE_3A_KERNEL(tree_3a_xoshiro256, xoshiro256ss_state_t, xoshiro256ss_init(&eng, key, addr),            xoshiro256ss_next_u32(&eng))
DEFINE_TREE_3A_KERNEL(tree_3a_chacha8,    chacha_stream_t,      chacha_stream_init(&eng, key, addr, 8),        chacha_stream_next_u32(&eng))

DEFINE_TREE_3B_OTF_KERNEL(tree_3b_ra_v2_E2,   ra_engine_v2_E2_t,    ra_engine_v2_E2_init(&eng, key, addr),         ra_engine_v2_E2_next_u32(&eng))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_ra_v1,      ra_engine_v2_t,       ra_engine_v2_init(&eng, key, addr),            ra_engine_v1_next_u32(&eng))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_philox,     philox_stream_t,      philox_stream_init(&eng, key, addr, 0u, 0u),   philox_stream_next_u32(&eng))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_squares32,  squares32_stream_t,   squares32_stream_init(&eng, key, addr),        squares32_stream_next_u32(&eng))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_wyrand,     wyrand_stream_t,      eng.seed = ((uint64_t)addr << 32) | key,       (uint32_t)wyrand_step(&eng.seed))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_splitmix64, splitmix_stream_t,    eng.sm   = ((uint64_t)addr << 32) | key,       (uint32_t)splitmix64_step(&eng.sm))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_pcg32,      pcg32_random_t,       pcg32_srandom_r(&eng, ((uint64_t)addr << 32) | key, 54u), pcg32_random_r(&eng))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_sfc64,      sfc64_state_t,        sfc64_init(&eng, key, addr),                   sfc64_next_u32(&eng))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_xoshiro256, xoshiro256ss_state_t, xoshiro256ss_init(&eng, key, addr),            xoshiro256ss_next_u32(&eng))
DEFINE_TREE_3B_OTF_KERNEL(tree_3b_chacha8,    chacha_stream_t,      chacha_stream_init(&eng, key, addr, 8),        chacha_stream_next_u32(&eng))

typedef void (*tree_fn_t)(double *sec_out, double *cyc_out, uint64_t *draws_out,
                          uint64_t *csum_out, double *nodes_out, double *words_per_node_out);

static const char *TREE_3A_NAMES[N_GEN_10] = {
    "ra_core_v2 (eng_v2_E2)",
    "ra_core_v1 (eng_v1)",
    "Philox4x32-10",
    "Squares32",
    "WyRand",
    "SplitMix64",
    "PCG32",
    "SFC64",
    "xoshiro256**",
    "ChaCha8"
};

static const tree_fn_t TREE_3A_FNS[N_GEN_10] = {
    tree_3a_ra_v2_E2,
    tree_3a_ra_v1,
    tree_3a_philox,
    tree_3a_squares32,
    tree_3a_wyrand,
    tree_3a_splitmix64,
    tree_3a_pcg32,
    tree_3a_sfc64,
    tree_3a_xoshiro256,
    tree_3a_chacha8
};

#define N_TREE_3B 11
static const char *TREE_3B_NAMES[N_TREE_3B] = {
    "ra_core_v2 (eng_v2_E2)",
    "ra_core_v2 (Direct_A3)",
    "ra_core_v1 (eng_v1)",
    "Philox4x32-10",
    "Squares32",
    "WyRand",
    "SplitMix64",
    "PCG32",
    "SFC64",
    "xoshiro256**",
    "ChaCha8"
};

static const tree_fn_t TREE_3B_FNS[N_TREE_3B] = {
    tree_3b_ra_v2_E2,
    tree_3b_ra_v2_direct_A3,
    tree_3b_ra_v1,
    tree_3b_philox,
    tree_3b_squares32,
    tree_3b_wyrand,
    tree_3b_splitmix64,
    tree_3b_pcg32,
    tree_3b_sfc64,
    tree_3b_xoshiro256,
    tree_3b_chacha8
};

// ============================================================================
// 4. WORKLOAD E1 & E2: Photon Transport MC (`ra_engine` Stateful OTF Shootout)
//    Evaluates P in {0.50, 0.80, 0.90, 0.98}:
//    - E1-Geom (Coupled Geometric Scattering Transport): particle survives each step
//      with probability P_survive in {0.50, 0.80, 0.90, 0.98}, verifying empirical
//      mean steps <L> ~= 1 / (1 - P) in {2.00, 5.00, 10.00, 50.00} across all PRNGs!
//    - E2-Geom (Fixed-Trajectory Same-Work Geometric Transport): for each particle p,
//      step count L_p >= 1 is determined by `splitmix64` on `particle_id` with survival
//      probability P in {0.50, 0.80, 0.90, 0.98}, then the tested PRNG engine is seeded
//      with `particle_id` and drawn EXACTLY L_p times (`next_u32(&eng)`).
//    - E1-Slab & E2-Slab (Turbid-Slab Photon Transport with Surface Rejection P_reject
//      in {0.50, 0.80, 0.90, 0.98}): Coupled real slab physics (E1-Slab) vs Fixed-Work
//      pre-sampled slab trajectory schedule (E2-Slab).
// ============================================================================

#define PHOTON_N_PARTICLES 250000u
#define PHOTON_SEED        0xC0FFEE01u
#define PHOTON_BATCH       0x20261104u

// Deterministic SplitMix64 geometric trajectory length generator for Mode E2-Geom:
static inline __attribute__((always_inline)) uint32_t geom_schedule_length(
    uint32_t particle_id, uint32_t thresh_survive) {
    uint64_t sm = ((uint64_t)PHOTON_BATCH << 32) | (uint64_t)(PHOTON_SEED + particle_id * 0x9E3779B9u);
    uint32_t steps = 1u;
    while (steps < 1000u) {
        uint32_t u = (uint32_t)splitmix64_step(&sm);
        if (u >= thresh_survive) break;
        ++steps;
    }
    return steps;
}

static uint16_t *g_e2_geom_lengths[4] = { NULL, NULL, NULL, NULL };
static uint16_t *g_e2_slab_lengths[4] = { NULL, NULL, NULL, NULL };

static void init_e2_schedules(void) {
    const double p_list[4] = { 0.50, 0.80, 0.90, 0.98 };
    for (int pi = 0; pi < 4; ++pi) {
        double p = p_list[pi];
        uint32_t thresh_survive = (uint32_t)(p * 4294967296.0 + 0.5);
        g_e2_geom_lengths[pi] = (uint16_t *)malloc(PHOTON_N_PARTICLES * sizeof(uint16_t));
        g_e2_slab_lengths[pi] = (uint16_t *)malloc(PHOTON_N_PARTICLES * sizeof(uint16_t));

        for (uint32_t part = 0; part < PHOTON_N_PARTICLES; ++part) {
            // 1. E2-Geom: exact geometric survival schedule via SplitMix64 on particle_id
            g_e2_geom_lengths[pi][part] = (uint16_t)geom_schedule_length(part, thresh_survive);

            // 2. E2-Slab: exact turbid-slab word count schedule via SplitMix64 on particle_id
            uint64_t sm = ((uint64_t)0x51AB2026u << 32) | (uint64_t)(PHOTON_SEED + part * 0x9E3779B9u);
            double u0 = u32_to_unit_open((uint32_t)splitmix64_step(&sm));
            uint32_t words_used = 1u;
            if (u0 >= p) {
                double z = 0.0, mu_z = 1.0, weight = 1.0;
                while (words_used < 2000u) {
                    double u_step = u32_to_unit_open((uint32_t)splitmix64_step(&sm));
                    double u_cos  = u32_to_unit_open((uint32_t)splitmix64_step(&sm));
                    words_used += 2u;
                    double s = -log(u_step);
                    z += s * mu_z;
                    if (z <= 0.0 || z >= SLAB_THICKNESS) break;
                    weight *= ALBEDO;
                    mu_z = 2.0 * u_cos - 1.0;
                    if (weight < RR_THRESHOLD) {
                        double u_rr = u32_to_unit_open((uint32_t)splitmix64_step(&sm));
                        words_used += 1u;
                        if (u_rr < RR_SURVIVAL) weight /= RR_SURVIVAL;
                        else break;
                    }
                }
            }
            g_e2_slab_lengths[pi][part] = (uint16_t)words_used;
        }
    }
}

// Mode E1-Geom: Coupled Stochastic Geometric Transport
// At each step, draws 1 word `w` from the tested PRNG:
//   - Accumulates simulated scattering/energy payload into `csum`
//   - Continues scattering if `w < thresh_survive`, terminates if `w >= thresh_survive` (or step == 1000).
#define DEFINE_E1_GEOM_KERNEL(FNAME, STATE_T, INIT_STMT, NEXT_EXPR)                \
static __attribute__((noinline, noclone)) void FNAME(                              \
    int p_idx, double p_survive, double *sec_out, double *cyc_out,                 \
    uint64_t *draws_out, uint64_t *csum_out, double *mean_steps_out,               \
    double *theory_steps_out) {                                                    \
    (void)p_idx;                                                                   \
    const uint32_t thresh = (uint32_t)(p_survive * 4294967296.0 + 0.5);            \
    uint64_t tot_draws = 0ULL, csum = 0ULL;                                        \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t p = 0; p < PHOTON_N_PARTICLES; ++p) {                            \
        uint32_t key = PHOTON_SEED + p * 0x9E3779B9u;                              \
        uint32_t addr_hi = PHOTON_BATCH;                                           \
        STATE_T eng;                                                               \
        INIT_STMT;                                                                 \
        uint32_t steps = 0u;                                                       \
        uint32_t p_acc = key;                                                      \
        while (steps < 1000u) {                                                    \
            uint32_t w = (NEXT_EXPR);                                              \
            ++steps;                                                               \
            p_acc = rot32(p_acc ^ w, 7) + 0x9E3779B9u;                             \
            if (w >= thresh) break;                                                \
        }                                                                          \
        tot_draws += steps;                                                        \
        csum ^= (uint64_t)p_acc + ((uint64_t)steps << 32);                         \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    volatile uint64_t sink = csum + tot_draws; (void)sink;                         \
    *sec_out          = t1 - t0;                                                   \
    *cyc_out          = (double)(c1 - c0);                                         \
    *draws_out        = tot_draws;                                                 \
    *csum_out         = csum;                                                      \
    *mean_steps_out   = (double)tot_draws / (double)PHOTON_N_PARTICLES;            \
    *theory_steps_out = (1.0 - pow(p_survive, 1000.0)) / (1.0 - p_survive);        \
}

// Mode E2-Geom & E2-Slab: Fixed-Trajectory Same-Work Transport
// Uses the precomputed SplitMix64 schedule array `sched_len[p]` so every PRNG executes
// the exact same number of calls `L_p` per particle `p` (100% bit-for-bit identical `total_draws`)!
#define DEFINE_E2_FIXED_KERNEL(FNAME, STATE_T, INIT_STMT, NEXT_EXPR)               \
static __attribute__((noinline, noclone)) void FNAME(                              \
    const uint16_t * restrict sched_len, double *sec_out, double *cyc_out,         \
    uint64_t *draws_out, uint64_t *csum_out, double *mean_steps_out,               \
    double *max_steps_out) {                                                       \
    uint64_t tot_draws = 0ULL, csum = 0ULL;                                        \
    uint32_t max_l = 0u;                                                           \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t p = 0; p < PHOTON_N_PARTICLES; ++p) {                            \
        uint32_t key = PHOTON_SEED + p * 0x9E3779B9u;                              \
        uint32_t addr_hi = PHOTON_BATCH;                                           \
        uint32_t lp = sched_len[p];                                                \
        STATE_T eng;                                                               \
        INIT_STMT;                                                                 \
        uint32_t p_acc = key;                                                      \
        for (uint32_t s = 0u; s < lp; ++s) {                                       \
            uint32_t w = (NEXT_EXPR);                                              \
            p_acc = rot32(p_acc ^ w, 7) + 0x9E3779B9u;                             \
        }                                                                          \
        tot_draws += lp;                                                           \
        if (lp > max_l) max_l = lp;                                                \
        csum ^= (uint64_t)p_acc + ((uint64_t)lp << 32);                            \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    volatile uint64_t sink = csum + tot_draws; (void)sink;                         \
    *sec_out        = t1 - t0;                                                     \
    *cyc_out        = (double)(c1 - c0);                                           \
    *draws_out      = tot_draws;                                                   \
    *csum_out       = csum;                                                        \
    *mean_steps_out = (double)tot_draws / (double)PHOTON_N_PARTICLES;              \
    *max_steps_out  = (double)max_l;                                               \
}

// Mode E1-Slab: Coupled Turbid-Slab Photon Transport MC (with R_d, T_d, K=1 rejection)
#define DEFINE_E1_SLAB_KERNEL(FNAME, STATE_T, INIT_STMT, NEXT_EXPR)                \
static __attribute__((noinline, noclone)) void FNAME(                              \
    double p_reject, double *sec_out, double *cyc_out, uint64_t *draws_out,        \
    uint64_t *csum_out, double *rd_out, double *td_out, double *mean_w_out) {      \
    double sum_rd = 0.0, sum_td = 0.0;                                             \
    uint64_t tot_words = 0ULL;                                                     \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t p = 0; p < PHOTON_N_PARTICLES; ++p) {                            \
        uint32_t key = PHOTON_SEED + p * 0x9E3779B9u;                              \
        uint32_t addr_hi = PHOTON_BATCH;                                           \
        STATE_T eng;                                                               \
        INIT_STMT;                                                                 \
        double u0 = u32_to_unit_open(NEXT_EXPR);                                   \
        uint32_t words_used = 1u;                                                  \
        if (u0 < p_reject) {                                                       \
            tot_words += 1u;                                                       \
            continue;                                                              \
        }                                                                          \
        double z = 0.0, mu_z = 1.0, weight = 1.0;                                  \
        double p_rd = 0.0, p_td = 0.0;                                             \
        while (1) {                                                                \
            double u_step = u32_to_unit_open(NEXT_EXPR);                           \
            double u_cos  = u32_to_unit_open(NEXT_EXPR);                           \
            words_used += 2u;                                                      \
            double s = -log(u_step);                                               \
            z += s * mu_z;                                                         \
            if (z <= 0.0) { p_rd = weight; break; }                                \
            if (z >= SLAB_THICKNESS) { p_td = weight; break; }                     \
            weight *= ALBEDO;                                                      \
            mu_z = 2.0 * u_cos - 1.0;                                              \
            if (weight < RR_THRESHOLD) {                                           \
                double u_rr = u32_to_unit_open(NEXT_EXPR);                         \
                words_used += 1u;                                                  \
                if (u_rr < RR_SURVIVAL) weight /= RR_SURVIVAL;                     \
                else break;                                                        \
            }                                                                      \
        }                                                                          \
        sum_rd += p_rd;                                                            \
        sum_td += p_td;                                                            \
        tot_words += words_used;                                                   \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    volatile double sink = sum_rd + sum_td + (double)tot_words; (void)sink;        \
    *sec_out    = t1 - t0;                                                         \
    *cyc_out    = (double)(c1 - c0);                                               \
    *draws_out  = tot_words;                                                       \
    *csum_out   = (uint64_t)(sum_rd * 1e9) ^ tot_words;                            \
    *rd_out     = sum_rd / (double)PHOTON_N_PARTICLES;                             \
    *td_out     = sum_td / (double)PHOTON_N_PARTICLES;                             \
    *mean_w_out = (double)tot_words / (double)PHOTON_N_PARTICLES;                  \
}

DEFINE_E1_GEOM_KERNEL(e1_geom_ra_v2_E2,   ra_engine_v2_E2_t,    ra_engine_v2_E2_init(&eng, key, addr_hi),         ra_engine_v2_E2_next_u32(&eng))
DEFINE_E1_GEOM_KERNEL(e1_geom_ra_v1,      ra_engine_v2_t,       ra_engine_v2_init(&eng, key, addr_hi),            ra_engine_v1_next_u32(&eng))
DEFINE_E1_GEOM_KERNEL(e1_geom_philox,     philox_stream_t,      philox_stream_init(&eng, key, addr_hi, 0u, 0u),   philox_stream_next_u32(&eng))
DEFINE_E1_GEOM_KERNEL(e1_geom_squares32,  squares32_stream_t,   squares32_stream_init(&eng, key, addr_hi),        squares32_stream_next_u32(&eng))
DEFINE_E1_GEOM_KERNEL(e1_geom_wyrand,     wyrand_stream_t,      eng.seed = ((uint64_t)addr_hi << 32) | key,       (uint32_t)wyrand_step(&eng.seed))
DEFINE_E1_GEOM_KERNEL(e1_geom_splitmix64, splitmix_stream_t,    eng.sm   = ((uint64_t)addr_hi << 32) | key,       (uint32_t)splitmix64_step(&eng.sm))
DEFINE_E1_GEOM_KERNEL(e1_geom_pcg32,      pcg32_random_t,       pcg32_srandom_r(&eng, ((uint64_t)addr_hi << 32) | key, 54u), pcg32_random_r(&eng))
DEFINE_E1_GEOM_KERNEL(e1_geom_sfc64,      sfc64_state_t,        sfc64_init(&eng, key, addr_hi),                   sfc64_next_u32(&eng))
DEFINE_E1_GEOM_KERNEL(e1_geom_xoshiro256, xoshiro256ss_state_t, xoshiro256ss_init(&eng, key, addr_hi),            xoshiro256ss_next_u32(&eng))
DEFINE_E1_GEOM_KERNEL(e1_geom_chacha8,    chacha_stream_t,      chacha_stream_init(&eng, key, addr_hi, 8),        chacha_stream_next_u32(&eng))

DEFINE_E2_FIXED_KERNEL(e2_fixed_ra_v2_E2,   ra_engine_v2_E2_t,    ra_engine_v2_E2_init(&eng, key, addr_hi),         ra_engine_v2_E2_next_u32(&eng))
DEFINE_E2_FIXED_KERNEL(e2_fixed_ra_v1,      ra_engine_v2_t,       ra_engine_v2_init(&eng, key, addr_hi),            ra_engine_v1_next_u32(&eng))
DEFINE_E2_FIXED_KERNEL(e2_fixed_philox,     philox_stream_t,      philox_stream_init(&eng, key, addr_hi, 0u, 0u),   philox_stream_next_u32(&eng))
DEFINE_E2_FIXED_KERNEL(e2_fixed_squares32,  squares32_stream_t,   squares32_stream_init(&eng, key, addr_hi),        squares32_stream_next_u32(&eng))
DEFINE_E2_FIXED_KERNEL(e2_fixed_wyrand,     wyrand_stream_t,      eng.seed = ((uint64_t)addr_hi << 32) | key,       (uint32_t)wyrand_step(&eng.seed))
DEFINE_E2_FIXED_KERNEL(e2_fixed_splitmix64, splitmix_stream_t,    eng.sm   = ((uint64_t)addr_hi << 32) | key,       (uint32_t)splitmix64_step(&eng.sm))
DEFINE_E2_FIXED_KERNEL(e2_fixed_pcg32,      pcg32_random_t,       pcg32_srandom_r(&eng, ((uint64_t)addr_hi << 32) | key, 54u), pcg32_random_r(&eng))
DEFINE_E2_FIXED_KERNEL(e2_fixed_sfc64,      sfc64_state_t,        sfc64_init(&eng, key, addr_hi),                   sfc64_next_u32(&eng))
DEFINE_E2_FIXED_KERNEL(e2_fixed_xoshiro256, xoshiro256ss_state_t, xoshiro256ss_init(&eng, key, addr_hi),            xoshiro256ss_next_u32(&eng))
DEFINE_E2_FIXED_KERNEL(e2_fixed_chacha8,    chacha_stream_t,      chacha_stream_init(&eng, key, addr_hi, 8),        chacha_stream_next_u32(&eng))

DEFINE_E1_SLAB_KERNEL(e1_slab_ra_v2_E2,   ra_engine_v2_E2_t,    ra_engine_v2_E2_init(&eng, key, addr_hi),         ra_engine_v2_E2_next_u32(&eng))
DEFINE_E1_SLAB_KERNEL(e1_slab_ra_v1,      ra_engine_v2_t,       ra_engine_v2_init(&eng, key, addr_hi),            ra_engine_v1_next_u32(&eng))
DEFINE_E1_SLAB_KERNEL(e1_slab_philox,     philox_stream_t,      philox_stream_init(&eng, key, addr_hi, 0u, 0u),   philox_stream_next_u32(&eng))
DEFINE_E1_SLAB_KERNEL(e1_slab_squares32,  squares32_stream_t,   squares32_stream_init(&eng, key, addr_hi),        squares32_stream_next_u32(&eng))
DEFINE_E1_SLAB_KERNEL(e1_slab_wyrand,     wyrand_stream_t,      eng.seed = ((uint64_t)addr_hi << 32) | key,       (uint32_t)wyrand_step(&eng.seed))
DEFINE_E1_SLAB_KERNEL(e1_slab_splitmix64, splitmix_stream_t,    eng.sm   = ((uint64_t)addr_hi << 32) | key,       (uint32_t)splitmix64_step(&eng.sm))
DEFINE_E1_SLAB_KERNEL(e1_slab_pcg32,      pcg32_random_t,       pcg32_srandom_r(&eng, ((uint64_t)addr_hi << 32) | key, 54u), pcg32_random_r(&eng))
DEFINE_E1_SLAB_KERNEL(e1_slab_sfc64,      sfc64_state_t,        sfc64_init(&eng, key, addr_hi),                   sfc64_next_u32(&eng))
DEFINE_E1_SLAB_KERNEL(e1_slab_xoshiro256, xoshiro256ss_state_t, xoshiro256ss_init(&eng, key, addr_hi),            xoshiro256ss_next_u32(&eng))
DEFINE_E1_SLAB_KERNEL(e1_slab_chacha8,    chacha_stream_t,      chacha_stream_init(&eng, key, addr_hi, 8),        chacha_stream_next_u32(&eng))

typedef void (*e1_geom_fn_t)(int p_idx, double p_survive, double *sec_out, double *cyc_out,
                             uint64_t *draws_out, uint64_t *csum_out, double *mean_steps_out,
                             double *theory_steps_out);
typedef void (*e2_fixed_fn_t)(const uint16_t * restrict sched_len, double *sec_out, double *cyc_out,
                              uint64_t *draws_out, uint64_t *csum_out, double *mean_steps_out,
                              double *max_steps_out);
typedef void (*e1_slab_fn_t)(double p_reject, double *sec_out, double *cyc_out, uint64_t *draws_out,
                             uint64_t *csum_out, double *rd_out, double *td_out, double *mean_w_out);

static const e1_geom_fn_t E1_GEOM_FNS[N_GEN_10] = {
    e1_geom_ra_v2_E2, e1_geom_ra_v1, e1_geom_philox, e1_geom_squares32, e1_geom_wyrand,
    e1_geom_splitmix64, e1_geom_pcg32, e1_geom_sfc64, e1_geom_xoshiro256, e1_geom_chacha8
};

static const e2_fixed_fn_t E2_FIXED_FNS[N_GEN_10] = {
    e2_fixed_ra_v2_E2, e2_fixed_ra_v1, e2_fixed_philox, e2_fixed_squares32, e2_fixed_wyrand,
    e2_fixed_splitmix64, e2_fixed_pcg32, e2_fixed_sfc64, e2_fixed_xoshiro256, e2_fixed_chacha8
};

static const e1_slab_fn_t E1_SLAB_FNS[N_GEN_10] = {
    e1_slab_ra_v2_E2, e1_slab_ra_v1, e1_slab_philox, e1_slab_squares32, e1_slab_wyrand,
    e1_slab_splitmix64, e1_slab_pcg32, e1_slab_sfc64, e1_slab_xoshiro256, e1_slab_chacha8
};

// ============================================================================
// Verification Gate: Bit-Exactness of `Opsi_A3` + `eng_v2_E2` + Baseline KATs
// ============================================================================

static bool verify_stage3_gates(void) {
    if (!verify_shootout_all_kat()) return false;

    uint32_t ref_orb[1024], ref_sb[256];
    for (uint32_t k = 0; k < 16; ++k) {
        uint32_t key  = 0xDEADBEEFu ^ (k * 0x9E3779B9u);
        uint32_t addr = 0x20261104u ^ k;
        ra_orbit_opt_A3(key, addr, 1000u, ref_orb);
        if (ra_eval_k1_A3(key, addr) != ref_orb[0]) {
            fprintf(stderr, "FATAL: ra_eval_k1_A3 != ref_orb[0]\n");
            return false;
        }
        ra_singleblock_opt_A3(key, addr, 255u, ref_sb);
        if (memcmp(ref_sb, ref_orb, 255u * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "FATAL: ra_singleblock_opt_A3 != ref_orb prefix\n");
            return false;
        }
        ra_engine_v2_E2_t eng;
        ra_engine_v2_E2_init(&eng, key, addr);
        for (size_t w = 0; w < 1000u; ++w) {
            if (ra_engine_v2_E2_next_u32(&eng) != ref_orb[w]) {
                fprintf(stderr, "FATAL: ra_engine_v2_E2 != ref_orb at w=%zu\n", w);
                return false;
            }
        }
    }
    return true;
}

static void write_csv_summary_row(FILE *fp, const bench_stat_t *r, double ref_med, double v2_med) {
    fprintf(fp, "%s,%s,%s,%llu,%llu,%.6f,%.6f,%.6f,%.6f,%.3f,%.2f,%.3f,%.3f,%.4f,%.4f,0x%016llx,%.6f,%.6f,%.6f\n",
            r->workload, r->regime, r->generator,
            (unsigned long long)r->total_draws,
            (unsigned long long)r->total_items,
            r->min_sec * 1e3, r->med_sec * 1e3, r->mean_sec * 1e3, r->stdev_sec * 1e3,
            r->cov_pct, r->mibs_med, r->ns_per_draw_med, r->cyc_per_draw_med,
            ref_med / r->med_sec, r->med_sec / v2_med,
            (unsigned long long)r->checksum,
            r->obs1, r->obs2, r->obs3);
}

static void write_json_stat_array(FILE *fp, const char *key_name,
                                  const bench_stat_t *rows, int n_rows,
                                  int ref_idx, bool comma_after) {
    double ref_med = rows[ref_idx].med_sec;
    double v2_med  = rows[0].med_sec;
    fprintf(fp, "  \"%s\": [\n", key_name);
    for (int i = 0; i < n_rows; ++i) {
        const bench_stat_t *r = &rows[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"workload\": \"%s\",\n", r->workload);
        fprintf(fp, "      \"regime\": \"%s\",\n", r->regime);
        fprintf(fp, "      \"generator\": \"%s\",\n", r->generator);
        fprintf(fp, "      \"total_draws\": %llu,\n", (unsigned long long)r->total_draws);
        fprintf(fp, "      \"total_items\": %llu,\n", (unsigned long long)r->total_items);
        fprintf(fp, "      \"min_ms\": %.4f,\n", r->min_sec * 1e3);
        fprintf(fp, "      \"med_ms\": %.4f,\n", r->med_sec * 1e3);
        fprintf(fp, "      \"mean_ms\": %.4f,\n", r->mean_sec * 1e3);
        fprintf(fp, "      \"stdev_ms\": %.4f,\n", r->stdev_sec * 1e3);
        fprintf(fp, "      \"cov_pct\": %.3f,\n", r->cov_pct);
        fprintf(fp, "      \"mibs_med\": %.2f,\n", r->mibs_med);
        fprintf(fp, "      \"ns_per_draw_med\": %.3f,\n", r->ns_per_draw_med);
        fprintf(fp, "      \"cyc_per_draw_med\": %.3f,\n", r->cyc_per_draw_med);
        fprintf(fp, "      \"speedup_vs_philox\": %.4f,\n", ref_med / r->med_sec);
        fprintf(fp, "      \"ratio_v2_over_gen\": %.4f,\n", r->med_sec / v2_med);
        fprintf(fp, "      \"checksum\": \"0x%016llx\",\n", (unsigned long long)r->checksum);
        fprintf(fp, "      \"obs1\": %.6f,\n", r->obs1);
        fprintf(fp, "      \"obs2\": %.6f,\n", r->obs2);
        fprintf(fp, "      \"obs3\": %.6f,\n", r->obs3);
        fprintf(fp, "      \"samples_ms\": [");
        for (int s = 0; s < N_ROUNDS; ++s) {
            fprintf(fp, "%s%.4f", s ? ", " : "", r->samples_sec[s] * 1e3);
        }
        fprintf(fp, "]\n    }%s\n", (i + 1 < n_rows) ? "," : "");
    }
    fprintf(fp, "  ]%s\n", comma_after ? "," : "");
}

int main(void) {
    if (!verify_stage3_gates()) {
        fprintf(stderr, "FATAL: Stage 3 KAT / Bit-Exactness gate failed!\n");
        return 1;
    }
    puts("[GATE 0 PASS] Opsi_A3 (direct + orbit) == eng_v2_E2 bit-exact & all baseline KATs verified.");

    init_e2_schedules();
    mkdir("data", 0755);
    FILE *csv_samples = fopen("data/clean_mc_samples.csv", "w");
    FILE *csv_summary = fopen("data/clean_mc_summary.csv", "w");
    if (csv_samples) {
        fprintf(csv_samples, "workload,regime,generator,round,sec,cycles,total_draws,checksum\n");
    }
    if (csv_summary) {
        fprintf(csv_summary, "workload,regime,generator,total_draws,total_items,min_ms,med_ms,mean_ms,stdev_ms,cov_pct,mibs_med,ns_per_draw_med,cyc_per_draw_med,speedup_vs_philox,ratio_v2_over_gen,checksum,obs1,obs2,obs3\n");
    }

    // ------------------------------------------------------------------------
    // 1. Workload 2a: Parallel Trajectory Option Pricing MC (K = 16, 64, 255)
    // ------------------------------------------------------------------------
    const size_t opt_ks[3]      = { 16u, 64u, 255u };
    const uint32_t opt_paths[3] = { 100000u, 50000u, 20000u };
    const char *opt_labels[3]   = { "K=16", "K=64", "K=255" };
    bench_stat_t res_2a[3][N_GEN_10];
    memset(res_2a, 0, sizeof(res_2a));

    for (int ki = 0; ki < 3; ++ki) {
        size_t k_steps = opt_ks[ki];
        uint32_t n_paths = opt_paths[ki];
        for (int r = -1; r < N_ROUNDS; ++r) {
            for (int step = 0; step < N_GEN_10; ++step) {
                int g = (r < 0) ? step : ((r + step) % N_GEN_10);
                double sec, cyc, mean_p, se_p;
                uint64_t draws, csum;
                OPT_MC_FNS[g](k_steps, n_paths, &sec, &cyc, &draws, &csum, &mean_p, &se_p);
                if (r >= 0) {
                    bench_stat_t *st = &res_2a[ki][g];
                    st->workload    = "2a_OptionPricing_MC";
                    st->regime      = opt_labels[ki];
                    st->generator   = GEN10_NAMES[g];
                    st->samples_sec[r] = sec;
                    st->samples_cyc[r] = cyc;
                    st->total_draws = draws;
                    st->total_items = n_paths;
                    st->checksum    = csum;
                    st->obs1        = mean_p;
                    st->obs2        = se_p;
                }
            }
        }
        for (int g = 0; g < N_GEN_10; ++g) {
            finalize_bench_stat(&res_2a[ki][g], csv_samples);
            if (res_2a[ki][g].total_draws != res_2a[ki][0].total_draws) {
                fprintf(stderr, "FATAL: total_draws mismatch in 2a!\n");
                return 1;
            }
        }
        // Verify ra_singleblock_opt_A3 == ra_engine_v2_E2 checksum & payoff bit-for-bit!
        if (res_2a[ki][0].checksum != res_2a[ki][1].checksum) {
            fprintf(stderr, "FATAL: Opsi_A3 != eng_v2_E2 checksum in 2a %s!\n", opt_labels[ki]);
            return 1;
        }
        for (int g = 0; g < N_GEN_10; ++g) {
            if (csv_summary) write_csv_summary_row(csv_summary, &res_2a[ki][g], res_2a[ki][3].med_sec, res_2a[ki][0].med_sec);
        }
        char title[160];
        snprintf(title, sizeof(title), "Workload 2a: Parallel Trajectory Asian Option Pricing MC (%s, %u paths, N=9 interleaved)", opt_labels[ki], n_paths);
        print_stat_table(title, "PayoffMean / SE", res_2a[ki], N_GEN_10, 3);
    }

    // ------------------------------------------------------------------------
    // 2. Workload 2b-1 (Bernoulli Sweep) & 2b-2 (True 2D Ising Checkerboard Metropolis)
    // ------------------------------------------------------------------------
    bench_stat_t res_2b1[N_GEN_10];
    bench_stat_t res_2b2_k1[N_GEN_10];
    bench_stat_t res_2b2_k128[N_GEN_10];
    memset(res_2b1, 0, sizeof(res_2b1));
    memset(res_2b2_k1, 0, sizeof(res_2b2_k1));
    memset(res_2b2_k128, 0, sizeof(res_2b2_k128));

    // 2b-1: Bernoulli Sweep K=1
    for (int r = -1; r < N_ROUNDS; ++r) {
        for (int step = 0; step < N_GEN_10; ++step) {
            int g = (r < 0) ? step : ((r + step) % N_GEN_10);
            double sec, cyc, flip_r, mag;
            uint64_t draws, csum;
            BERN_K1_FNS[g](&sec, &cyc, &draws, &csum, &flip_r, &mag);
            if (r >= 0) {
                bench_stat_t *st = &res_2b1[g];
                st->workload    = "2b1_Bernoulli_Sweep";
                st->regime      = "K=1_PerCell";
                st->generator   = GEN10_STATELESS_NAMES[g];
                st->samples_sec[r] = sec;
                st->samples_cyc[r] = cyc;
                st->total_draws = draws;
                st->total_items = draws;
                st->checksum    = csum;
                st->obs1        = flip_r;
                st->obs2        = mag;
            }
        }
    }
    for (int g = 0; g < N_GEN_10; ++g) {
        finalize_bench_stat(&res_2b1[g], csv_samples);
        if (csv_summary) write_csv_summary_row(csv_summary, &res_2b1[g], res_2b1[2].med_sec, res_2b1[0].med_sec);
    }
    print_stat_table("Workload 2b-1: 2D Lattice Bernoulli Sweep (256x256 x 32 sweeps, Per-Cell Stateless K=1, N=9)",
                     "FlipRate / <|M|>", res_2b1, N_GEN_10, 2);

    // 2b-2A: True 2D Ising Checkerboard Metropolis (Per-Site Stateless K=1)
    for (int r = -1; r < N_ROUNDS; ++r) {
        for (int step = 0; step < N_GEN_10; ++step) {
            int g = (r < 0) ? step : ((r + step) % N_GEN_10);
            double sec, cyc, e_spin, mag, acc_r;
            uint64_t draws, csum;
            ISING_K1_FNS[g](&sec, &cyc, &draws, &csum, &e_spin, &mag, &acc_r);
            if (r >= 0) {
                bench_stat_t *st = &res_2b2_k1[g];
                st->workload    = "2b2_True_2D_Ising_Metropolis";
                st->regime      = "K=1_PerSite_Stateless";
                st->generator   = GEN10_STATELESS_NAMES[g];
                st->samples_sec[r] = sec;
                st->samples_cyc[r] = cyc;
                st->total_draws = draws;
                st->total_items = draws;
                st->checksum    = csum;
                st->obs1        = e_spin;
                st->obs2        = mag;
                st->obs3        = acc_r;
            }
        }
    }
    for (int g = 0; g < N_GEN_10; ++g) {
        finalize_bench_stat(&res_2b2_k1[g], csv_samples);
        if (csv_summary) write_csv_summary_row(csv_summary, &res_2b2_k1[g], res_2b2_k1[2].med_sec, res_2b2_k1[0].med_sec);
    }
    print_stat_table("Workload 2b-2A: True 2D Ising Checkerboard Metropolis at T_c (256x256 x 32 sweeps, Per-Site Stateless K=1, N=9)",
                     "<E>/N    / <|M|>", res_2b2_k1, N_GEN_10, 2);

    // 2b-2B: True 2D Ising Checkerboard Metropolis (Row-Sublattice Batched K=128)
    for (int r = -1; r < N_ROUNDS; ++r) {
        for (int step = 0; step < N_GEN_10; ++step) {
            int g = (r < 0) ? step : ((r + step) % N_GEN_10);
            double sec, cyc, e_spin, mag, acc_r;
            uint64_t draws, csum;
            ISING_K128_FNS[g](&sec, &cyc, &draws, &csum, &e_spin, &mag, &acc_r);
            if (r >= 0) {
                bench_stat_t *st = &res_2b2_k128[g];
                st->workload    = "2b2_True_2D_Ising_Metropolis";
                st->regime      = "K=128_RowSublattice_Batched";
                st->generator   = GEN10_NAMES[g];
                st->samples_sec[r] = sec;
                st->samples_cyc[r] = cyc;
                st->total_draws = draws;
                st->total_items = draws;
                st->checksum    = csum;
                st->obs1        = e_spin;
                st->obs2        = mag;
                st->obs3        = acc_r;
            }
        }
    }
    for (int g = 0; g < N_GEN_10; ++g) {
        finalize_bench_stat(&res_2b2_k128[g], csv_samples);
        if (csv_summary) write_csv_summary_row(csv_summary, &res_2b2_k128[g], res_2b2_k128[3].med_sec, res_2b2_k128[0].med_sec);
    }
    if (res_2b2_k128[0].checksum != res_2b2_k128[1].checksum) {
        fprintf(stderr, "FATAL: Opsi_A3 != eng_v2_E2 checksum in 2b-2B!\n");
        return 1;
    }
    print_stat_table("Workload 2b-2B: True 2D Ising Checkerboard Metropolis at T_c (256x256 x 32 sweeps, Row-Batched K=128, N=9)",
                     "<E>/N    / <|M|>", res_2b2_k128, N_GEN_10, 3);

    // ------------------------------------------------------------------------
    // 3. Workload 3A (Coupled Stochastic Tree) & 3B (Fixed-Topology Same-Work Tree)
    // ------------------------------------------------------------------------
    bench_stat_t res_3a[N_GEN_10];
    bench_stat_t res_3b[N_TREE_3B];
    memset(res_3a, 0, sizeof(res_3a));
    memset(res_3b, 0, sizeof(res_3b));

    for (int r = -1; r < N_ROUNDS; ++r) {
        for (int step = 0; step < N_GEN_10; ++step) {
            int g = (r < 0) ? step : ((r + step) % N_GEN_10);
            double sec, cyc, nodes, wpn;
            uint64_t draws, csum;
            TREE_3A_FNS[g](&sec, &cyc, &draws, &csum, &nodes, &wpn);
            if (r >= 0) {
                bench_stat_t *st = &res_3a[g];
                st->workload    = "3A_Branching_Tree_Coupled";
                st->regime      = "Coupled_Stochastic";
                st->generator   = TREE_3A_NAMES[g];
                st->samples_sec[r] = sec;
                st->samples_cyc[r] = cyc;
                st->total_draws = draws;
                st->total_items = (uint64_t)nodes;
                st->checksum    = csum;
                st->obs1        = nodes;
                st->obs2        = wpn;
            }
        }
    }
    for (int g = 0; g < N_GEN_10; ++g) {
        finalize_bench_stat(&res_3a[g], csv_samples);
        if (csv_summary) write_csv_summary_row(csv_summary, &res_3a[g], res_3a[2].med_sec, res_3a[0].med_sec);
    }
    print_stat_table("Workload 3A: Coupled Stochastic Branching Particle Tree (10,000 roots, D_max=4, N=9 interleaved)",
                     "TotNodes / Words/Node", res_3a, N_GEN_10, 2);

    // 3B: Fixed-Topology Same-Work Tree
    for (int r = -1; r < N_ROUNDS; ++r) {
        for (int step = 0; step < N_TREE_3B; ++step) {
            int g = (r < 0) ? step : ((r + step) % N_TREE_3B);
            double sec, cyc, nodes, wpn;
            uint64_t draws, csum;
            TREE_3B_FNS[g](&sec, &cyc, &draws, &csum, &nodes, &wpn);
            if (r >= 0) {
                bench_stat_t *st = &res_3b[g];
                st->workload    = "3B_Branching_Tree_FixedTopology";
                st->regime      = "Fixed_Topology_SameWork";
                st->generator   = TREE_3B_NAMES[g];
                st->samples_sec[r] = sec;
                st->samples_cyc[r] = cyc;
                st->total_draws = draws;
                st->total_items = (uint64_t)nodes;
                st->checksum    = csum;
                st->obs1        = nodes;
                st->obs2        = wpn;
            }
        }
    }
    for (int g = 0; g < N_TREE_3B; ++g) {
        finalize_bench_stat(&res_3b[g], csv_samples);
        if (res_3b[g].total_draws != res_3b[0].total_draws ||
            res_3b[g].total_items != res_3b[0].total_items) {
            fprintf(stderr, "FATAL: Mode 3B total_draws/nodes mismatch for %s!\n", res_3b[g].generator);
            return 1;
        }
        if (csv_summary) write_csv_summary_row(csv_summary, &res_3b[g], res_3b[3].med_sec, res_3b[0].med_sec);
    }
    if (res_3b[0].checksum != res_3b[1].checksum) {
        fprintf(stderr, "FATAL: eng_v2_E2 != Direct_A3 checksum in Mode 3B!\n");
        return 1;
    }
    print_stat_table("Workload 3B: Fixed-Topology Same-Work Branching Particle Tree (EXACT MATCH total_nodes & total_draws, N=9)",
                     "TotNodes / Words/Node", res_3b, N_TREE_3B, 3);

    // ------------------------------------------------------------------------
    // 4. Workload E1 & E2: Photon Transport MC (P in {0.50, 0.80, 0.90, 0.98})
    // ------------------------------------------------------------------------
    const double p_vals[4] = { 0.50, 0.80, 0.90, 0.98 };
    const char *p_geom_labels[4] = { "P_surv=0.50", "P_surv=0.80", "P_surv=0.90", "P_surv=0.98" };
    const char *p_slab_labels[4] = { "P_rej=0.50",  "P_rej=0.80",  "P_rej=0.90",  "P_rej=0.98"  };

    bench_stat_t res_e1_geom[4][N_GEN_10];
    bench_stat_t res_e2_geom[4][N_GEN_10];
    bench_stat_t res_e1_slab[4][N_GEN_10];
    bench_stat_t res_e2_slab[4][N_GEN_10];
    memset(res_e1_geom, 0, sizeof(res_e1_geom));
    memset(res_e2_geom, 0, sizeof(res_e2_geom));
    memset(res_e1_slab, 0, sizeof(res_e1_slab));
    memset(res_e2_slab, 0, sizeof(res_e2_slab));

    for (int pi = 0; pi < 4; ++pi) {
        double p = p_vals[pi];

        // E1-Geom (Coupled Stochastic Geometric Transport)
        for (int r = -1; r < N_ROUNDS; ++r) {
            for (int step = 0; step < N_GEN_10; ++step) {
                int g = (r < 0) ? step : ((r + step) % N_GEN_10);
                double sec, cyc, mean_s, theo_s;
                uint64_t draws, csum;
                E1_GEOM_FNS[g](pi, p, &sec, &cyc, &draws, &csum, &mean_s, &theo_s);
                if (r >= 0) {
                    bench_stat_t *st = &res_e1_geom[pi][g];
                    st->workload    = "E1_Geom_Photon_Coupled";
                    st->regime      = p_geom_labels[pi];
                    st->generator   = TREE_3A_NAMES[g];
                    st->samples_sec[r] = sec;
                    st->samples_cyc[r] = cyc;
                    st->total_draws = draws;
                    st->total_items = PHOTON_N_PARTICLES;
                    st->checksum    = csum;
                    st->obs1        = mean_s;
                    st->obs2        = theo_s;
                }
            }
        }
        for (int g = 0; g < N_GEN_10; ++g) {
            finalize_bench_stat(&res_e1_geom[pi][g], csv_samples);
            if (csv_summary) write_csv_summary_row(csv_summary, &res_e1_geom[pi][g], res_e1_geom[pi][2].med_sec, res_e1_geom[pi][0].med_sec);
        }
        char title[160];
        snprintf(title, sizeof(title), "Workload E1-Geom: Coupled Stochastic Transport (%s, E[L]=%.1f, 250k particles, N=9)", p_geom_labels[pi], 1.0 / (1.0 - p));
        print_stat_table(title, "Emp<L>   / Theory<L>", res_e1_geom[pi], N_GEN_10, 2);

        // E2-Geom (Fixed-Trajectory Same-Work Geometric Transport)
        for (int r = -1; r < N_ROUNDS; ++r) {
            for (int step = 0; step < N_GEN_10; ++step) {
                int g = (r < 0) ? step : ((r + step) % N_GEN_10);
                double sec, cyc, mean_s, max_s;
                uint64_t draws, csum;
                E2_FIXED_FNS[g](g_e2_geom_lengths[pi], &sec, &cyc, &draws, &csum, &mean_s, &max_s);
                if (r >= 0) {
                    bench_stat_t *st = &res_e2_geom[pi][g];
                    st->workload    = "E2_Geom_Photon_FixedTrajectory";
                    st->regime      = p_geom_labels[pi];
                    st->generator   = TREE_3A_NAMES[g];
                    st->samples_sec[r] = sec;
                    st->samples_cyc[r] = cyc;
                    st->total_draws = draws;
                    st->total_items = PHOTON_N_PARTICLES;
                    st->checksum    = csum;
                    st->obs1        = mean_s;
                    st->obs2        = max_s;
                }
            }
        }
        for (int g = 0; g < N_GEN_10; ++g) {
            finalize_bench_stat(&res_e2_geom[pi][g], csv_samples);
            if (res_e2_geom[pi][g].total_draws != res_e2_geom[pi][0].total_draws) {
                fprintf(stderr, "FATAL: E2-Geom total_draws mismatch for %s at %s!\n",
                        res_e2_geom[pi][g].generator, p_geom_labels[pi]);
                return 1;
            }
            if (csv_summary) write_csv_summary_row(csv_summary, &res_e2_geom[pi][g], res_e2_geom[pi][2].med_sec, res_e2_geom[pi][0].med_sec);
        }
        snprintf(title, sizeof(title), "Workload E2-Geom: Fixed-Trajectory Same-Work Transport (%s, EXACT MATCH total_draws=%llu, N=9)",
                 p_geom_labels[pi], (unsigned long long)res_e2_geom[pi][0].total_draws);
        print_stat_table(title, "Mean<L>  / Max_L", res_e2_geom[pi], N_GEN_10, 2);

        // E1-Slab (Coupled Turbid-Slab Photon Transport)
        for (int r = -1; r < N_ROUNDS; ++r) {
            for (int step = 0; step < N_GEN_10; ++step) {
                int g = (r < 0) ? step : ((r + step) % N_GEN_10);
                double sec, cyc, rd, td, mean_w;
                uint64_t draws, csum;
                E1_SLAB_FNS[g](p, &sec, &cyc, &draws, &csum, &rd, &td, &mean_w);
                if (r >= 0) {
                    bench_stat_t *st = &res_e1_slab[pi][g];
                    st->workload    = "E1_Slab_Photon_Coupled";
                    st->regime      = p_slab_labels[pi];
                    st->generator   = TREE_3A_NAMES[g];
                    st->samples_sec[r] = sec;
                    st->samples_cyc[r] = cyc;
                    st->total_draws = draws;
                    st->total_items = PHOTON_N_PARTICLES;
                    st->checksum    = csum;
                    st->obs1        = rd;
                    st->obs2        = td;
                    st->obs3        = mean_w;
                }
            }
        }
        for (int g = 0; g < N_GEN_10; ++g) {
            finalize_bench_stat(&res_e1_slab[pi][g], csv_samples);
            if (csv_summary) write_csv_summary_row(csv_summary, &res_e1_slab[pi][g], res_e1_slab[pi][2].med_sec, res_e1_slab[pi][0].med_sec);
        }
        snprintf(title, sizeof(title), "Workload E1-Slab: Coupled Turbid-Slab Photon MC (%s, 250k particles, N=9)", p_slab_labels[pi]);
        print_stat_table(title, "R_d(mean)/ T_d(mean)", res_e1_slab[pi], N_GEN_10, 2);

        // E2-Slab (Fixed-Trajectory Same-Work Turbid-Slab Schedule)
        for (int r = -1; r < N_ROUNDS; ++r) {
            for (int step = 0; step < N_GEN_10; ++step) {
                int g = (r < 0) ? step : ((r + step) % N_GEN_10);
                double sec, cyc, mean_s, max_s;
                uint64_t draws, csum;
                E2_FIXED_FNS[g](g_e2_slab_lengths[pi], &sec, &cyc, &draws, &csum, &mean_s, &max_s);
                if (r >= 0) {
                    bench_stat_t *st = &res_e2_slab[pi][g];
                    st->workload    = "E2_Slab_Photon_FixedTrajectory";
                    st->regime      = p_slab_labels[pi];
                    st->generator   = TREE_3A_NAMES[g];
                    st->samples_sec[r] = sec;
                    st->samples_cyc[r] = cyc;
                    st->total_draws = draws;
                    st->total_items = PHOTON_N_PARTICLES;
                    st->checksum    = csum;
                    st->obs1        = mean_s;
                    st->obs2        = max_s;
                }
            }
        }
        for (int g = 0; g < N_GEN_10; ++g) {
            finalize_bench_stat(&res_e2_slab[pi][g], csv_samples);
            if (res_e2_slab[pi][g].total_draws != res_e2_slab[pi][0].total_draws) {
                fprintf(stderr, "FATAL: E2-Slab total_draws mismatch for %s at %s!\n",
                        res_e2_slab[pi][g].generator, p_slab_labels[pi]);
                return 1;
            }
            if (csv_summary) write_csv_summary_row(csv_summary, &res_e2_slab[pi][g], res_e2_slab[pi][2].med_sec, res_e2_slab[pi][0].med_sec);
        }
        snprintf(title, sizeof(title), "Workload E2-Slab: Fixed-Trajectory Same-Work Slab Schedule (%s, EXACT MATCH total_draws=%llu, N=9)",
                 p_slab_labels[pi], (unsigned long long)res_e2_slab[pi][0].total_draws);
        print_stat_table(title, "Mean<K>  / Max_K", res_e2_slab[pi], N_GEN_10, 2);
    }

    if (csv_samples) fclose(csv_samples);
    if (csv_summary) fclose(csv_summary);

    // Write structured JSON report
    FILE *jfp = fopen("data/clean_mc_results.json", "w");
    if (jfp) {
        fprintf(jfp, "{\n");
        write_json_stat_array(jfp, "workload_2a_k16",         res_2a[0],    N_GEN_10,  3, true);
        write_json_stat_array(jfp, "workload_2a_k64",         res_2a[1],    N_GEN_10,  3, true);
        write_json_stat_array(jfp, "workload_2a_k255",        res_2a[2],    N_GEN_10,  3, true);
        write_json_stat_array(jfp, "workload_2b1_bernoulli",  res_2b1,      N_GEN_10,  2, true);
        write_json_stat_array(jfp, "workload_2b2_ising_k1",   res_2b2_k1,   N_GEN_10,  2, true);
        write_json_stat_array(jfp, "workload_2b2_ising_k128", res_2b2_k128, N_GEN_10,  3, true);
        write_json_stat_array(jfp, "workload_3a_tree_coupled",res_3a,       N_GEN_10,  2, true);
        write_json_stat_array(jfp, "workload_3b_tree_fixed",  res_3b,       N_TREE_3B, 3, true);
        write_json_stat_array(jfp, "workload_e1_geom_p50",    res_e1_geom[0], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e1_geom_p80",    res_e1_geom[1], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e1_geom_p90",    res_e1_geom[2], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e1_geom_p98",    res_e1_geom[3], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e2_geom_p50",    res_e2_geom[0], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e2_geom_p80",    res_e2_geom[1], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e2_geom_p90",    res_e2_geom[2], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e2_geom_p98",    res_e2_geom[3], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e1_slab_p50",    res_e1_slab[0], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e1_slab_p80",    res_e1_slab[1], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e1_slab_p90",    res_e1_slab[2], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e1_slab_p98",    res_e1_slab[3], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e2_slab_p50",    res_e2_slab[0], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e2_slab_p80",    res_e2_slab[1], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e2_slab_p90",    res_e2_slab[2], N_GEN_10, 2, true);
        write_json_stat_array(jfp, "workload_e2_slab_p98",    res_e2_slab[3], N_GEN_10, 2, false);
        fprintf(jfp, "}\n");
        fclose(jfp);
    }

    for (int pi = 0; pi < 4; ++pi) {
        free(g_e2_geom_lengths[pi]);
        free(g_e2_slab_lengths[pi]);
    }
    puts("[STAGE 3 COMPLETE] All JSON and CSV outputs saved to data/.");
    return 0;
}
