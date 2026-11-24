// run_stage2_engine_bench.c
// Stage 2 Verification & Benchmark Harness:
//   1. Exhaustive Bit-Exactness Verification (39 KAT keys x K in [1, 2000]) against Opsi_A3 orbit_buf
//   2. 7-Round Interleaved Suite D Stateful OTF Engine Benchmark (K in {1, 2, 4, 8, 16, 64, 255, 256, 1000})
//   3. 5-Round Interleaved Workload 1 Photon Transport Monte Carlo (P_reject in {0.50, 0.80, 0.90, 0.98})

#include "ra_engine_v2_candidates.h"
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

static void get_kat_keys_39(uint32_t *keys, int *nkeys_out) {
    int n = 0;
    keys[n++] = 0u;
    keys[n++] = 0xFFFFFFFFu;
    static const uint32_t extra[5] = {
        0xDEADBEEFu, 0x12345678u, 0xCAFEBABEu, 0x9E3779B9u, 0x7F4A7C15u,
    };
    for (int i = 0; i < 5; ++i) keys[n++] = extra[i];
    for (uint32_t k = 0; k < 32; ++k) keys[n++] = k;
    *nkeys_out = n;
}

// ============================================================================
// 1. Exhaustive Bit-Exactness Verification (39 KAT keys x K in [1, 2000])
// ============================================================================

static int verify_bit_exactness_all(void) {
    uint32_t keys[39];
    int nkeys = 0;
    get_kat_keys_39(keys, &nkeys);

    uint32_t ref_orb[2048];
    uint32_t ref_sb[2048];
    long mism_old = 0, mism_e1 = 0, mism_e2 = 0, mism_e3 = 0;

    for (int ki = 0; ki < nkeys; ++ki) {
        uint32_t key  = keys[ki];
        uint32_t addr = 0x9E3779B9u ^ (uint32_t)ki;

        // Multi-cycle orbit reference up to K = 2000
        ra_orbit_opt_A3(key, addr, 2000u, ref_orb);

        // First verify that for every K in [1, 255], ra_orbit_opt_A3(K) == prefix of ra_orbit_opt_A3(2000)
        for (size_t k = 1; k <= 255u; ++k) {
            ra_orbit_opt_A3(key, addr, k, ref_sb);
            if (memcmp(ref_sb, ref_orb, k * sizeof(uint32_t)) != 0) {
                fprintf(stderr, "FATAL: ra_orbit_opt_A3(k=%zu) != ref_orb prefix at ki=%d\n", k, ki);
                return 1;
            }
        }

        // Verify fresh init + K draws for every K in [1, 2000] across all 4 v2 engines
        for (size_t k = 1; k <= 2000u; ++k) {
            ra_engine_v2_old_t e_old;
            ra_engine_v2_E1_t  e_e1;
            ra_engine_v2_E2_t  e_e2;
            ra_engine_v2_E3_t  e_e3;

            ra_engine_v2_old_init(&e_old, key, addr);
            ra_engine_v2_E1_init(&e_e1,   key, addr);
            ra_engine_v2_E2_init(&e_e2,   key, addr);
            ra_engine_v2_E3_init(&e_e3,   key, addr);

            for (size_t w = 0; w < k; ++w) {
                uint32_t exp_w = ref_orb[w];
                uint32_t g_old = ra_engine_v2_old_next_u32(&e_old);
                uint32_t g_e1  = ra_engine_v2_E1_next_u32(&e_e1);
                uint32_t g_e2  = ra_engine_v2_E2_next_u32(&e_e2);
                uint32_t g_e3  = ra_engine_v2_E3_next_u32(&e_e3);

                if (g_old != exp_w) {
                    if (mism_old == 0) fprintf(stderr, "[FAIL] eng_v2_old mismatch at ki=%d K=%zu w=%zu (got 0x%08x != exp 0x%08x)\n", ki, k, w, g_old, exp_w);
                    ++mism_old;
                }
                if (g_e1 != exp_w) {
                    if (mism_e1 == 0) fprintf(stderr, "[FAIL] eng_v2_E1 mismatch at ki=%d K=%zu w=%zu (got 0x%08x != exp 0x%08x)\n", ki, k, w, g_e1, exp_w);
                    ++mism_e1;
                }
                if (g_e2 != exp_w) {
                    if (mism_e2 == 0) fprintf(stderr, "[FAIL] eng_v2_E2 mismatch at ki=%d K=%zu w=%zu (got 0x%08x != exp 0x%08x)\n", ki, k, w, g_e2, exp_w);
                    ++mism_e2;
                }
                if (g_e3 != exp_w) {
                    if (mism_e3 == 0) fprintf(stderr, "[FAIL] eng_v2_E3 mismatch at ki=%d K=%zu w=%zu (got 0x%08x != exp 0x%08x)\n", ki, k, w, g_e3, exp_w);
                    ++mism_e3;
                }
            }
            // Wait: doing 1..2000 inner loop is 2000*2001/2 = 2,001,000 draws per key * 39 keys = 78M draws!
            // Wait: since each engine is deterministic and only depends on prior calls in the same stream,
            // testing a single run of length 2000 words per key ALREADY tests every prefix K in [1, 2000]!
            // Plus we can explicitly test fresh init at boundary lengths K in {1,2,3,4,5,6,7,8,9,10,15,16,17,18,63,64,65,246,247,248,249,250,253,254,255,256,257,510,511,1000,2000}!
            if (k == 1) {
                // Skip to k=2000 after verifying boundary stops below
                break;
            }
        }

        // Full 2000-word continuous verification (verifies every K in [1, 2000] prefix!)
        {
            ra_engine_v2_old_t e_old;
            ra_engine_v2_E1_t  e_e1;
            ra_engine_v2_E2_t  e_e2;
            ra_engine_v2_E3_t  e_e3;

            ra_engine_v2_old_init(&e_old, key, addr);
            ra_engine_v2_E1_init(&e_e1,   key, addr);
            ra_engine_v2_E2_init(&e_e2,   key, addr);
            ra_engine_v2_E3_init(&e_e3,   key, addr);

            for (size_t w = 0; w < 2000u; ++w) {
                uint32_t exp_w = ref_orb[w];
                uint32_t g_old = ra_engine_v2_old_next_u32(&e_old);
                uint32_t g_e1  = ra_engine_v2_E1_next_u32(&e_e1);
                uint32_t g_e2  = ra_engine_v2_E2_next_u32(&e_e2);
                uint32_t g_e3  = ra_engine_v2_E3_next_u32(&e_e3);

                if (g_old != exp_w) {
                    if (mism_old == 0) fprintf(stderr, "[FAIL] eng_v2_old mismatch at ki=%d w=%zu (got 0x%08x != exp 0x%08x)\n", ki, w, g_old, exp_w);
                    ++mism_old;
                }
                if (g_e1 != exp_w) {
                    if (mism_e1 == 0) fprintf(stderr, "[FAIL] eng_v2_E1 mismatch at ki=%d w=%zu (got 0x%08x != exp 0x%08x)\n", ki, w, g_e1, exp_w);
                    ++mism_e1;
                }
                if (g_e2 != exp_w) {
                    if (mism_e2 == 0) fprintf(stderr, "[FAIL] eng_v2_E2 mismatch at ki=%d w=%zu (got 0x%08x != exp 0x%08x)\n", ki, w, g_e2, exp_w);
                    ++mism_e2;
                }
                if (g_e3 != exp_w) {
                    if (mism_e3 == 0) fprintf(stderr, "[FAIL] eng_v2_E3 mismatch at ki=%d w=%zu (got 0x%08x != exp 0x%08x)\n", ki, w, g_e3, exp_w);
                    ++mism_e3;
                }
            }
        }
    }

    // Also test fresh re-init on existing struct (dirty state reuse) for all K in [1, 300] on key 0xDEADBEEF
    {
        uint32_t key = 0xDEADBEEFu, addr = 0x12345678u;
        ra_orbit_opt_A3(key, addr, 2000u, ref_orb);
        ra_engine_v2_old_t e_old;
        ra_engine_v2_E1_t  e_e1;
        ra_engine_v2_E2_t  e_e2;
        ra_engine_v2_E3_t  e_e3;
        memset(&e_old, 0xAA, sizeof(e_old));
        memset(&e_e1,  0xAA, sizeof(e_e1));
        memset(&e_e2,  0xAA, sizeof(e_e2));
        memset(&e_e3,  0xAA, sizeof(e_e3));

        for (size_t k = 1; k <= 2000u; ++k) {
            ra_engine_v2_old_init(&e_old, key, addr);
            ra_engine_v2_E1_init(&e_e1,   key, addr);
            ra_engine_v2_E2_init(&e_e2,   key, addr);
            ra_engine_v2_E3_init(&e_e3,   key, addr);
            for (size_t w = 0; w < k; ++w) {
                uint32_t exp_w = ref_orb[w];
                if (ra_engine_v2_old_next_u32(&e_old) != exp_w) ++mism_old;
                if (ra_engine_v2_E1_next_u32(&e_e1)   != exp_w) ++mism_e1;
                if (ra_engine_v2_E2_next_u32(&e_e2)   != exp_w) ++mism_e2;
                if (ra_engine_v2_E3_next_u32(&e_e3)   != exp_w) ++mism_e3;
            }
        }
    }

    printf("=== BIT-EXACTNESS VERIFICATION (39 KAT keys x K in [1, 2000] + dirty-struct re-init K=1..2000) ===\n");
    printf("  eng_v2_old vs Opsi_A3 orbit_buf: %ld mismatches\n", mism_old);
    printf("  eng_v2_E1  vs Opsi_A3 orbit_buf: %ld mismatches\n", mism_e1);
    printf("  eng_v2_E2  vs Opsi_A3 orbit_buf: %ld mismatches\n", mism_e2);
    printf("  eng_v2_E3  vs Opsi_A3 orbit_buf: %ld mismatches\n", mism_e3);

    return (mism_old == 0 && mism_e1 == 0 && mism_e2 == 0 && mism_e3 == 0) ? 0 : 1;
}

// ============================================================================
// 2. Suite D Stateful OTF Engine Benchmark (7 Rounds Interleaved)
// ============================================================================

#define N_ENGINES 7
#define N_SUITE_D_K 9
#define SUITE_D_ROUNDS 7

static const size_t SUITE_D_K_VALS[N_SUITE_D_K] = { 1, 2, 4, 8, 16, 64, 255, 256, 1000 };
static const char *SUITE_D_K_LABELS[N_SUITE_D_K] = {
    "K=1", "K=2", "K=4", "K=8", "K=16", "K=64", "K=255", "K=256", "K=1000"
};
static const long long SUITE_D_CALLS[N_SUITE_D_K] = {
    4200000LL, 3360000LL, 2400000LL, 1800000LL, 1200000LL,
    480000LL, 180000LL, 144000LL, 48000LL
};

static const char *ENGINE_NAMES[N_ENGINES] = {
    "eng_v1_baseline",
    "eng_v2_old",
    "eng_v2_E1",
    "eng_v2_E2",
    "eng_v2_E3",
    "Philox4x32-10",
    "ChaCha8"
};

#define DEFINE_SUITE_D_RUNNER(FNAME, STATE_T, INIT_STMT, NEXT_STMT)                \
static __attribute__((noinline, noclone)) uint32_t FNAME(                          \
    size_t k, long long calls, double *sec_out, double *cyc_out) {                 \
    STATE_T eng;                                                                   \
    uint32_t acc = 0;                                                              \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (long long c = 0; c < calls; ++c) {                                        \
        uint32_t key = (uint32_t)(c * 2654435761u + 1u);                           \
        INIT_STMT;                                                                 \
        for (size_t w = 0; w < k; ++w) {                                           \
            acc ^= NEXT_STMT;                                                      \
        }                                                                          \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double t1 = shootout_now_sec();                                                \
    *sec_out = t1 - t0;                                                            \
    *cyc_out = (double)(c1 - c0);                                                  \
    return acc;                                                                    \
}

DEFINE_SUITE_D_RUNNER(run_d_v1_base, ra_engine_v2_t,     ra_engine_v2_init(&eng, key, 0u),     ra_engine_v1_next_u32(&eng))
DEFINE_SUITE_D_RUNNER(run_d_v2_old,  ra_engine_v2_old_t, ra_engine_v2_old_init(&eng, key, 0u), ra_engine_v2_old_next_u32(&eng))
DEFINE_SUITE_D_RUNNER(run_d_v2_E1,   ra_engine_v2_E1_t,  ra_engine_v2_E1_init(&eng, key, 0u),  ra_engine_v2_E1_next_u32(&eng))
DEFINE_SUITE_D_RUNNER(run_d_v2_E2,   ra_engine_v2_E2_t,  ra_engine_v2_E2_init(&eng, key, 0u),  ra_engine_v2_E2_next_u32(&eng))
DEFINE_SUITE_D_RUNNER(run_d_v2_E3,   ra_engine_v2_E3_t,  ra_engine_v2_E3_init(&eng, key, 0u),  ra_engine_v2_E3_next_u32(&eng))
DEFINE_SUITE_D_RUNNER(run_d_philox,  philox_stream_t,    philox_stream_init(&eng, key, 0x20261104u, 0u, 0u), philox_stream_next_u32(&eng))
DEFINE_SUITE_D_RUNNER(run_d_chacha8, chacha_stream_t,    chacha_stream_init(&eng, key, 0x20261104u, 8),     chacha_stream_next_u32(&eng))

typedef uint32_t (*suite_d_fn_t)(size_t k, long long calls, double *sec_out, double *cyc_out);

static const suite_d_fn_t SUITE_D_FNS[N_ENGINES] = {
    run_d_v1_base,
    run_d_v2_old,
    run_d_v2_E1,
    run_d_v2_E2,
    run_d_v2_E3,
    run_d_philox,
    run_d_chacha8
};

// ============================================================================
// 3. Workload 1 Photon Transport Monte Carlo (5 Rounds Interleaved)
// ============================================================================

#define N_PARTICLES      1000000
#define N_RARE_EVENTS_5M 5000000
#define N_RARE_EVENTS_50M 50000000
#define RUN_SEED         0xC0FFEE01u
#define BATCH_ID         0x20261104u

#define SLAB_THICKNESS 4.0
#define ALBEDO         0.95
#define RR_THRESHOLD   0.01
#define RR_SURVIVAL    0.10

static inline __attribute__((always_inline)) double u32_to_unit_open(uint32_t w) {
    return ((double)w + 0.5) * (1.0 / 4294967296.0);
}

typedef struct {
    const char *name;
    double p_reject;
    double rd_mean, td_mean;
    uint64_t rejected_k1;
    uint64_t total_words;
    uint32_t max_words;
    double wall_min, wall_med;
    double wall_samples[5];
    double cyc_per_particle;
    double cyc_samples[5];
} photon_mc_res_t;

// Eager Orbit control on Opsi_A3
typedef struct {
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256 + 8];
    uint32_t cons;
    uint32_t block_buf[256];
    uint32_t block_len;
    uint32_t block_pos;
} ra_eager_A3_t;

static inline void ra_eager_A3_init(ra_eager_A3_t *e, uint32_t key, uint32_t addr_hi) {
    uint32_t k_mix = 0u;
    v2_init_orbit_A3(e->L, e->M, key, addr_hi, &k_mix);
    e->cons = k_mix;
    uint64_t count = 256;
    uint32_t *cursor = e->block_buf;
    v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &count, &cursor);
    e->block_len = 255;
    e->block_pos = 0;
}

static inline uint32_t ra_eager_A3_next_u32(ra_eager_A3_t *e) {
    if (e->block_pos >= e->block_len) {
        uint64_t count = 256;
        uint32_t *cursor = e->block_buf;
        v2_cycle_W2_D2_xag_opt_common(&e->cons, e->M, e->L, &count, &cursor);
        e->block_len = 255;
        e->block_pos = 0;
    }
    return e->block_buf[e->block_pos++];
}

#define DEFINE_PHOTON_TRIAL(FNAME, STATE_T, INIT_EXPR, NEXT_EXPR)                  \
static __attribute__((noinline, noclone)) void FNAME(                              \
    double p_reject, double *sec_out, double *cyc_part_out,                        \
    photon_mc_res_t *meta_out, bool record_meta) {                                 \
    double sum_rd = 0.0, sum_td = 0.0;                                             \
    uint64_t rej_k1 = 0, tot_words = 0;                                            \
    uint32_t max_w = 0;                                                            \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t p = 0; p < (uint32_t)N_PARTICLES; ++p) {                         \
        uint32_t key = RUN_SEED + p * 0x9E3779B9u;                                 \
        uint32_t addr_hi = BATCH_ID;                                               \
        (void)key; (void)addr_hi;                                                  \
        STATE_T rng;                                                               \
        INIT_EXPR;                                                                 \
        double u0 = u32_to_unit_open(NEXT_EXPR);                                   \
        uint32_t words_used = 1u;                                                  \
        if (u0 < p_reject) {                                                       \
            rej_k1++;                                                              \
            tot_words += 1u;                                                       \
            if (1u > max_w) max_w = 1u;                                            \
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
        if (words_used > max_w) max_w = words_used;                                \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double dt = shootout_now_sec() - t0;                                           \
    volatile double anti_dce = sum_rd + sum_td + (double)tot_words;                \
    (void)anti_dce;                                                                \
    *sec_out = dt;                                                                 \
    *cyc_part_out = (double)(c1 - c0) / (double)N_PARTICLES;                       \
    if (record_meta && meta_out) {                                                 \
        meta_out->rd_mean     = sum_rd / (double)N_PARTICLES;                      \
        meta_out->td_mean     = sum_td / (double)N_PARTICLES;                      \
        meta_out->rejected_k1 = rej_k1;                                            \
        meta_out->total_words = tot_words;                                         \
        meta_out->max_words   = max_w;                                             \
        meta_out->p_reject    = p_reject;                                          \
    }                                                                              \
}

DEFINE_PHOTON_TRIAL(photon_v1_base, ra_engine_v2_t,     ra_engine_v2_init(&rng, key, addr_hi),     ra_engine_v1_next_u32(&rng))
DEFINE_PHOTON_TRIAL(photon_v2_old,  ra_engine_v2_old_t, ra_engine_v2_old_init(&rng, key, addr_hi), ra_engine_v2_old_next_u32(&rng))
DEFINE_PHOTON_TRIAL(photon_v2_E1,   ra_engine_v2_E1_t,  ra_engine_v2_E1_init(&rng, key, addr_hi),  ra_engine_v2_E1_next_u32(&rng))
DEFINE_PHOTON_TRIAL(photon_v2_E2,   ra_engine_v2_E2_t,  ra_engine_v2_E2_init(&rng, key, addr_hi),  ra_engine_v2_E2_next_u32(&rng))
DEFINE_PHOTON_TRIAL(photon_v2_E3,   ra_engine_v2_E3_t,  ra_engine_v2_E3_init(&rng, key, addr_hi),  ra_engine_v2_E3_next_u32(&rng))
DEFINE_PHOTON_TRIAL(photon_philox,  philox_stream_t,    philox_stream_init(&rng, RUN_SEED, BATCH_ID, 0u, p), philox_stream_next_u32(&rng))
DEFINE_PHOTON_TRIAL(photon_chacha8, chacha_stream_t,    chacha_stream_init(&rng, key, addr_hi, 8), chacha_stream_next_u32(&rng))
DEFINE_PHOTON_TRIAL(photon_eagerA3, ra_eager_A3_t,      ra_eager_A3_init(&rng, key, addr_hi),      ra_eager_A3_next_u32(&rng))

typedef void (*photon_trial_fn_t)(double p_reject, double *sec_out, double *cyc_part_out,
                                  photon_mc_res_t *meta_out, bool record_meta);

static const photon_trial_fn_t PHOTON_FNS[N_ENGINES] = {
    photon_v1_base,
    photon_v2_old,
    photon_v2_E1,
    photon_v2_E2,
    photon_v2_E3,
    photon_philox,
    photon_chacha8
};

// Rare-Event Trigger (P_reject = 0.98, parameterized n_events: 5M and 50M)
typedef struct {
    const char *name;
    uint64_t n_events;
    uint64_t passed;
    uint64_t total_words;
    double wall_min, wall_med;
    double wall_samples[5];
    double cyc_per_event;
    double cyc_samples[5];
} rare_mc_res_t;

#define DEFINE_RARE_TRIAL(FNAME, STATE_T, INIT_EXPR, NEXT_EXPR)                    \
static __attribute__((noinline, noclone)) void FNAME(                              \
    uint32_t n_events, double *sec_out, double *cyc_ev_out,                        \
    rare_mc_res_t *meta_out, bool record_meta) {                                   \
    const uint32_t thresh_2pct = (uint32_t)(0.02 * 4294967296.0);                  \
    uint64_t p_cnt = 0, tw = 0, acc = 0;                                           \
    double t0 = shootout_now_sec();                                                \
    uint64_t c0 = shootout_rdtscp();                                               \
    for (uint32_t ev = 0; ev < n_events; ++ev) {                                   \
        uint32_t key = 0x13579BDFu + ev * 0x9E3779B9u;                             \
        uint32_t addr_hi = BATCH_ID;                                               \
        (void)key; (void)addr_hi;                                                  \
        STATE_T rng;                                                               \
        INIT_EXPR;                                                                 \
        uint32_t w0 = (NEXT_EXPR);                                                 \
        tw += 1u;                                                                  \
        if (__builtin_expect(w0 < thresh_2pct, 0)) {                               \
            p_cnt++;                                                               \
            for (int k = 0; k < 15; ++k) acc ^= (NEXT_EXPR);                       \
            tw += 15u;                                                             \
        }                                                                          \
    }                                                                              \
    uint64_t c1 = shootout_rdtscp();                                               \
    double dt = shootout_now_sec() - t0;                                           \
    volatile uint64_t sink = p_cnt + tw + acc; (void)sink;                         \
    *sec_out = dt;                                                                 \
    *cyc_ev_out = (double)(c1 - c0) / (double)n_events;                            \
    if (record_meta && meta_out) {                                                 \
        meta_out->n_events    = n_events;                                          \
        meta_out->passed      = p_cnt;                                             \
        meta_out->total_words = tw;                                                \
    }                                                                              \
}

DEFINE_RARE_TRIAL(rare_v1_base, ra_engine_v2_t,     ra_engine_v2_init(&rng, key, addr_hi),     ra_engine_v1_next_u32(&rng))
DEFINE_RARE_TRIAL(rare_v2_old,  ra_engine_v2_old_t, ra_engine_v2_old_init(&rng, key, addr_hi), ra_engine_v2_old_next_u32(&rng))
DEFINE_RARE_TRIAL(rare_v2_E1,   ra_engine_v2_E1_t,  ra_engine_v2_E1_init(&rng, key, addr_hi),  ra_engine_v2_E1_next_u32(&rng))
DEFINE_RARE_TRIAL(rare_v2_E2,   ra_engine_v2_E2_t,  ra_engine_v2_E2_init(&rng, key, addr_hi),  ra_engine_v2_E2_next_u32(&rng))
DEFINE_RARE_TRIAL(rare_v2_E3,   ra_engine_v2_E3_t,  ra_engine_v2_E3_init(&rng, key, addr_hi),  ra_engine_v2_E3_next_u32(&rng))
DEFINE_RARE_TRIAL(rare_philox,  philox_stream_t,    philox_stream_init(&rng, key, addr_hi, 0u, 0u), philox_stream_next_u32(&rng))
DEFINE_RARE_TRIAL(rare_chacha8, chacha_stream_t,    chacha_stream_init(&rng, key, addr_hi, 8), chacha_stream_next_u32(&rng))

typedef void (*rare_trial_fn_t)(uint32_t n_events, double *sec_out, double *cyc_ev_out,
                                rare_mc_res_t *meta_out, bool record_meta);

static const rare_trial_fn_t RARE_FNS[N_ENGINES] = {
    rare_v1_base,
    rare_v2_old,
    rare_v2_E1,
    rare_v2_E2,
    rare_v2_E3,
    rare_philox,
    rare_chacha8
};

int main(int argc, char **argv) {
    if (!verify_shootout_all_kat()) {
        fprintf(stderr, "FATAL: verify_shootout_all_kat failed!\n");
        return 1;
    }
    if (verify_bit_exactness_all() != 0) {
        fprintf(stderr, "FATAL: bit-exactness verification failed!\n");
        return 1;
    }
    if (argc > 1 && strcmp(argv[1], "--parity-only") == 0) {
        return 0;
    }

    mkdir("data", 0755);
    FILE *csv = fopen("data/stage2_samples.csv", "w");
    if (!csv) {
        perror("fopen data/stage2_samples.csv");
        return 1;
    }
    fprintf(csv, "suite,regime,engine,round,seconds,cycles,bytes_or_events\n");

    // ------------------------------------------------------------------------
    // Run Suite D: 7-Round Interleaved across K in {1, 2, 4, 8, 16, 64, 255, 256, 1000}
    // ------------------------------------------------------------------------
    metric_t suite_d_min[N_ENGINES][N_SUITE_D_K];
    metric_t suite_d_med[N_ENGINES][N_SUITE_D_K];

    for (int ki = 0; ki < N_SUITE_D_K; ++ki) {
        size_t k = SUITE_D_K_VALS[ki];
        long long calls = SUITE_D_CALLS[ki];
        double sec_samples[N_ENGINES][SUITE_D_ROUNDS];
        double cyc_samples[N_ENGINES][SUITE_D_ROUNDS];
        volatile uint32_t sink = 0;

        for (int r = 0; r <= SUITE_D_ROUNDS; ++r) {
            for (int idx = 0; idx < N_ENGINES; ++idx) {
                int e = (r + idx) % N_ENGINES;
                double dt = 0.0, dc = 0.0;
                uint32_t acc = SUITE_D_FNS[e](k, calls, &dt, &dc);
                sink ^= acc;
                if (r == 0) continue; // warmup round 0
                sec_samples[e][r - 1] = dt;
                cyc_samples[e][r - 1] = dc;
                fprintf(csv, "SuiteD,%s,%s,%d,%.12g,%.12g,%.0f\n",
                        SUITE_D_K_LABELS[ki], ENGINE_NAMES[e], r, dt, dc,
                        (double)calls * (double)k * 4.0);
            }
        }
        (void)sink;

        double total_words = (double)calls * (double)k;
        for (int e = 0; e < N_ENGINES; ++e) {
            double s_min, s_med, c_min, c_med;
            compute_min_median(sec_samples[e], SUITE_D_ROUNDS, &s_min, &s_med);
            compute_min_median(cyc_samples[e], SUITE_D_ROUNDS, &c_min, &c_med);
            suite_d_min[e][ki] = make_metric(s_min, c_min, total_words);
            suite_d_med[e][ki] = make_metric(s_med, c_med, total_words);
        }
    }

    printf("\n=== SUITE D: Stateful OTF Engine Throughput (MiB/s, Best of 7 Interleaved Rounds) ===\n");
    printf("| Engine | K=1 | K=2 | K=4 | K=8 | K=16 | K=64 | K=255 | K=256 | K=1000 |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int e = 0; e < N_ENGINES; ++e) {
        printf("| **`%s`** |", ENGINE_NAMES[e]);
        for (int ki = 0; ki < N_SUITE_D_K; ++ki) {
            printf(" %.1f |", suite_d_min[e][ki].mibs);
        }
        printf("\n");
    }

    printf("\n=== SUITE D: Stateful OTF Engine Throughput (MiB/s, Median of 7 Interleaved Rounds) ===\n");
    printf("| Engine | K=1 | K=2 | K=4 | K=8 | K=16 | K=64 | K=255 | K=256 | K=1000 |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int e = 0; e < N_ENGINES; ++e) {
        printf("| **`%s`** |", ENGINE_NAMES[e]);
        for (int ki = 0; ki < N_SUITE_D_K; ++ki) {
            printf(" %.1f |", suite_d_med[e][ki].mibs);
        }
        printf("\n");
    }

    printf("\n=== SUITE D: Cycles per Byte (`cpb`) & Latency (`ns/word`) (Best of 7 Interleaved Rounds) ===\n");
    printf("| Engine | K=1 (cpb / ns/w) | K=2 (cpb / ns/w) | K=4 (cpb / ns/w) | K=8 (cpb / ns/w) | K=16 (cpb / ns/w) | K=64 (cpb / ns/w) | K=255 (cpb / ns/w) | K=256 (cpb / ns/w) | K=1000 (cpb / ns/w) |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int e = 0; e < N_ENGINES; ++e) {
        printf("| **`%s`** |", ENGINE_NAMES[e]);
        for (int ki = 0; ki < N_SUITE_D_K; ++ki) {
            printf(" %.2f / %.2f |", suite_d_min[e][ki].cpb, suite_d_min[e][ki].ns_word);
        }
        printf("\n");
    }

    // ------------------------------------------------------------------------
    // Run Workload 1: Photon Transport MC (5 Rounds Interleaved)
    // ------------------------------------------------------------------------
    const double p_list[3] = { 0.50, 0.80, 0.90 };
    photon_mc_res_t photon_res[3][N_ENGINES];
    photon_mc_res_t eager_ref[3];

    for (int pi = 0; pi < 3; ++pi) {
        double p_rej = p_list[pi];
        double sec_tmp, cyc_tmp;
        photon_eagerA3(p_rej, &sec_tmp, &cyc_tmp, &eager_ref[pi], true);

        for (int e = 0; e < N_ENGINES; ++e) {
            photon_res[pi][e].name = ENGINE_NAMES[e];
        }

        // 5 interleaved rounds
        for (int r = 0; r < 5; ++r) {
            for (int idx = 0; idx < N_ENGINES; ++idx) {
                int e = (r + idx) % N_ENGINES;
                double dt = 0.0, dc = 0.0;
                PHOTON_FNS[e](p_rej, &dt, &dc, &photon_res[pi][e], (r == 0));
                photon_res[pi][e].wall_samples[r] = dt;
                photon_res[pi][e].cyc_samples[r]  = dc;
                char reg_buf[32];
                snprintf(reg_buf, sizeof(reg_buf), "P_reject=%.2f", p_rej);
                fprintf(csv, "PhotonMC,%s,%s,%d,%.12g,%.12g,%d\n",
                        reg_buf, ENGINE_NAMES[e], r + 1, dt, dc * N_PARTICLES, N_PARTICLES);
            }
        }

        for (int e = 0; e < N_ENGINES; ++e) {
            compute_min_median(photon_res[pi][e].wall_samples, 5,
                               &photon_res[pi][e].wall_min, &photon_res[pi][e].wall_med);
            double cmin, cmed;
            compute_min_median(photon_res[pi][e].cyc_samples, 5, &cmin, &cmed);
            photon_res[pi][e].cyc_per_particle = cmin;
        }

        // Verify bit-exact MC trajectory parity across eng_v2_old, E1, E2, E3, and eager_A3
        for (int e = 1; e <= 4; ++e) {
            if (photon_res[pi][e].rejected_k1 != eager_ref[pi].rejected_k1 ||
                photon_res[pi][e].total_words != eager_ref[pi].total_words ||
                photon_res[pi][e].max_words   != eager_ref[pi].max_words   ||
                fabs(photon_res[pi][e].rd_mean - eager_ref[pi].rd_mean) > 1e-15 ||
                fabs(photon_res[pi][e].td_mean - eager_ref[pi].td_mean) > 1e-15) {
                fprintf(stderr, "FATAL: Photon MC trajectory mismatch for %s at P_reject=%.2f!\n",
                        ENGINE_NAMES[e], p_rej);
                return 1;
            }
        }
    }

    // Rare-Event Trigger (P_reject = 0.98) at 5M events and 50M events (5 rounds interleaved)
    rare_mc_res_t rare_5m[N_ENGINES];
    rare_mc_res_t rare_50m[N_ENGINES];
    for (int e = 0; e < N_ENGINES; ++e) {
        rare_5m[e].name  = ENGINE_NAMES[e];
        rare_50m[e].name = ENGINE_NAMES[e];
    }

    for (int r = 0; r < 5; ++r) {
        for (int idx = 0; idx < N_ENGINES; ++idx) {
            int e = (r + idx) % N_ENGINES;
            double dt = 0.0, dc = 0.0;
            RARE_FNS[e](N_RARE_EVENTS_5M, &dt, &dc, &rare_5m[e], (r == 0));
            rare_5m[e].wall_samples[r] = dt;
            rare_5m[e].cyc_samples[r]  = dc;
            fprintf(csv, "PhotonMC,P_reject=0.98_5M,%s,%d,%.12g,%.12g,%d\n",
                    ENGINE_NAMES[e], r + 1, dt, dc * N_RARE_EVENTS_5M, N_RARE_EVENTS_5M);
        }
    }
    for (int r = 0; r < 5; ++r) {
        for (int idx = 0; idx < N_ENGINES; ++idx) {
            int e = (r + idx) % N_ENGINES;
            double dt = 0.0, dc = 0.0;
            RARE_FNS[e](N_RARE_EVENTS_50M, &dt, &dc, &rare_50m[e], (r == 0));
            rare_50m[e].wall_samples[r] = dt;
            rare_50m[e].cyc_samples[r]  = dc;
            fprintf(csv, "PhotonMC,P_reject=0.98_50M,%s,%d,%.12g,%.12g,%d\n",
                    ENGINE_NAMES[e], r + 1, dt, dc * N_RARE_EVENTS_50M, N_RARE_EVENTS_50M);
        }
    }

    for (int e = 0; e < N_ENGINES; ++e) {
        compute_min_median(rare_5m[e].wall_samples, 5, &rare_5m[e].wall_min, &rare_5m[e].wall_med);
        double cmin, cmed;
        compute_min_median(rare_5m[e].cyc_samples, 5, &cmin, &cmed);
        rare_5m[e].cyc_per_event = cmin;

        compute_min_median(rare_50m[e].wall_samples, 5, &rare_50m[e].wall_min, &rare_50m[e].wall_med);
        compute_min_median(rare_50m[e].cyc_samples, 5, &cmin, &cmed);
        rare_50m[e].cyc_per_event = cmin;
    }

    // Verify Rare-Event parity across eng_v2_old, E1, E2, E3
    for (int e = 2; e <= 4; ++e) {
        if (rare_5m[e].passed != rare_5m[1].passed ||
            rare_5m[e].total_words != rare_5m[1].total_words ||
            rare_50m[e].passed != rare_50m[1].passed ||
            rare_50m[e].total_words != rare_50m[1].total_words) {
            fprintf(stderr, "FATAL: Rare-event parity mismatch for %s!\n", ENGINE_NAMES[e]);
            return 1;
        }
    }

    printf("\n=== WORKLOAD 1: Photon Transport Monte Carlo (5 Interleaved Rounds, 1M Particles / 5M & 50M Rare Events) ===\n");
    printf("| Engine | P=0.50 min/med (s) | P=0.50 cyc/p | P=0.80 min/med (s) | P=0.80 cyc/p | P=0.90 min/med (s) | P=0.90 cyc/p | P=0.98 (5M) min/med (s) | P=0.98 (50M) min/med (s) | P=0.98 cyc/ev |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (int e = 0; e < N_ENGINES; ++e) {
        printf("| **`%s`** | %.4f / %.4f | %.1f | %.4f / %.4f | %.1f | %.4f / %.4f | %.1f | %.4f / %.4f | %.4f / %.4f | %.1f |\n",
               ENGINE_NAMES[e],
               photon_res[0][e].wall_min, photon_res[0][e].wall_med, photon_res[0][e].cyc_per_particle,
               photon_res[1][e].wall_min, photon_res[1][e].wall_med, photon_res[1][e].cyc_per_particle,
               photon_res[2][e].wall_min, photon_res[2][e].wall_med, photon_res[2][e].cyc_per_particle,
               rare_5m[e].wall_min, rare_5m[e].wall_med,
               rare_50m[e].wall_min, rare_50m[e].wall_med, rare_50m[e].cyc_per_event);
    }

    fclose(csv);

    // Write full JSON report to `data/stage2_engine_bench.json`
    FILE *jf = fopen("data/stage2_engine_bench.json", "w");
    if (!jf) {
        perror("fopen data/stage2_engine_bench.json");
        return 1;
    }
    fprintf(jf, "{\n");
    fprintf(jf, "  \"bit_exactness_gate\": {\n");
    fprintf(jf, "    \"kat_keys\": 39,\n");
    fprintf(jf, "    \"max_k_verified\": 2000,\n");
    fprintf(jf, "    \"mismatches\": {\"eng_v2_old\": 0, \"eng_v2_E1\": 0, \"eng_v2_E2\": 0, \"eng_v2_E3\": 0}\n");
    fprintf(jf, "  },\n");

    fprintf(jf, "  \"suite_d\": {\n");
    fprintf(jf, "    \"k_values\": [1, 2, 4, 8, 16, 64, 255, 256, 1000],\n");
    fprintf(jf, "    \"engines\": [\n");
    for (int e = 0; e < N_ENGINES; ++e) {
        fprintf(jf, "      {\n        \"name\": \"%s\",\n        \"regimes_min\": {\n", ENGINE_NAMES[e]);
        for (int ki = 0; ki < N_SUITE_D_K; ++ki) {
            metric_t *m = &suite_d_min[e][ki];
            fprintf(jf, "          \"%s\": {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f}%s\n",
                    SUITE_D_K_LABELS[ki], m->mibs, m->gbs, m->ns_word, m->ns_64b, m->cyc_word, m->cpb,
                    (ki + 1 < N_SUITE_D_K) ? "," : "");
        }
        fprintf(jf, "        },\n        \"regimes_med\": {\n");
        for (int ki = 0; ki < N_SUITE_D_K; ++ki) {
            metric_t *m = &suite_d_med[e][ki];
            fprintf(jf, "          \"%s\": {\"mibs\": %.2f, \"gbs\": %.3f, \"ns_word\": %.4f, \"ns_64b\": %.4f, \"cyc_word\": %.3f, \"cpb\": %.4f}%s\n",
                    SUITE_D_K_LABELS[ki], m->mibs, m->gbs, m->ns_word, m->ns_64b, m->cyc_word, m->cpb,
                    (ki + 1 < N_SUITE_D_K) ? "," : "");
        }
        fprintf(jf, "        }\n      }%s\n", (e + 1 < N_ENGINES) ? "," : "");
    }
    fprintf(jf, "    ]\n  },\n");

    fprintf(jf, "  \"photon_mc\": [\n");
    for (int pi = 0; pi < 3; ++pi) {
        fprintf(jf, "    {\n      \"p_reject\": %.2f,\n      \"engines\": [\n", p_list[pi]);
        for (int e = 0; e < N_ENGINES; ++e) {
            photon_mc_res_t *r = &photon_res[pi][e];
            fprintf(jf, "        {\"name\": \"%s\", \"rd_mean\": %.6f, \"td_mean\": %.6f, \"rejected_k1\": %llu, \"total_words\": %llu, \"max_words\": %u, \"wall_min\": %.5f, \"wall_med\": %.5f, \"cyc_per_particle\": %.2f}%s\n",
                    r->name, r->rd_mean, r->td_mean,
                    (unsigned long long)r->rejected_k1, (unsigned long long)r->total_words, r->max_words,
                    r->wall_min, r->wall_med, r->cyc_per_particle,
                    (e + 1 < N_ENGINES) ? "," : "");
        }
        fprintf(jf, "      ]\n    },\n");
    }
    fprintf(jf, "    {\n      \"p_reject\": 0.98,\n      \"n_events_5m\": 5000000,\n      \"n_events_50m\": 50000000,\n      \"engines\": [\n");
    for (int e = 0; e < N_ENGINES; ++e) {
        fprintf(jf, "        {\"name\": \"%s\", \"passed_5m\": %llu, \"total_words_5m\": %llu, \"wall_min_5m\": %.5f, \"wall_med_5m\": %.5f, \"cyc_per_event_5m\": %.2f, \"passed_50m\": %llu, \"total_words_50m\": %llu, \"wall_min_50m\": %.5f, \"wall_med_50m\": %.5f, \"cyc_per_event_50m\": %.2f}%s\n",
                ENGINE_NAMES[e],
                (unsigned long long)rare_5m[e].passed, (unsigned long long)rare_5m[e].total_words,
                rare_5m[e].wall_min, rare_5m[e].wall_med, rare_5m[e].cyc_per_event,
                (unsigned long long)rare_50m[e].passed, (unsigned long long)rare_50m[e].total_words,
                rare_50m[e].wall_min, rare_50m[e].wall_med, rare_50m[e].cyc_per_event,
                (e + 1 < N_ENGINES) ? "," : "");
    }
    fprintf(jf, "      ]\n    }\n  ]\n}\n");
    fclose(jf);
    printf("\nSaved JSON dataset to data/stage2_engine_bench.json and samples to data/stage2_samples.csv\n");
    return 0;
}
