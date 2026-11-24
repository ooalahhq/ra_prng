// mc_workload1_photon_v2.c
// Workload 1: Variable-Length Photon Transport in Turbid Slab (`P_reject` in {0.50, 0.80, 0.90})
// + High-Rejection Rare-Event Calorimeter Trigger (`P_reject` = 0.98, 5,000,000 events).
//
// Compares `ra_core v2` (`engine_v2_W2_D2_xag_opt` 3-Stage OTF + `ra_core_v2_eager_orbit`)
// side-by-side against `ra_core v1` (`engine_v1` 3-Stage OTF + `ra_core_v1_eager_orbit`),
// `Philox4x32-10`, `ChaCha8`, `xoshiro256**`, `xoroshiro128++`, `PCG32`, and `MT19937`.
//
// Outputs summary tables to stdout and structured JSON to `data/mc_workload1_v2.json`.

#include "baselines_shootout.h"
#include <sys/stat.h>

#define N_PARTICLES   1000000
#define N_RARE_EVENTS 5000000
#define RUN_SEED      0xC0FFEE01u
#define BATCH_ID      0x20261104u

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
    double rd_mean, rd_se;
    double td_mean, td_se;
    double rd_cond_mean, rd_cond_se;
    double td_cond_mean, td_cond_se;
    uint64_t rejected_k1;
    uint64_t total_words;
    uint32_t max_words;
    double wall_min, wall_med;
    double wall_samples[5];
    double cyc_per_particle;
} photon_res_t;

// Eager Orbit controls for `v2` and `v1` (initializes L[256] + M[256] unconditionally at particle start)
typedef struct {
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256 + 8];
    uint32_t cons;
    uint32_t block_buf[256];
    uint32_t block_len;
    uint32_t block_pos;
} ra_eager_v2_t;

static inline void ra_eager_v2_init(ra_eager_v2_t *e, uint32_t key, uint32_t addr_hi) {
    v2_init_orbit_W2(e->L, e->M, key, addr_hi);
    e->cons = key;
    uint64_t count = 256;
    uint32_t *cursor = e->block_buf;
    v2_cycle_W2_D2_xag_opt(&e->cons, e->M, e->L, &count, &cursor);
    e->block_len = 255;
    e->block_pos = 0;
}

static inline uint32_t ra_eager_v2_next_u32(ra_eager_v2_t *e) {
    if (e->block_pos >= e->block_len) {
        uint64_t count = 256;
        uint32_t *cursor = e->block_buf;
        v2_cycle_W2_D2_xag_opt(&e->cons, e->M, e->L, &count, &cursor);
        e->block_len = 255;
        e->block_pos = 0;
    }
    return e->block_buf[e->block_pos++];
}

typedef struct {
    alignas(64) uint32_t L[256];
    alignas(64) uint32_t M[256];
    uint32_t cons;
    uint32_t block_buf[256];
    uint32_t block_len;
    uint32_t block_pos;
} ra_eager_v1_t;

static inline void ra_eager_v1_init(ra_eager_v1_t *e, uint32_t key, uint32_t addr_hi) {
    v1_init_state_orbit(e->L, e->M, key, addr_hi);
    e->cons = key;
    uint64_t count = 256;
    uint32_t *cursor = e->block_buf;
    v1_permutation_cycle_orbit_buf(e->cons, e->M, e->L, &count, &cursor);
    e->cons = v1_reseed(e->M, e->L);
    e->block_len = 255;
    e->block_pos = 0;
}

static inline uint32_t ra_eager_v1_next_u32(ra_eager_v1_t *e) {
    if (e->block_pos >= e->block_len) {
        uint64_t count = 256;
        uint32_t *cursor = e->block_buf;
        v1_permutation_cycle_orbit_buf(e->cons, e->M, e->L, &count, &cursor);
        e->cons = v1_reseed(e->M, e->L);
        e->block_len = 255;
        e->block_pos = 0;
    }
    return e->block_buf[e->block_pos++];
}

#define DEFINE_PHOTON_SIM(FUNC_NAME, STATE_TYPE, INIT_EXPR, NEXT_U32_EXPR)         \
static __attribute__((noinline, noclone)) void FUNC_NAME(                          \
    double p_reject, photon_res_t *res) {                                          \
    double times[5], cycs[5];                                                      \
    for (int trial = 0; trial < 5; ++trial) {                                      \
        double sum_rd = 0.0, sum_rd2 = 0.0;                                        \
        double sum_td = 0.0, sum_td2 = 0.0;                                        \
        uint64_t rej_k1 = 0, tot_words = 0;                                        \
        uint32_t max_w = 0;                                                        \
        double t0 = shootout_now_sec();                                            \
        uint64_t c0 = shootout_rdtscp();                                           \
        for (uint32_t p = 0; p < (uint32_t)N_PARTICLES; ++p) {                     \
            uint32_t key = RUN_SEED + p * 0x9E3779B9u;                             \
            uint32_t addr_hi = BATCH_ID;                                           \
            (void)key; (void)addr_hi;                                              \
            STATE_TYPE rng;                                                        \
            INIT_EXPR;                                                             \
            double u0 = u32_to_unit_open(NEXT_U32_EXPR);                           \
            uint32_t words_used = 1u;                                              \
            if (u0 < p_reject) {                                                   \
                rej_k1++;                                                          \
                tot_words += 1u;                                                   \
                if (1u > max_w) max_w = 1u;                                        \
                continue;                                                          \
            }                                                                      \
            double z = 0.0, mu_z = 1.0, weight = 1.0;                              \
            double p_rd = 0.0, p_td = 0.0;                                         \
            while (1) {                                                            \
                double u_step = u32_to_unit_open(NEXT_U32_EXPR);                   \
                double u_cos  = u32_to_unit_open(NEXT_U32_EXPR);                   \
                words_used += 2u;                                                  \
                double s = -log(u_step);                                           \
                z += s * mu_z;                                                     \
                if (z <= 0.0) { p_rd = weight; break; }                            \
                if (z >= SLAB_THICKNESS) { p_td = weight; break; }                 \
                weight *= ALBEDO;                                                  \
                mu_z = 2.0 * u_cos - 1.0;                                          \
                if (weight < RR_THRESHOLD) {                                       \
                    double u_rr = u32_to_unit_open(NEXT_U32_EXPR);                 \
                    words_used += 1u;                                              \
                    if (u_rr < RR_SURVIVAL) weight /= RR_SURVIVAL;                 \
                    else break;                                                    \
                }                                                                  \
            }                                                                      \
            sum_rd  += p_rd; sum_rd2 += p_rd * p_rd;                               \
            sum_td  += p_td; sum_td2 += p_td * p_td;                               \
            tot_words += words_used;                                               \
            if (words_used > max_w) max_w = words_used;                            \
        }                                                                          \
        uint64_t c1 = shootout_rdtscp();                                           \
        times[trial] = shootout_now_sec() - t0;                                    \
        cycs[trial]  = (double)(c1 - c0) / (double)N_PARTICLES;                    \
        volatile double anti_dce = sum_rd + sum_td + (double)tot_words;            \
        (void)anti_dce;                                                            \
        if (trial == 0) {                                                          \
            double n = (double)N_PARTICLES;                                        \
            res->rd_mean = sum_rd / n;                                             \
            res->rd_se   = sqrt(((sum_rd2 - (sum_rd * sum_rd) / n) / (n - 1.0)) / n); \
            res->td_mean = sum_td / n;                                             \
            res->td_se   = sqrt(((sum_td2 - (sum_td * sum_td) / n) / (n - 1.0)) / n); \
            double n_ent = (double)(N_PARTICLES - rej_k1);                         \
            res->rd_cond_mean = sum_rd / n_ent;                                    \
            res->rd_cond_se   = sqrt(((sum_rd2 - (sum_rd * sum_rd) / n_ent) / (n_ent - 1.0)) / n_ent); \
            res->td_cond_mean = sum_td / n_ent;                                    \
            res->td_cond_se   = sqrt(((sum_td2 - (sum_td * sum_td) / n_ent) / (n_ent - 1.0)) / n_ent); \
            res->rejected_k1 = rej_k1;                                             \
            res->total_words = tot_words;                                          \
            res->max_words   = max_w;                                              \
            res->p_reject    = p_reject;                                           \
        }                                                                          \
    }                                                                              \
    compute_min_median(times, 5, &res->wall_min, &res->wall_med);                  \
    for (int i = 0; i < 5; ++i) res->wall_samples[i] = times[i];                  \
    double cmin, cmed;                                                             \
    compute_min_median(cycs, 5, &cmin, &cmed);                                     \
    res->cyc_per_particle = cmin;                                                  \
}

DEFINE_PHOTON_SIM(sim_ra_v2_otf,   ra_engine_v2_t,         ra_engine_v2_init(&rng, key, addr_hi),    ra_engine_v2_next_u32(&rng))
DEFINE_PHOTON_SIM(sim_ra_v1_otf,   ra_engine_v2_t,         ra_engine_v2_init(&rng, key, addr_hi),    ra_engine_v1_next_u32(&rng))
DEFINE_PHOTON_SIM(sim_ra_v2_eager, ra_eager_v2_t,          ra_eager_v2_init(&rng, key, addr_hi),     ra_eager_v2_next_u32(&rng))
DEFINE_PHOTON_SIM(sim_ra_v1_eager, ra_eager_v1_t,          ra_eager_v1_init(&rng, key, addr_hi),     ra_eager_v1_next_u32(&rng))
DEFINE_PHOTON_SIM(sim_philox,      philox_stream_t,        philox_stream_init(&rng, RUN_SEED, BATCH_ID, 0u, p), philox_stream_next_u32(&rng))
DEFINE_PHOTON_SIM(sim_chacha8,     chacha_stream_t,        chacha_stream_init(&rng, key, addr_hi, 8), chacha_stream_next_u32(&rng))
DEFINE_PHOTON_SIM(sim_xoshiro256,  xoshiro256ss_state_t,   xoshiro256ss_init(&rng, key, addr_hi),    xoshiro256ss_next_u32(&rng))
DEFINE_PHOTON_SIM(sim_xoroshiro128,xoroshiro128pp_state_t, xoroshiro128pp_init(&rng, key, addr_hi),  xoroshiro128pp_next_u32(&rng))
DEFINE_PHOTON_SIM(sim_pcg32,       pcg32_random_t,         pcg32_srandom_r(&rng, ((uint64_t)addr_hi << 32) | key, 54u), pcg32_random_r(&rng))
DEFINE_PHOTON_SIM(sim_mt19937,     mt19937_state_t,        mt19937_init_genrand(&rng, key ^ addr_hi), mt19937_genrand_int32(&rng))

// Rare-Event Trigger (P_reject = 0.98, 5,000,000 events, 16-word hit reconstruction)
typedef struct {
    const char *name;
    uint64_t passed;
    uint64_t total_words;
    double wall_min, wall_med;
    double wall_samples[5];
    double cyc_per_event;
} rare_trigger_res_t;

#define DEFINE_RARE_TRIGGER_SIM(FNAME, STATE_T, INIT_EXPR, NEXT_EXPR)              \
static __attribute__((noinline, noclone)) void FNAME(rare_trigger_res_t *res) {    \
    double times[5], cycs[5];                                                      \
    const uint32_t thresh_2pct = (uint32_t)(0.02 * 4294967296.0);                  \
    for (int trial = 0; trial < 5; ++trial) {                                      \
        uint64_t p_cnt = 0, tw = 0, acc = 0;                                       \
        double t0 = shootout_now_sec();                                            \
        uint64_t c0 = shootout_rdtscp();                                           \
        for (uint32_t ev = 0; ev < (uint32_t)N_RARE_EVENTS; ++ev) {                \
            uint32_t key = 0x13579BDFu + ev * 0x9E3779B9u;                         \
            uint32_t addr_hi = BATCH_ID;                                           \
            (void)key; (void)addr_hi;                                              \
            STATE_T rng;                                                           \
            INIT_EXPR;                                                             \
            uint32_t w0 = (NEXT_EXPR);                                             \
            tw += 1u;                                                              \
            if (__builtin_expect(w0 < thresh_2pct, 0)) {                           \
                p_cnt++;                                                           \
                for (int k = 0; k < 15; ++k) acc ^= (NEXT_EXPR);                   \
                tw += 15u;                                                         \
            }                                                                      \
        }                                                                          \
        uint64_t c1 = shootout_rdtscp();                                           \
        times[trial] = shootout_now_sec() - t0;                                    \
        cycs[trial]  = (double)(c1 - c0) / (double)N_RARE_EVENTS;                  \
        volatile uint64_t sink = p_cnt + tw + acc; (void)sink;                     \
        if (trial == 0) { res->passed = p_cnt; res->total_words = tw; }            \
    }                                                                              \
    compute_min_median(times, 5, &res->wall_min, &res->wall_med);                  \
    for (int i = 0; i < 5; ++i) res->wall_samples[i] = times[i];                  \
    double cmin, cmed;                                                             \
    compute_min_median(cycs, 5, &cmin, &cmed);                                     \
    res->cyc_per_event = cmin;                                                     \
}

DEFINE_RARE_TRIGGER_SIM(rare_ra_v2_otf,    ra_engine_v2_t,         ra_engine_v2_init(&rng, key, addr_hi),    ra_engine_v2_next_u32(&rng))
DEFINE_RARE_TRIGGER_SIM(rare_ra_v1_otf,    ra_engine_v2_t,         ra_engine_v2_init(&rng, key, addr_hi),    ra_engine_v1_next_u32(&rng))
DEFINE_RARE_TRIGGER_SIM(rare_philox,       philox_stream_t,        philox_stream_init(&rng, key, addr_hi, 0u, 0u), philox_stream_next_u32(&rng))
DEFINE_RARE_TRIGGER_SIM(rare_chacha8,      chacha_stream_t,        chacha_stream_init(&rng, key, addr_hi, 8), chacha_stream_next_u32(&rng))
DEFINE_RARE_TRIGGER_SIM(rare_xoshiro256,   xoshiro256ss_state_t,   xoshiro256ss_init(&rng, key, addr_hi),    xoshiro256ss_next_u32(&rng))
DEFINE_RARE_TRIGGER_SIM(rare_xoroshiro128, xoroshiro128pp_state_t, xoroshiro128pp_init(&rng, key, addr_hi),  xoroshiro128pp_next_u32(&rng))
DEFINE_RARE_TRIGGER_SIM(rare_pcg32,        pcg32_random_t,         pcg32_srandom_r(&rng, ((uint64_t)addr_hi << 32) | key, 54u), pcg32_random_r(&rng))

static bool verify_photon_stream_parity(void) {
    const size_t lengths[] = {1, 2, 7, 8, 15, 63, 127, 255, 256, 300};
    for (uint32_t p = 0; p < 32; ++p) {
        uint32_t key = RUN_SEED + p * 0x9E3779B9u;
        for (size_t t = 0; t < sizeof(lengths) / sizeof(lengths[0]); ++t) {
            ra_engine_v2_t v2_otf, v1_otf;
            ra_eager_v2_t v2_eager;
            ra_eager_v1_t v1_eager;
            ra_engine_v2_init(&v2_otf, key, BATCH_ID);
            ra_engine_v2_init(&v1_otf, key, BATCH_ID);
            ra_eager_v2_init(&v2_eager, key, BATCH_ID);
            ra_eager_v1_init(&v1_eager, key, BATCH_ID);
            for (size_t w = 0; w < lengths[t]; ++w) {
                if (ra_engine_v2_next_u32(&v2_otf) != ra_eager_v2_next_u32(&v2_eager) ||
                    ra_engine_v1_next_u32(&v1_otf) != ra_eager_v1_next_u32(&v1_eager)) {
                    fprintf(stderr, "FATAL: photon OTF/eager stream mismatch at p=%u K=%zu word=%zu\n",
                            p, lengths[t], w);
                    return false;
                }
            }
        }
    }
    return true;
}

int main(void) {
    if (!verify_shootout_all_kat() || !verify_photon_stream_parity()) {
        fprintf(stderr, "FATAL: KAT verification failed!\n");
        return 1;
    }
    puts("[PHOTON STREAM PARITY] v1 and v2 OTF/eager word streams match: PASS");

    const double p_list[3] = { 0.50, 0.80, 0.90 };
    photon_res_t results[3][10];

    for (int pi = 0; pi < 3; ++pi) {
        double p_rej = p_list[pi];
        results[pi][0].name = "ra_core_v2_otf (3-Stage W2_D2)";
        results[pi][1].name = "ra_core_v1_otf (3-Stage W0_D0)";
        results[pi][2].name = "ra_core_v2_orbit (Eager Init)";
        results[pi][3].name = "ra_core_v1_orbit (Eager Init)";
        results[pi][4].name = "Philox4x32-10";
        results[pi][5].name = "ChaCha8";
        results[pi][6].name = "xoshiro256**";
        results[pi][7].name = "xoroshiro128++";
        results[pi][8].name = "PCG32";
        results[pi][9].name = "MT19937";

        sim_ra_v2_otf(p_rej,    &results[pi][0]);
        sim_ra_v1_otf(p_rej,    &results[pi][1]);
        sim_ra_v2_eager(p_rej,  &results[pi][2]);
        sim_ra_v1_eager(p_rej,  &results[pi][3]);
        sim_philox(p_rej,       &results[pi][4]);
        sim_chacha8(p_rej,      &results[pi][5]);
        sim_xoshiro256(p_rej,   &results[pi][6]);
        sim_xoroshiro128(p_rej, &results[pi][7]);
        sim_pcg32(p_rej,        &results[pi][8]);
        sim_mt19937(p_rej,      &results[pi][9]);

        // Verify bit-exact parity between OTF and Eager for both v2 and v1
        if (results[pi][0].rejected_k1 != results[pi][2].rejected_k1 ||
            results[pi][0].total_words != results[pi][2].total_words ||
            results[pi][0].max_words != results[pi][2].max_words ||
            fabs(results[pi][0].rd_mean - results[pi][2].rd_mean) > 1e-15 ||
            fabs(results[pi][0].td_mean - results[pi][2].td_mean) > 1e-15) {
            fprintf(stderr, "FATAL: v2 OTF != v2 Eager at P_reject=%.2f\n", p_rej);
            return 1;
        }
        if (results[pi][1].rejected_k1 != results[pi][3].rejected_k1 ||
            results[pi][1].total_words != results[pi][3].total_words ||
            results[pi][1].max_words != results[pi][3].max_words ||
            fabs(results[pi][1].rd_mean - results[pi][3].rd_mean) > 1e-15 ||
            fabs(results[pi][1].td_mean - results[pi][3].td_mean) > 1e-15) {
            fprintf(stderr, "FATAL: v1 OTF != v1 Eager at P_reject=%.2f\n", p_rej);
            return 1;
        }

        printf("=================================================================================================================\n");
        printf("Workload 1: Photon Transport in Turbid Slab (1,000,000 particles, P_reject = %.2f)\n", p_rej);
        printf("=================================================================================================================\n");
        printf("%-32s | %9s | %9s | %8s | %10s | %5s | %8s | %8s | %9s\n",
               "Generator", "R_d(mean)", "T_d(mean)", "K=1 Rej%", "TotalWords", "Max K", "min(5)s", "med(5)s", "cyc/part");
        printf("---------------------------------+-----------+-----------+----------+------------+-------+----------+----------+----------\n");
        for (int g = 0; g < 10; ++g) {
            photon_res_t *r = &results[pi][g];
            printf("%-32s | %9.6f | %9.6f | %7.2f%% | %10llu | %5u | %7.4fs | %7.4fs | %8.1f\n",
                   r->name, r->rd_mean, r->td_mean,
                   100.0 * (double)r->rejected_k1 / (double)N_PARTICLES,
                   (unsigned long long)r->total_words, r->max_words,
                   r->wall_min, r->wall_med, r->cyc_per_particle);
        }
        printf("\n");
    }

    rare_trigger_res_t rare_res[7] = {
        { .name = "ra_core_v2_otf (3-Stage W2_D2)" },
        { .name = "ra_core_v1_otf (3-Stage W0_D0)" },
        { .name = "Philox4x32-10" },
        { .name = "ChaCha8" },
        { .name = "xoshiro256**" },
        { .name = "xoroshiro128++" },
        { .name = "PCG32" }
    };
    rare_ra_v2_otf(&rare_res[0]);
    rare_ra_v1_otf(&rare_res[1]);
    rare_philox(&rare_res[2]);
    rare_chacha8(&rare_res[3]);
    rare_xoshiro256(&rare_res[4]);
    rare_xoroshiro128(&rare_res[5]);
    rare_pcg32(&rare_res[6]);

    printf("=================================================================================================================\n");
    printf("Workload 1B: Rare-Event Calorimeter Trigger (5,000,000 events, P_reject = 0.98, 16-word shower on hit)\n");
    printf("=================================================================================================================\n");
    printf("%-32s | %10s | %12s | %9s | %9s | %9s\n",
           "Generator", "Passed", "TotalWords", "min(5) s", "med(5) s", "cyc/event");
    printf("---------------------------------+------------+--------------+-----------+-----------+----------\n");
    for (int g = 0; g < 7; ++g) {
        printf("%-32s | %10llu | %12llu | %8.4fs | %8.4fs | %8.1f\n",
               rare_res[g].name,
               (unsigned long long)rare_res[g].passed,
               (unsigned long long)rare_res[g].total_words,
               rare_res[g].wall_min, rare_res[g].wall_med, rare_res[g].cyc_per_event);
    }

    mkdir("data", 0755);
    FILE *fp = fopen("data/mc_workload1_v2.json", "w");
    if (fp) {
        fprintf(fp, "{\n  \"photon_transport\": [\n");
        for (int pi = 0; pi < 3; ++pi) {
            fprintf(fp, "    {\n      \"p_reject\": %.2f,\n      \"generators\": [\n", p_list[pi]);
            for (int g = 0; g < 10; ++g) {
                photon_res_t *r = &results[pi][g];
                fprintf(fp, "        {\"name\": \"%s\", \"rd_mean\": %.6f, \"rd_se\": %.6f, \"td_mean\": %.6f, \"td_se\": %.6f, \"rejected_k1\": %llu, \"total_words\": %llu, \"max_words\": %u, \"wall_min\": %.5f, \"wall_med\": %.5f, \"cyc_per_particle\": %.2f, \"samples_s\": [",
                        r->name, r->rd_mean, r->rd_se, r->td_mean, r->td_se,
                        (unsigned long long)r->rejected_k1, (unsigned long long)r->total_words, r->max_words,
                        r->wall_min, r->wall_med, r->cyc_per_particle);
                for (int i = 0; i < 5; ++i) fprintf(fp, "%s%.12g", i ? ", " : "", r->wall_samples[i]);
                fprintf(fp, "]}%s\n", (g + 1 < 10) ? "," : "");
            }
            fprintf(fp, "      ]\n    }%s\n", (pi + 1 < 3) ? "," : "");
        }
        fprintf(fp, "  ],\n  \"rare_event_trigger\": [\n");
        for (int g = 0; g < 7; ++g) {
            fprintf(fp, "    {\"name\": \"%s\", \"passed\": %llu, \"total_words\": %llu, \"wall_min\": %.5f, \"wall_med\": %.5f, \"cyc_per_event\": %.2f, \"samples_s\": [",
                    rare_res[g].name, (unsigned long long)rare_res[g].passed, (unsigned long long)rare_res[g].total_words,
                    rare_res[g].wall_min, rare_res[g].wall_med, rare_res[g].cyc_per_event);
            for (int i = 0; i < 5; ++i) fprintf(fp, "%s%.12g", i ? ", " : "", rare_res[g].wall_samples[i]);
            fprintf(fp, "]}%s\n", (g + 1 < 7) ? "," : "");
        }
        fprintf(fp, "  ]\n}\n");
        fclose(fp);
    }
    return 0;
}
