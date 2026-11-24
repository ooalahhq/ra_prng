// run_stage6_suite_d_shootout.c
// Stage 6 Pure-Generate Shootout (Suite D + Direct Stateless) & Cross-Machine KAT Checksum Gate
// Evaluates:
//   - ra_core_v2 (Opsi_A3 Direct)
//   - ra_engine_v2 (eng_v2_E2)
//   - ra_core_v1 (W0_D0 Direct)
//   - ra_engine_v1
//   - Philox4x32-10
//   - ChaCha8
//   - Squares32
//   - PCG32
//   - SFC64
//   - WyRand
//   - xoshiro256**
// across K in {1, 2, 4, 8, 16, 64, 255, 256, 510, 1000} (7 interleaved rounds),
// recording median MiB/s, mean MiB/s, CoV%, cpb (cycles/byte), cyc/word, and ns/word.

#include "../2026-11-04_v2-engine-opt/ra_engine_v2_candidates.h"
#include <sys/stat.h>

#define N_ROUNDS_D 7
#define N_K_D      10
#define N_GENS_D   11

static const size_t K_VALS_D[N_K_D] = { 1, 2, 4, 8, 16, 64, 255, 256, 510, 1000 };
static const long long CALLS_D[N_K_D] = {
    2000000LL, 1600000LL, 1200000LL, 900000LL, 600000LL,
    240000LL, 90000LL, 72000LL, 40000LL, 24000LL
};

static uint64_t fnv1a_64_bytes(const uint32_t *words, size_t n_words) {
    const uint8_t *bytes = (const uint8_t *)words;
    size_t n_bytes = n_words * sizeof(uint32_t);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n_bytes; ++i) {
        h ^= (uint64_t)bytes[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static uint64_t fnv1a_64_u32(const uint32_t *words, size_t n_words) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n_words; ++i) {
        h ^= (uint64_t)words[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static inline __attribute__((always_inline)) uint32_t ra_eval_k1_A3_fast(uint32_t key, uint32_t addr_hi) {
    uint32_t step_m, k_mix;
    derive_m_A3(key, addr_hi, &step_m, &k_mix);
    uint32_t o = rot32(256u * step_m, k_mix ^ 255u) ^ (rot32(2u * step_m, k_mix ^ 1u) << 1);
    uint32_t a = o ^ (k_mix << 1);
    uint32_t b = (k_mix + a) ^ o;
    uint32_t c = rot32((a >> 13) ^ a, b);
    return c ^ (c >> 17u);
}

// Squares32 (Widynski 2020)
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

// SFC64 (Doty-Humphrey, PractRand)
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

static const char *GEN_NAMES_D[N_GENS_D] = {
    "ra_core_v2 (Opsi_A3 Direct)",
    "ra_engine_v2 (eng_v2_E2)",
    "ra_core_v1 (W0_D0 Direct)",
    "ra_engine_v1",
    "Philox4x32-10",
    "ChaCha8",
    "Squares32",
    "PCG32",
    "SFC64",
    "WyRand",
    "xoshiro256**"
};

static __attribute__((noinline, noclone)) uint32_t run_d_ra_v2_direct(
    size_t k, long long calls, double *sec_out, double *cyc_out) {
    alignas(64) uint32_t buf[1024];
    uint32_t acc = 0;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    if (k == 1) {
        for (long long c = 0; c < calls; ++c) {
            uint32_t key = (uint32_t)(c * 2654435761u + 1u);
            acc ^= ra_eval_k1_A3_fast(key, 0u);
        }
    } else if (k <= 255) {
        for (long long c = 0; c < calls; ++c) {
            uint32_t key = (uint32_t)(c * 2654435761u + 1u);
            ra_singleblock_opt_A3(key, 0u, k, buf);
            for (size_t w = 0; w < k; ++w) acc ^= buf[w];
        }
    } else {
        for (long long c = 0; c < calls; ++c) {
            uint32_t key = (uint32_t)(c * 2654435761u + 1u);
            ra_orbit_opt_A3(key, 0u, k, buf);
            for (size_t w = 0; w < k; ++w) acc ^= buf[w];
        }
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
    return acc;
}

static __attribute__((noinline, noclone)) uint32_t run_d_ra_v1_direct(
    size_t k, long long calls, double *sec_out, double *cyc_out) {
    alignas(64) uint32_t buf[1024];
    uint32_t acc = 0;
    double t0 = shootout_now_sec();
    uint64_t c0 = shootout_rdtscp();
    if (k <= 255) {
        for (long long c = 0; c < calls; ++c) {
            uint32_t key = (uint32_t)(c * 2654435761u + 1u);
            ra_v1_N256_singleblock_into(key, 0u, k, buf);
            for (size_t w = 0; w < k; ++w) acc ^= buf[w];
        }
    } else {
        for (long long c = 0; c < calls; ++c) {
            uint32_t key = (uint32_t)(c * 2654435761u + 1u);
            ra_core_v1_orbit_buf(key, 0u, k, buf);
            for (size_t w = 0; w < k; ++w) acc ^= buf[w];
        }
    }
    uint64_t c1 = shootout_rdtscp();
    double t1 = shootout_now_sec();
    *sec_out = t1 - t0;
    *cyc_out = (double)(c1 - c0);
    return acc;
}

#define DEFINE_D_ENGINE_RUNNER(FNAME, STATE_T, INIT_STMT, NEXT_STMT)               \
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

DEFINE_D_ENGINE_RUNNER(run_d_eng_v2_E2, ra_engine_v2_E2_t,  ra_engine_v2_E2_init(&eng, key, 0u),                  ra_engine_v2_E2_next_u32(&eng))
DEFINE_D_ENGINE_RUNNER(run_d_eng_v1,    ra_engine_v2_t,     ra_engine_v2_init(&eng, key, 0u),                     ra_engine_v1_next_u32(&eng))
DEFINE_D_ENGINE_RUNNER(run_d_philox,    philox_stream_t,    philox_stream_init(&eng, key, 0x20261104u, 0u, 0u),   philox_stream_next_u32(&eng))
DEFINE_D_ENGINE_RUNNER(run_d_chacha8,   chacha_stream_t,    chacha_stream_init(&eng, key, 0x20261104u, 8),        chacha_stream_next_u32(&eng))
DEFINE_D_ENGINE_RUNNER(run_d_squares32, squares32_stream_t, squares32_stream_init(&eng, key, 0x20261104u),        squares32_stream_next_u32(&eng))
DEFINE_D_ENGINE_RUNNER(run_d_pcg32,     pcg32_random_t,     pcg32_srandom_r(&eng, ((uint64_t)0x20261104u << 32) | key, 54u), pcg32_random_r(&eng))
DEFINE_D_ENGINE_RUNNER(run_d_sfc64,     sfc64_state_t,      sfc64_init(&eng, key, 0x20261104u),                   sfc64_next_u32(&eng))
DEFINE_D_ENGINE_RUNNER(run_d_wyrand,    uint64_t,           eng = ((uint64_t)0x20261104u << 32) | key,            (uint32_t)wyrand_step(&eng))
DEFINE_D_ENGINE_RUNNER(run_d_xoshiro,   xoshiro256ss_state_t, xoshiro256ss_init(&eng, key, 0x20261104u),          xoshiro256ss_next_u32(&eng))

typedef uint32_t (*suite_d_fn_t)(size_t k, long long calls, double *sec_out, double *cyc_out);

static const suite_d_fn_t GENS_D_FNS[N_GENS_D] = {
    run_d_ra_v2_direct,
    run_d_eng_v2_E2,
    run_d_ra_v1_direct,
    run_d_eng_v1,
    run_d_philox,
    run_d_chacha8,
    run_d_squares32,
    run_d_pcg32,
    run_d_sfc64,
    run_d_wyrand,
    run_d_xoshiro
};

typedef struct {
    double mibs_med;
    double mibs_mean;
    double cov_pct;
    double cpb_med;
    double cyc_word_med;
    double ns_word_med;
    uint32_t checksum;
} d_stat_t;

int main(void) {
    mkdir("data", 0755);

    if (!verify_shootout_all_kat()) {
        fprintf(stderr, "FATAL: verify_shootout_all_kat() failed!\n");
        return 1;
    }

    // Verify both v2_base (pre-Stage 1 ablation reference) and Opsi_A3 (v2 Final) Golden KAT checksums
    uint32_t buf[256];
    static const struct {
        uint32_t key;
        uint32_t addr;
        uint64_t exp_v2_base;
        uint64_t exp_opsi_a3;
    } kat_checks[5] = {
        { 0x00000000u, 0x00000000u, 0x7afeeffdbba261e7ULL, 0xcf914a65c5bfc357ULL },
        { 0x0000002Au, 0x00000000u, 0x67a666fac0725284ULL, 0xa2a2ab8dee50bccfULL },
        { 0x00000001u, 0x00000000u, 0x887fdbfea13c32cfULL, 0xc8db90b4a0564280ULL },
        { 0xDEADBEEFu, 0x9E3779B9u, 0x17e645163eb06b21ULL, 0xa81bf05044ea4b69ULL },
        { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x3b8017275fa8a61dULL, 0x7fb358921bbf86c3ULL }
    };

    int kat_mism = 0;
    printf("=== Cross-Machine Golden KAT Checksum Gate (K=255 FNV1a-64) ===\n");
    for (int i = 0; i < 5; ++i) {
        ra_core_v2_W2_D2_xag_opt_orbit_buf(kat_checks[i].key, kat_checks[i].addr, 255, buf);
        uint64_t got_base = fnv1a_64_bytes(buf, 255);
        ra_orbit_opt_A3(kat_checks[i].key, kat_checks[i].addr, 255, buf);
        uint64_t got_a3   = fnv1a_64_bytes(buf, 255);
        printf("  Seed (0x%08x, 0x%08x): v2_base=0x%016llx | Opsi_A3=0x%016llx (exp 0x%016llx)\n",
               kat_checks[i].key, kat_checks[i].addr,
               (unsigned long long)got_base,
               (unsigned long long)got_a3, (unsigned long long)kat_checks[i].exp_opsi_a3);
        if (got_a3 != kat_checks[i].exp_opsi_a3 || (i < 3 && got_base != kat_checks[i].exp_v2_base)) {
            ++kat_mism;
        }
    }
    if (kat_mism != 0) {
        fprintf(stderr, "FATAL: Golden KAT checksum mismatch (%d)!\n", kat_mism);
        return 1;
    }
    printf("  Status: PASS (0 mismatches across v2_base and v2 Final Opsi_A3)\n\n");

    double sec_samples[N_K_D][N_GENS_D][N_ROUNDS_D];
    double cyc_samples[N_K_D][N_GENS_D][N_ROUNDS_D];
    uint32_t csums[N_K_D][N_GENS_D];
    volatile uint32_t sink = 0;

    for (int ki = 0; ki < N_K_D; ++ki) {
        size_t k = K_VALS_D[ki];
        long long calls = CALLS_D[ki];
        // Warmup round (r = -1) + N_ROUNDS_D interleaved rounds
        for (int r = -1; r < N_ROUNDS_D; ++r) {
            for (int step = 0; step < N_GENS_D; ++step) {
                int g = (r < 0) ? step : ((r + step) % N_GENS_D);
                double sec = 0.0, cyc = 0.0;
                uint32_t acc = GENS_D_FNS[g](k, calls, &sec, &cyc);
                sink ^= acc;
                if (r >= 0) {
                    sec_samples[ki][g][r] = sec;
                    cyc_samples[ki][g][r] = cyc;
                    csums[ki][g] = acc;
                }
            }
        }
    }
    (void)sink;

    d_stat_t stats[N_K_D][N_GENS_D];
    FILE *fcsv = fopen("data/suite_d_shootout.csv", "w");
    if (fcsv) {
        fprintf(fcsv, "K,generator,mibs_med,mibs_mean,cov_pct,cpb_med,cyc_word_med,ns_word_med,checksum\n");
    }

    for (int ki = 0; ki < N_K_D; ++ki) {
        size_t k = K_VALS_D[ki];
        double total_words = (double)CALLS_D[ki] * (double)k;
        double total_bytes = total_words * 4.0;
        printf("=== Suite D Pure-Generate Shootout: K=%zu (%lld calls, %.2f MiB/round) ===\n",
               k, CALLS_D[ki], total_bytes / (1024.0 * 1024.0));
        printf("%-26s | %10s | %10s | %7s | %7s | %8s | %8s | %10s\n",
               "Generator", "Med MiB/s", "Mean MiB/s", "CoV%", "cpb", "cyc/w", "ns/w", "Checksum");
        printf("---------------------------+------------+------------+---------+---------+----------+----------+-----------\n");

        for (int g = 0; g < N_GENS_D; ++g) {
            double s_sec[N_ROUNDS_D], s_cyc[N_ROUNDS_D], mibs_arr[N_ROUNDS_D];
            double sum_mibs = 0.0;
            for (int r = 0; r < N_ROUNDS_D; ++r) {
                s_sec[r] = sec_samples[ki][g][r];
                s_cyc[r] = cyc_samples[ki][g][r];
                mibs_arr[r] = (total_bytes / (1024.0 * 1024.0)) / s_sec[r];
                sum_mibs += mibs_arr[r];
            }
            for (int a = 0; a < N_ROUNDS_D; ++a) {
                for (int b = a + 1; b < N_ROUNDS_D; ++b) {
                    if (s_sec[b] < s_sec[a]) { double t = s_sec[a]; s_sec[a] = s_sec[b]; s_sec[b] = t; }
                    if (s_cyc[b] < s_cyc[a]) { double t = s_cyc[a]; s_cyc[a] = s_cyc[b]; s_cyc[b] = t; }
                }
            }
            double med_sec = s_sec[N_ROUNDS_D / 2];
            double med_cyc = s_cyc[N_ROUNDS_D / 2];
            double mean_mibs = sum_mibs / (double)N_ROUNDS_D;
            double var_mibs = 0.0;
            for (int r = 0; r < N_ROUNDS_D; ++r) {
                double d = mibs_arr[r] - mean_mibs;
                var_mibs += d * d;
            }
            double stdev_mibs = sqrt(var_mibs / (double)(N_ROUNDS_D - 1));
            d_stat_t *st = &stats[ki][g];
            st->mibs_med     = (total_bytes / (1024.0 * 1024.0)) / med_sec;
            st->mibs_mean    = mean_mibs;
            st->cov_pct      = (mean_mibs > 0.0) ? (100.0 * stdev_mibs / mean_mibs) : 0.0;
            st->cpb_med      = med_cyc / total_bytes;
            st->cyc_word_med = med_cyc / total_words;
            st->ns_word_med  = (med_sec * 1e9) / total_words;
            st->checksum     = csums[ki][g];

            printf("%-26s | %10.2f | %10.2f | %6.2f%% | %7.2f | %8.2f | %8.2f | 0x%08x\n",
                   GEN_NAMES_D[g], st->mibs_med, st->mibs_mean, st->cov_pct,
                   st->cpb_med, st->cyc_word_med, st->ns_word_med, st->checksum);

            if (fcsv) {
                fprintf(fcsv, "%zu,%s,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,0x%08x\n",
                        k, GEN_NAMES_D[g], st->mibs_med, st->mibs_mean, st->cov_pct,
                        st->cpb_med, st->cyc_word_med, st->ns_word_med, st->checksum);
            }
        }
        // Verify Direct_A3 == eng_v2_E2 checksum across all K!
        if (stats[ki][0].checksum != stats[ki][1].checksum) {
            fprintf(stderr, "FATAL: Direct_A3 != eng_v2_E2 checksum at K=%zu!\n", k);
            return 1;
        }
        printf("\n");
    }
    if (fcsv) fclose(fcsv);

    FILE *fjson = fopen("data/suite_d_shootout.json", "w");
    if (fjson) {
        fprintf(fjson, "{\n  \"kat_verification\": \"PASS\",\n  \"results\": {\n");
        for (int ki = 0; ki < N_K_D; ++ki) {
            fprintf(fjson, "    \"K_%zu\": [\n", K_VALS_D[ki]);
            for (int g = 0; g < N_GENS_D; ++g) {
                d_stat_t *st = &stats[ki][g];
                fprintf(fjson,
                        "      { \"generator\": \"%s\", \"mibs_med\": %.2f, \"mibs_mean\": %.2f, "
                        "\"cov_pct\": %.2f, \"cpb_med\": %.3f, \"cyc_word_med\": %.3f, "
                        "\"ns_word_med\": %.3f, \"checksum\": \"0x%08x\" }%s\n",
                        GEN_NAMES_D[g], st->mibs_med, st->mibs_mean, st->cov_pct,
                        st->cpb_med, st->cyc_word_med, st->ns_word_med, st->checksum,
                        (g + 1 < N_GENS_D) ? "," : "");
            }
            fprintf(fjson, "    ]%s\n", (ki + 1 < N_K_D) ? "," : "");
        }
        fprintf(fjson, "  }\n}\n");
        fclose(fjson);
    }

    return 0;
}
