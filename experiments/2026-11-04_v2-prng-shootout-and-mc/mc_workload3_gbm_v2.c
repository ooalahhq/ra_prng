// mc_workload3_gbm_v2.c
// Workload 3: Long-Trajectory Steady-State Asian Call Option GBM
// (1,000 paths x 10,000 timesteps = 10,000,000 words, S0=100, K=100, r=5%, sigma=20%, T=1.0)
//
// Compares `ra_core v2` (`ra_core_v2_W2_D2_xag_opt_orbit_buf` & `ra_engine_v2_t`)
// against `ra_core v1` (`ra_core_v1_orbit_buf` & `ra_engine_v1_t`),
// `Philox4x32-10` (both `philox_key_only` and idiomatic `philox_2d_counter`),
// `ChaCha8`, `xoshiro256**`, `xoroshiro128++`, `PCG32`, and `MT19937`.
//
// Outputs summary tables to stdout and structured JSON to `data/mc_workload3_v2.json`.

#include "baselines_shootout.h"
#include <sys/stat.h>

#define N_PATHS   1000
#define N_STEPS   10000
#define S0        100.0
#define K_STRIKE  100.0
#define RISK_FREE 0.05
#define SIGMA     0.20
#define MATURITY  1.0

static double asian_payoff_from_words(const uint32_t *w) {
    const double dt = MATURITY / (double)N_STEPS;
    const double drift = (RISK_FREE - 0.5 * SIGMA * SIGMA) * dt;
    const double vol = SIGMA * sqrt(dt);
    double S = S0;
    double sum = 0.0;
    for (int t = 0; t < N_STEPS; t += 2) {
        double u1 = ((double)w[t] + 0.5) / 4294967296.0;
        double u2 = ((double)w[t + 1] + 0.5) / 4294967296.0;
        double R = sqrt(-2.0 * log(u1));
        double theta = 2.0 * M_PI * u2;
        double z0 = R * cos(theta);
        double z1 = R * sin(theta);
        S *= exp(drift + vol * z0);
        sum += S;
        S *= exp(drift + vol * z1);
        sum += S;
    }
    double avg = sum / (double)N_STEPS;
    double payoff = avg - K_STRIKE;
    if (payoff < 0.0) payoff = 0.0;
    return exp(-RISK_FREE * MATURITY) * payoff;
}

static double pearson_corr(const uint32_t *a, const uint32_t *b, size_t n) {
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < n; i++) {
        mean_a += (double)a[i];
        mean_b += (double)b[i];
    }
    mean_a /= (double)n;
    mean_b /= (double)n;
    double cov = 0.0, var_a = 0.0, var_b = 0.0;
    for (size_t i = 0; i < n; i++) {
        double da = (double)a[i] - mean_a;
        double db = (double)b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    return cov / sqrt(var_a * var_b);
}

typedef struct {
    const char *name;
    uint32_t *words;
    double payoffs[N_PATHS];
    double mean, se;
    double gen_min, gen_med;
    double e2e_min, e2e_med;
    double gen_samples[5], e2e_samples[5];
    double gen_mibs;
    int viol_50, viol_500;
    double max_r_50, max_r_500;
} asian_res_t;

static void compute_stats_and_independence(asian_res_t *r) {
    double sum = 0.0;
    for (int p = 0; p < N_PATHS; p++) sum += r->payoffs[p];
    r->mean = sum / (double)N_PATHS;
    double var = 0.0;
    for (int p = 0; p < N_PATHS; p++) {
        double d = r->payoffs[p] - r->mean;
        var += d * d;
    }
    var /= (double)(N_PATHS - 1);
    r->se = sqrt(var / (double)N_PATHS);
    r->gen_mibs = ((double)N_PATHS * (double)N_STEPS * 4.0 / (1024.0 * 1024.0)) / r->gen_min;

    double threshold = 1.96 / sqrt((double)N_STEPS);

    {
        uint64_t lcg = 0x2545F4914F6CDD1DULL;
        int violations = 0;
        double max_abs_corr = 0.0;
        for (int k = 0; k < 50; k++) {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            int i = (int)((lcg >> 33) % N_PATHS);
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)((lcg >> 33) % N_PATHS);
            if (i == j) { k--; continue; }
            double corr = pearson_corr(&r->words[(size_t)i * N_STEPS],
                                       &r->words[(size_t)j * N_STEPS], N_STEPS);
            double a = fabs(corr);
            if (a > max_abs_corr) max_abs_corr = a;
            if (a > threshold) violations++;
        }
        r->viol_50 = violations;
        r->max_r_50 = max_abs_corr;
    }

    {
        uint64_t lcg = 0x2545F4914F6CDD1DULL;
        int violations = 0;
        double max_abs_corr = 0.0;
        for (int k = 0; k < 500; k++) {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            int i = (int)((lcg >> 33) % N_PATHS);
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)((lcg >> 33) % N_PATHS);
            if (i == j) { k--; continue; }
            double corr = pearson_corr(&r->words[(size_t)i * N_STEPS],
                                       &r->words[(size_t)j * N_STEPS], N_STEPS);
            double a = fabs(corr);
            if (a > max_abs_corr) max_abs_corr = a;
            if (a > threshold) violations++;
        }
        r->viol_500 = violations;
        r->max_r_500 = max_abs_corr;
    }
}

#define DEFINE_ASIAN_RUNNER(FNAME, ...)                                            \
static __attribute__((noinline, noclone)) void FNAME(asian_res_t *r) {             \
    r->words = malloc((size_t)N_PATHS * N_STEPS * sizeof(uint32_t));               \
    double g_times[5], e_times[5];                                                 \
    for (int trial = 0; trial < 5; ++trial) {                                      \
        double t0 = shootout_now_sec();                                            \
        for (int p = 0; p < N_PATHS; ++p) {                                        \
            uint32_t *buf = &r->words[(size_t)p * N_STEPS];                        \
            __VA_ARGS__;                                                           \
        }                                                                          \
        double t1 = shootout_now_sec();                                            \
        for (int p = 0; p < N_PATHS; ++p) {                                        \
            r->payoffs[p] = asian_payoff_from_words(&r->words[(size_t)p * N_STEPS]); \
        }                                                                          \
        double t2 = shootout_now_sec();                                            \
        volatile double anti_dce = r->payoffs[0] + r->payoffs[N_PATHS - 1];        \
        (void)anti_dce;                                                            \
        g_times[trial] = t1 - t0;                                                  \
        e_times[trial] = t2 - t0;                                                  \
    }                                                                              \
    compute_min_median(g_times, 5, &r->gen_min, &r->gen_med);                      \
    compute_min_median(e_times, 5, &r->e2e_min, &r->e2e_med);                      \
    for (int i = 0; i < 5; ++i) { r->gen_samples[i] = g_times[i]; r->e2e_samples[i] = e_times[i]; } \
    compute_stats_and_independence(r);                                             \
}

DEFINE_ASIAN_RUNNER(run_ra_v2_orbit_into, {
    uint32_t key = 0xC0FFEEu + (uint32_t)p * 0x9e3779b7u;
    ra_core_v2_W2_D2_xag_opt_orbit_buf(key, 0u, N_STEPS, buf);
})

DEFINE_ASIAN_RUNNER(run_ra_v2_engine, {
    uint32_t key = 0xC0FFEEu + (uint32_t)p * 0x9e3779b7u;
    ra_engine_v2_t eng;
    ra_engine_v2_init(&eng, key, 0u);
    for (int i = 0; i < N_STEPS; ++i) buf[i] = ra_engine_v2_next_u32(&eng);
})

DEFINE_ASIAN_RUNNER(run_ra_v1_orbit_into, {
    uint32_t key = 0xC0FFEEu + (uint32_t)p * 0x9e3779b7u;
    ra_core_v1_orbit_buf(key, 0u, N_STEPS, buf);
})

DEFINE_ASIAN_RUNNER(run_philox_2d, {
    uint32_t ctr[4] = { 0u, (uint32_t)p, 0u, 0u };
    uint32_t pkey[2] = { 0xC0FFEEu, 0xFEEDC0DEu };
    for (int b = 0; b < N_STEPS / 4; ++b) {
        philox4x32_10(&buf[b * 4], ctr, pkey);
        ++ctr[0];
    }
})

DEFINE_ASIAN_RUNNER(run_philox_key_only, {
    uint32_t key = (uint32_t)((uint32_t)p * 2654435761u + 1u);
    uint32_t ctr[4] = { 0u, 0u, 0u, 0u };
    uint32_t pkey[2] = { key, 0xFEEDC0DEu };
    for (int b = 0; b < N_STEPS / 4; ++b) {
        philox4x32_10(&buf[b * 4], ctr, pkey);
        ++ctr[0];
    }
})

DEFINE_ASIAN_RUNNER(run_chacha8, {
    uint32_t key = 0xC0FFEEu + (uint32_t)p * 0x9e3779b7u;
    chacha8_into(key, (uint32_t)p, N_STEPS, buf);
})

DEFINE_ASIAN_RUNNER(run_xoshiro256, {
    uint32_t key = (uint32_t)((uint32_t)p * 2654435761u + 1u);
    xoshiro256ss_into(key, 0u, N_STEPS, buf);
})

DEFINE_ASIAN_RUNNER(run_xoroshiro128, {
    uint32_t key = (uint32_t)((uint32_t)p * 2654435761u + 1u);
    xoroshiro128pp_into(key, 0u, N_STEPS, buf);
})

DEFINE_ASIAN_RUNNER(run_pcg32, {
    uint32_t key = (uint32_t)((uint32_t)p * 2654435761u + 1u);
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, (uint64_t)key, (uint64_t)p + 54u);
    for (int i = 0; i < N_STEPS; ++i) buf[i] = pcg32_random_r(&rng);
})

DEFINE_ASIAN_RUNNER(run_mt19937, {
    uint32_t key = (uint32_t)((uint32_t)p * 2654435761u + 1u);
    mt19937_into(key, 0u, N_STEPS, buf);
})

int main(void) {
    if (!verify_shootout_all_kat()) {
        fprintf(stderr, "FATAL: KAT verification failed!\n");
        return 1;
    }

    asian_res_t all[10] = {
        { .name = "ra_core_v2_orbit_into (W2_D2)" },
        { .name = "ra_core_v2_engine (Stateful)"  },
        { .name = "ra_core_v1_orbit_into (W0_D0)" },
        { .name = "Philox4x32-10 (2D counter)"    },
        { .name = "Philox4x32-10 (key-only ctrl)" },
        { .name = "ChaCha8 (ARX counter)"         },
        { .name = "xoshiro256**"                  },
        { .name = "xoroshiro128++"                },
        { .name = "PCG32"                         },
        { .name = "MT19937"                       }
    };

    run_ra_v2_orbit_into(&all[0]);
    run_ra_v2_engine(&all[1]);
    run_ra_v1_orbit_into(&all[2]);
    run_philox_2d(&all[3]);
    run_philox_key_only(&all[4]);
    run_chacha8(&all[5]);
    run_xoshiro256(&all[6]);
    run_xoroshiro128(&all[7]);
    run_pcg32(&all[8]);
    run_mt19937(&all[9]);

    if (memcmp(all[0].words, all[1].words, (size_t)N_PATHS * N_STEPS * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "FATAL: ra_core_v2_orbit_into != ra_core_v2_engine!\n");
        return 1;
    }

    printf("==================================================================================================================================================\n");
    printf("Workload 3: Asian Call Option GBM (N_PATHS=%d, N_STEPS=%d, 10,000,000 words, S0=100, K=100, r=5%%, sigma=20%%, T=1.0)\n", N_PATHS, N_STEPS);
    printf("==================================================================================================================================================\n");
    printf("%-32s | %11s | %9s | %11s | %12s | %8s | %9s | %8s | %8s\n",
           "Generator", "Mean Payoff", "SE", "Viol (n=50)", "Viol (n=500)", "GenMin(s)", "Gen MiB/s", "E2EMin(s)", "E2EMed(s)");
    printf("---------------------------------+-------------+-----------+-------------+--------------+----------+-----------+----------+---------\n");
    for (int g = 0; g < 10; ++g) {
        asian_res_t *r = &all[g];
        printf("%-32s | %11.6f | %9.6f | %2d/50 (%2.0f%%) | %2d/500 (%4.1f%%) | %7.4fs | %8.1f | %7.4fs | %7.4fs\n",
               r->name, r->mean, r->se,
               r->viol_50, 100.0 * r->viol_50 / 50.0,
               r->viol_500, 100.0 * r->viol_500 / 500.0,
               r->gen_min, r->gen_mibs, r->e2e_min, r->e2e_med);
    }

    mkdir("data", 0755);
    FILE *fp = fopen("data/mc_workload3_v2.json", "w");
    if (fp) {
        fprintf(fp, "{\n  \"asian_option_gbm\": [\n");
        for (int g = 0; g < 10; ++g) {
            asian_res_t *r = &all[g];
            fprintf(fp, "    {\"name\": \"%s\", \"mean_payoff\": %.6f, \"se\": %.6f, \"viol_50\": %d, \"viol_500\": %d, \"max_r_50\": %.5f, \"max_r_500\": %.5f, \"gen_min\": %.5f, \"gen_med\": %.5f, \"gen_mibs\": %.2f, \"e2e_min\": %.5f, \"e2e_med\": %.5f, \"gen_samples_s\": [",
                    r->name, r->mean, r->se, r->viol_50, r->viol_500, r->max_r_50, r->max_r_500,
                    r->gen_min, r->gen_med, r->gen_mibs, r->e2e_min, r->e2e_med);
            for (int i = 0; i < 5; ++i) fprintf(fp, "%s%.12g", i ? ", " : "", r->gen_samples[i]);
            fprintf(fp, "], \"e2e_samples_s\": [");
            for (int i = 0; i < 5; ++i) fprintf(fp, "%s%.12g", i ? ", " : "", r->e2e_samples[i]);
            fprintf(fp, "]}%s\n", (g + 1 < 10) ? "," : "");
        }
        fprintf(fp, "  ]\n}\n");
        fclose(fp);
    }

    for (int g = 0; g < 10; ++g) free(all[g].words);
    return 0;
}
