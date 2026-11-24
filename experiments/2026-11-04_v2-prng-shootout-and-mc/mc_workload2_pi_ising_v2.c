// mc_workload2_pi_ising_v2.c
// Workload 2:
//   Part A — Scalar Consumer Bounded-Burst Monte Carlo pi (1,000,000 trials x K in {255, 63, 31, 15})
//            across N in {256, 128, 64, 32, 16} for `ra_core v2` & `ra_core v1` vs Competitors.
//   Part B — Matched 8-Way AVX2 SIMD Consumer Bounded-Burst Monte Carlo pi (K in {255, 63, 31, 15})
//            for `ra_core_v2_x8` (N=256..16) vs `ra_core_v1_x8` vs `Philox4x32-10_x8` vs `ChaCha8_x8`
//            vs `xoshiro256**_x8` vs `xoroshiro128++_x8` vs `PCG32_x8` vs `SplitMix64_x8`.
//   Part C — 8-lane Bernoulli flip diagnostic (not an Ising lattice simulation)
//            across K in {255, 63, 31, 15} and N in {256, 128, 64, 32, 16}.
//
// Outputs summary tables to stdout and structured JSON to `data/mc_workload2_v2.json`.

#include "baselines_shootout.h"
#include <sys/stat.h>

#define N_TRIALS 1000000LL
_Static_assert(N_TRIALS % 8 == 0, "AVX2 path requires complete eight-lane batches");

typedef struct {
    const char *name;
    size_t k_words;
    uint64_t inside_count;
    uint64_t total_points;
    double pi_est;
    double err;
    double se;
    double wall_min;
    double wall_med;
    double wall_samples[5];
    double mibs;
} pi_res_t;

#define DEFINE_PI_SCALAR_BENCH(BENCH_NAME, GEN_CALL)                               \
static __attribute__((noinline, noclone)) void BENCH_NAME(                         \
    const char *label, size_t k_words, pi_res_t *res) {                            \
    res->name = label;                                                             \
    const int n_pairs = (int)(k_words / 2);                                        \
    double times[5];                                                               \
    for (int trial = 0; trial < 5; ++trial) {                                      \
        uint64_t inside = 0;                                                       \
        alignas(64) uint32_t buf[256];                                             \
        double t0 = shootout_now_sec();                                            \
        for (long long t = 0; t < N_TRIALS; ++t) {                                 \
            uint32_t key = (uint32_t)(t * 2654435761u + 1u);                       \
            GEN_CALL(key, 0u, k_words, buf);                                       \
            for (int p = 0; p < n_pairs; ++p) {                                    \
                uint32_t wx = buf[p * 2];                                          \
                uint32_t wy = buf[p * 2 + 1];                                      \
                float x = ((float)(wx >> 8) + 0.5f) * (2.0f / 16777216.0f) - 1.0f; \
                float y = ((float)(wy >> 8) + 0.5f) * (2.0f / 16777216.0f) - 1.0f; \
                if (x * x + y * y <= 1.0f) inside++;                              \
            }                                                                      \
        }                                                                          \
        times[trial] = shootout_now_sec() - t0;                                    \
        volatile uint64_t anti_dce = inside; (void)anti_dce;                       \
        if (trial == 0) {                                                          \
            res->k_words = k_words;                                                \
            res->inside_count = inside;                                            \
            res->total_points = (uint64_t)N_TRIALS * (uint64_t)n_pairs;            \
            res->pi_est = 4.0 * (double)inside / (double)res->total_points;        \
            res->se = sqrt(M_PI * (4.0 - M_PI) / (double)res->total_points);       \
            res->err = fabs(res->pi_est - M_PI);                                   \
        }                                                                          \
    }                                                                              \
    compute_min_median(times, 5, &res->wall_min, &res->wall_med);                  \
    for (int i = 0; i < 5; ++i) res->wall_samples[i] = times[i];                  \
    res->mibs = ((double)N_TRIALS * (double)k_words * 4.0 / (1024.0 * 1024.0)) / res->wall_min; \
}

#define DEFINE_PI_X8_SCALAR_CONSUMER_BENCH(BENCH_NAME, GEN_X8_CALL)                \
static __attribute__((noinline, noclone)) void BENCH_NAME(                         \
    const char *label, size_t k_words, pi_res_t *res) {                            \
    res->name = label;                                                             \
    const int n_pairs = (int)(k_words / 2);                                        \
    const long long n_batches = N_TRIALS / 8;                                      \
    const uint32_t addr_zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};                       \
    double times[5];                                                               \
    for (int trial = 0; trial < 5; ++trial) {                                      \
        uint64_t inside = 0;                                                       \
        alignas(64) uint32_t buf8[256 * 8];                                        \
        double t0 = shootout_now_sec();                                            \
        for (long long b = 0; b < n_batches; ++b) {                                \
            long long base_t = b * 8;                                              \
            alignas(32) uint32_t keys8[8] = {                                      \
                (uint32_t)((base_t + 0) * 2654435761u + 1u),                       \
                (uint32_t)((base_t + 1) * 2654435761u + 1u),                       \
                (uint32_t)((base_t + 2) * 2654435761u + 1u),                       \
                (uint32_t)((base_t + 3) * 2654435761u + 1u),                       \
                (uint32_t)((base_t + 4) * 2654435761u + 1u),                       \
                (uint32_t)((base_t + 5) * 2654435761u + 1u),                       \
                (uint32_t)((base_t + 6) * 2654435761u + 1u),                       \
                (uint32_t)((base_t + 7) * 2654435761u + 1u),                       \
            };                                                                     \
            GEN_X8_CALL(keys8, addr_zeros, k_words, buf8);                         \
            for (int p = 0; p < n_pairs; ++p) {                                    \
                const uint32_t *wx_row = &buf8[(p * 2) * 8];                       \
                const uint32_t *wy_row = &buf8[(p * 2 + 1) * 8];                   \
                for (int lane = 0; lane < 8; ++lane) {                             \
                    float x = ((float)(wx_row[lane] >> 8) + 0.5f) * (2.0f / 16777216.0f) - 1.0f; \
                    float y = ((float)(wy_row[lane] >> 8) + 0.5f) * (2.0f / 16777216.0f) - 1.0f; \
                    if (x * x + y * y <= 1.0f) inside++;                          \
                }                                                                  \
            }                                                                      \
        }                                                                          \
        times[trial] = shootout_now_sec() - t0;                                    \
        volatile uint64_t anti_dce = inside; (void)anti_dce;                       \
        if (trial == 0) {                                                          \
            res->k_words = k_words;                                                \
            res->inside_count = inside;                                            \
            res->total_points = (uint64_t)N_TRIALS * (uint64_t)n_pairs;            \
            res->pi_est = 4.0 * (double)inside / (double)res->total_points;        \
            res->se = sqrt(M_PI * (4.0 - M_PI) / (double)res->total_points);       \
            res->err = fabs(res->pi_est - M_PI);                                   \
        }                                                                          \
    }                                                                              \
    compute_min_median(times, 5, &res->wall_min, &res->wall_med);                  \
    for (int i = 0; i < 5; ++i) res->wall_samples[i] = times[i];                  \
    res->mibs = ((double)N_TRIALS * (double)k_words * 4.0 / (1024.0 * 1024.0)) / res->wall_min; \
}

DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v2_N256, ra_v2_N256_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v2_N128, ra_v2_N128_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v2_N64,  ra_v2_N64_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v2_N32,  ra_v2_N32_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v2_N16,  ra_v2_N16_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v1_N256, ra_v1_N256_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v1_N128, ra_v1_N128_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v1_N64,  ra_v1_N64_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_ra_v1_N32,  ra_v1_N32_singleblock_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_philox,     philox_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_chacha8,    chacha8_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_xoshiro256, xoshiro256ss_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_xoroshiro,  xoroshiro128pp_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_pcg32,      pcg32_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_splitmix64, splitmix64_into)
DEFINE_PI_SCALAR_BENCH(pi_scalar_wyrand,     wyrand_into)

DEFINE_PI_X8_SCALAR_CONSUMER_BENCH(pi_x8_sc_ra_v2_N256, ra_v2_N256_singleblock_x8_into)
DEFINE_PI_X8_SCALAR_CONSUMER_BENCH(pi_x8_sc_ra_v2_N128, ra_v2_N128_singleblock_x8_into)
DEFINE_PI_X8_SCALAR_CONSUMER_BENCH(pi_x8_sc_ra_v2_N64,  ra_v2_N64_singleblock_x8_into)
DEFINE_PI_X8_SCALAR_CONSUMER_BENCH(pi_x8_sc_ra_v2_N32,  ra_v2_N32_singleblock_x8_into)
DEFINE_PI_X8_SCALAR_CONSUMER_BENCH(pi_x8_sc_ra_v2_N16,  ra_v2_N16_singleblock_x8_into)
DEFINE_PI_X8_SCALAR_CONSUMER_BENCH(pi_x8_sc_ra_v1_N256, ra_v1_N256_singleblock_x8_into)

// Part B: Matched AVX2 SIMD Consumer for Bounded-Burst pi
typedef void (*avx2_gen_fn_t)(const uint32_t keys[8], const uint32_t addr_his[8], size_t rng, uint32_t * restrict out);

static __attribute__((noinline, noclone)) void bench_pi_avx2_consumer(
    const char *label, avx2_gen_fn_t gen_x8, size_t k_words, pi_res_t *res) {
    res->name = label;
    const int n_pairs = (int)(k_words / 2);
    const long long n_batches = N_TRIALS / 8;
    const uint32_t addr_zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double times[5];
    uint64_t inside_total = 0;

    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 scale = _mm256_set1_ps(2.0f / 16777216.0f);

    for (int trial = 0; trial < 5; ++trial) {
        __m256i v_inside = _mm256_setzero_si256();
        alignas(64) uint32_t buf8[256 * 8];
        double t0 = shootout_now_sec();
        for (long long b = 0; b < n_batches; ++b) {
            long long base_t = b * 8;
            alignas(32) uint32_t keys8[8] = {
                (uint32_t)((base_t + 0) * 2654435761u + 1u),
                (uint32_t)((base_t + 1) * 2654435761u + 1u),
                (uint32_t)((base_t + 2) * 2654435761u + 1u),
                (uint32_t)((base_t + 3) * 2654435761u + 1u),
                (uint32_t)((base_t + 4) * 2654435761u + 1u),
                (uint32_t)((base_t + 5) * 2654435761u + 1u),
                (uint32_t)((base_t + 6) * 2654435761u + 1u),
                (uint32_t)((base_t + 7) * 2654435761u + 1u),
            };
            gen_x8(keys8, addr_zeros, k_words, buf8);
            for (int p = 0; p < n_pairs; ++p) {
                __m256i wx = _mm256_load_si256((const __m256i *)&buf8[(p * 2) * 8]);
                __m256i wy = _mm256_load_si256((const __m256i *)&buf8[(p * 2 + 1) * 8]);
                __m256 fx = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(wx, 8)), half), scale), one);
                __m256 fy = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(wy, 8)), half), scale), one);
                __m256 r2 = _mm256_add_ps(_mm256_mul_ps(fx, fx), _mm256_mul_ps(fy, fy));
                __m256 cmp = _mm256_cmp_ps(r2, one, _CMP_LE_OQ);
                v_inside = _mm256_sub_epi32(v_inside, _mm256_castps_si256(cmp));
            }
        }
        uint32_t lanes[8];
        _mm256_storeu_si256((__m256i *)lanes, v_inside);
        uint64_t tot = 0;
        for (int l = 0; l < 8; ++l) tot += lanes[l];
        times[trial] = shootout_now_sec() - t0;
        volatile uint64_t sink = tot; (void)sink;
        if (trial == 0) inside_total = tot;
    }
    res->k_words = k_words;
    res->inside_count = inside_total;
    res->total_points = (uint64_t)N_TRIALS * (uint64_t)n_pairs;
    res->pi_est = 4.0 * (double)inside_total / (double)res->total_points;
    res->se = sqrt(M_PI * (4.0 - M_PI) / (double)res->total_points);
    res->err = fabs(res->pi_est - M_PI);
    compute_min_median(times, 5, &res->wall_min, &res->wall_med);
    for (int i = 0; i < 5; ++i) res->wall_samples[i] = times[i];
    res->mibs = ((double)N_TRIALS * (double)k_words * 4.0 / (1024.0 * 1024.0)) / res->wall_min;
}

// Part C: Eight-lane Bernoulli flip diagnostic with an unsigned threshold.
typedef struct {
    const char *name;
    size_t k_sweeps;
    uint64_t flips;
    double wall_min;
    double wall_med;
    double wall_samples[5];
    double mibs;
} ising_res_t;

static __attribute__((noinline, noclone)) void bench_ising_avx2(
    const char *label, avx2_gen_fn_t gen_x8, size_t k_sweeps, ising_res_t *res) {
    res->name = label;
    res->k_sweeps = k_sweeps;
    const __m256i thresh_flip = _mm256_set1_epi32((int)735442293u);
    const __m256i sign_flip   = _mm256_set1_epi32((int)0x80000000u);
    const long long n_batches = 500000LL / 8;
    const uint32_t addr_zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double times[5];
    uint64_t flip_total = 0;

    for (int trial = 0; trial < 5; ++trial) {
        __m256i v_flips = _mm256_setzero_si256();
        __m256i spin_state = _mm256_set1_epi32(1);
        alignas(64) uint32_t buf8[256 * 8];
        double t0 = shootout_now_sec();
        for (long long b = 0; b < n_batches; ++b) {
            long long base_t = b * 8;
            alignas(32) uint32_t keys8[8] = {
                (uint32_t)(base_t + 0 + 101u), (uint32_t)(base_t + 1 + 101u),
                (uint32_t)(base_t + 2 + 101u), (uint32_t)(base_t + 3 + 101u),
                (uint32_t)(base_t + 4 + 101u), (uint32_t)(base_t + 5 + 101u),
                (uint32_t)(base_t + 6 + 101u), (uint32_t)(base_t + 7 + 101u),
            };
            gen_x8(keys8, addr_zeros, k_sweeps, buf8);
            for (size_t s = 0; s < k_sweeps; ++s) {
                __m256i u = _mm256_load_si256((const __m256i *)&buf8[s * 8]);
                __m256i cmp = _mm256_cmpgt_epi32(
                    _mm256_xor_si256(thresh_flip, sign_flip),
                    _mm256_xor_si256(u, sign_flip));
                spin_state = _mm256_xor_si256(spin_state, cmp);
                v_flips = _mm256_sub_epi32(v_flips, cmp);
            }
        }
        uint32_t lanes[8];
        _mm256_storeu_si256((__m256i *)lanes, v_flips);
        uint64_t tot = 0;
        for (int l = 0; l < 8; ++l) tot += lanes[l];
        times[trial] = shootout_now_sec() - t0;
        volatile uint64_t sink = tot + (uint32_t)_mm256_extract_epi32(spin_state, 0);
        (void)sink;
        if (trial == 0) flip_total = tot;
    }
    res->flips = flip_total;
    compute_min_median(times, 5, &res->wall_min, &res->wall_med);
    for (int i = 0; i < 5; ++i) res->wall_samples[i] = times[i];
    res->mibs = (500000.0 * (double)k_sweeps * 4.0 / (1024.0 * 1024.0)) / res->wall_min;
}

static void print_pi_list(const char *title, pi_res_t *list, int n) {
    printf("=======================================================================================================================\n");
    printf("%s\n", title);
    printf("=======================================================================================================================\n");
    printf("%-34s | %4s | %12s | %9s | %9s | %9s | %6s | %8s | %9s\n",
           "Generator", "K", "inside_count", "pi_est", "|err|", "4*SE", "Status", "min(5)s", "MiB/s");
    printf("-----------------------------------+------+--------------+-----------+-----------+-----------+--------+----------+----------\n");
    for (int i = 0; i < n; ++i) {
        pi_res_t *r = &list[i];
        bool pass = r->err < 4.0 * r->se;
        printf("%-34s | %4zu | %12llu | %9.6f | %9.6f | %9.6f | %-6s | %7.4fs | %8.1f\n",
               r->name, r->k_words, (unsigned long long)r->inside_count,
               r->pi_est, r->err, 4.0 * r->se, pass ? "PASS" : "FAIL",
               r->wall_min, r->mibs);
    }
    printf("\n");
}

static void print_ising_list(const char *title, ising_res_t *list, int n) {
    printf("========================================================================================================\n");
    printf("%s\n", title);
    printf("========================================================================================================\n");
    printf("%-34s | %5s | %12s | %9s | %9s | %9s\n",
           "Generator (8-Way AVX2)", "K", "Flips", "min(5) s", "med(5) s", "MiB/s");
    printf("-----------------------------------+-------+--------------+-----------+-----------+----------\n");
    for (int i = 0; i < n; ++i) {
        ising_res_t *r = &list[i];
        printf("%-34s | %5zu | %12llu | %8.4fs | %8.4fs | %8.1f\n",
               r->name, r->k_sweeps, (unsigned long long)r->flips,
               r->wall_min, r->wall_med, r->mibs);
    }
    printf("\n");
}

int main(void) {
    if (!verify_shootout_all_kat()) {
        fprintf(stderr, "FATAL: KAT verification failed!\n");
        return 1;
    }

    // ------------------------------------------------------------------------
    // Part A: Scalar Consumer Bounded-Burst pi (K=255, K=63, K=31, K=15)
    // ------------------------------------------------------------------------
    pi_res_t s255[10];
    pi_scalar_ra_v2_N256("ra_core_v2_sb (N=256, Scalar)",  255, &s255[0]);
    pi_x8_sc_ra_v2_N256 ("ra_core_v2_x8 (N=256, AVX2)",    255, &s255[1]);
    pi_scalar_ra_v1_N256("ra_core_v1_sb (N=256, Scalar)",  255, &s255[2]);
    pi_x8_sc_ra_v1_N256 ("ra_core_v1_x8 (N=256, AVX2)",    255, &s255[3]);
    pi_scalar_philox    ("Philox4x32-10",                  255, &s255[4]);
    pi_scalar_chacha8   ("ChaCha8",                        255, &s255[5]);
    pi_scalar_xoshiro256("xoshiro256**",                   255, &s255[6]);
    pi_scalar_xoroshiro ("xoroshiro128++",                 255, &s255[7]);
    pi_scalar_pcg32     ("PCG32",                          255, &s255[8]);
    pi_scalar_splitmix64("SplitMix64",                     255, &s255[9]);
    if (s255[0].inside_count != s255[1].inside_count || s255[2].inside_count != s255[3].inside_count) {
        fprintf(stderr, "FATAL: Scalar vs AVX2 inside_count mismatch at K=255!\n");
        return 1;
    }
    print_pi_list("Workload 2A (K=255, Scalar Consumer): 1,000,000 trials x 255 words", s255, 10);

    pi_res_t s63[9];
    pi_scalar_ra_v2_N256("ra_core_v2_sb (N=256, Scalar)",  63, &s63[0]);
    pi_scalar_ra_v2_N128("ra_core_v2_sb (N=128, Scalar)",  63, &s63[1]);
    pi_scalar_ra_v2_N64 ("ra_core_v2_sb (N=64,  Scalar)",  63, &s63[2]);
    pi_scalar_ra_v1_N64 ("ra_core_v1_sb (N=64,  Scalar)",  63, &s63[3]);
    pi_scalar_philox    ("Philox4x32-10",                  63, &s63[4]);
    pi_scalar_chacha8   ("ChaCha8",                        63, &s63[5]);
    pi_scalar_xoshiro256("xoshiro256**",                   63, &s63[6]);
    pi_scalar_xoroshiro ("xoroshiro128++",                 63, &s63[7]);
    pi_scalar_pcg32     ("PCG32",                          63, &s63[8]);
    print_pi_list("Workload 2A (K=63, Scalar Consumer): Pareto State-Size Study", s63, 9);

    pi_res_t s31[9];
    pi_scalar_ra_v2_N256("ra_core_v2_sb (N=256, Scalar)",  31, &s31[0]);
    pi_scalar_ra_v2_N64 ("ra_core_v2_sb (N=64,  Scalar)",  31, &s31[1]);
    pi_scalar_ra_v2_N32 ("ra_core_v2_sb (N=32,  Scalar)",  31, &s31[2]);
    pi_scalar_ra_v1_N32 ("ra_core_v1_sb (N=32,  Scalar)",  31, &s31[3]);
    pi_scalar_philox    ("Philox4x32-10",                  31, &s31[4]);
    pi_scalar_chacha8   ("ChaCha8",                        31, &s31[5]);
    pi_scalar_xoshiro256("xoshiro256**",                   31, &s31[6]);
    pi_scalar_xoroshiro ("xoroshiro128++",                 31, &s31[7]);
    pi_scalar_pcg32     ("PCG32",                          31, &s31[8]);
    print_pi_list("Workload 2A (K=31, Scalar Consumer): Pareto State-Size Study", s31, 9);

    pi_res_t s15[8];
    pi_scalar_ra_v2_N256("ra_core_v2_sb (N=256, Scalar)",  15, &s15[0]);
    pi_scalar_ra_v2_N32 ("ra_core_v2_sb (N=32,  Scalar)",  15, &s15[1]);
    pi_scalar_ra_v2_N16 ("ra_core_v2_sb (N=16,  Scalar)",  15, &s15[2]);
    pi_scalar_philox    ("Philox4x32-10",                  15, &s15[3]);
    pi_scalar_chacha8   ("ChaCha8",                        15, &s15[4]);
    pi_scalar_xoshiro256("xoshiro256**",                   15, &s15[5]);
    pi_scalar_xoroshiro ("xoroshiro128++",                 15, &s15[6]);
    pi_scalar_pcg32     ("PCG32",                          15, &s15[7]);
    print_pi_list("Workload 2A (K=15, Scalar Consumer): N=16 (128 B) Safe-Floor Validation", s15, 8);

    // ------------------------------------------------------------------------
    // Part B: Matched AVX2 SIMD Consumer for Bounded-Burst pi (K=255, 63, 31, 15)
    // ------------------------------------------------------------------------
    pi_res_t v255[8];
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=256)", ra_v2_N256_singleblock_x8_into, 255, &v255[0]);
    bench_pi_avx2_consumer("ra_core_v1_x8 (N=256)", ra_v1_N256_singleblock_x8_into, 255, &v255[1]);
    bench_pi_avx2_consumer("Philox4x32-10_x8",      philox4x32_x8_into,             255, &v255[2]);
    bench_pi_avx2_consumer("ChaCha8_x8",            chacha8_x8_into,                255, &v255[3]);
    bench_pi_avx2_consumer("xoshiro256**_x8",       xoshiro256ss_x8_into,           255, &v255[4]);
    bench_pi_avx2_consumer("xoroshiro128++_x8",     xoroshiro128pp_x8_into,         255, &v255[5]);
    bench_pi_avx2_consumer("PCG32_x8",              pcg32_x8_into,                  255, &v255[6]);
    bench_pi_avx2_consumer("SplitMix64_x8",         splitmix64_x8_into,             255, &v255[7]);
    print_pi_list("Workload 2B (K=255, Matched 8-Way AVX2 Consumer): Apples-to-Apples SIMD", v255, 8);

    pi_res_t v63[9];
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=256)", ra_v2_N256_singleblock_x8_into, 63, &v63[0]);
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=128)", ra_v2_N128_singleblock_x8_into, 63, &v63[1]);
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=64)",  ra_v2_N64_singleblock_x8_into,  63, &v63[2]);
    bench_pi_avx2_consumer("ra_core_v1_x8 (N=64)",  ra_v1_N64_singleblock_x8_into,  63, &v63[3]);
    bench_pi_avx2_consumer("Philox4x32-10_x8",      philox4x32_x8_into,             63, &v63[4]);
    bench_pi_avx2_consumer("ChaCha8_x8",            chacha8_x8_into,                63, &v63[5]);
    bench_pi_avx2_consumer("xoshiro256**_x8",       xoshiro256ss_x8_into,           63, &v63[6]);
    bench_pi_avx2_consumer("xoroshiro128++_x8",     xoroshiro128pp_x8_into,         63, &v63[7]);
    bench_pi_avx2_consumer("PCG32_x8",              pcg32_x8_into,                  63, &v63[8]);
    print_pi_list("Workload 2B (K=63, Matched 8-Way AVX2 Consumer): Apples-to-Apples SIMD", v63, 9);

    pi_res_t v31[9];
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=256)", ra_v2_N256_singleblock_x8_into, 31, &v31[0]);
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=64)",  ra_v2_N64_singleblock_x8_into,  31, &v31[1]);
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=32)",  ra_v2_N32_singleblock_x8_into,  31, &v31[2]);
    bench_pi_avx2_consumer("ra_core_v1_x8 (N=32)",  ra_v1_N32_singleblock_x8_into,  31, &v31[3]);
    bench_pi_avx2_consumer("Philox4x32-10_x8",      philox4x32_x8_into,             31, &v31[4]);
    bench_pi_avx2_consumer("ChaCha8_x8",            chacha8_x8_into,                31, &v31[5]);
    bench_pi_avx2_consumer("xoshiro256**_x8",       xoshiro256ss_x8_into,           31, &v31[6]);
    bench_pi_avx2_consumer("xoroshiro128++_x8",     xoroshiro128pp_x8_into,         31, &v31[7]);
    bench_pi_avx2_consumer("PCG32_x8",              pcg32_x8_into,                  31, &v31[8]);
    print_pi_list("Workload 2B (K=31, Matched 8-Way AVX2 Consumer): Apples-to-Apples SIMD", v31, 9);

    pi_res_t v15[8];
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=256)", ra_v2_N256_singleblock_x8_into, 15, &v15[0]);
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=32)",  ra_v2_N32_singleblock_x8_into,  15, &v15[1]);
    bench_pi_avx2_consumer("ra_core_v2_x8 (N=16)",  ra_v2_N16_singleblock_x8_into,  15, &v15[2]);
    bench_pi_avx2_consumer("Philox4x32-10_x8",      philox4x32_x8_into,             15, &v15[3]);
    bench_pi_avx2_consumer("ChaCha8_x8",            chacha8_x8_into,                15, &v15[4]);
    bench_pi_avx2_consumer("xoshiro256**_x8",       xoshiro256ss_x8_into,           15, &v15[5]);
    bench_pi_avx2_consumer("xoroshiro128++_x8",     xoroshiro128pp_x8_into,         15, &v15[6]);
    bench_pi_avx2_consumer("PCG32_x8",              pcg32_x8_into,                  15, &v15[7]);
    print_pi_list("Workload 2B (K=15, Matched 8-Way AVX2 Consumer): N=16 (128 B) Safe-Floor SIMD", v15, 8);

    // Same keys, same streams and same top-24-bit float mapping must give the
    // same physical consumer result, regardless of scalar or AVX2 execution.
    if (s255[0].inside_count != v255[0].inside_count ||
        s255[2].inside_count != v255[1].inside_count ||
        s63[0].inside_count != v63[0].inside_count ||
        s63[1].inside_count != v63[1].inside_count ||
        s63[2].inside_count != v63[2].inside_count ||
        s31[0].inside_count != v31[0].inside_count ||
        s31[1].inside_count != v31[1].inside_count ||
        s31[2].inside_count != v31[2].inside_count ||
        s15[0].inside_count != v15[0].inside_count ||
        s15[1].inside_count != v15[1].inside_count ||
        s15[2].inside_count != v15[2].inside_count ||
        s255[4].inside_count != v255[2].inside_count ||
        s255[5].inside_count != v255[3].inside_count ||
        s255[6].inside_count != v255[4].inside_count ||
        s255[7].inside_count != v255[5].inside_count ||
        s255[8].inside_count != v255[6].inside_count ||
        s255[9].inside_count != v255[7].inside_count ||
        s63[4].inside_count != v63[4].inside_count ||
        s63[5].inside_count != v63[5].inside_count ||
        s63[6].inside_count != v63[6].inside_count ||
        s63[7].inside_count != v63[7].inside_count ||
        s63[8].inside_count != v63[8].inside_count ||
        s31[4].inside_count != v31[4].inside_count ||
        s31[5].inside_count != v31[5].inside_count ||
        s31[6].inside_count != v31[6].inside_count ||
        s31[7].inside_count != v31[7].inside_count ||
        s31[8].inside_count != v31[8].inside_count ||
        s15[3].inside_count != v15[3].inside_count ||
        s15[4].inside_count != v15[4].inside_count ||
        s15[5].inside_count != v15[5].inside_count ||
        s15[6].inside_count != v15[6].inside_count ||
        s15[7].inside_count != v15[7].inside_count) {
        fprintf(stderr, "FATAL: scalar/AVX2 pi consumer parity failed\n");
        return 1;
    }
    puts("[PI CONSUMER PARITY] Scalar and AVX2 inside counts match for all shared generators and K values: PASS");

    // ------------------------------------------------------------------------
    // Part C: Eight-lane Bernoulli flip diagnostic (K=255, 63, 31, 15)
    // ------------------------------------------------------------------------
    ising_res_t is255[8];
    bench_ising_avx2("ra_core_v2_x8 (N=256)", ra_v2_N256_singleblock_x8_into, 255, &is255[0]);
    bench_ising_avx2("ra_core_v1_x8 (N=256)", ra_v1_N256_singleblock_x8_into, 255, &is255[1]);
    bench_ising_avx2("Philox4x32-10_x8",      philox4x32_x8_into,             255, &is255[2]);
    bench_ising_avx2("ChaCha8_x8",            chacha8_x8_into,                255, &is255[3]);
    bench_ising_avx2("xoshiro256**_x8",       xoshiro256ss_x8_into,           255, &is255[4]);
    bench_ising_avx2("xoroshiro128++_x8",     xoroshiro128pp_x8_into,         255, &is255[5]);
    bench_ising_avx2("PCG32_x8",              pcg32_x8_into,                  255, &is255[6]);
    bench_ising_avx2("SplitMix64_x8",         splitmix64_x8_into,             255, &is255[7]);
    print_ising_list("Workload 2C (K=255): 8-lane Bernoulli flip diagnostic", is255, 8);

    ising_res_t is63[8];
    bench_ising_avx2("ra_core_v2_x8 (N=64)",  ra_v2_N64_singleblock_x8_into,  63, &is63[0]);
    bench_ising_avx2("ra_core_v1_x8 (N=64)",  ra_v1_N64_singleblock_x8_into,  63, &is63[1]);
    bench_ising_avx2("Philox4x32-10_x8",      philox4x32_x8_into,             63, &is63[2]);
    bench_ising_avx2("ChaCha8_x8",            chacha8_x8_into,                63, &is63[3]);
    bench_ising_avx2("xoshiro256**_x8",       xoshiro256ss_x8_into,           63, &is63[4]);
    bench_ising_avx2("xoroshiro128++_x8",     xoroshiro128pp_x8_into,         63, &is63[5]);
    bench_ising_avx2("PCG32_x8",              pcg32_x8_into,                  63, &is63[6]);
    bench_ising_avx2("SplitMix64_x8",         splitmix64_x8_into,             63, &is63[7]);
    print_ising_list("Workload 2C (K=63, N=64): 8-lane Bernoulli flip diagnostic", is63, 8);

    ising_res_t is31[8];
    bench_ising_avx2("ra_core_v2_x8 (N=32)",  ra_v2_N32_singleblock_x8_into,  31, &is31[0]);
    bench_ising_avx2("ra_core_v1_x8 (N=32)",  ra_v1_N32_singleblock_x8_into,  31, &is31[1]);
    bench_ising_avx2("Philox4x32-10_x8",      philox4x32_x8_into,             31, &is31[2]);
    bench_ising_avx2("ChaCha8_x8",            chacha8_x8_into,                31, &is31[3]);
    bench_ising_avx2("xoshiro256**_x8",       xoshiro256ss_x8_into,           31, &is31[4]);
    bench_ising_avx2("xoroshiro128++_x8",     xoroshiro128pp_x8_into,         31, &is31[5]);
    bench_ising_avx2("PCG32_x8",              pcg32_x8_into,                  31, &is31[6]);
    bench_ising_avx2("SplitMix64_x8",         splitmix64_x8_into,             31, &is31[7]);
    print_ising_list("Workload 2C (K=31, N=32): 8-lane Bernoulli flip diagnostic", is31, 8);

    ising_res_t is15[8];
    bench_ising_avx2("ra_core_v2_x8 (N=16)",  ra_v2_N16_singleblock_x8_into,  15, &is15[0]);
    bench_ising_avx2("ra_core_v1_x8 (N=16)",  ra_v1_N16_singleblock_x8_into,  15, &is15[1]);
    bench_ising_avx2("Philox4x32-10_x8",      philox4x32_x8_into,             15, &is15[2]);
    bench_ising_avx2("ChaCha8_x8",            chacha8_x8_into,                15, &is15[3]);
    bench_ising_avx2("xoshiro256**_x8",       xoshiro256ss_x8_into,           15, &is15[4]);
    bench_ising_avx2("xoroshiro128++_x8",     xoroshiro128pp_x8_into,         15, &is15[5]);
    bench_ising_avx2("PCG32_x8",              pcg32_x8_into,                  15, &is15[6]);
    bench_ising_avx2("SplitMix64_x8",         splitmix64_x8_into,             15, &is15[7]);
    print_ising_list("Workload 2C (K=15, N=16): 8-lane Bernoulli flip diagnostic", is15, 8);

    // Save JSON
    mkdir("data", 0755);
    FILE *fp = fopen("data/mc_workload2_v2.json", "w");
    if (fp) {
        fprintf(fp, "{\n  \"pi_conversion\": \"top24_float_midpoint\",\n");
        #define DUMP_PI_ARR(KEY, ARR, LEN) \
            fprintf(fp, "  \"%s\": [\n", KEY); \
            for (int i = 0; i < (LEN); ++i) { \
                fprintf(fp, "    {\"name\": \"%s\", \"k\": %zu, \"inside\": %llu, \"generated_words\": %.0f, \"consumed_words\": %.0f, \"pi_est\": %.6f, \"err\": %.6f, \"se\": %.6f, \"wall_min\": %.5f, \"wall_med\": %.5f, \"mibs\": %.2f, \"samples_s\": [", \
                        (ARR)[i].name, (ARR)[i].k_words, (unsigned long long)(ARR)[i].inside_count, \
                        (double)N_TRIALS * (double)(ARR)[i].k_words, (double)(ARR)[i].total_points * 2.0, \
                        (ARR)[i].pi_est, (ARR)[i].err, (ARR)[i].se, (ARR)[i].wall_min, (ARR)[i].wall_med, (ARR)[i].mibs); \
                for (int j = 0; j < 5; ++j) fprintf(fp, "%s%.12g", j ? ", " : "", (ARR)[i].wall_samples[j]); \
                fprintf(fp, "]}%s\n", (i + 1 < (LEN)) ? "," : ""); \
            } \
            fprintf(fp, "  ],\n");
        DUMP_PI_ARR("scalar_consumer_k255", s255, 10)
        DUMP_PI_ARR("scalar_consumer_k63",  s63,  9)
        DUMP_PI_ARR("scalar_consumer_k31",  s31,  9)
        DUMP_PI_ARR("scalar_consumer_k15",  s15,  8)
        DUMP_PI_ARR("avx2_consumer_k255",   v255, 8)
        DUMP_PI_ARR("avx2_consumer_k63",    v63,  9)
        DUMP_PI_ARR("avx2_consumer_k31",    v31,  9)
        DUMP_PI_ARR("avx2_consumer_k15",    v15,  8)

        #define DUMP_ISING_ARR(KEY, ARR, LEN, LAST) \
            fprintf(fp, "  \"%s\": [\n", KEY); \
            for (int i = 0; i < (LEN); ++i) { \
                fprintf(fp, "    {\"name\": \"%s\", \"k\": %zu, \"flips\": %llu, \"wall_min\": %.5f, \"wall_med\": %.5f, \"mibs\": %.2f, \"samples_s\": [", \
                        (ARR)[i].name, (ARR)[i].k_sweeps, (unsigned long long)(ARR)[i].flips, \
                        (ARR)[i].wall_min, (ARR)[i].wall_med, (ARR)[i].mibs); \
                for (int j = 0; j < 5; ++j) fprintf(fp, "%s%.12g", j ? ", " : "", (ARR)[i].wall_samples[j]); \
                fprintf(fp, "]}%s\n", (i + 1 < (LEN)) ? "," : ""); \
            } \
            fprintf(fp, "  ]%s\n", (LAST) ? "" : ",");
        DUMP_ISING_ARR("bernoulli_flip_avx2_k255", is255, 8, 0)
        DUMP_ISING_ARR("bernoulli_flip_avx2_k63",  is63,  8, 0)
        DUMP_ISING_ARR("bernoulli_flip_avx2_k31",  is31,  8, 0)
        DUMP_ISING_ARR("bernoulli_flip_avx2_k15",  is15,  8, 1)
        fprintf(fp, "}\n");
        fclose(fp);
    }
    return 0;
}
