// run_stage4_sac_bic_perturbation.c
// Stage 4 Cryptographic Diffusion, 2D SAC (64x32, N=100,000), BIC (32x32),
// and Minimum Perturbation Distance Evaluation for `ra_core v2 Final (Opsi_A3)`.
//
// Compile:
//   gcc -O3 -march=native -mavx2 -std=gnu17 -include stdalign.h -fopenmp \
//       run_stage4_sac_bic_perturbation.c -lm -o run_stage4_sac_bic_perturbation

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "../../src/ra_core/c/ra_core_engine.c"

static inline uint64_t splitmix64_stateless(uint64_t idx) {
    uint64_t z = (idx + 1ULL) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

typedef struct {
    double mean_prob;
    double min_prob;
    double max_prob;
    double max_abs_bias;
    double rms_bias;
    double chi2;
    double chi2_per_dof;
} sac_summary_t;

static void summarize_sac_cells(const uint64_t *counts, int in_start, int in_end, int n_trials, sac_summary_t *out) {
    double sum_p = 0.0;
    double min_p = 1.0;
    double max_p = 0.0;
    double max_bias = 0.0;
    double sum_sq_bias = 0.0;
    double chi2 = 0.0;
    int n_cells = (in_end - in_start) * 32;
    double expected = 0.5 * (double)n_trials;
    double var = 0.25 * (double)n_trials;

    for (int ib = in_start; ib < in_end; ++ib) {
        for (int ob = 0; ob < 32; ++ob) {
            uint64_t c = counts[ib * 32 + ob];
            double p = (double)c / (double)n_trials;
            double bias = fabs(p - 0.5);
            sum_p += p;
            if (p < min_p) min_p = p;
            if (p > max_p) max_p = p;
            if (bias > max_bias) max_bias = bias;
            sum_sq_bias += (p - 0.5) * (p - 0.5);
            double diff = (double)c - expected;
            chi2 += (diff * diff) / var;
        }
    }
    out->mean_prob = sum_p / (double)n_cells;
    out->min_prob = min_p;
    out->max_prob = max_p;
    out->max_abs_bias = max_bias;
    out->rms_bias = sqrt(sum_sq_bias / (double)n_cells);
    out->chi2 = chi2;
    out->chi2_per_dof = chi2 / (double)n_cells;
}

int main(void) {
    // ========================================================================
    // 1. 2D SAC (64 input bits x 32 output bits) with N = 100,000 sample pairs
    //    Positions evaluated:
    //      - pos 0: K=1 / Word #0 (`i = 255`)
    //      - pos 1: K=255 Word #127 (`i = 128`, mid-Cycle 0)
    //      - pos 2: K=255 Word #254 (`i = 1`, end of Cycle 0)
    //      - pos 3: K=256 Word #255 (Cycle 1 Word #0, post-reseed/guard-fold)
    // ========================================================================
    const int N_SAC = 100000;
    const int POS_INDICES[4] = { 0, 127, 254, 255 };
    const char *POS_NAMES[4] = {
        "K1_word0",
        "K255_word127",
        "K255_word254",
        "K256_word255_cycle1"
    };

    uint64_t sac_counts[4][64 * 32];
    memset(sac_counts, 0, sizeof(sac_counts));

    #pragma omp parallel
    {
        uint64_t local_counts[4][64 * 32];
        memset(local_counts, 0, sizeof(local_counts));
        uint32_t base_w[256], mut_w[256];

        #pragma omp for schedule(static)
        for (int t = 0; t < N_SAC; ++t) {
            uint64_t seed = splitmix64_stateless((uint64_t)t);
            uint32_t bkey  = (uint32_t)(seed & 0xFFFFFFFFULL);
            uint32_t baddr = (uint32_t)(seed >> 32);

            ra_core_orbit_into(bkey, baddr, 256, base_w);

            for (int ib = 0; ib < 64; ++ib) {
                uint32_t mk = (ib < 32) ? (bkey ^ (1u << ib)) : bkey;
                uint32_t ma = (ib >= 32) ? (baddr ^ (1u << (ib - 32))) : baddr;
                ra_core_orbit_into(mk, ma, 256, mut_w);

                for (int pi = 0; pi < 4; ++pi) {
                    uint32_t diff = base_w[POS_INDICES[pi]] ^ mut_w[POS_INDICES[pi]];
                    uint64_t *row = &local_counts[pi][ib * 32];
                    for (int ob = 0; ob < 32; ++ob) {
                        row[ob] += (diff >> ob) & 1u;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int pi = 0; pi < 4; ++pi) {
                for (int idx = 0; idx < 64 * 32; ++idx) {
                    sac_counts[pi][idx] += local_counts[pi][idx];
                }
            }
        }
    }

    // ========================================================================
    // 2. Bit Independence Criterion (BIC 32x32)
    //    Evaluated over M_BIC = 10,000 random base addresses x 32 key bit flips
    //    (320,000 flips) and x 32 addr_hi bit flips (320,000 flips), plus
    //    combined 64 bit flips (640,000 flips) at words t in {0, 1, 7, 127, 254, 255}
    //    and lag-1 / lag-2 cross-step at t = 0, 127, 254.
    // ========================================================================
    const int M_BIC = 10000;
    const int BIC_STEPS[6] = { 0, 1, 7, 127, 254, 255 };
    // mode: 0 = key_flips (32), 1 = addr_hi_flips (32), 2 = all_64_flips (64)
    uint64_t bic_sum_u[3][6][32];
    uint64_t bic_sum_uv[3][6][32][32];
    // Cross-step lag-1 and lag-2 at t in {0, 127, 254} (all 64 flips)
    uint64_t bic_lag_u[2][3][32];
    uint64_t bic_lag_v[2][3][32];
    uint64_t bic_lag_uv[2][3][32][32];

    memset(bic_sum_u, 0, sizeof(bic_sum_u));
    memset(bic_sum_uv, 0, sizeof(bic_sum_uv));
    memset(bic_lag_u, 0, sizeof(bic_lag_u));
    memset(bic_lag_v, 0, sizeof(bic_lag_v));
    memset(bic_lag_uv, 0, sizeof(bic_lag_uv));

    #pragma omp parallel
    {
        uint64_t loc_u[3][6][32];
        uint64_t loc_uv[3][6][32][32];
        uint64_t loc_lag_u[2][3][32];
        uint64_t loc_lag_v[2][3][32];
        uint64_t loc_lag_uv[2][3][32][32];
        memset(loc_u, 0, sizeof(loc_u));
        memset(loc_uv, 0, sizeof(loc_uv));
        memset(loc_lag_u, 0, sizeof(loc_lag_u));
        memset(loc_lag_v, 0, sizeof(loc_lag_v));
        memset(loc_lag_uv, 0, sizeof(loc_lag_uv));

        uint32_t base_w[258], mut_w[258];
        const int LAG_T[3] = { 0, 127, 254 };

        #pragma omp for schedule(static)
        for (int tr = 0; tr < M_BIC; ++tr) {
            uint64_t seed = splitmix64_stateless((uint64_t)(tr + 500000));
            uint32_t bkey  = (uint32_t)(seed & 0xFFFFFFFFULL);
            uint32_t baddr = (uint32_t)(seed >> 32);
            ra_core_orbit_into(bkey, baddr, 258, base_w);

            for (int ib = 0; ib < 64; ++ib) {
                int dir = (ib < 32) ? 0 : 1;
                uint32_t mk = (ib < 32) ? (bkey ^ (1u << ib)) : bkey;
                uint32_t ma = (ib >= 32) ? (baddr ^ (1u << (ib - 32))) : baddr;
                ra_core_orbit_into(mk, ma, 258, mut_w);

                for (int si = 0; si < 6; ++si) {
                    uint32_t dy = base_w[BIC_STEPS[si]] ^ mut_w[BIC_STEPS[si]];
                    for (int u = 0; u < 32; ++u) {
                        uint32_t bu = (dy >> u) & 1u;
                        if (bu) {
                            loc_u[dir][si][u]++;
                            loc_u[2][si][u]++;
                            for (int v = 0; v < 32; ++v) {
                                uint32_t bv = (dy >> v) & 1u;
                                loc_uv[dir][si][u][v] += bv;
                                loc_uv[2][si][u][v]   += bv;
                            }
                        }
                    }
                }

                for (int li = 0; li < 3; ++li) {
                    int t = LAG_T[li];
                    uint32_t dy0 = base_w[t] ^ mut_w[t];
                    uint32_t dyl[2] = {
                        base_w[t + 1] ^ mut_w[t + 1],
                        base_w[t + 2] ^ mut_w[t + 2]
                    };
                    for (int lag = 0; lag < 2; ++lag) {
                        for (int u = 0; u < 32; ++u) {
                            uint32_t bu = (dy0 >> u) & 1u;
                            uint32_t bvu = (dyl[lag] >> u) & 1u;
                            loc_lag_u[lag][li][u] += bu;
                            loc_lag_v[lag][li][u] += bvu;
                            if (bu) {
                                for (int v = 0; v < 32; ++v) {
                                    loc_lag_uv[lag][li][u][v] += (dyl[lag] >> v) & 1u;
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int d = 0; d < 3; ++d)
                for (int s = 0; s < 6; ++s)
                    for (int u = 0; u < 32; ++u) {
                        bic_sum_u[d][s][u] += loc_u[d][s][u];
                        for (int v = 0; v < 32; ++v)
                            bic_sum_uv[d][s][u][v] += loc_uv[d][s][u][v];
                    }
            for (int l = 0; l < 2; ++l)
                for (int s = 0; s < 3; ++s)
                    for (int u = 0; u < 32; ++u) {
                        bic_lag_u[l][s][u] += loc_lag_u[l][s][u];
                        bic_lag_v[l][s][u] += loc_lag_v[l][s][u];
                        for (int v = 0; v < 32; ++v)
                            bic_lag_uv[l][s][u][v] += loc_lag_uv[l][s][u][v];
                    }
        }
    }

    // ========================================================================
    // 3. Minimum Perturbation Distance (1-bit flip, +1 increment, Gray code)
    //    over N_PERT = 10,000 pairs + 510-word Perturbation Curves
    // ========================================================================
    const int N_PERT = 10000;
    // Schemes:
    //   0: 1-bit flip in `key` (32 bits x N_PERT)
    //   1: 1-bit flip in `addr_hi` (32 bits x N_PERT)
    //   2: `key + 1` increment (N_PERT)
    //   3: `addr_hi + 1` increment (N_PERT)
    //   4: Gray code step on `key`: `k = m ^ (m >> 1)` vs `k' = (m+1) ^ ((m+1) >> 1)`
    //   5: Gray code step on `addr_hi`
    double pert_k1_sum_bits[6] = {0};
    int    pert_k1_min_bits[6] = { 32, 32, 32, 32, 32, 32 };
    int    pert_k1_max_bits[6] = { 0, 0, 0, 0, 0, 0 };
    double pert_c0_sum_bits[6] = {0}; // across 255 words (8160 bits)
    int    pert_c0_min_bits[6] = { 8160, 8160, 8160, 8160, 8160, 8160 };
    int    pert_c0_max_bits[6] = { 0, 0, 0, 0, 0, 0 };
    uint64_t pert_counts[6]    = { 0, 0, 0, 0, 0, 0 };

    for (int t = 0; t < N_PERT; ++t) {
        uint64_t seed = splitmix64_stateless((uint64_t)(t + 900000));
        uint32_t bkey  = (uint32_t)(seed & 0xFFFFFFFFULL);
        uint32_t baddr = (uint32_t)(seed >> 32);
        uint32_t base_w[255], mut_w[255];
        ra_core_singleblock_into(bkey, baddr, 255, base_w);

        // 0: 1-bit flip in key (sample bit t % 32, plus full 32 bits for first 500 seeds)
        int n_b = (t < 500) ? 32 : 1;
        for (int bi = 0; bi < n_b; ++bi) {
            int bit = (t < 500) ? bi : (t & 31);
            ra_core_singleblock_into(bkey ^ (1u << bit), baddr, 255, mut_w);
            int hd0 = __builtin_popcount(base_w[0] ^ mut_w[0]);
            int hdc = 0;
            for (int j = 0; j < 255; ++j) hdc += __builtin_popcount(base_w[j] ^ mut_w[j]);
            pert_k1_sum_bits[0] += hd0;
            if (hd0 < pert_k1_min_bits[0]) pert_k1_min_bits[0] = hd0;
            if (hd0 > pert_k1_max_bits[0]) pert_k1_max_bits[0] = hd0;
            pert_c0_sum_bits[0] += hdc;
            if (hdc < pert_c0_min_bits[0]) pert_c0_min_bits[0] = hdc;
            if (hdc > pert_c0_max_bits[0]) pert_c0_max_bits[0] = hdc;
            pert_counts[0]++;

            ra_core_singleblock_into(bkey, baddr ^ (1u << bit), 255, mut_w);
            hd0 = __builtin_popcount(base_w[0] ^ mut_w[0]);
            hdc = 0;
            for (int j = 0; j < 255; ++j) hdc += __builtin_popcount(base_w[j] ^ mut_w[j]);
            pert_k1_sum_bits[1] += hd0;
            if (hd0 < pert_k1_min_bits[1]) pert_k1_min_bits[1] = hd0;
            if (hd0 > pert_k1_max_bits[1]) pert_k1_max_bits[1] = hd0;
            pert_c0_sum_bits[1] += hdc;
            if (hdc < pert_c0_min_bits[1]) pert_c0_min_bits[1] = hdc;
            if (hdc > pert_c0_max_bits[1]) pert_c0_max_bits[1] = hdc;
            pert_counts[1]++;
        }

        // 2: key + 1
        ra_core_singleblock_into(bkey + 1u, baddr, 255, mut_w);
        {
            int hd0 = __builtin_popcount(base_w[0] ^ mut_w[0]);
            int hdc = 0;
            for (int j = 0; j < 255; ++j) hdc += __builtin_popcount(base_w[j] ^ mut_w[j]);
            pert_k1_sum_bits[2] += hd0;
            if (hd0 < pert_k1_min_bits[2]) pert_k1_min_bits[2] = hd0;
            if (hd0 > pert_k1_max_bits[2]) pert_k1_max_bits[2] = hd0;
            pert_c0_sum_bits[2] += hdc;
            if (hdc < pert_c0_min_bits[2]) pert_c0_min_bits[2] = hdc;
            if (hdc > pert_c0_max_bits[2]) pert_c0_max_bits[2] = hdc;
            pert_counts[2]++;
        }

        // 3: addr_hi + 1
        ra_core_singleblock_into(bkey, baddr + 1u, 255, mut_w);
        {
            int hd0 = __builtin_popcount(base_w[0] ^ mut_w[0]);
            int hdc = 0;
            for (int j = 0; j < 255; ++j) hdc += __builtin_popcount(base_w[j] ^ mut_w[j]);
            pert_k1_sum_bits[3] += hd0;
            if (hd0 < pert_k1_min_bits[3]) pert_k1_min_bits[3] = hd0;
            if (hd0 > pert_k1_max_bits[3]) pert_k1_max_bits[3] = hd0;
            pert_c0_sum_bits[3] += hdc;
            if (hdc < pert_c0_min_bits[3]) pert_c0_min_bits[3] = hdc;
            if (hdc > pert_c0_max_bits[3]) pert_c0_max_bits[3] = hdc;
            pert_counts[3]++;
        }

        // 4: Gray code step on key (m = t + 1)
        uint32_t g0 = (uint32_t)t ^ ((uint32_t)t >> 1);
        uint32_t g1 = (uint32_t)(t + 1) ^ ((uint32_t)(t + 1) >> 1);
        ra_core_singleblock_into(g0, baddr, 255, base_w);
        ra_core_singleblock_into(g1, baddr, 255, mut_w);
        {
            int hd0 = __builtin_popcount(base_w[0] ^ mut_w[0]);
            int hdc = 0;
            for (int j = 0; j < 255; ++j) hdc += __builtin_popcount(base_w[j] ^ mut_w[j]);
            pert_k1_sum_bits[4] += hd0;
            if (hd0 < pert_k1_min_bits[4]) pert_k1_min_bits[4] = hd0;
            if (hd0 > pert_k1_max_bits[4]) pert_k1_max_bits[4] = hd0;
            pert_c0_sum_bits[4] += hdc;
            if (hdc < pert_c0_min_bits[4]) pert_c0_min_bits[4] = hdc;
            if (hdc > pert_c0_max_bits[4]) pert_c0_max_bits[4] = hdc;
            pert_counts[4]++;
        }

        // 5: Gray code step on addr_hi
        ra_core_singleblock_into(bkey, g0, 255, base_w);
        ra_core_singleblock_into(bkey, g1, 255, mut_w);
        {
            int hd0 = __builtin_popcount(base_w[0] ^ mut_w[0]);
            int hdc = 0;
            for (int j = 0; j < 255; ++j) hdc += __builtin_popcount(base_w[j] ^ mut_w[j]);
            pert_k1_sum_bits[5] += hd0;
            if (hd0 < pert_k1_min_bits[5]) pert_k1_min_bits[5] = hd0;
            if (hd0 > pert_k1_max_bits[5]) pert_k1_max_bits[5] = hd0;
            pert_c0_sum_bits[5] += hdc;
            if (hdc < pert_c0_min_bits[5]) pert_c0_min_bits[5] = hdc;
            if (hdc > pert_c0_max_bits[5]) pert_c0_max_bits[5] = hdc;
            pert_counts[5]++;
        }
    }

    // Write JSON and print summary
    FILE *fj = fopen("data/sac_bic_perturbation.json", "w");
    if (!fj) {
        perror("fopen data/sac_bic_perturbation.json");
        return 1;
    }

    fprintf(fj, "{\n");
    fprintf(fj, "  \"sac_2d_64x32\": {\n");
    fprintf(fj, "    \"n_trials\": %d,\n", N_SAC);
    printf("\n=== Stage 4: 2D SAC (64x32 Matrix, N = %d Pairs) ===\n", N_SAC);
    printf("%-22s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n",
           "Position / Subset", "Mean p", "Min Cell", "Max Cell", "Max |Bias|", "Chi^2", "Chi^2/dof");
    printf("-----------------------+------------+------------+------------+------------+------------+-----------\n");

    for (int pi = 0; pi < 4; ++pi) {
        sac_summary_t s_key, s_addr, s_all;
        summarize_sac_cells(sac_counts[pi], 0, 32, N_SAC, &s_key);
        summarize_sac_cells(sac_counts[pi], 32, 64, N_SAC, &s_addr);
        summarize_sac_cells(sac_counts[pi], 0, 64, N_SAC, &s_all);

        printf("%-14s [key]   | %10.6f | %10.6f | %10.6f | %10.6f | %10.2f | %9.4f\n",
               POS_NAMES[pi], s_key.mean_prob, s_key.min_prob, s_key.max_prob, s_key.max_abs_bias, s_key.chi2, s_key.chi2_per_dof);
        printf("%-14s [addr]  | %10.6f | %10.6f | %10.6f | %10.6f | %10.2f | %9.4f\n",
               POS_NAMES[pi], s_addr.mean_prob, s_addr.min_prob, s_addr.max_prob, s_addr.max_abs_bias, s_addr.chi2, s_addr.chi2_per_dof);
        printf("%-14s [64x32] | %10.6f | %10.6f | %10.6f | %10.6f | %10.2f | %9.4f\n",
               POS_NAMES[pi], s_all.mean_prob, s_all.min_prob, s_all.max_prob, s_all.max_abs_bias, s_all.chi2, s_all.chi2_per_dof);

        fprintf(fj, "    \"%s\": {\n", POS_NAMES[pi]);
        fprintf(fj, "      \"key_32x32\": { \"mean_p\": %.6f, \"min_p\": %.6f, \"max_p\": %.6f, \"max_abs_bias\": %.6f, \"rms_bias\": %.6f, \"chi2\": %.2f, \"chi2_per_dof\": %.4f },\n",
                s_key.mean_prob, s_key.min_prob, s_key.max_prob, s_key.max_abs_bias, s_key.rms_bias, s_key.chi2, s_key.chi2_per_dof);
        fprintf(fj, "      \"addr_hi_32x32\": { \"mean_p\": %.6f, \"min_p\": %.6f, \"max_p\": %.6f, \"max_abs_bias\": %.6f, \"rms_bias\": %.6f, \"chi2\": %.2f, \"chi2_per_dof\": %.4f },\n",
                s_addr.mean_prob, s_addr.min_prob, s_addr.max_prob, s_addr.max_abs_bias, s_addr.rms_bias, s_addr.chi2, s_addr.chi2_per_dof);
        fprintf(fj, "      \"combined_64x32\": { \"mean_p\": %.6f, \"min_p\": %.6f, \"max_p\": %.6f, \"max_abs_bias\": %.6f, \"rms_bias\": %.6f, \"chi2\": %.2f, \"chi2_per_dof\": %.4f }\n",
                s_all.mean_prob, s_all.min_prob, s_all.max_prob, s_all.max_abs_bias, s_all.rms_bias, s_all.chi2, s_all.chi2_per_dof);
        fprintf(fj, "    }%s\n", (pi < 3) ? "," : "");
    }
    fprintf(fj, "  },\n");

    // BIC summary
    fprintf(fj, "  \"bic_32x32\": {\n");
    fprintf(fj, "    \"m_base_seeds\": %d,\n", M_BIC);
    printf("\n=== Stage 4: 32x32 Bit Independence Criterion (BIC, M = %d Seeds) ===\n", M_BIC);
    printf("%-16s | %-18s | %-18s | %-18s\n",
           "Word Position", "Key Max/Mean |rho|", "Addr Max/Mean |rho|", "64b Max/Mean |rho|");
    printf("-----------------+--------------------+--------------------+-------------------\n");

    const char *dir_labels[3] = { "key_flips", "addr_hi_flips", "combined_64_flips" };
    const double n_samps[3] = { (double)M_BIC * 32.0, (double)M_BIC * 32.0, (double)M_BIC * 64.0 };

    for (int si = 0; si < 6; ++si) {
        double max_r[3] = {0}, mean_r[3] = {0};
        for (int d = 0; d < 3; ++d) {
            double ns = n_samps[d];
            double sum_r = 0.0;
            int cnt = 0;
            for (int u = 0; u < 32; ++u) {
                double pu = (double)bic_sum_u[d][si][u] / ns;
                double su = sqrt(pu * (1.0 - pu));
                for (int v = 0; v < 32; ++v) {
                    if (u == v) continue;
                    double pv = (double)bic_sum_u[d][si][v] / ns;
                    double sv = sqrt(pv * (1.0 - pv));
                    double puv = (double)bic_sum_uv[d][si][u][v] / ns;
                    double rho = (su > 0 && sv > 0) ? fabs((puv - pu * pv) / (su * sv)) : 0.0;
                    if (rho > max_r[d]) max_r[d] = rho;
                    sum_r += rho;
                    ++cnt;
                }
            }
            mean_r[d] = sum_r / (double)cnt;
        }
        printf("Word #%-10d | %.5f / %.5f  | %.5f / %.5f  | %.5f / %.5f\n",
               BIC_STEPS[si], max_r[0], mean_r[0], max_r[1], mean_r[1], max_r[2], mean_r[2]);
        fprintf(fj, "    \"word_%d\": {\n", BIC_STEPS[si]);
        for (int d = 0; d < 3; ++d) {
            fprintf(fj, "      \"%s\": { \"max_abs_rho\": %.6f, \"mean_abs_rho\": %.6f }%s\n",
                    dir_labels[d], max_r[d], mean_r[d], (d < 2) ? "," : "");
        }
        fprintf(fj, "    },\n");
    }

    // Cross-step Lag-1 and Lag-2 BIC
    const int LAG_T[3] = { 0, 127, 254 };
    fprintf(fj, "    \"cross_step_bic\": {\n");
    for (int lag = 0; lag < 2; ++lag) {
        fprintf(fj, "      \"lag_%d\": {\n", lag + 1);
        for (int li = 0; li < 3; ++li) {
            double ns = (double)M_BIC * 64.0;
            double max_rho = 0.0, sum_rho = 0.0;
            for (int u = 0; u < 32; ++u) {
                double pu = (double)bic_lag_u[lag][li][u] / ns;
                double su = sqrt(pu * (1.0 - pu));
                for (int v = 0; v < 32; ++v) {
                    double pv = (double)bic_lag_v[lag][li][v] / ns;
                    double sv = sqrt(pv * (1.0 - pv));
                    double puv = (double)bic_lag_uv[lag][li][u][v] / ns;
                    double rho = (su > 0 && sv > 0) ? fabs((puv - pu * pv) / (su * sv)) : 0.0;
                    if (rho > max_rho) max_rho = rho;
                    sum_rho += rho;
                }
            }
            double mean_rho = sum_rho / 1024.0;
            printf("Lag-%d (t=%-3d)    | %-18s | %-18s | %.5f / %.5f\n",
                   lag + 1, LAG_T[li], "-", "-", max_rho, mean_rho);
            fprintf(fj, "        \"t_%d\": { \"max_abs_rho\": %.6f, \"mean_abs_rho\": %.6f }%s\n",
                    LAG_T[li], max_rho, mean_rho, (li < 2) ? "," : "");
        }
        fprintf(fj, "      }%s\n", (lag < 1) ? "," : "");
    }
    fprintf(fj, "    }\n");
    fprintf(fj, "  },\n");

    // Perturbation Distance summary
    static const char *pert_names[6] = {
        "key_1bit_flip",
        "addr_hi_1bit_flip",
        "key_plus_1",
        "addr_hi_plus_1",
        "key_gray_step",
        "addr_hi_gray_step"
    };
    fprintf(fj, "  \"minimum_perturbation_distance\": {\n");
    printf("\n=== Stage 4: Perturbation Hamming Distance Summary ===\n");
    printf("%-20s | %-22s | %-26s\n", "Perturbation Mode", "K=1 Word #0 (of 32b)", "Cycle-0 255w (of 8160b)");
    printf("---------------------+------------------------+---------------------------\n");
    for (int m = 0; m < 6; ++m) {
        double mean_k1 = pert_k1_sum_bits[m] / (double)pert_counts[m];
        double mean_c0 = pert_c0_sum_bits[m] / (double)pert_counts[m];
        printf("%-20s | mean=%5.2f [%2d..%2d]   | mean=%7.2f (%.5f) [%4d..%4d]\n",
               pert_names[m], mean_k1, pert_k1_min_bits[m], pert_k1_max_bits[m],
               mean_c0, mean_c0 / 8160.0, pert_c0_min_bits[m], pert_c0_max_bits[m]);
        fprintf(fj, "    \"%s\": {\n", pert_names[m]);
        fprintf(fj, "      \"n_pairs\": %llu,\n", (unsigned long long)pert_counts[m]);
        fprintf(fj, "      \"k1_mean_bits\": %.4f, \"k1_mean_frac\": %.6f, \"k1_min_bits\": %d, \"k1_max_bits\": %d,\n",
                mean_k1, mean_k1 / 32.0, pert_k1_min_bits[m], pert_k1_max_bits[m]);
        fprintf(fj, "      \"cycle0_mean_bits\": %.2f, \"cycle0_mean_frac\": %.6f, \"cycle0_min_bits\": %d, \"cycle0_max_bits\": %d\n",
                mean_c0, mean_c0 / 8160.0, pert_c0_min_bits[m], pert_c0_max_bits[m]);
        fprintf(fj, "    }%s\n", (m < 5) ? "," : "");
    }
    fprintf(fj, "  }\n");
    fprintf(fj, "}\n");
    fclose(fj);
    return 0;
}
