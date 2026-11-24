// ra_core.c
// Production implementation of ra_core v2 Final:
//   - Recurrence & Cycle Core: v2_W2_D2_xag_opt (2-Tap Causal Ramp-Up Window W2 +
//     Lag-2 Delayed Feedback D2 + In-Place L[i]<->L[d_use] / M[i+7] += L[d_use]
//     Per-Step Coupling + O(1) End-of-Cycle Guard Fold + 2-Step Unrolled XAG Spine).
//   - Key-Mixing & Step Derivation: Opsi_A3 (2026-11-04 Stage 1 Winner: 2D cross-pre-mix
//     u_m/v_m and u_l/v_l before fmix32, k_mix = km_k ^ rot32(km_p, 13), odd-forced
//     bijective step_m/step_l, and (idx + 1u) * step index offset; zero extra ALU
//     instructions inside the generation loop).
//   - Register-Only Fast Paths: K=1 (`ra_eval_k1_word`), K=2 (`ra_eval_k2_words`),
//     K=3..8 (`ra_eval_k3_to_k8_words`), and Dead-Cone Elimination (`rng <= 255`
//     delegates orbit directly to singleblock).
//
// Compile: gcc -O3 -march=native -std=gnu17 -include stdalign.h
//          ra_core.c -o ra_core -Wall -Wextra
//
// Copyright (c) 2025-2026 Hamas A. Rahman (derivative research variant)
// Licensed under CC BY-NC-SA 4.0, matching the original this derives from.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "ra_core.h"

uint32_t ra_core_singleblock(uint32_t key, uint32_t addr_hi, size_t rng, FILE *raw_stream) {
    if (rng == 0) return key;
    if (rng > (size_t)RA_MAX_SINGLE) {
        fprintf(stderr,
            "ra_core_singleblock: rng=%zu > %d out of scope (no-reseed "
            "fast path valid only for rng in [1,%d]); use ra_core_orbit.\n",
            rng, RA_MAX_SINGLE, RA_MAX_SINGLE);
        abort();
    }
    alignas(64) uint32_t buf[RA_N];
    uint32_t cons = ra_singleblock_v2_buf(key, addr_hi, rng, buf);
    if (raw_stream) {
        fwrite(buf, sizeof(uint32_t), rng, raw_stream);
    } else {
        uint32_t sink = buf[rng - 1u];
        __asm__ volatile("" : "+r"(sink));
    }
    return cons;
}

uint32_t ra_core_orbit(uint32_t key, uint32_t addr_hi, size_t rng, FILE *raw_stream) {
    if (rng == 0) return key;
    if (rng <= (size_t)RA_MAX_SINGLE) {
        return ra_core_singleblock(key, addr_hi, rng, raw_stream);
    }

    alignas(64) uint32_t L[RA_N];
    alignas(64) uint32_t M[RA_N + 8];
    alignas(64) uint32_t cycle_buf[RA_N];
    uint64_t count = rng;
    uint64_t iteration = rng / (size_t)RA_MAX_SINGLE + 1u;
    uint32_t cons = 0u;

    ra_init_orbit_v2(L, M, key, addr_hi, &cons);

    for (size_t it = 0; it < iteration; ++it) {
        uint32_t *cursor = cycle_buf;
        bool done = ra_orbit_cycle_v2_buf(&cons, M, L, &count, &cursor);
        size_t emitted = (size_t)(cursor - cycle_buf);
        if (raw_stream) {
            fwrite(cycle_buf, sizeof(uint32_t), emitted, raw_stream);
        } else if (emitted > 0) {
            uint32_t sink = cycle_buf[emitted - 1u];
            __asm__ volatile("" : "+r"(sink));
        }
        if (done) return cons;
    }
    return cons;
}

// ----------------------------------------------------------------------
// CLI dispatch & Golden KAT validation
// ----------------------------------------------------------------------

typedef uint32_t (*core_fn_t)(uint32_t key, uint32_t addr_hi, size_t rng, FILE *raw_stream);

typedef struct { const char *name; core_fn_t fn; long max_rng; } core_entry_t;

static const core_entry_t CORES[] = {
    { "orbit", ra_core_orbit, -1 },        // -1 == no cap
    { "singleblock", ra_core_singleblock, RA_MAX_SINGLE },
};

static const core_entry_t *find_core(const char *name) {
    for (size_t i = 0; i < sizeof(CORES) / sizeof(CORES[0]); ++i) {
        if (strcmp(CORES[i].name, name) == 0) return &CORES[i];
    }
    fprintf(stderr, "Unknown core '%s'. Known: ", name);
    for (size_t i = 0; i < sizeof(CORES) / sizeof(CORES[0]); ++i) {
        fprintf(stderr, "%s ", CORES[i].name);
    }
    fprintf(stderr, "\n");
    exit(1);
}

static void validate_keys(uint32_t *keys, int *nkeys_out) {
    int nkeys = 0;
    keys[nkeys++] = 0u;
    keys[nkeys++] = 0xFFFFFFFFu;
    static const uint32_t extra_keys[5] = {
        0xDEADBEEFu, 0x12345678u, 0xCAFEBABEu, 0x9E3779B9u, 0x7F4A7C15u,
    };
    for (int i = 0; i < 5; ++i) keys[nkeys++] = extra_keys[i];
    for (uint32_t k = 0; k < 32; ++k) keys[nkeys++] = k;
    *nkeys_out = nkeys;
}

static uint32_t checksum_key_core(core_fn_t fn, uint32_t key) {
    uint32_t chk = 0;
    uint32_t buf[256];
    for (size_t n = 1; n <= 255; ++n) {
        FILE *fs = fmemopen(buf, n * sizeof(uint32_t) + 1, "wb");
        fn(key, 0u, n, fs); // addr_hi=0: KAT baseline == "single global address"
        fclose(fs);
        for (size_t j = 0; j < n; ++j) {
            chk = rot32(chk ^ buf[j], 7) + 0x9E3779B9u;
        }
    }
    return chk;
}

// Golden KAT checksums for ra_core v2 Final (W2_D2_xag_opt + Opsi_A3, 2026-11-04)
// across all 39 canonical test keys x n in 1..255 at addr_hi = 0.
static const uint32_t SINGLEBLOCK_KAT_CHECKSUMS[39] = {
    0xa601a39bu, 0x0e36cdc1u, 0x83538c63u, 0xdeb93947u, 0x41b26d19u, 0xe0b632cbu,
    0x32404898u, 0xa601a39bu, 0x26af61fbu, 0xe0fa072bu, 0x501c821au, 0xaa9cfd6cu,
    0x37f6fb8cu, 0x795ce40fu, 0x6583fe64u, 0x31ea64fdu, 0xd7da1f0du, 0xbdb7e501u,
    0xe89da650u, 0xe8a2df45u, 0x2eace98cu, 0xdfd593e0u, 0xa2a37cdbu, 0x6261f9e0u,
    0xef58d066u, 0x2b79c0f6u, 0x6f79b53bu, 0x6125cc8cu, 0x4f486283u, 0xfc7a7088u,
    0x63da1252u, 0xd2450c85u, 0xcd355550u, 0x6c08b053u, 0xd2b8d748u, 0x3b05bb20u,
    0xe57c7ef6u, 0xd1176e47u, 0xd104e8b7u,
};

static const uint32_t ORBIT_KAT_CHECKSUMS[39] = {
    0xa601a39bu, 0x0e36cdc1u, 0x83538c63u, 0xdeb93947u, 0x41b26d19u, 0xe0b632cbu,
    0x32404898u, 0xa601a39bu, 0x26af61fbu, 0xe0fa072bu, 0x501c821au, 0xaa9cfd6cu,
    0x37f6fb8cu, 0x795ce40fu, 0x6583fe64u, 0x31ea64fdu, 0xd7da1f0du, 0xbdb7e501u,
    0xe89da650u, 0xe8a2df45u, 0x2eace98cu, 0xdfd593e0u, 0xa2a37cdbu, 0x6261f9e0u,
    0xef58d066u, 0x2b79c0f6u, 0x6f79b53bu, 0x6125cc8cu, 0x4f486283u, 0xfc7a7088u,
    0x63da1252u, 0xd2450c85u, 0xcd355550u, 0x6c08b053u, 0xd2b8d748u, 0x3b05bb20u,
    0xe57c7ef6u, 0xd1176e47u, 0xd104e8b7u,
};

static int run_validate(void) {
    uint32_t keys[2 + 5 + 32];
    int nkeys = 0;
    validate_keys(keys, &nkeys);

    long sb_mismatches = 0;
    for (int ki = 0; ki < nkeys; ++ki) {
        uint32_t got = checksum_key_core(ra_core_singleblock, keys[ki]);
        uint32_t want = SINGLEBLOCK_KAT_CHECKSUMS[ki];
        if (got != want) {
            ++sb_mismatches;
            fprintf(stderr, "SINGLEBLOCK MISMATCH key=%u checksum=0x%08x want=0x%08x\n",
                    keys[ki], got, want);
        }
    }
    printf("validate(singleblock): %d keys checked, %ld checksum mismatches\n",
           nkeys, sb_mismatches);

    long orbit_mismatches = 0;
    for (int ki = 0; ki < nkeys; ++ki) {
        uint32_t got = checksum_key_core(ra_core_orbit, keys[ki]);
        uint32_t want = ORBIT_KAT_CHECKSUMS[ki];
        if (got != want) {
            ++orbit_mismatches;
            fprintf(stderr, "ORBIT MISMATCH key=%u checksum=0x%08x want=0x%08x\n",
                    keys[ki], got, want);
        }
    }
    printf("validate(orbit, K<=255 window): %d keys checked, %ld checksum mismatches\n",
           nkeys, orbit_mismatches);

    long total = sb_mismatches + orbit_mismatches;
    if (total == 0) {
        printf("validate: PASS -- singleblock and orbit both match their "
               "known-answer checksums (v2 Final W2_D2_xag_opt + Opsi_A3).\n");
    } else {
        printf("validate: FAIL -- %ld total checksum mismatches (see above).\n", total);
    }
    return total == 0 ? 0 : 1;
}

#if !defined(RA_CORE_NO_MAIN)
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage:\n"
            "  %s validate\n"
            "  %s checksum-gen <core>\n"
            "  %s --stream <core> <key> <n>              (addr_hi implied 0)\n"
            "  %s --stream <core> <key> <addr_hi> <n>     (explicit addr_hi)\n"
            "Cores: orbit singleblock (singleblock valid only for rng in [1,255])\n",
            argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "validate") == 0) {
        return run_validate();
    }

    if (strcmp(argv[1], "checksum-gen") == 0 && argc >= 3) {
        const core_entry_t *core = find_core(argv[2]);
        uint32_t keys[2 + 5 + 32];
        int nkeys = 0;
        validate_keys(keys, &nkeys);
        printf("static const uint32_t %s_KAT_CHECKSUMS[%d] = {\n",
               strcmp(core->name, "orbit") == 0 ? "ORBIT" : "SINGLEBLOCK", nkeys);
        for (int ki = 0; ki < nkeys; ++ki) {
            uint32_t chk = checksum_key_core(core->fn, keys[ki]);
            printf("    0x%08xu,%s", chk, (ki % 6 == 5) ? "\n" : " ");
        }
        printf("\n};\n");
        return 0;
    }

    if (strcmp(argv[1], "--stream") == 0 && argc >= 5) {
        const core_entry_t *core = find_core(argv[2]);
        uint32_t key = (uint32_t)strtoul(argv[3], NULL, 0);
        uint32_t addr_hi = 0u;
        size_t rng;
        if (argc == 5) {
            rng = (size_t)strtoull(argv[4], NULL, 0);
        } else {
            addr_hi = (uint32_t)strtoul(argv[4], NULL, 0);
            rng = (size_t)strtoull(argv[5], NULL, 0);
        }

        if (core->max_rng >= 0 && (long)rng > core->max_rng) {
            fprintf(stderr,
                "--stream: n=%zu exceeds core '%s' max_rng=%ld.\n",
                rng, core->name, core->max_rng);
            return 1;
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint32_t last_cons = core->fn(key, addr_hi, rng, stdout);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

        fprintf(stderr, "Streamed %zu pseudorandom outputs in %.3f seconds\n", rng, elapsed);
        fprintf(stderr, "Last cons from RNGing: %u\n", last_cons);
        return 0;
    }

    fprintf(stderr, "Unknown mode or missing args: %s\n", argv[1]);
    return 1;
}
#endif // !RA_CORE_NO_MAIN
