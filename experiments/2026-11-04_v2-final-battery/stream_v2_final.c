// stream_v2_final.c
// High-throughput binary stream emitter on top of promoted `src/ra_core/c/ra_core_engine.c`
// for Dieharder (`dieharder -a -g 200`) and PractRand (`RNG_test stdin32`).
//
// Modes:
//   ./stream_v2_final continuous [key] [addr_hi]
//   ./stream_v2_final k1_weyl
//   ./stream_v2_final k16_seq_key
//   ./stream_v2_final k64_seq_key
//   ./stream_v2_final k255_seq_key
//   ./stream_v2_final k255_seq_addr

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#include "../../src/ra_core/c/ra_core_engine.c"

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    const char *mode = (argc >= 2) ? argv[1] : "continuous";
    uint32_t key = (argc >= 3) ? (uint32_t)strtoul(argv[2], NULL, 0) : 42u;
    uint32_t addr_hi = (argc >= 4) ? (uint32_t)strtoul(argv[3], NULL, 0) : 0u;

    #define BUF_WORDS 16320u // 64 * 255 words = 65,280 bytes
    alignas(64) uint32_t buf[BUF_WORDS];

    if (strcmp(mode, "continuous") == 0) {
        ra_core_engine_t eng;
        ra_engine_seed_ex(&eng, key, addr_hi);
        for (;;) {
            ra_engine_fill_u32(&eng, buf, BUF_WORDS);
            if (fwrite(buf, sizeof(uint32_t), BUF_WORDS, stdout) != BUF_WORDS) break;
        }
    } else if (strcmp(mode, "k1_weyl") == 0) {
        uint32_t k = 1u, a = 0u;
        for (;;) {
            for (size_t i = 0; i < BUF_WORDS; ++i) {
                buf[i] = ra_core_eval_k1(k, a);
                k += 0x9E3779B9u;
                a += 0x85EBCA6Bu;
            }
            if (fwrite(buf, sizeof(uint32_t), BUF_WORDS, stdout) != BUF_WORDS) break;
        }
    } else if (strcmp(mode, "k16_seq_key") == 0) {
        uint32_t m = 1u;
        for (;;) {
            for (size_t i = 0; i < BUF_WORDS; i += 16u) {
                ra_core_singleblock_into(m++, 0u, 16u, buf + i);
            }
            if (fwrite(buf, sizeof(uint32_t), BUF_WORDS, stdout) != BUF_WORDS) break;
        }
    } else if (strcmp(mode, "k64_seq_key") == 0) {
        uint32_t m = 1u;
        for (;;) {
            for (size_t i = 0; i < BUF_WORDS; i += 64u) {
                ra_core_singleblock_into(m++, 0u, 64u, buf + i);
            }
            if (fwrite(buf, sizeof(uint32_t), BUF_WORDS, stdout) != BUF_WORDS) break;
        }
    } else if (strcmp(mode, "k255_seq_key") == 0) {
        uint32_t m = 1u;
        for (;;) {
            for (size_t i = 0; i < BUF_WORDS; i += 255u) {
                ra_core_singleblock_into(m++, 0u, 255u, buf + i);
            }
            if (fwrite(buf, sizeof(uint32_t), BUF_WORDS, stdout) != BUF_WORDS) break;
        }
    } else if (strcmp(mode, "k255_seq_addr") == 0) {
        uint32_t m = 0u;
        for (;;) {
            for (size_t i = 0; i < BUF_WORDS; i += 255u) {
                ra_core_singleblock_into(0u, m++, 255u, buf + i);
            }
            if (fwrite(buf, sizeof(uint32_t), BUF_WORDS, stdout) != BUF_WORDS) break;
        }
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }
    return 0;
}
