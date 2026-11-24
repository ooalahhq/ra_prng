// Local differential test against the canonical production C header.
// This file is not needed in the self-contained Colab notebook.
#define restrict __restrict__
#include "../../src/ra_core/c/ra_core.h"
#undef restrict
#include "batch1_core.h"

#include <cstdint>
#include <cstdio>

static uint32_t next(uint32_t &x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

template<int K>
static bool check(uint32_t key, uint32_t addr) {
    uint32_t gpu_port[8] = {}, canonical[8] = {};
    batch1::ra_burst<K>(key, addr, gpu_port);
    if constexpr (K == 1) canonical[0] = ra_eval_k1_word(key, addr);
    else if constexpr (K == 2) ra_eval_k2_words(key, addr, canonical);
    else ra_eval_k3_to_k8_words(key, addr, K, canonical);
    for (int i = 0; i < K; ++i) if (gpu_port[i] != canonical[i]) {
        std::fprintf(stderr, "FAIL K=%d key=%08x addr=%08x index=%d got=%08x want=%08x\n",
                     K, key, addr, i, gpu_port[i], canonical[i]);
        return false;
    }
    return true;
}

int main() {
    uint32_t x = 0x6d2b79f5u;
    for (int i = 0; i < 10000; ++i) {
        const uint32_t key = next(x), addr = next(x);
        if (!check<1>(key, addr) || !check<2>(key, addr) ||
            !check<4>(key, addr) || !check<8>(key, addr)) return 1;
    }
    std::puts("PASS 10,000 key/address pairs x K={1,2,4,8} against production header");
    return 0;
}
