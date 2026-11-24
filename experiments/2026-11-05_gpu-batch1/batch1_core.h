// Shared host/device formulas for the first ra_core v2 Final GPU batch.
// Adapted from src/ra_core/c/ra_core.h and the audited CPU shootout.
// Copyright (c) 2026 Hamas A. Rahman. Repository license applies.
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define B1_HD __host__ __device__ __forceinline__
#else
#define B1_HD inline
#endif

namespace batch1 {

constexpr uint32_t KEY = 0x12345678u;
constexpr uint32_t SWEEP_BASE = 0x20261104u;
constexpr uint32_t TC_DE4 = 736900509u;
constexpr uint32_t TC_DE8 = 126432030u;

B1_HD uint32_t rot32(uint32_t x, uint32_t r) {
    const uint32_t s = r & 31u;
    return (x << s) | (x >> ((0u - s) & 31u));
}

B1_HD uint32_t fmix32(uint32_t h) {
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

B1_HD void derive_m(uint32_t key, uint32_t addr, uint32_t &step, uint32_t &mix) {
    const uint32_t p = 0x9e3779b1u * addr;
    const uint32_t km_k = fmix32(0x6c26fc92u ^ (0x06a0dd9bu * key + p));
    const uint32_t km_p = fmix32(0x392456deu ^ (p ^ rot32(key, 17)));
    mix = km_k ^ rot32(km_p, 13);
    step = (0x06a0dd9bu * (km_k + km_p)) | 1u;
}

B1_HD uint32_t ra_k1(uint32_t key, uint32_t addr) {
    uint32_t step, mix;
    derive_m(key, addr, step, mix);
    const uint32_t o = rot32(256u * step, mix ^ 255u)
                     ^ (rot32(2u * step, mix ^ 1u) << 1);
    const uint32_t a = o ^ (mix << 1);
    const uint32_t b = (mix + a) ^ o;
    const uint32_t c = rot32((a >> 13) ^ a, b);
    return c ^ (c >> 17);
}

template<int K>
B1_HD void ra_burst(uint32_t key, uint32_t addr, uint32_t out[8]) {
    static_assert(K == 1 || K == 2 || K == 4 || K == 8, "supported K: 1,2,4,8");
    if constexpr (K == 1) {
        out[0] = ra_k1(key, addr);
    } else {
        uint32_t step, mix;
        derive_m(key, addr, step, mix);
        const uint32_t m1 = rot32(2u * step, mix ^ 1u);
        if constexpr (K == 2) {
            const uint32_t o0 = rot32(256u * step, mix ^ 255u) ^ (m1 << 1);
            const uint32_t o1 = rot32(255u * step, mix ^ 254u) ^ (o0 << 1);
            const uint32_t d1 = (rot32(mix, 15) ^ 0x55u) & 255u;
            const uint32_t a0 = o0 ^ (mix << 1);
            const uint32_t s1 = mix + a0;
            const uint32_t a1 = (d1 ^ o1) ^ s1;
            const uint32_t s2 = mix + a1;
            const uint32_t b0 = s1 ^ o0;
            const uint32_t b1 = s2 ^ (o1 + d1);
            const uint32_t c0 = rot32((a0 >> 13) ^ a0, b0);
            const uint32_t c1 = rot32((a1 >> 13) ^ a1, b1);
            out[0] = c0 ^ (c0 >> 17);
            out[1] = c1 ^ (c1 >> 17);
        } else {
            uint32_t i = 255u;
            uint32_t o0 = rot32(256u * step, mix ^ 255u) ^ (m1 << 1);
            uint32_t s = mix << 1;
            uint32_t d0 = 0u;
            uint32_t d1 = (rot32(mix, 15) ^ 0x55u) & 255u;
            int produced = 0;
            while (produced < K) {
                const uint32_t i1 = i - 1u;
                const uint32_t mim1 = rot32((i1 + 1u) * step, mix ^ i1);
                const uint32_t o1 = i == 249u
                    ? (mim1 ^ ((o0 ^ (m1 << 7)) << 1))
                    : (mim1 ^ (o0 << 1));
                const uint32_t a0 = (d0 ^ o0) ^ s;
                const uint32_t s1 = mix + a0;
                const uint32_t a1 = (d1 ^ o1) ^ s1;
                const uint32_t s2 = mix + a1;
                const uint32_t b0 = s1 ^ (o0 + d0);
                const uint32_t b1 = s2 ^ (o1 + d1);
                const uint32_t c0 = rot32((a0 >> 13) ^ a0, b0);
                const uint32_t c1 = rot32((a1 >> 13) ^ a1, b1);
                out[produced] = c0 ^ (c0 >> 17);
                out[produced + 1] = c1 ^ (c1 >> 17);
                produced += 2;
                if (produced == K) break;
                s = s2;
                d0 = out[produced - 2] & 255u;
                d1 = out[produced - 1] & 255u;
                const uint32_t i2 = i - 2u;
                const uint32_t mim2 = rot32((i2 + 1u) * step, mix ^ i2);
                o0 = mim2 ^ (o1 << 1);
                i -= 2u;
            }
        }
    }
}

B1_HD void philox_block(uint32_t out[4], const uint32_t in[4], const uint32_t kin[2]) {
    uint32_t c0 = in[0], c1 = in[1], c2 = in[2], c3 = in[3];
    uint32_t k0 = kin[0], k1 = kin[1];
#ifdef __CUDACC__
#pragma unroll 10
#endif
    for (int round = 0; round < 10; ++round) {
        const uint64_t p0 = uint64_t(c0) * 0xd2511f53u;
        const uint64_t p1 = uint64_t(c2) * 0xcd9e8d57u;
        const uint32_t n0 = uint32_t(p1 >> 32) ^ c1 ^ k0;
        const uint32_t n1 = uint32_t(p1);
        const uint32_t n2 = uint32_t(p0 >> 32) ^ c3 ^ k1;
        const uint32_t n3 = uint32_t(p0);
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += 0x9e3779b9u; k1 += 0xbb67ae85u;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

template<int K>
B1_HD void philox_burst(uint32_t key, uint32_t addr, uint32_t out[8]) {
    static_assert(K == 1 || K == 2 || K == 4 || K == 8, "supported K: 1,2,4,8");
    const uint32_t pkey[2] = {key, addr ^ 0xfeedc0deu};
    for (int b = 0; b < (K + 3) / 4; ++b) {
        const uint32_t ctr[4] = {uint32_t(b), 0u, 0u, 0u};
        uint32_t block[4];
        philox_block(block, ctr, pkey);
        for (int j = 0; j < 4 && 4 * b + j < K; ++j) out[4 * b + j] = block[j];
    }
}

B1_HD uint32_t philox_k1(uint32_t key, uint32_t addr) {
    uint32_t out[8]; philox_burst<1>(key, addr, out); return out[0];
}

template<int K, bool RA>
B1_HD uint32_t burst_accept_count(uint32_t key, uint32_t addr) {
    uint32_t words[8];
    if constexpr (RA) ra_burst<K>(key, addr, words);
    else philox_burst<K>(key, addr, words);
    uint32_t count = 0;
    for (int j = 0; j < K; ++j) count += words[j] < TC_DE4;
    return count;
}

B1_HD uint32_t site_key(uint32_t cell) { return cell * 2654435761u + 1u; }
B1_HD uint32_t sweep_addr(uint32_t sw) { return SWEEP_BASE + sw; }

} // namespace batch1
