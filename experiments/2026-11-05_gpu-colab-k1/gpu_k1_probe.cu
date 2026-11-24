// ra_core v2 Final K=1 GPU probe. Exploratory: one draw per independent address.
// Uses the canonical A3 K=1 formula and the existing shootout's Philox4x32-10.
// Copyright (c) 2026 Hamas A. Rahman. Repository license applies.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef RA_HOST_ONLY
#include <cuda_runtime.h>
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

HD uint32_t rot32(uint32_t x, uint32_t r) {
    const uint32_t s = r & 31u;
    return (x << s) | (x >> ((0u - s) & 31u));
}

HD uint32_t fmix32(uint32_t h) {
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

HD uint32_t ra_k1(uint32_t key, uint32_t addr_hi) {
    const uint32_t p_addr = 0x9e3779b1u * addr_hi;
    const uint32_t u_m = 0x06a0dd9bu * key + p_addr;
    const uint32_t v_m = p_addr ^ rot32(key, 17);
    const uint32_t km_k = fmix32(0x6c26fc92u ^ u_m);
    const uint32_t km_p = fmix32(0x392456deu ^ v_m);
    const uint32_t k_mix = km_k ^ rot32(km_p, 13);
    const uint32_t step_m = (0x06a0dd9bu * (km_k + km_p)) | 1u;
    const uint32_t o = rot32(256u * step_m, k_mix ^ 255u)
                     ^ (rot32(2u * step_m, k_mix ^ 1u) << 1);
    const uint32_t a = o ^ (k_mix << 1);
    const uint32_t b = (k_mix + a) ^ o;
    const uint32_t c = rot32((a >> 13) ^ a, b);
    return c ^ (c >> 17);
}

HD void philox4x32_10(uint32_t out[4], const uint32_t in[4], const uint32_t kin[2]) {
    uint32_t c0 = in[0], c1 = in[1], c2 = in[2], c3 = in[3];
    uint32_t k0 = kin[0], k1 = kin[1];
#ifndef RA_HOST_ONLY
#pragma unroll 10
#endif
    for (int round = 0; round < 10; ++round) {
        const uint64_t p0 = uint64_t(c0) * 0xD2511F53u;
        const uint64_t p1 = uint64_t(c2) * 0xCD9E8D57u;
        const uint32_t n0 = uint32_t(p1 >> 32) ^ c1 ^ k0;
        const uint32_t n1 = uint32_t(p1);
        const uint32_t n2 = uint32_t(p0 >> 32) ^ c3 ^ k1;
        const uint32_t n3 = uint32_t(p0);
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += 0x9e3779b9u; k1 += 0xbb67ae85u;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

HD uint32_t philox_k1(uint32_t key, uint32_t addr_hi) {
    const uint32_t ctr[4] = {0u, 0u, 0u, 0u};
    const uint32_t pkey[2] = {key, addr_hi ^ 0xfeedc0deu};
    uint32_t out[4];
    philox4x32_10(out, ctr, pkey);
    return out[0];
}

static bool host_kat() {
    if (ra_k1(0u, 0u) != 0xc1dbbb12u || ra_k1(42u, 0u) != 0x9533450cu) {
        std::fprintf(stderr, "FAIL ra_core v2 Final frozen K=1 KAT\n");
        return false;
    }
    struct Kat { uint32_t ctr[4], key[2], want[4]; };
    const Kat cases[] = {
        {{0,0,0,0}, {0,0}, {0x6627e8d5u,0xe169c58du,0xbc57ac4cu,0x9b00dbd8u}},
        {{0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu},
         {0xffffffffu,0xffffffffu},
         {0x408f276du,0x41c83b0eu,0xa20bc7c6u,0x6d5451fdu}},
        {{0x243f6a88u,0x85a308d3u,0x13198a2eu,0x03707344u},
         {0xa4093822u,0x299f31d0u},
         {0xd16cfe09u,0x94fdccebu,0x5001e420u,0x24126ea1u}}
    };
    for (const Kat &k : cases) {
        uint32_t got[4];
        philox4x32_10(got, k.ctr, k.key);
        for (int i = 0; i < 4; ++i) if (got[i] != k.want[i]) {
            std::fprintf(stderr, "FAIL Philox4x32-10 KAT\n");
            return false;
        }
    }
    std::puts("PASS frozen ra_core K=1 and official Philox KAT");
    return true;
}

#ifdef RA_HOST_ONLY
int main() { return host_kat() ? 0 : 1; }
#else

#define CUDA_OK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::exit(2); \
    } \
} while (0)

constexpr int BLOCK = 256;
constexpr uint32_t KEY = 0x12345678u;
constexpr uint32_t CUT = 0x66666666u; // approximately 40% Bernoulli acceptance

__global__ void raw_ra(uint32_t *out, size_t n) {
    const size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < n) out[id] = ra_k1(KEY, uint32_t(id));
}

__global__ void raw_philox(uint32_t *out, size_t n) {
    const size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < n) out[id] = philox_k1(KEY, uint32_t(id));
}

__global__ void bern_ra(uint32_t *counts, size_t n) {
    __shared__ uint32_t votes[BLOCK];
    const size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    votes[threadIdx.x] = id < n && ra_k1(KEY, uint32_t(id)) < CUT;
    __syncthreads();
    for (int s = BLOCK / 2; s; s >>= 1) {
        if (threadIdx.x < s) votes[threadIdx.x] += votes[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[blockIdx.x] = votes[0];
}

__global__ void bern_philox(uint32_t *counts, size_t n) {
    __shared__ uint32_t votes[BLOCK];
    const size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    votes[threadIdx.x] = id < n && philox_k1(KEY, uint32_t(id)) < CUT;
    __syncthreads();
    for (int s = BLOCK / 2; s; s >>= 1) {
        if (threadIdx.x < s) votes[threadIdx.x] += votes[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[blockIdx.x] = votes[0];
}

static void launch(bool bernoulli, bool ra, uint32_t *out, size_t n) {
    const unsigned blocks = unsigned((n + BLOCK - 1) / BLOCK);
    if (bernoulli) {
        if (ra) bern_ra<<<blocks, BLOCK>>>(out, n);
        else bern_philox<<<blocks, BLOCK>>>(out, n);
    } else {
        if (ra) raw_ra<<<blocks, BLOCK>>>(out, n);
        else raw_philox<<<blocks, BLOCK>>>(out, n);
    }
    CUDA_OK(cudaGetLastError());
}

static float one_ms(bool bernoulli, bool ra, uint32_t *out, size_t n,
                    cudaEvent_t start, cudaEvent_t stop) {
    CUDA_OK(cudaEventRecord(start));
    launch(bernoulli, ra, out, n);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static void report(const char *mode, const char *generator, std::vector<float> times, size_t n) {
    std::sort(times.begin(), times.end());
    std::printf("RESULT mode=%s generator=%s n=%zu median_ms=%.4f min_ms=%.4f max_ms=%.4f samples_ms=",
                mode, generator, n, times[times.size()/2], times.front(), times.back());
    for (float v : times) std::printf("%.4f,", v);
    std::puts("");
}

static void benchmark(bool bernoulli, uint32_t *ra_out, uint32_t *philox_out, size_t n) {
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    for (int i = 0; i < 3; ++i) {
        launch(bernoulli, true, ra_out, n);
        launch(bernoulli, false, philox_out, n);
    }
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> ra_ms, philox_ms;
    for (int i = 0; i < 9; ++i) {
        if (i & 1) {
            philox_ms.push_back(one_ms(bernoulli, false, philox_out, n, start, stop));
            ra_ms.push_back(one_ms(bernoulli, true, ra_out, n, start, stop));
        } else {
            ra_ms.push_back(one_ms(bernoulli, true, ra_out, n, start, stop));
            philox_ms.push_back(one_ms(bernoulli, false, philox_out, n, start, stop));
        }
    }
    const char *mode = bernoulli ? "bernoulli_k1" : "raw_k1";
    report(mode, "ra_core_v2_final", ra_ms, n);
    report(mode, "philox4x32_10", philox_ms, n);
    if (bernoulli) {
        const size_t blocks = (n + BLOCK - 1) / BLOCK;
        std::vector<uint32_t> rc(blocks), pc(blocks);
        CUDA_OK(cudaMemcpy(rc.data(), ra_out, blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(pc.data(), philox_out, blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint64_t rs = 0, ps = 0;
        for (size_t i = 0; i < blocks; ++i) { rs += rc[i]; ps += pc[i]; }
        std::printf("CHECK bernoulli_count ra=%llu philox=%llu n=%zu\n",
                    (unsigned long long)rs, (unsigned long long)ps, n);
    }
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
}

int main(int argc, char **argv) {
    if (!host_kat()) return 1;
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf("DEVICE name=%s cc=%d.%d memory_bytes=%zu\n",
                prop.name, prop.major, prop.minor, size_t(prop.totalGlobalMem));
    constexpr size_t CHECK_N = 1024;
    uint32_t *ra_out = nullptr, *philox_out = nullptr;
    CUDA_OK(cudaMalloc(&ra_out, CHECK_N * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&philox_out, CHECK_N * sizeof(uint32_t)));
    launch(false, true, ra_out, CHECK_N);
    launch(false, false, philox_out, CHECK_N);
    std::vector<uint32_t> ra(CHECK_N), philox(CHECK_N);
    CUDA_OK(cudaMemcpy(ra.data(), ra_out, CHECK_N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(philox.data(), philox_out, CHECK_N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < CHECK_N; ++i) {
        if (ra[i] != ra_k1(KEY, uint32_t(i)) ||
            philox[i] != philox_k1(KEY, uint32_t(i))) {
            std::fprintf(stderr, "FAIL GPU/CPU parity at address %zu\n", i);
            return 1;
        }
    }
    std::puts("PASS first 1024 GPU/CPU K=1 outputs for both generators");
    CUDA_OK(cudaFree(ra_out)); CUDA_OK(cudaFree(philox_out));
    if (argc < 2 || std::string(argv[1]) == "--check") return 0;
    if (std::string(argv[1]) != "--bench") {
        std::fprintf(stderr, "Usage: gpu_k1_probe [--check | --bench [N]]\n");
        return 2;
    }
    size_t n = (size_t(1) << 24);
    if (argc >= 3) {
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(argv[2], &end, 10);
        if (!end || *end || parsed < 1024 || parsed > (1ull << 28)) {
            std::fprintf(stderr, "N must be an integer in [1024, 268435456]\n");
            return 2;
        }
        n = size_t(parsed);
    }
    const size_t bytes = n * sizeof(uint32_t);
    CUDA_OK(cudaMalloc(&ra_out, bytes));
    CUDA_OK(cudaMalloc(&philox_out, bytes));
    benchmark(false, ra_out, philox_out, n);
    benchmark(true, ra_out, philox_out, n);
    CUDA_OK(cudaFree(ra_out)); CUDA_OK(cudaFree(philox_out));
    return 0;
}
#endif
