// First GPU batch: KAT, Ising K1, short bursts, and small/large crossover.
// Copyright (c) 2026 Hamas A. Rahman. Repository license applies.
#include "batch1_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace batch1;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static bool host_kat() {
    struct RaKat { uint32_t key, addr, words[8]; };
    const RaKat kats[] = {
        {0u,0u,{0xc1dbbb12u,0xfbede3b9u,0x93b1e2c8u,0x2067ac16u,
                 0xc477241fu,0x982a158du,0xd77dd2e4u,0x89763004u}},
        {42u,0u,{0x9533450cu,0x288d3bd2u,0x7e833e19u,0xff96b720u,
                  0x8cd7be6bu,0xe91267f5u,0xacc46c12u,0xe7271ff4u}},
        {1u,0u,{0xd8d07063u,0xd78f6289u,0x180d4a60u,0xe7ddc2c3u,
                 0x508aa139u,0x11c8ea1eu,0x9d8ce965u,0x37953664u}},
        {0xdeadbeefu,0x9e3779b9u,{0x5612413du,0xc09648c9u,0xd533833fu,0xee2fc833u,
                              0x09e9cec7u,0x3b89f8afu,0xf08fbf4au,0x648a24c5u}},
        {0xffffffffu,0xffffffffu,{0x4ada0f26u,0x78b8b5deu,0xc07bf57cu,0xb049a08eu,
                                   0xd1fa8279u,0xac4e4438u,0x66c12ecfu,0x90e051f1u}}
    };
    for (const RaKat &k : kats) {
        uint32_t out[8] = {};
        ra_burst<1>(k.key, k.addr, out);
        if (out[0] != k.words[0]) return false;
        ra_burst<2>(k.key, k.addr, out);
        for (int j = 0; j < 2; ++j) if (out[j] != k.words[j]) return false;
        ra_burst<4>(k.key, k.addr, out);
        for (int j = 0; j < 4; ++j) if (out[j] != k.words[j]) return false;
        ra_burst<8>(k.key, k.addr, out);
        for (int j = 0; j < 8; ++j) if (out[j] != k.words[j]) return false;
    }
    struct PhiloxKat {uint32_t ctr[4], key[2], words[4];};
    const PhiloxKat pkats[] = {
        {{0,0,0,0},{0,0},{0x6627e8d5u,0xe169c58du,0xbc57ac4cu,0x9b00dbd8u}},
        {{0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu},
         {0xffffffffu,0xffffffffu},{0x408f276du,0x41c83b0eu,0xa20bc7c6u,0x6d5451fdu}},
        {{0x243f6a88u,0x85a308d3u,0x13198a2eu,0x03707344u},
         {0xa4093822u,0x299f31d0u},{0xd16cfe09u,0x94fdccebu,0x5001e420u,0x24126ea1u}}
    };
    for (const PhiloxKat &k : pkats) {
        uint32_t got[4]; philox_block(got, k.ctr, k.key);
        for (int j = 0; j < 4; ++j) if (got[j] != k.words[j]) return false;
    }
    std::puts("PASS host frozen ra_core K1/K2/K4/K8 and official Philox KAT");
    return true;
}

static bool address_mapping_check() {
    // Odd multiplication is a bijection modulo 2^32; verify a bounded sample
    // of the exact (sweep address, site key) pairs used by Ising.
    std::unordered_set<uint64_t> seen;
    for (uint32_t sw = 0; sw < 4; ++sw) {
        for (uint32_t cell = 0; cell < 64u * 64u; ++cell) {
            const uint64_t pair = (uint64_t(sweep_addr(sw)) << 32) | site_key(cell);
            if (!seen.insert(pair).second) return false;
        }
    }
    std::puts("PASS unique address mapping for sampled Ising site/sweep pairs");
    return true;
}

struct IsingResult {
    std::vector<int8_t> spins;
    uint32_t rng_xor = 0;
    uint64_t accepted = 0;
    double energy_per_spin = 0;
    double abs_mag_per_spin = 0;
};

static std::vector<int8_t> initial_lattice(int L) {
    std::vector<int8_t> spins(size_t(L) * L);
    for (int cell = 0; cell < L * L; ++cell) {
        const uint32_t h = fmix32(uint32_t(cell) ^ SWEEP_BASE);
        spins[size_t(cell)] = (h & 3u) == 0u ? int8_t(-1) : int8_t(1);
    }
    return spins;
}

static void observables(IsingResult &r, int L) {
    int64_t energy = 0, mag = 0;
    const int mask = L - 1;
    for (int i = 0; i < L; ++i) {
        const int ip = (i + 1) & mask;
        for (int j = 0; j < L; ++j) {
            const int jp = (j + 1) & mask;
            const int s = r.spins[size_t(i) * L + j];
            mag += s;
            energy -= int64_t(s) * (r.spins[size_t(ip) * L + j] + r.spins[size_t(i) * L + jp]);
        }
    }
    r.energy_per_spin = double(energy) / double(L * L);
    r.abs_mag_per_spin = std::fabs(double(mag)) / double(L * L);
}

template<bool RA>
static IsingResult cpu_ising(const std::vector<int8_t> &initial, int L, int sweeps) {
    IsingResult r;
    r.spins = initial;
    const int mask = L - 1;
    for (int sw = 0; sw < sweeps; ++sw) {
        const uint32_t addr = sweep_addr(uint32_t(sw));
        for (int parity = 0; parity < 2; ++parity) {
            for (int i = 0; i < L; ++i) {
                const int im = (i - 1) & mask, ip = (i + 1) & mask;
                for (int j = (i ^ parity) & 1; j < L; j += 2) {
                    const int jm = (j - 1) & mask, jp = (j + 1) & mask;
                    const uint32_t key = site_key(uint32_t(i * L + j));
                    const uint32_t u = RA ? ra_k1(key, addr) : philox_k1(key, addr);
                    r.rng_xor ^= u;
                    const int s = r.spins[size_t(i) * L + j];
                    const int nn = r.spins[size_t(im) * L + j] + r.spins[size_t(ip) * L + j]
                                 + r.spins[size_t(i) * L + jm] + r.spins[size_t(i) * L + jp];
                    const int de = 2 * s * nn;
                    const bool accept = (de <= 0) | (u < (de == 4 ? TC_DE4 : TC_DE8));
                    if (accept) { r.spins[size_t(i) * L + j] = int8_t(-s); ++r.accepted; }
                }
            }
        }
    }
    observables(r, L);
    return r;
}

static bool host_ising_golden() {
    const auto initial = initial_lattice(256);
    const IsingResult ra = cpu_ising<true>(initial, 256, 32);
    const IsingResult ph = cpu_ising<false>(initial, 256, 32);
    const uint32_t ra_final = ra.rng_xor ^ uint32_t(ra.accepted);
    const uint32_t ph_final = ph.rng_xor ^ uint32_t(ph.accepted);
    std::printf("HOST_ISING_GOLDEN L=256 sweeps=32 ra_checksum=0x%08x philox_checksum=0x%08x ra_E=%.6f ra_M=%.6f philox_E=%.6f philox_M=%.6f\n",
                ra_final, ph_final, ra.energy_per_spin, ra.abs_mag_per_spin,
                ph.energy_per_spin, ph.abs_mag_per_spin);
    if (ra_final != 0x8210691eu || ph_final != 0xf8a28877u ||
        std::fabs(ra.energy_per_spin + 1.417175) > 1e-6 ||
        std::fabs(ra.abs_mag_per_spin - 0.582092) > 1e-6 ||
        std::fabs(ph.energy_per_spin + 1.401489) > 1e-6 ||
        std::fabs(ph.abs_mag_per_spin - 0.570770) > 1e-6) {
        std::fprintf(stderr, "FAIL audited CPU Ising checksum parity\n");
        return false;
    }
    std::puts("PASS audited CPU Ising checksum parity");
    return true;
}

#ifdef RA_HOST_ONLY
int main() {
    return host_kat() && address_mapping_check() && host_ising_golden() ? 0 : 1;
}
#else

#include <cuda_runtime.h>

#define CUDA_OK(expr) do { \
    const cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(2); \
    } \
} while (0)

constexpr int BLOCK = 256;
constexpr int ISING_SWEEPS = 32;

struct BlockAudit { uint32_t rng_xor, accepted; };

template<int K, bool RA>
__global__ void burst_kernel(uint32_t *out, size_t n) {
    const size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < n) out[id] = burst_accept_count<K, RA>(KEY, uint32_t(id));
}

template<int L, bool RA>
__global__ void ising_half_kernel(int8_t *spins, BlockAudit *audits, int sw, int parity,
                                  int pass, int blocks_per_pass) {
    // All batch sizes have L*L/2 divisible by BLOCK; every lane participates.
    const int id = int(blockIdx.x) * BLOCK + int(threadIdx.x);
    const int row = id / (L / 2);
    const int col = 2 * (id % (L / 2)) + ((row ^ parity) & 1);
    const int cell = row * L + col;
    const int im = (row - 1) & (L - 1), ip = (row + 1) & (L - 1);
    const int jm = (col - 1) & (L - 1), jp = (col + 1) & (L - 1);
    const uint32_t key = site_key(uint32_t(cell));
    const uint32_t addr = sweep_addr(uint32_t(sw));
    const uint32_t u = RA ? ra_k1(key, addr) : philox_k1(key, addr);
    const int s = spins[cell];
    const int nn = spins[im * L + col] + spins[ip * L + col]
                 + spins[row * L + jm] + spins[row * L + jp];
    const int de = 2 * s * nn;
    const uint32_t accept = uint32_t((de <= 0) | (u < (de == 4 ? TC_DE4 : TC_DE8)));
    if (accept) spins[cell] = int8_t(-s);

    // Consume every draw, including unconditional Metropolis accepts, while
    // producing one small correctness record per block and pass.
    uint32_t x = u, a = accept;
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        x ^= __shfl_xor_sync(0xffffffffu, x, off);
        a += __shfl_down_sync(0xffffffffu, a, off);
    }
    __shared__ uint32_t warp_xor[BLOCK / 32], warp_accept[BLOCK / 32];
    if (lane == 0) { warp_xor[warp] = x; warp_accept[warp] = a; }
    __syncthreads();
    if (warp == 0) {
        x = lane < BLOCK / 32 ? warp_xor[lane] : 0u;
        a = lane < BLOCK / 32 ? warp_accept[lane] : 0u;
        for (int off = 16; off > 0; off >>= 1) {
            x ^= __shfl_xor_sync(0xffffffffu, x, off);
            a += __shfl_down_sync(0xffffffffu, a, off);
        }
        if (lane == 0) audits[size_t(pass) * blocks_per_pass + blockIdx.x] = {x, a};
    }
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct BurstGpuRun { double kernel_ms, total_ms; };

static volatile uint64_t output_sink = 0;

__attribute__((noinline)) static uint64_t sum_output(const std::vector<uint32_t> &values) {
    uint64_t sum = 0;
    for (uint32_t v : values) sum += v;
    return sum;
}

template<int K, bool RA>
static BurstGpuRun gpu_burst_once(uint32_t *dout, std::vector<uint32_t> &host, size_t n,
                                  cudaEvent_t start, cudaEvent_t stop) {
    const auto wall0 = Clock::now();
    CUDA_OK(cudaEventRecord(start));
    const unsigned blocks = unsigned((n + BLOCK - 1) / BLOCK);
    burst_kernel<K, RA><<<blocks, BLOCK>>>(dout, n);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CUDA_OK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_OK(cudaMemcpy(host.data(), dout, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const auto wall1 = Clock::now();
    output_sink ^= sum_output(host);
    return {double(kernel_ms), elapsed_ms(wall0, wall1)};
}

template<int K, bool RA>
static bool burst_gpu_parity(uint32_t *dout, std::vector<uint32_t> &host,
                             cudaEvent_t start, cudaEvent_t stop) {
    gpu_burst_once<K, RA>(dout, host, 1024, start, stop);
    for (uint32_t i = 0; i < 1024; ++i) {
        const uint32_t expected = burst_accept_count<K, RA>(KEY, i);
        if (host[i] != expected) {
            std::fprintf(stderr, "FAIL GPU/CPU burst parity K=%d gen=%s address=%u\n",
                         K, RA ? "ra" : "philox", i);
            return false;
        }
    }
    return true;
}

static bool gpu_burst_check() {
    uint32_t *dout = nullptr;
    CUDA_OK(cudaMalloc(&dout, 1024 * sizeof(uint32_t)));
    std::vector<uint32_t> host(1024);
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    const bool okay = burst_gpu_parity<1, true>(dout, host, start, stop)
                   && burst_gpu_parity<1, false>(dout, host, start, stop)
                   && burst_gpu_parity<2, true>(dout, host, start, stop)
                   && burst_gpu_parity<2, false>(dout, host, start, stop)
                   && burst_gpu_parity<4, true>(dout, host, start, stop)
                   && burst_gpu_parity<4, false>(dout, host, start, stop)
                   && burst_gpu_parity<8, true>(dout, host, start, stop)
                   && burst_gpu_parity<8, false>(dout, host, start, stop);
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
    CUDA_OK(cudaFree(dout));
    if (okay) std::puts("PASS GPU/CPU K1/K2/K4/K8 burst consumer parity, first 1024 addresses");
    return okay;
}

template<int L>
struct IsingGpuContext {
    static constexpr int blocks_per_pass = (L * L / 2) / BLOCK;
    int8_t *dspins = nullptr;
    BlockAudit *daudits = nullptr;
    std::vector<int8_t> initial;
    std::vector<BlockAudit> audits;
    cudaEvent_t start = nullptr, stop = nullptr;

    explicit IsingGpuContext(int sweeps) : initial(initial_lattice(L)),
        audits(size_t(sweeps) * 2u * size_t(blocks_per_pass)) {
        static_assert(((L * L / 2) % BLOCK) == 0, "no partial warps");
        CUDA_OK(cudaMalloc(&dspins, size_t(L) * L * sizeof(int8_t)));
        CUDA_OK(cudaMalloc(&daudits, audits.size() * sizeof(BlockAudit)));
        CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    }
    ~IsingGpuContext() {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(dspins); cudaFree(daudits);
    }
};

struct IsingGpuRun { IsingResult result; double kernel_ms, total_ms; };

template<int L>
static IsingGpuRun gpu_ising_once(IsingGpuContext<L> &ctx, bool ra, int sweeps) {
    const auto wall0 = Clock::now();
    CUDA_OK(cudaMemcpy(ctx.dspins, ctx.initial.data(), ctx.initial.size() * sizeof(int8_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(ctx.start));
    for (int sw = 0; sw < sweeps; ++sw) {
        for (int parity = 0; parity < 2; ++parity) {
            const int pass = 2 * sw + parity;
            if (ra) ising_half_kernel<L, true><<<ctx.blocks_per_pass, BLOCK>>>(
                ctx.dspins, ctx.daudits, sw, parity, pass, ctx.blocks_per_pass);
            else ising_half_kernel<L, false><<<ctx.blocks_per_pass, BLOCK>>>(
                ctx.dspins, ctx.daudits, sw, parity, pass, ctx.blocks_per_pass);
            CUDA_OK(cudaGetLastError());
        }
    }
    CUDA_OK(cudaEventRecord(ctx.stop));
    CUDA_OK(cudaEventSynchronize(ctx.stop));
    float kernel_ms = 0;
    CUDA_OK(cudaEventElapsedTime(&kernel_ms, ctx.start, ctx.stop));
    IsingResult result;
    result.spins.resize(ctx.initial.size());
    CUDA_OK(cudaMemcpy(result.spins.data(), ctx.dspins,
                       result.spins.size() * sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ctx.audits.data(), ctx.daudits,
                       ctx.audits.size() * sizeof(BlockAudit), cudaMemcpyDeviceToHost));
    for (const BlockAudit &b : ctx.audits) {
        result.rng_xor ^= b.rng_xor;
        result.accepted += b.accepted;
    }
    observables(result, L);
    const auto wall1 = Clock::now();
    return {std::move(result), double(kernel_ms), elapsed_ms(wall0, wall1)};
}

static bool same_ising(const IsingResult &gpu, const IsingResult &cpu, int L,
                       int sweeps, const char *generator) {
    if (gpu.spins != cpu.spins || gpu.rng_xor != cpu.rng_xor || gpu.accepted != cpu.accepted ||
        gpu.energy_per_spin != cpu.energy_per_spin || gpu.abs_mag_per_spin != cpu.abs_mag_per_spin) {
        std::fprintf(stderr, "FAIL Ising GPU/CPU parity L=%d sweeps=%d gen=%s\n",
                     L, sweeps, generator);
        return false;
    }
    return true;
}

static bool gpu_ising_check() {
    constexpr int L = 64, SWEEPS = 2;
    IsingGpuContext<L> ctx(SWEEPS);
    const IsingResult ra_cpu = cpu_ising<true>(ctx.initial, L, SWEEPS);
    const IsingResult ph_cpu = cpu_ising<false>(ctx.initial, L, SWEEPS);
    const IsingGpuRun ra_gpu = gpu_ising_once(ctx, true, SWEEPS);
    const IsingGpuRun ph_gpu = gpu_ising_once(ctx, false, SWEEPS);
    const bool okay = same_ising(ra_gpu.result, ra_cpu, L, SWEEPS, "ra")
                   && same_ising(ph_gpu.result, ph_cpu, L, SWEEPS, "philox");
    if (okay) std::puts("PASS GPU/CPU true Ising spins, draw XOR, accepts and observables");
    return okay;
}

template<int K, bool RA>
static double cpu_burst_once(std::vector<uint32_t> &host) {
    const auto t0 = Clock::now();
    for (size_t i = 0; i < host.size(); ++i)
        host[i] = burst_accept_count<K, RA>(KEY, uint32_t(i));
    const auto t1 = Clock::now();
    output_sink ^= sum_output(host);
    return elapsed_ms(t0, t1);
}

template<int K>
static void bench_burst(size_t n, bool include_cpu) {
    uint32_t *ra_out = nullptr, *ph_out = nullptr;
    CUDA_OK(cudaMalloc(&ra_out, n * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&ph_out, n * sizeof(uint32_t)));
    std::vector<uint32_t> ra_host(n), ph_host(n);
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    gpu_burst_once<K, true>(ra_out, ra_host, n, start, stop);
    gpu_burst_once<K, false>(ph_out, ph_host, n, start, stop);
    std::vector<double> ra_kernel, ph_kernel, ra_total, ph_total;
    for (int rep = 0; rep < 7; ++rep) {
        auto run_ra = [&] {
            const BurstGpuRun r = gpu_burst_once<K, true>(ra_out, ra_host, n, start, stop);
            ra_kernel.push_back(r.kernel_ms); ra_total.push_back(r.total_ms);
            std::printf("SAMPLE test=burst k=%d n=%zu gen=ra impl=gpu rep=%d kernel_ms=%.6f total_ms=%.6f\n",
                        K, n, rep, r.kernel_ms, r.total_ms);
        };
        auto run_ph = [&] {
            const BurstGpuRun r = gpu_burst_once<K, false>(ph_out, ph_host, n, start, stop);
            ph_kernel.push_back(r.kernel_ms); ph_total.push_back(r.total_ms);
            std::printf("SAMPLE test=burst k=%d n=%zu gen=philox impl=gpu rep=%d kernel_ms=%.6f total_ms=%.6f\n",
                        K, n, rep, r.kernel_ms, r.total_ms);
        };
        if (rep & 1) { run_ph(); run_ra(); }
        else { run_ra(); run_ph(); }
    }
    std::printf("SUMMARY test=burst k=%d n=%zu gen=ra kernel_median_ms=%.6f total_median_ms=%.6f\n",
                K, n, median(ra_kernel), median(ra_total));
    std::printf("SUMMARY test=burst k=%d n=%zu gen=philox kernel_median_ms=%.6f total_median_ms=%.6f\n",
                K, n, median(ph_kernel), median(ph_total));
    if (include_cpu) {
        std::vector<double> ra_cpu, ph_cpu;
        for (int rep = 0; rep < 3; ++rep) {
            auto run_ra = [&] {
                const double ms = cpu_burst_once<K, true>(ra_host);
                ra_cpu.push_back(ms);
                std::printf("SAMPLE test=burst k=%d n=%zu gen=ra impl=cpu rep=%d total_ms=%.6f\n", K, n, rep, ms);
            };
            auto run_ph = [&] {
                const double ms = cpu_burst_once<K, false>(ph_host);
                ph_cpu.push_back(ms);
                std::printf("SAMPLE test=burst k=%d n=%zu gen=philox impl=cpu rep=%d total_ms=%.6f\n", K, n, rep, ms);
            };
            if (rep & 1) { run_ph(); run_ra(); }
            else { run_ra(); run_ph(); }
        }
        std::printf("SUMMARY test=burst k=%d n=%zu gen=ra impl=cpu total_median_ms=%.6f\n",
                    K, n, median(ra_cpu));
        std::printf("SUMMARY test=burst k=%d n=%zu gen=philox impl=cpu total_median_ms=%.6f\n",
                    K, n, median(ph_cpu));
    }
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
    CUDA_OK(cudaFree(ra_out)); CUDA_OK(cudaFree(ph_out));
}

template<int L>
static void bench_ising() {
    constexpr int sweeps = ISING_SWEEPS;
    IsingGpuContext<L> ctx(sweeps);
    std::vector<double> ra_cpu_ms, ph_cpu_ms;
    IsingResult ra_ref, ph_ref;
    for (int rep = 0; rep < 3; ++rep) {
        auto run_ra = [&] {
            const auto t0 = Clock::now();
            IsingResult r = cpu_ising<true>(ctx.initial, L, sweeps);
            const auto t1 = Clock::now();
            if (rep == 0) ra_ref = r;
            else if (r.spins != ra_ref.spins || r.rng_xor != ra_ref.rng_xor || r.accepted != ra_ref.accepted) {
                std::fprintf(stderr, "FAIL CPU Ising determinism, ra L=%d\n", L); std::exit(1);
            }
            const double ms = elapsed_ms(t0, t1);
            ra_cpu_ms.push_back(ms);
            std::printf("SAMPLE test=ising L=%d sweeps=%d gen=ra impl=cpu rep=%d total_ms=%.6f\n",
                        L, sweeps, rep, ms);
        };
        auto run_ph = [&] {
            const auto t0 = Clock::now();
            IsingResult r = cpu_ising<false>(ctx.initial, L, sweeps);
            const auto t1 = Clock::now();
            if (rep == 0) ph_ref = r;
            else if (r.spins != ph_ref.spins || r.rng_xor != ph_ref.rng_xor || r.accepted != ph_ref.accepted) {
                std::fprintf(stderr, "FAIL CPU Ising determinism, philox L=%d\n", L); std::exit(1);
            }
            const double ms = elapsed_ms(t0, t1);
            ph_cpu_ms.push_back(ms);
            std::printf("SAMPLE test=ising L=%d sweeps=%d gen=philox impl=cpu rep=%d total_ms=%.6f\n",
                        L, sweeps, rep, ms);
        };
        if (rep & 1) { run_ph(); run_ra(); }
        else { run_ra(); run_ph(); }
    }
    std::printf("PHYSICS test=ising L=%d sweeps=%d gen=ra E=%.6f abs_M=%.6f accepted=%llu draws=%llu checksum=0x%08x\n",
                L, sweeps, ra_ref.energy_per_spin, ra_ref.abs_mag_per_spin,
                (unsigned long long)ra_ref.accepted, (unsigned long long)L * L * sweeps,
                ra_ref.rng_xor ^ uint32_t(ra_ref.accepted));
    std::printf("PHYSICS test=ising L=%d sweeps=%d gen=philox E=%.6f abs_M=%.6f accepted=%llu draws=%llu checksum=0x%08x\n",
                L, sweeps, ph_ref.energy_per_spin, ph_ref.abs_mag_per_spin,
                (unsigned long long)ph_ref.accepted, (unsigned long long)L * L * sweeps,
                ph_ref.rng_xor ^ uint32_t(ph_ref.accepted));

    const IsingGpuRun warm_ra = gpu_ising_once(ctx, true, sweeps);
    const IsingGpuRun warm_ph = gpu_ising_once(ctx, false, sweeps);
    if (!same_ising(warm_ra.result, ra_ref, L, sweeps, "ra") ||
        !same_ising(warm_ph.result, ph_ref, L, sweeps, "philox")) std::exit(1);
    std::vector<double> ra_kernel, ph_kernel, ra_total, ph_total;
    for (int rep = 0; rep < 5; ++rep) {
        auto run_ra = [&] {
            const IsingGpuRun r = gpu_ising_once(ctx, true, sweeps);
            if (!same_ising(r.result, ra_ref, L, sweeps, "ra")) std::exit(1);
            ra_kernel.push_back(r.kernel_ms); ra_total.push_back(r.total_ms);
            std::printf("SAMPLE test=ising L=%d sweeps=%d gen=ra impl=gpu rep=%d kernel_ms=%.6f total_ms=%.6f\n",
                        L, sweeps, rep, r.kernel_ms, r.total_ms);
        };
        auto run_ph = [&] {
            const IsingGpuRun r = gpu_ising_once(ctx, false, sweeps);
            if (!same_ising(r.result, ph_ref, L, sweeps, "philox")) std::exit(1);
            ph_kernel.push_back(r.kernel_ms); ph_total.push_back(r.total_ms);
            std::printf("SAMPLE test=ising L=%d sweeps=%d gen=philox impl=gpu rep=%d kernel_ms=%.6f total_ms=%.6f\n",
                        L, sweeps, rep, r.kernel_ms, r.total_ms);
        };
        if (rep & 1) { run_ph(); run_ra(); }
        else { run_ra(); run_ph(); }
    }
    std::printf("SUMMARY test=ising L=%d sweeps=%d gen=ra impl=gpu kernel_median_ms=%.6f total_median_ms=%.6f cpu_median_ms=%.6f\n",
                L, sweeps, median(ra_kernel), median(ra_total), median(ra_cpu_ms));
    std::printf("SUMMARY test=ising L=%d sweeps=%d gen=philox impl=gpu kernel_median_ms=%.6f total_median_ms=%.6f cpu_median_ms=%.6f\n",
                L, sweeps, median(ph_kernel), median(ph_total), median(ph_cpu_ms));
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (!host_kat() || !address_mapping_check() || !host_ising_golden()) return 1;
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    int runtime_version = 0, driver_version = 0;
    CUDA_OK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_OK(cudaDriverGetVersion(&driver_version));
    std::printf("DEVICE name=%s cc=%d.%d memory_bytes=%zu cuda_runtime=%d cuda_driver=%d\n",
                prop.name, prop.major, prop.minor, size_t(prop.totalGlobalMem),
                runtime_version, driver_version);
    if (!gpu_burst_check() || !gpu_ising_check()) return 1;
    if (argc == 1 || std::string(argv[1]) == "--check") return 0;
    if (std::string(argv[1]) != "--bench") {
        std::fprintf(stderr, "Usage: batch1 [--check | --bench]\n"); return 2;
    }
    std::puts("BATCH_START tests=ising,burst,crossover sequential=true");
    bench_ising<64>();
    bench_ising<256>();
    bench_ising<1024>();
    bench_burst<2>(size_t(1) << 22, false);
    bench_burst<4>(size_t(1) << 22, false);
    bench_burst<8>(size_t(1) << 22, false);
    for (size_t n : {size_t(4096), size_t(65536), size_t(1) << 20, size_t(1) << 22})
        bench_burst<1>(n, true);
    std::printf("BATCH_DONE output_sink=%llu\n", (unsigned long long)output_sink);
    return 0;
}

#endif
