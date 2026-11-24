// ra_core_engine.h
// Public C prototypes for ra_core_engine.c (ra_core v2 Final: Opsi_A3 + eng_v2_E2),
// consumed by the Cython BitGenerator extension in pip-ra-core/ra_core/_bitgen.pyx,
// the ctypes layer in pip-ra-core/ra_core/_engine.py, and C consumers.
#ifndef RA_CORE_ENGINE_H
#define RA_CORE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

typedef struct ra_core_engine_s ra_core_engine_t;

size_t ra_core_engine_struct_size(void);

// Canonical stateful engine API (pip-ra-core & Cython BitGenerator)
void ra_core_engine_init(ra_core_engine_t *e, uint32_t key, uint32_t addr_hi);
void ra_core_engine_next(ra_core_engine_t *e, size_t n, uint32_t *out);
uint32_t ra_core_engine_next_u32(ra_core_engine_t *e);
uint64_t ra_core_engine_next_u64(ra_core_engine_t *e);
double ra_core_engine_next_double(ra_core_engine_t *e);
void ra_core_engine_discard(ra_core_engine_t *e, uint64_t z);
uint64_t ra_core_engine_position(const ra_core_engine_t *e);
void ra_core_engine_fork(const ra_core_engine_t *src, ra_core_engine_t *child);
void ra_core_engine_save(const ra_core_engine_t *e, uint8_t *out);
void ra_core_engine_load(ra_core_engine_t *e, const uint8_t *in);
uint32_t ra_core_engine_key(const ra_core_engine_t *e);
uint32_t ra_core_engine_addr_hi(const ra_core_engine_t *e);

// Canonical stateless buffer & SIMD API
uint32_t ra_core_eval_k1(uint32_t key, uint32_t addr_hi);
void ra_core_orbit_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t *out);
void ra_core_singleblock_into(uint32_t key, uint32_t addr_hi, size_t rng, uint32_t *out);
void ra_core_default_into(uint32_t key, uint32_t addr_hi, uint32_t *out, size_t rng);
void ra_core_singleblock_x8_into(const uint32_t keys[8], const uint32_t addr_his[8],
                                 size_t rng, uint32_t *out_interleaved);
void ra_core_x8(const uint32_t keys[8], const uint32_t addr_his[8],
                size_t rng, uint32_t *out_interleaved);
void ra_core_shuffle_indices(uint32_t key, size_t n, uint32_t *out_indices);

// Convenience short-name aliases (`ra_engine_*`)
void ra_engine_seed(ra_core_engine_t *e, uint32_t key);
void ra_engine_seed_ex(ra_core_engine_t *e, uint32_t key, uint32_t addr_hi);
uint32_t ra_engine_next_u32(ra_core_engine_t *e);
uint64_t ra_engine_next_u64(ra_core_engine_t *e);
double ra_engine_next_f64(ra_core_engine_t *e);
void ra_engine_fill_u32(ra_core_engine_t *e, uint32_t *out, size_t n);
void ra_engine_discard(ra_core_engine_t *e, uint64_t z);

#endif // RA_CORE_ENGINE_H
