# `ra_core v2`: Deferred Prefix Computation and 2D Cross-Pre-Mixed Seeding for Addressable Permutation-Stream Pseudorandom Number Generation

**Author:** Hamas A. Rahman  
**Version:** v2.0 Final (`W2_D2_xag_opt` + `Opsi_A3` + `eng_v2_E2`, 2026-11-05)  
**Repository:** `ra_prng` (`src/ra_core/c/`, `pip-ra-core/`)  
**License:** CC BY-NC-SA 4.0

---

## Abstract

Monte Carlo simulations, parallel lattice updates, and variable-length stochastic algorithms frequently require pseudorandom streams indexed by logical 2D coordinates $(\text{key}, \text{addr\_hi})$ whose consumed burst length $K$ ranges from one or two words ($K=1, 2$) to long continuous trajectories ($K \ge 256$). Counter-based random-access PRNGs (such as Philox4x32-10 in Random123 and ChaCha8) provide stateless $O(1)$ logical addressing and schedule-independent reproducibility, but evaluate full multi-round 4-word or 16-word cipher blocks even when a trajectory terminates after $K=1$ or $K=2$ draws. Conversely, stateful linear or permutation-array generators (such as Xoshiro256**, PCG64-DXSM, and the predecessor hybrid `ra_core v1`) amortize sequential generation over a multi-word state, but normally require $O(N)$ state initialization per coordinate or exhibit structural correlation defects when seeded ad-hoc on structured 2D coordinates. Specifically, `ra_core v1` suffered from three diagnostic bottlenecks: (i) independent 1D pre-mixing before `fmix32` and an unshifted $i \cdot \text{step}$ origin that left $M[0]=0$ invariant and collapsed $(k_{\text{mix}}, \text{step}_m)$ onto a 1D affine relationship whenever either coordinate was held constant; (ii) a 7-tap cyclic bootstrap, a lag-1 serial feedback dependency $c_t \to d_t \to o_{t+1} \to a_{t+1}$, and an $O(N)$ end-of-cycle reseed scan (`ra_reseed` + `ra_hash`); and (iii) a scalar one-word-at-a-time On-The-Fly (OTF) engine that regressed at mid-$K$ burst lengths ($K=8..255$).

We present **`ra_core v2`**, a mathematical and systems redesign centered on **deferred computation of a stream prefix** within a permutation-coupled state architecture:
1. **`Opsi_A3` 2D Cross-Pre-Mix Key-Mixing with Odd-Forced Bijective Index Stepping:** Cross-mixes $(\text{key}, \text{addr\_hi})$ into both branches ($u_m, v_m$ and $u_l, v_l$) *before* the two independent `fmix32` avalanches, enforces odd Weyl increments ($\text{step}_m, \text{step}_l \equiv 1 \pmod 2$) so that the per-index map $i \mapsto (i+1)\cdot\text{step} \bmod 2^{32}$ is bijective over $i$, and eliminates the fixed-point zero slot at $i=0$ with **zero extra ALU instructions** inside the generation loop.
2. **`W2_D2_xag_opt` Causal 2-Tap Ramp-Up, 2-Step Unrolled Recurrence, & In-Place Coupling:** Replaces the 7-tap cyclic wrap-around bootstrap with a **2-tap causal ramp-up window ($W2$)** initialized with zeroed guard slots ($M[257..262]=0$) so that word #1 ($K=1$) depends on only 2 state words and word #2 ($K=2$) on 3 state words; introduces **lag-2 delayed feedback ($D2$, $d_{\text{use}, t} = d_{t-1}$)** to decouple adjacent scrambler evaluations into a **2-step unrolled spine**; and fuses the inter-cycle $L \to M$ coupling directly into the orbit loop ($M[i+7] \mathrel{+}= L[d_{\text{use}}]$) followed by an **$O(1)$ 7-word guard fold** at cycle boundaries—eliminating `ra_reseed` and `ra_hash`.
3. **`eng_v2_E2` Stateful On-The-Fly (OTF) Engine for Deferred Prefix Computation:** Defers the computational work of $O(N)$ array initialization within a preallocated `3,520`-byte engine struct by evaluating $K=1$ and $K=2$ in registers, stepping $K=3..254$ in **Even-$K$-Aligned 2-Step Coupled OTF Pairs** over an 8-word ring buffer (`m_ring[8]`, caching 1 word across odd-to-even calls), and materializing the full $M, L$ arrays and `255`-word block buffer only if a trajectory reaches $K = 256$.

Empirically, `ra_core v2` passes **57/57 non-repeating statistical and adversarial seeding regimes** in PractRand (combining **42 baseline regimes** and **15 non-periodic adversarial regimes** healed by `Opsi_A3`, plus **8 GiB PASS** on the canonical orbit, `CLEAN` from 128 MiB to 8 GiB), heals downscaled $N=16$ sub-orbit boundaries to **15/15 PASS (`4 GiB CLEAN` at $K=14$)**, increases exact Brent cycle lengths on reduced toy models by **$1.98\times\text{--}2.34\times$** (raising minimum cycle length at $(n=4, w=8)$ by **$6.40\times$** from $11{,}072$ to $70{,}876$), achieves ideal 2D Strict Avalanche ($\bar{p} = 0.499959$, $\chi^2/\text{dof} = 1.1070$) and Bit Independence ($\max |\rho| \le 0.00477$), and passes the **114-subtest Dieharder battery (`113 PASS / 1 WEAK / 0 FAIL`, `27/27` Good families PASS)**. On CPU, `ra_core v2` reaches **$1{,}596.5\text{ MiB/s}$** (`1,435.8 MiB/s` in Suite D on the Intel Core i3-1115G4 client; `917.4 MiB/s` on the 1.995 GHz Xeon KVM server) in scalar stateless mode at $K=255$, lifts stateful OTF throughput at $K=255$ from `918.9 MiB/s` (`v2_old`) to **$1{,}219.4\text{ MiB/s}$ ($2.34\text{ cpb}$, $1.33\times$ over `v2_old`, $1{,}162.9\text{ MiB/s}$ median)** while achieving **$1.38\times\text{--}1.74\times$ over `v1` OTF at $K=1$**, and reaches **$10{,}687.9\text{ MiB/s}$ ($10.44\text{ GiB/s}$ client; $6{,}180.4\text{ MiB/s}$ / $6.04\text{ GiB/s}$ server)** in 8-way AVX2 SIMD. Across both CPU platforms, stateless `ra_core v2` (`Opsi_A3`) outperforms **Philox4x32-10** and **ChaCha8** in pure-generate bursts ($K=1..255$ and $K=510..1000$, **$2.65\times\text{--}2.66\times$ faster than Philox at $K=1$**), 2D Lattice Bernoulli sweeps (**$1.304\times\text{--}1.342\times$**), and 32-sweep True 2D Ising Checkerboard Metropolis at $T_c$ (**$1.250\times\text{--}1.276\times$ faster than Philox at $K=1$**), while stateful `eng_v2_E2` outperforms Philox on fixed-trajectory Photon Transport (`E2-Slab`, **$1.242\times\text{--}1.528\times$**). Finally, an exploratory CUDA port on an **NVIDIA RTX PRO 6000 Blackwell GPU (`sm_120`)** verifies **100% bit-exact CPU–GPU parity** across $K=1..8$ bursts and 32-sweep critical 2D Ising lattices ($L=64, 256, 1024$), showing tied GPU kernel execution times at $K=1$ and Ising ($1.001\times\text{--}1.010\times$) while documenting where 4-word block ciphers retain a GPU kernel advantage ($K=4, 8$).

---

## 1. Introduction and Architectural Motivation

Modern stochastic computing workloads span a continuum of access patterns parameterized by the burst length $K$—the number of 32-bit pseudorandom words drawn per logical 64-bit coordinate $(\text{key}, \text{addr\_hi})$:
- **Short Stateless Random Access ($K = 1..2$):** Parallel lattice updates (e.g., per-site checkerboard Metropolis sweeps on a 2D Ising spin lattice), spatial shader noise, and stateless branching decisions where each coordinate $(\text{key}, \text{addr\_hi})$ emits one or two words.
- **Short-to-Medium Sub-Orbit Bursts ($K = 3..255$):** Path-wise financial Monte Carlo (e.g., discrete-monitored Asian options with $K \in \{16, 64, 255\}$ steps per path), stochastic ray/photon transport where trajectories terminate early via Russian roulette, and branching particle cascades.
- **Continuous Multi-Cycle Streaming ($K \ge 256$):** Bulk array population, long-horizon molecular or spin dynamics, and incremental stateful calls (`next_u32()`, `fill_u32()`).

In practice, combining logical addressability with sequential streaming does not require a hybrid state-array architecture: counter-based PRNGs (such as Philox4x32-10 and Threefry in Random123, or ChaCha8, and device-portable libraries such as OpenRAND) already map application-level tuples $(\text{stream}, \text{substream}, \text{draw})$ directly to pseudorandom blocks, guaranteeing schedule-independent reproducibility across threads and CPU/GPU architectures. Likewise, fast stateful generators (such as Xoshiro256**, PCG32/PCG64-DXSM, SFC64, and WyRand) can be seeded per coordinate via a hash or `SplitMix64` mixer. However, when workloads mix **structured 2D coordinate addressing** with **short or unknown-length prefix draws ($K=1..255$) on CPUs**, two practical costs arise:
1. **Fixed-Block Computation on Short Prefixes in Counter-Based Ciphers:** Counter-based block ciphers execute 8 to 10 rounds of 128-bit (4-word) or 512-bit (16-word) Feistel/ARX mixing per block regardless of $K$. On scalar x86-64 CPUs, this imposes a flat throughput ceiling ($\sim 0.77\text{--}1.08\text{ GiB/s}$) and discards 75% of the computed block when only $K=1$ word is needed per coordinate, or when an incremental trajectory terminates mid-block.
2. **Initialization Cost vs. Coordinate-Seeding Sensitivity in Stateful Engines:** Traditional state-array generators require initializing their multi-word state before emitting the first word ($K=1$). Conversely, if a lightweight 1-step multiplicative mixer is seeded directly with un-premixed 2D lattice coordinates $(\text{site}, \text{sweep})$ to avoid initialization overhead, structured correlations between adjacent coordinates can leak into the simulation trajectory (as illustrated by a diagnostic 32-sweep critical 2D Ising realization in Section 4.3, where ad-hoc per-site seeded WyRand drops to $\langle |M| \rangle = 0.2888$ versus $\approx 0.56\text{--}0.67$ for well-diffused generators).

This motivates the central design question of `ra_core v2`: **Can a permutation-coupled state-array generator ($N=256$ words each of $M$ and $L$) be structured so that the computational cost of evaluating a $K$-word stream prefix scales progressively with $K$—evaluating only 2 state words at $K=1$, 3 words at $K=2$, an 8-word ring buffer for $K \le 255$, and materializing the full $M, L$ permutation arrays only when $K \ge 256$—while producing the exact same bit stream as full-state multi-cycle generation and resisting structured 2D seeding?**

### 1.1 Limitations of the `ra_core v1` Baseline

The first-generation `ra_core v1` architecture introduced a dual-mode design over $N = 256$ words ($M[0..255]$ and $L[0..255]$), where Cycle 0 (`singleblock`, $K \le 255$) bypassed $L$ via Dead-Cone Elimination and multi-cycle streaming (`orbit`, $K \ge 256$) permuted $L$ and mixed it into $M$ at cycle boundaries. Despite competitive small-$K$ throughput via register-only fast paths, a systematic four-stage diagnostic investigation exposed three structural flaws in `ra_core v1`:

1. **Affine 1D Key-Mixing Vulnerability & Zero-Slot Fixed Point:** In `v1`, the 64-bit coordinate $(\text{key}, \text{addr\_hi})$ was mixed via two separate 1D `fmix32` calls: $k_{\text{mix}} = \operatorname{fmix32}(G_M \oplus \text{key}) \oplus \operatorname{rot32}(\operatorname{fmix32}(G_{M2} \oplus \text{addr\_hi}), 13)$, and initial state words were seeded as $M[i] = \operatorname{rot32}(i \cdot \text{step}_m,\; k_{\text{mix}} \oplus i)$. When one coordinate was held constant (e.g., $\text{addr\_hi} = 0$ across sequential keys, or $\text{key} = 0$ across sequential $\text{addr\_hi}$), one of the two `fmix32` branches froze to a constant. Both $k_{\text{mix}}$ and $\text{step}_m$ then became deterministic affine functions of a single 32-bit `fmix32` output without a nonlinear barrier between them. Moreover, $\text{step}_m$ could take even values (reducing the subgroup order of $i \cdot \text{step}_m \bmod 2^{32}$), and at index $i = 0$, $M[0] = \operatorname{rot32}(0, k_{\text{mix}}) = 0$ was identically zero for *all* $2^{64}$ seed pairs. Consequently, `v1` and unpatched `v2_base` failed PractRand on sequential-key bursts (`2C_seq_K255`, `4A3_contig_seq_K16/K64`), Gray-code keys (`2D1_gray_K255`), bit-plane grids (`2E1_grid_bp_K1/K255`), and fixed-key sequential address streams (`2F_addr_seq_ctrl`).
2. **Serial Lag-1 Feedback Bottleneck ($D1$) & $O(N)$ End-of-Cycle Reseed Overhead:** In `v1`, step $t$ computed $c_t$, extracted $d_t = c_t \bmod 256$, and immediately injected $d_t$ into step $t+1$'s spine update $a_{t+1} = (d_t \oplus o_{t+1}) \oplus s_{t+1}$. Because $c_t$ depends on a data-dependent rotation $\operatorname{rot32}(u_t, b_t)$ followed by a xorshift, the critical path per word was strictly serial ($\sim 7\text{--}8$ ALU cycles), preventing instruction-level parallelism (ILP) between consecutive words. Furthermore, at the end of every 255-word cycle, `v1` executed two full $N=256$ loops (`ra_reseed` and `ra_hash`, costing $>512$ array reads/writes and rotations), creating a latency cliff at $K = 256$.
3. **Mid-$K$ Stateful OTF Engine Regression in Unrolled `v2_old` ($K = 8..255$):** When the `v1` one-word-at-a-time stateful OTF engine (`ra_core_engine_t`, `m_ring[8]`) was initially adapted to `v2` (`eng_v2_old`), an inverted branch hint (`__builtin_expect(i >= 250u, 1)`) and serial 1-word evaluation prevented the OTF engine from exploiting `v2`'s lag-2 ($D2$) parallelism, causing $K=255$ throughput to drop to $918.9\text{ MiB/s}$ (versus $1{,}303.6\text{ MiB/s}$ for `eng_v1_baseline` in the Stage 2 ablation harness).

### 1.2 Summary of Contributions in `ra_core v2`

This paper presents the design, structural analysis, and cross-platform characterization of **`ra_core v2` (`W2_D2_xag_opt` + `Opsi_A3` + `eng_v2_E2`)**:
- **Deferred Computation of a Stream Prefix via `W2_D2_xag_opt` and `eng_v2_E2` (Sections 2.3–2.7):** Introduces a **2-tap causal ramp-up window ($W2$)** with zeroed guard slots so that the causal dependency cone of Cycle 0 grows progressively from 2 state words at $K=1$ and 3 words at $K=2$, **lag-2 delayed feedback ($D2$)** to unroll adjacent steps into a 2-step coupled spine, and **in-place per-step coupling ($M[i+7] \mathrel{+}= L[d_{\text{use}}]$)** with an **$O(1)$ end-of-cycle guard fold (`END_CYCLE_M_W2`)**. In the stateful engine `eng_v2_E2` (`sizeof(ra_core_engine_t) = 3,520` bytes statically allocated), initialization work is deferred dynamically: $K=1, 2$ execute in registers, $K=3..254$ step in coupled 2-word OTF pairs over `m_ring[8]` (caching 1 word on odd calls), and full $M, L$ array materialization occurs only at $K=256$, recovering $K=255$ OTF throughput to **$1{,}219.4\text{ MiB/s}$ ($1.33\times$ best-of-7 over `eng_v2_old`, $1{,}162.9\text{ MiB/s}$ median)** while outperforming `ra_engine_v1` by **$1.38\times\text{--}1.74\times$ at $K=1$**.
- **`Opsi_A3` 2D Cross-Pre-Mix Key-Mixing (Section 2.2):** Mixes both $\text{key}$ and $\text{addr\_hi}$ into both input branches ($u, v$) *prior* to the dual `fmix32` avalanches, forces odd Weyl increments ($\text{step}_m, \text{step}_l \equiv 1 \pmod 2$, making the index-to-word map bijective over $i$), and shifts the index origin to $(i + 1) \cdot \text{step}$. This heals **16/16 non-periodic adversarial seeding regimes** (`57/57 PASS` overall, combining 42 baseline and 15 adversarial regimes), cures the downscaled $N=16$ $K=14$ anomaly to `4 GiB CLEAN`, and doubles mean Brent cycle lengths on reduced-state toy models with **zero runtime cost inside the generation loop**.
- **Controlled Same-Work Monte Carlo Evaluation & Cross-Platform CPU/GPU Characterization (Section 4):** Applies a **Dual-Mode Monte Carlo Evaluation Protocol**—combining *Coupled Stochastic Physics* with *Fixed-Topology Same-Work Traces* ($0.000\%$ work drift, analogous to controlled common-random-number experimental designs)—across two x86-64 CPU platforms (Intel Core i3-1115G4 client and Xeon KVM 1.995 GHz server) and an exploratory CUDA port on an **NVIDIA RTX PRO 6000 Blackwell GPU (`sm_120`)**. We verify **100% bit-exact parity** across CPU scalar, 8-way AVX2, stateful OTF, and GPU CUDA execution, and delineate the hardware regimes where `ra_core v2` accelerates CPU simulations ($1.250\times\text{--}1.276\times$ over Philox on $K=1$ 2D Ising; $1.242\times\text{--}1.528\times$ on `E2-Slab` photon transport) versus GPU regimes where execution is tied ($1.001\times\text{--}1.010\times$ on GPU Ising and $K=1$) or favors 4-word block ciphers ($K=4, 8$).

---

## 2. Mathematical Specification of `ra_core v2`

### 2.1 Primitives, Parameters, and Guard Constants

Let $\mathbb{Z}_{2^{32}}$ denote the ring of 32-bit unsigned integers (`uint32_t`) with arithmetic modulo $2^{32}$, bitwise XOR $\oplus$, bitwise OR $\mid$, bitwise AND $\&$, left shift $\ll$, logical right shift $\gg$, and 32-bit left barrel rotation:
$$\operatorname{rot32}(x, r) = (x \ll (r \bmod 32)) \mid (x \gg ((-r) \bmod 32)).$$
The default state size is $N = 256$ (`RA_N = 256`, mask $\text{RA\_MASK} = N - 1 = 255$, single-cycle capacity $K_{\max} = N - 1 = 255$, index offset $\text{RA\_IDX\_OFF} = 1$). The MurmurHash3 32-bit finalizer $\operatorname{fmix32}: \mathbb{Z}_{2^{32}} \to \mathbb{Z}_{2^{32}}$ is defined as:
$$\begin{aligned}
h_1 &= (h \oplus (h \gg 16)) \cdot \mathtt{0x85ebca6b}, \\
h_2 &= (h_1 \oplus (h_1 \gg 13)) \cdot \mathtt{0xc2b2ae35}, \\
\operatorname{fmix32}(h) &= h_2 \oplus (h_2 \gg 16).
\end{aligned}$$
Four balanced 32-bit guard constants (each having exact Hamming weight $\operatorname{popcount} = 16$) provide domain separation across the four mixing channels:
$$G_L = \mathtt{0x38916df4}, \quad G_M = \mathtt{0x6c26fc92}, \quad G_{L2} = \mathtt{0x3eb13b90}, \quad G_{M2} = \mathtt{0x392456de}.$$

---

### 2.2 `Opsi_A3`: 2D Cross-Pre-Mix Key-Mixing and Odd-Step Bijective Initialization

Given a 64-bit coordinate $(\text{key}, \text{addr\_hi}) \in \mathbb{Z}_{2^{32}} \times \mathbb{Z}_{2^{32}}$, `ra_core v2` derives the mixed seed constant $k_{\text{mix}}$ and the odd Weyl increments $\text{step}_m, \text{step}_l$ via **`Opsi_A3` (`ra_derive_m_A3` and `ra_derive_l_A3`)**:

1. **Primary State $M$ Parameter Derivation (`ra_derive_m_A3`):**
   $$p_{\text{addr}, M} = \mathtt{0x9e3779b1} \cdot \text{addr\_hi}$$
   $$u_m = (\mathtt{0x06a0dd9b} \cdot \text{key}) + p_{\text{addr}, M}, \qquad v_m = p_{\text{addr}, M} \oplus \operatorname{rot32}(\text{key}, 17)$$
   $$\text{km}_k = \operatorname{fmix32}(G_M \oplus u_m), \qquad \text{km}_p = \operatorname{fmix32}(G_{M2} \oplus v_m)$$
   $$k_{\text{mix}} = \text{km}_k \oplus \operatorname{rot32}(\text{km}_p, 13), \qquad \text{step}_m = \big(\mathtt{0x06a0dd9b} \cdot (\text{km}_k + \text{km}_p)\big) \mid 1$$

2. **Coupling Permutation Array $L$ Parameter Derivation (`ra_derive_l_A3`):**
   $$p_{\text{addr}, L} = \mathtt{0x85ebca77} \cdot \text{addr\_hi}$$
   $$u_l = (\mathtt{0x9e3779b7} \cdot \text{key}) + p_{\text{addr}, L}, \qquad v_l = p_{\text{addr}, L} \oplus \operatorname{rot32}(\text{key}, 15)$$
   $$\text{kl}_k = \operatorname{fmix32}(G_L \oplus u_l), \qquad \text{kl}_p = \operatorname{fmix32}(G_{L2} \oplus v_l)$$
   $$\text{step}_l = \big(\mathtt{0x9e3779b7} \cdot (\text{kl}_k + \text{kl}_p)\big) \mid 1$$

3. **Bijective $+1$ Offset Array Initialization ($i \in \{0, 1, \dots, N-1\}$):**
   $$M_{\text{init}}[i] = \operatorname{rot32}\big((i + 1) \cdot \text{step}_m,\; k_{\text{mix}} \oplus i\big), \qquad L_{\text{init}}[i] = \operatorname{rot32}\big((i + 1) \cdot \text{step}_l,\; k_{\text{mix}} \oplus i\big), \qquad \text{cons}_0 = k_{\text{mix}}.$$

#### Why 2D Cross-Pre-Mix Before `fmix32` Is Essential

In `v1` and unpatched `v2_base`, $\text{km}_k = \operatorname{fmix32}(G_M \oplus \text{key})$ depended exclusively on $\text{key}$, while $\text{km}_p = \operatorname{fmix32}(G_{M2} \oplus \text{addr\_hi})$ depended exclusively on $\text{addr\_hi}$. Consider what happens when one coordinate is held constant—the standard operating mode in almost every real application:
- **Case A ($\text{addr\_hi} = c_0$ constant, $\text{key} = 0, 1, 2, \dots$ varying):** $\text{km}_p = C$ becomes a fixed constant across the entire key space! Then $k_{\text{mix}}(\text{key}) = \text{km}_k(\text{key}) \oplus \operatorname{rot32}(C, 13)$ and $\text{step}_m(\text{key}) = \mathtt{0x06a0dd9b} \cdot (\text{km}_k(\text{key}) + C)$ are **both deterministic functions of the single 32-bit word $\text{km}_k(\text{key})$** with *no nonlinear barrier between them*. Consequently, knowing $k_{\text{mix}}$ uniquely determines $\text{step}_m$ via an affine relation, collapsing the effective 64-bit internal seed state $(k_{\text{mix}}, \text{step}_m)$ onto a 32-bit 1D manifold.
- **Case B ($\text{key} = c_0$ constant, $\text{addr\_hi} = 0, 1, 2, \dots$ varying):** Symmetrically, $\text{km}_k = C'$ freezes to a constant, so both $k_{\text{mix}}$ and $\text{step}_m$ become affine transforms of the single word $\text{km}_p(\text{addr\_hi})$.

By introducing the **2D cross-pre-mix** $u_m = (\mathtt{0x06a0dd9b} \cdot \text{key}) + p_{\text{addr}, M}$ (modular addition of odd-multiplied coordinates) and $v_m = p_{\text{addr}, M} \oplus \operatorname{rot32}(\text{key}, 17)$ (XOR with rotated coordinate) *before* $\operatorname{fmix32}$, **both $\text{km}_k$ and $\text{km}_p$ undergo full, independent 3-round MurmurHash3 avalanches even when either $\text{key}$ or $\text{addr\_hi}$ is held constant**. Combined with:
- **Odd-forced bijective index step ($\mid 1$):** Guarantees $\gcd(\text{step}_m, 2^{32}) = 1$ and $\gcd(\text{step}_l, 2^{32}) = 1$, so the per-index map $i \mapsto (i+1)\cdot\text{step} \bmod 2^{32}$ is a strict bijection of $\mathbb{Z}_{2^{32}}$ with maximal 2-adic period $2^{32}$ over the index $i$ for every seed (avoiding even-step subgroup collapse, without claiming that the global coordinate-to-stream mapping or full generator orbit is bijective); and
- **$+1$ Index Offset ($(i+1)\cdot\text{step}$):** Eliminates the invariant $M[0] = L[0] = 0$ zero slot at $i=0$, ensuring all $N=256$ state words carry full 32-bit entropy from step 0—

`Opsi_A3` achieves thorough 2D coordinate pre-diffusion while adding **zero instructions** inside the per-word generation loop.

---

### 2.3 Causal 2-Tap Ramp-Up Window ($W2$) vs. `v1` Cyclic Bootstrap

In `ra_core v1`, the 7-tap running-XOR recurrence over $M$ required copying 6 words at the start of every cycle (`M[257..262] = M[1..6]`, `M[256] = M[0]`) and seeding the initial window with a 7-way XOR $o_{255} = \bigoplus_{k=0}^{6} (M[255+k] \ll k)$. For short stateless bursts ($K \le 8$), computing 7 distinct $M$ taps imposed unnecessary register pressure and initialization work.

`ra_core v2` introduces the **2-Tap Causal Ramp-Up Window ($W2$)**:
1. At initialization (`ra_init_orbit_v2`), the guard extension slots are set to:
   $$M[N] = M[1], \qquad M[N+1] = M[N+2] = \dots = M[N+6] = 0.$$
2. Each cycle begins at index $i = N - 1 = 255$ with only **two active taps**:
   $$o_{255} = M[255] \oplus (M[N] \ll 1) = M[255] \oplus (M[1] \ll 1).$$
3. As $i$ steps downward from $255$ to $1$, the running 7-tap polynomial shifts left and absorbs $M[i-1]$ at bit 0 while evicting $M[i+6] \ll 7$ from the top of the 7-tap window:
   $$o_{i-1} = M[i-1] \oplus \big((o_i \oplus (M[i+6] \ll 7)) \ll 1\big).$$

**Algebraic Consequence for Cycle 0 (`singleblock`, $K \le 255$):** Because $M[257..262] = 0$ in Cycle 0, the eviction term $o_{\text{term7}}(i) = M[i+6] \ll 7$ is **identically zero** for the first 6 steps ($i = 255, 254, 253, 252, 251, 250$), and at $i = 249$ ($N-7$) it equals $M[256] \ll 7 = M[1] \ll 7$. Thus:
- At **$K = 1$** ($i = 255$), only **two words** ($M[255]$ and $M[1]$) are ever referenced!
- At **$K = 2$** ($i = 255, 254$), only **three words** ($M[255], M[254], M[1]$) are referenced!
- For the first 3 unrolled pairs ($K = 1..6$, $i = 255..250$), the recurrence simplifies to $o_{i-1} = M[i-1] \oplus (o_i \ll 1)$ with zero eviction loads.

---

### 2.4 Lag-2 Delayed Feedback ($D2$) and the 2-Step Unrolled XAG Recurrence Core (`W2_D2_xag_opt`)

In `ra_core v1`, the feedback index $d_t = c_t \ \& \ 255$ from step $t$'s scrambler output $c_t$ was immediately needed at step $t+1$ to compute $a_{t+1} = (d_t \oplus o_{t+1}) \oplus s_{t+1}$. This created a tight 1-step loop-carried dependency chain through the barrel rotator $\operatorname{rot32}((a_t \gg 13) \oplus a_t, b_t)$ and finalizer $c_t \oplus (c_t \gg 17)$.

`ra_core v2` replaces lag-1 feedback with **Lag-2 Delayed Feedback ($D2$)**, maintaining a 2-element feedback queue $(d_0, d_1)$ initialized at the start of each cycle from the cycle constant $\text{cons}$:
$$s^{(0)} = \text{cons} \ll 1, \qquad d_0^{(0)} = 0, \qquad d_1^{(0)} = \big(\operatorname{rot32}(\text{cons}, 15) \oplus \mathtt{0x55}\big) \ \& \ \text{RA\_MASK}.$$

Because $d_0$ and $d_1$ are * both known at the start of an even/odd step pair $(i, i-1)$*, two consecutive steps can be evaluated as a **2-Step Coupled Pair**:

1. **2-Step Window Update:**
   $$o_1 = M[i-1] \oplus \big((o_0 \oplus (M[i+6] \ll 7)) \ll 1\big).$$

2. **Coupled 2-Step Spine (serial cost: only 2 XORs + 2 additions across the pair):**
   $$\begin{aligned}
   a_0 &= (d_0 \oplus o_0) \oplus s, &\qquad s_1 &= \text{cons} + a_0, \\
   a_1 &= (d_1 \oplus o_1) \oplus s_1, &\qquad s_2 &= \text{cons} + a_1.
   \end{aligned}$$

3. **Dual Parallel XAG Scramblers (100% independent between word 0 and word 1):**
   $$\begin{aligned}
   b_0 &= s_1 \oplus (o_0 + d_0), &\qquad b_1 &= s_2 \oplus (o_1 + d_1), \\
   u_0 &= (a_0 \gg 13) \oplus a_0, &\qquad u_1 &= (a_1 \gg 13) \oplus a_1, \\
   r_0 &= \operatorname{rot32}(u_0, b_0), &\qquad r_1 &= \operatorname{rot32}(u_1, b_1), \\
   c_0 &= r_0 \oplus (r_0 \gg 17), &\qquad c_1 &= r_1 \oplus (r_1 \gg 17).
   \end{aligned}$$

4. **Lag-2 Feedback Advance for Next Pair:**
   $$d_2 = c_0 \ \& \ \text{RA\_MASK}, \qquad d_3 = c_1 \ \& \ \text{RA\_MASK}, \qquad (s, d_0, d_1) \leftarrow (s_2, d_2, d_3).$$

Notice the hardware synergy: while the spine $(a_0 \to s_1 \to a_1 \to s_2)$ preserves full sequential state coupling across every single step at a cost of just **2 clock cycles per word**, the expensive nonlinear operations—additions $o + d$, xorshifts $(a \gg 13) \oplus a$, variable barrel rotations $\operatorname{rot32}(u, b)$, and output xorshifts $r \oplus (r \gg 17)$—execute in **two completely independent parallel superscalar pipelines**!

---

### 2.5 In-Place Per-Step $L \to M$ Coupling and $O(1)$ End-of-Cycle Guard Fold (`END_CYCLE_M_W2`)

For multi-cycle streaming ($K > 255$, `ra_core_orbit`), `ra_core v1` permuted $L$ during the cycle and then executed two separate $O(N)$ loops (`ra_reseed` and `ra_hash`) between cycles. `ra_core v2` eliminates both inter-cycle loops by **fusing the $L \to M$ state injection directly into the 2-step generation loop** at offset $i+7$ (which has already exited the active 7-tap window $[i-1 .. i+5]$!):

During each 2-step pair $(i, i-1)$ for $i \in \{255, 253, \dots, 3\}$, using $d_1$ (the feedback index paired with step $i$) and $d_2 = c_0 \ \& \ \text{RA\_MASK}$ (the feedback index paired with step $i-1$):
$$\begin{aligned}
\text{Step } i:\quad & \text{swap}\big(L[i],\; L[d_1]\big), &\quad M[i + 7] &\mathrel{+}= L[i] \quad (\text{the post-swap value } L_{\text{old}}[d_1]), \\
\text{Step } i-1:\quad & \text{swap}\big(L[i-1],\; L[d_2]\big), &\quad M[i + 6] &\mathrel{+}= L[i-1] \quad (\text{the post-swap value } L_{\text{old}}[d_2]).
\end{aligned}$$
At the final odd step $i = 1$ (the 255th word of the cycle), $c_{\text{last}}$ is emitted and:
$$\text{swap}\big(L[1],\; L[d_1]\big), \qquad M[8] \mathrel{+}= L[1].$$

Observe where the in-place updates land as $i$ descends from $255$ down to $1$:
- Steps $i = 255..250$ accumulate $L$ values into the **guard overflow slots** $M[262..257]$ (`M[RA_N + 6]` down to `M[RA_N + 1]`).
- Step $i = 249$ accumulates into $M[256]$ (`M[RA_N]`, which initially held $M_{\text{old}}[1]$).
- Steps $i = 248..1$ accumulate directly into $M[255..8]$ in-place!

Thus, at the end of the 255-word cycle, wrapping the guard overflow slots back into $M[0..7]$ and refreshing the cycle constant $\text{cons}$ requires only an **$O(1)$ 7-word macro (`END_CYCLE_M_W2`)**:
$$\begin{aligned}
M[0] &\mathrel{+}= M[N] - M[1], \\
M[g] &\mathrel{+}= M[N + g], \quad M[N + g] \leftarrow 0 \qquad \text{for } g \in \{1, 2, 3, 4, 5, 6\}, \\
M[7] &\mathrel{+}= L[0], \\
M[N] &\leftarrow M[1], \\
\text{cons} &\leftarrow \text{cons} \oplus (c_{\text{last}} + s_{\text{last}} + L[0]).
\end{aligned}$$
This reduces the end-of-cycle transition from $>512$ loop iterations (`ra_reseed` + `ra_hash`) to **15 scalar instructions ($O(1)$)** while maintaining continuous permutation coupling between $L$ and $M$.

---

### 2.6 Register-Only Algebraic Fast-Paths ($K=1$, $K=2$, $K=3..8$) and Dead-Cone Elimination

For any burst length $K \le 255$ (Cycle 0), none of the mutations to $L[1..255]$ or $M[8..262]$ can ever causally influence the first 255 output words (**Dead-Cone Elimination**). Therefore, `ra_core_orbit` delegates $K \le 255$ directly to `ra_core_singleblock`. Furthermore, for $K \le 8$, `ra_core v2` bypasses stack allocation of $M[0..255]$ altogether:

1. **Exact $K = 1$ Fast-Path (`ra_eval_k1_word`):** Evaluates `ra_derive_m_A3(key, addr_hi, &step_m, &k_mix)`, computes only the two required taps $M[255]$ and $M[1]$ directly in CPU registers, and applies the single-step scrambler:
   $$\begin{aligned}
   o_0 &= \operatorname{rot32}(256 \cdot \text{step}_m,\; k_{\text{mix}} \oplus 255) \oplus \big(\operatorname{rot32}(2 \cdot \text{step}_m,\; k_{\text{mix}} \oplus 1) \ll 1\big), \\
   a_0 &= o_0 \oplus (k_{\text{mix}} \ll 1), \qquad b_0 = (k_{\text{mix}} + a_0) \oplus o_0, \\
   c_0 &= \operatorname{rot32}((a_0 \gg 13) \oplus a_0,\; b_0), \qquad \text{out}[0] = c_0 \oplus (c_0 \gg 17).
   \end{aligned}$$
2. **Exact $K = 2$ Fast-Path (`ra_eval_k2_words`):** Computes $M[255]$, $M[254]$, and $M[1]$ in registers and executes a single 2-step coupled pair $(o_0, o_1)$ with zero branches and zero stack arrays.
3. **Register-Only $K = 3..8$ Fast-Path (`ra_eval_k3_to_k8_words`):** Evaluates $M[i-1]$ and $M[i-2]$ on the fly in registers across 1 to 4 unrolled pairs, handling the single eviction tap $M[1] \ll 7$ at $i = 249$ ($K=7, 8$) via a branchless ternary select.
4. **Partial-Init $K = 9..255$ Path (`ra_singleblock_v2_buf`):** Materializes only the active slice $M[\max(1, 256-K) .. 255]$ plus $M[1]$, executing the first 3 pairs ($K=1..6$) without eviction loads before entering the general 2-step loop.

---

### 2.7 `eng_v2_E2`: Deferred Computation of a Stream Prefix via Even-$K$-Aligned 2-Step Coupled OTF Evaluation

When a caller consumes pseudorandom numbers incrementally (`ra_engine_next_u32(eng)` or `ra_engine_fill_u32(eng, out, count)`) without knowing the total draw count $K$ in advance, the stateful `ra_core_engine_t` must produce the exact same sequence as `ra_core_orbit` ($0$ mismatches across all $K \in [1, \infty)$) while deferring the $O(N)$ computational work of initializing $M$ and $L$ when $K$ turns out to be small. Note that in the production C implementation (`src/ra_core/c/ra_core_engine.c`), `ra_core_engine_t` statically reserves `sizeof(ra_core_engine_t) = 3,520` bytes (holding scalar registers, `m_ring[8]`, `d_hist[256]`, `L[256]`, `M[264]`, and `block_buf[256]`); thus, **what is deferred is the computational work of array initialization and permutation, not the struct's memory footprint**.

In Stage 2, we designed **`eng_v2_E2`**, which solves the mid-$K$ OTF bottleneck by aligning the ring-buffer transition at an **even step boundary ($K=2$, $i=254$)** so that all subsequent steps in Cycle 0 ($i = 253$ down to $2$) execute as **2-Step Coupled OTF Pairs** using a single 32-bit `cached_word` slot inside `ra_core_engine_t`:

| Engine Mode | Active Step(s) | Architectural Action in `eng_v2_E2` |
| :---: | :---: | :--- |
| **`mode = 0`** | Word #1 ($K=1, i=255$) | Evaluates only 2 taps ($M[255]$ and $M[1]$) in registers; initializes $s, d_0, d_1, o_0$; records `d_hist[255]`; returns $c_0$ immediately. **Zero ring buffer or array writes.** |
| **`mode = 1`** | Word #2 ($K=2, i=254$) | Evaluates $M[254]$ in registers; populates the 8-word rolling window `m_ring[0..7]` once (`m_ring[1] = M[1]`, `m_ring[2..7] = 0`, `m_ring[0] = M[253]`); records `d_hist[254]`; sets `i_cur = 253`, `mode = 3`, and returns $c_1$. |
| **`mode = 3`** (Odd call) | Odd Call $K \in \{3, 5, \dots, 253\}$ ($i \in \{253, 251, \dots, 3\}$) | Executes a **full 2-Step Coupled OTF Pair $(i, i-1)$**: generates $M[i-1]$ and $M[i-2]$ on the fly via `acc_m -= step_m`, updates `m_ring` and `d_hist` for both steps, evaluates the 2-step coupled spine $(a_0, s_1, a_1, s_2)$ and **Dual Parallel Scramblers** $(c_0, c_1)$, stores `cached_word = c_1`, transitions to `mode = 2`, advances $i \leftarrow i - 2$, and returns $c_0$ (if a trajectory stops at an odd $K \ge 3$, one extra word $c_1$ was computed and cached). |
| **`mode = 2`** (Even call) | Even Call $K \in \{4, 6, \dots, 254\}$ ($i$ already advanced) | **Zero-compute cache hit:** transitions back to `mode = 3` and immediately returns `cached_word` ($c_1$). |
| **`mode = 3`** ($i = 1$) | Word #255 ($K=255, i=1$) | Evaluates the final single odd step of Cycle 0 ($i=1$), records `d_hist[1]`, sets `i_cur = 0`, and returns $c_{254}$ with zero wasted words. |
| **`mode = 4`** | Multi-Cycle ($K \ge 256$) | On the 256th word, executes **Retroactive Cycle 0 Materialization (`ra_core_engine_promote`)**: initializes $L[0..255]$ and $M[0..262]$, replays Cycle 0's recorded `d_hist[255..1]` swaps ($L[i] \leftrightarrow L[d]$) and $M[i+7] \mathrel{+}= L[i]$ accumulations, applies `END_CYCLE_M_W2(M, L)`, and fills the 255-word `block_buf` via `ra_orbit_cycle_v2_buf`. |

Moreover, in `ra_engine_fill_u32(eng, out, count)`, whenever `eng` is in `mode = 3` (`i_cur >= 2`) and `count >= 2`, `eng_v2_E2` executes a **tight unrolled 2-step OTF pair loop** that writes $(c_0, c_1)$ directly to `out[0..1]` without touching `cached_word` or dispatching per-word branches.

---

## 3. Empirical Statistical Validation and Structural Analysis

### 3.1 Stage 1 & Stage 4 Seeding Sensitivity: 16/16 Adversarial Regimes and 57/57 Full Matrix

To stress-test the algebraic independence of adjacent streams across the 64-bit $(\text{key}, \text{addr\_hi})$ space, we evaluated `ra_core v2` across **16 non-periodic adversarial seeding regimes** in Stage 1 and the **combined 57-regime multi-key matrix** in Stage 4 using PractRand v0.94 (`RNG_test stdin32 -tlmin 16MB -tlmax 2GB -multithreaded`):

| Category | Regime ID | Stream Construction & $(K, \text{key}, \text{addr\_hi})$ Pattern | `v1` Baseline | `v2_base` (1D `fmix32`) | **`ra_core v2` (`Opsi_A3`)** |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. Continuous** | `1A_orbit_k42` | Single seed `(42, 0)` continuous multi-cycle orbit (**8 GiB**) | PASS | PASS | **8 GiB PASS (0 anomalies)** |
| **1. Continuous** | `1B_orbit_k0` | All-zero seed `(0, 0)` continuous multi-cycle orbit | PASS | PASS | **1 GiB PASS (CLEAN)** |
| **2. Single-Word** | `2A_weyl_K1` | $K=1$ per key, Weyl sequence $\text{key}_n = n \cdot \mathtt{0x9e3779b9}$ | PASS | PASS | **2 GiB PASS (CLEAN)** |
| **2. Sequential** | `2B_seq_K1` | $K=1$ per key, $\text{key} = 0, 1, 2, \dots, \text{addr\_hi} = 0$ | PASS | PASS | **1 GiB PASS (CLEAN)** |
| **2. Sequential** | `2C_seq_K255` | $K=255$ per key, $\text{key} = 0, 1, 2, \dots, \text{addr\_hi} = 0$ | **FAIL (64M)** | **FAIL (128M)** | **2 GiB PASS (CLEAN)** |
| **2. Gray Code** | `2D1_gray_K1` | $K=1$ per key, Gray code $\text{key}_n = n \oplus (n \gg 1)$ | PASS | PASS | **1 GiB PASS (CLEAN)** |
| **2. Gray Code** | `2D1_gray_K255` | $K=255$ per key, Gray code $\text{key}_n = n \oplus (n \gg 1)$ | **FAIL (64M)** | **FAIL (128M)** | **1 GiB PASS (CLEAN)** |
| **2. Low-HW** | `2D2_sparse_K255` | $K=255$ per seed, low Hamming weight $\operatorname{HW}(\text{key}), \operatorname{HW}(\text{addr\_hi}) \le 3$ | PASS | PASS | **1 GiB PASS (CLEAN)** |
| **2. Low-HW** | `2D2_sparse_hw4_K1` | $K=1$ per seed, $\operatorname{HW}(\text{key}), \operatorname{HW}(\text{addr\_hi}) \le 4$ ($508.5\text{M}$ pairs)* | — | — | **2 GiB PASS (CLEAN)** |
| **2. Bit-Plane** | `2E1_grid_bp_K1` | $K=1$ 2D bit-plane grid $(\text{key} \in [0, 2^{16}), \text{addr\_hi} \in [0, 2^{16}))$ | **FAIL (256M)** | **FAIL (256M)** | **1 GiB PASS (CLEAN)** |
| **2. Bit-Plane** | `2E1_grid_bp_K255` | $K=255$ 2D bit-plane grid $(\text{key}, \text{addr\_hi})$ | **FAIL (64M)** | **FAIL (128M)** | **1 GiB PASS (CLEAN)** |
| **2. Hi-Nibble** | `2E2_hi_nibble_K1` | $K=1$ upper-bit counters ($\text{key} \ll 16, \text{addr\_hi} \ll 16$) | PASS | PASS | **1 GiB PASS (CLEAN)** |
| **2. Addr-Seq** | `2F_addr_seq_ctrl` | **Positive Control:** $\text{key} = 0$ fixed, $\text{addr\_hi} = 0, 1, 2, \dots$ ($K=1$ & $K=255$) | **FAIL (16M)** | **FAIL (16M)** | **2 GiB PASS (CLEAN)** |
| **3. Striping** | `3B_stripe_8_K255` | 8-lane round-robin interleaved sequential keys ($K=255$) | **FAIL (512M)** | PASS | **1 GiB PASS (CLEAN)** |
| **4. Sub-Orbit** | `4A2_contig_weyl_K16` | Contiguous $K=16$ bursts, Weyl key sequence | PASS | PASS | **1 GiB PASS (CLEAN)** |
| **4. Sub-Orbit** | `4A3_contig_seq_K16` | Contiguous $K=16$ bursts, sequential $\text{key} = 0, 1, 2, \dots$ | **FAIL (32M)** | **FAIL (64M)** | **2 GiB PASS (CLEAN)** |
| **4. Sub-Orbit** | `4A3_contig_seq_K64` | Contiguous $K=64$ bursts, sequential $\text{key} = 0, 1, 2, \dots$ | **FAIL (64M)** | **FAIL (128M)** | **2 GiB PASS (CLEAN)** |

*\*Combinatorial Note on `2D2_sparse_K1`:* In the original `2D2_sparse_K1` test harness, the generator enumerated 32-bit words with Hamming weight $\operatorname{HW} \le 3$. Because there are only $\binom{32}{0} + \binom{32}{1} + \binom{32}{2} + \binom{32}{3} = 1 + 32 + 496 + 4{,}960 = 5{,}489$ such 32-bit integers, the Cartesian product $(\text{key}, \text{addr\_hi})$ contains exactly $5{,}489^2 = 30{,}129{,}121$ unique seed pairs. At $K=1$ (4 bytes per pair), the entire finite input space is exhausted at $30{,}129{,}121 \times 4\text{ B} = 120{,}516{,}484\text{ B} \approx 114.93\text{ MiB}$, after which the test harness wraps around (`idx % 30,129,121`) and repeats the exact 114.93 MiB buffer verbatim—causing any deterministic function (including SHA-256 or Philox) to trigger `DC6`/`BCFN`/`FPF` periodicity failures at 256 MiB. When expanded to $\operatorname{HW} \le 4$ ($41{,}449$ words $\implies 1.718 \times 10^9$ pairs $\approx 6.40\text{ GiB}$ at $K=1$), `Opsi_A3` passes **2 GiB completely clean**. Across all **57 non-repeating regimes** in the full evaluation matrix, `ra_core v2` (`Opsi_A3`) achieves **57 / 57 PASS (`0 FAIL`)**.

---

### 3.2 Downscaled $N = 16$ Stress Test and Exact Toy-Model Brent Cycle Lengths

#### Downscaled $N = 16$ Stress Test (`RA_N = 16`, $K \in [1, 15]$)
Reducing the state array size by $16\times$ to $N = 16$ (`RA_MASK = 15`, `RA_MAX_SINGLE = 15`) magnifies any boundary-coupling defect between Cycle 0 and Cycle 1. Across all 15 burst lengths $K \in \{1, 2, \dots, 15\}$:
- **`ra_core v2` (`Opsi_A3`) achieves `15 / 15 PASS` (`0 FAIL`, `0 SUSP`)**.
- Crucially, at $K = 14$ (just one step before the $N-1=15$ reseed boundary), where unpatched `v2_base` exhibited a suspicious anomaly (`SUSP: 1` on `BCFN(2+1,13-1,T)` at $p = 3.2 \times 10^{-5}$), `Opsi_A3` is **100% CLEAN all the way to 4 GiB**.

#### Exact Brent Cycle Detection on Reduced-State Toy Models
To verify that `Opsi_A3` and `W2_D2_xag_opt` improve the underlying permutation orbit structure rather than merely masking output correlations, we performed exact Brent cycle-length and tail-length ($\lambda, \mu$) decompositions over 64 seeds on three tractable reduced-state models $(n, w)$ (where $N = 2^n$ words of $w$ bits each):

| Toy Configuration $(n, w)$ | Variant | Mean Cycle $\bar{\lambda}$ | Median $\lambda$ | Min $\lambda$ | Max $\lambda$ | Mean Tail $\bar{\mu}$ | Ratio vs. `v2_base` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$(n=3, w=4)$** ($N=8$, 4-bit words) | `v1` baseline | $874.2$ | $1{,}005.0$ | $168$ | $1{,}764$ | $597.4$ | $0.82\times$ |
| **$(n=3, w=4)$** | `v2_base` | $1{,}062.4$ | $850.0$ | $189$ | $3{,}059$ | $642.1$ | $1.00\times$ |
| **$(n=3, w=4)$** | **`v2` (`Opsi_A3`)** | **$2{,}326.8$** | **$2{,}142.0$** | **$315$** | **$5{,}201$** | $1{,}104.2$ | **$2.19\times$** |
| **$(n=4, w=4)$** ($N=16$, 4-bit words) | `v1` baseline | $1{,}190.5$ | $1{,}125.0$ | $225$ | $2{,}850$ | $710.8$ | $0.81\times$ |
| **$(n=4, w=4)$** | `v2_base` | $1{,}469.8$ | $1{,}410.0$ | $330$ | $3{,}465$ | $885.3$ | $1.00\times$ |
| **$(n=4, w=4)$** | **`v2` (`Opsi_A3`)** | **$2{,}907.7$** | **$2{,}790.0$** | **$585$** | **$6{,}810$** | $1{,}412.6$ | **$1.98\times$** |
| **$(n=2, w=8)$** ($N=4$, 8-bit words) | `v1` baseline | $4{,}112.0$ | $3{,}890.0$ | $510$ | $9{,}840$ | $2{,}105.0$ | $0.83\times$ |
| **$(n=2, w=8)$** | `v2_base` | $4{,}983.7$ | $4{,}530.0$ | $702$ | $12{,}450$ | $2{,}640.5$ | $1.00\times$ |
| **$(n=2, w=8)$** | **`v2` (`Opsi_A3`)** | **$11{,}673.8$** | **$11{,}124.0$** | **$1{,}836$** | **$26{,}190$** | $5{,}418.9$ | **$2.34\times$** |
| **$(n=4, w=8)$** ($N=16$, 8-bit words, 16-seed cap) | `v2_base` | $\ge 786{,}432$ | $1{,}048{,}576$ | $11{,}072$ | $> 1{,}048{,}576$ | — | $1.00\times$ |
| **$(n=4, w=8)$** | **`v2` (`Opsi_A3`)** | **$\ge 912{,}500$** | **$1{,}048{,}576$** | **$70{,}876$** | **$> 1{,}048{,}576$** | — | **Min $\lambda$ $6.40\times$** |

Across all tractable configurations, `Opsi_A3` expands the mean cycle length $\bar{\lambda}$ by **$1.98\times\text{--}2.34\times$** over `v2_base` ($2.44\times\text{--}2.84\times$ over `v1`), and raises the worst-case minimum cycle length at $(n=4, w=8)$ by **$6.40\times$** (from $11{,}072$ to $70{,}876$ steps) by eliminating even-step subgroup collapse and the $i=0$ zero fixed point.

---

### 3.3 ENT Byte-Level $\chi^2$ Distribution, 2D SAC ($64\times 32$), BIC ($32\times 32$), and Dieharder

1. **ENT Byte-Level $\chi^2$ Multi-Seed Distribution:** In `v1` and `v2_base`, the single canonical seed `(42, 0)` at 1 GiB produced an outlier byte-level $\chi^2 = 319.99$ ($p = 0.0022$). Under `ra_core v2` (`Opsi_A3`), the canonical 1 GiB stream (`key=42, addr_hi=0`) yields **$\chi^2 = 250.45$ ($p = 0.5614$)** with entropy $7.9999998\text{ bits/byte}$, arithmetic mean $127.5001$, and serial correlation $-0.000015$. Across 32 independent seeds (`100 MiB` each), **32 / 32 seeds (`100%`) fall cleanly inside $p \in [0.01, 0.99]$**, with ensemble mean $\bar{\chi}^2 = 255.21 \pm 20.96$ (matching the theoretical $\chi^2_{255}$ expectation of $\mu = 255.0, \sigma = \sqrt{510} \approx 22.58$).
2. **2D Strict Avalanche Criterion (SAC $64 \times 32 = 2{,}048$ cells, $N = 100{,}000$ seed pairs):** Flipping each of the 64 input bits ($\text{key}[0..31]$ and $\text{addr\_hi}[0..31]$) flips every output bit $j \in [0, 31]$ with probability $p_{i,j}$ indistinguishable from $0.5$:
   - **$K=1$ (`word #0`):** $\bar{p} = 0.499959$, $p_{i,j} \in [0.494200, 0.506620]$, $\max |p - 0.5| = 0.006620$, $\chi^2/\text{dof} = 1.1070$.
   - **$K=255$ (`word #127`, mid-Cycle 0):** $\bar{p} = 0.500065$, $\max |p - 0.5| = 0.005970$, $\chi^2/\text{dof} = 0.9723$.
   - **$K=255$ (`word #254`, end of Cycle 0):** $\bar{p} = 0.499916$, $\max |p - 0.5| = 0.005560$, $\chi^2/\text{dof} = 1.0113$.
   - **$K=256$ (`word #255`, Cycle 1 post-`END_CYCLE_M_W2`):** $\bar{p} = 0.499970$, $\max |p - 0.5| = 0.005460$, $\chi^2/\text{dof} = 1.0084$.
3. **Bit Independence Criterion (BIC $32 \times 32$, $10{,}000$ base seeds $\times 64$ bit flips $= 640{,}000$ avalanche vectors):** Across all $\binom{32}{2} = 496$ off-diagonal output bit pairs $(j, k)$, the maximum empirical Pearson correlation is **$\max |\rho_{j,k}| \le 0.00477$** (mean $|\rho| = 0.00098\text{--}0.00103$, matching the $1/\sqrt{640{,}000} \approx 0.00125$ finite-sample noise floor) both within words (`Word #0, #1, #7, #127, #254, #255`) and across consecutive steps (`Lag-1` and `Lag-2` cross-word BIC).
4. **Minimum Perturbation Distance (`1-bit flip`, `+1 increment`, `Gray-code step`):** Across $100{,}000$ transitions on both `key` and `addr_hi`, the mean output Hamming distance is **$15.96\text{--}16.03\text{ bits}$** out of $32$ at $K=1$ and **$4{,}079.5\text{--}4{,}080.4\text{ bits}$ ($0.49993\text{--}0.50005$)** out of $8{,}160\text{ bits}$ across Cycle 0 ($K=255$).
5. **Dieharder Full 114-Subtest Battery (`dieharder -a -g 200`):** Both the continuous orbit stream (`key=42, addr_hi=0`) and the stateless $K=1$ Weyl stream (`k1_weyl`) achieve **`113 PASS / 1 WEAK / 0 FAIL` out of 114 subtests** and **`27 / 27 PASS` across all 27 "Good"-reliability test families**.

---

### 3.4 Frozen Golden KAT Vectors (`ra_core v2 Final`: `Opsi_A3` + `eng_v2_E2`)

Gate 0 cross-verified **`2,720,934` output words** with **0 mismatches** across `ra_core_singleblock`, `ra_core_orbit`, 1-step unoptimized reference, 8-way AVX2 `ra_core_singleblock_x8_into`, and stateful `ra_core_engine_t` (`next_u32`, `fill_u32`, `discard`, `fork`, `save`/`load`). The frozen Golden KAT vectors and 64-bit FNV-1a stream checksums for `ra_core v2 Final` are tabulated below:

| Seed Label | `(key, addr_hi)` | Derived `(k_mix, step_m, step_l)` | Words `[0..3]` (`K=1..4`) | Cycle Boundary `[254, 255, 256]` | `FNV1a-64 (K=1)` | `FNV1a-64 (K=255)` |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **`key0_addr0`** | `(0x00000000, 0x00000000)` | `(0x4ea2c4a4, 0x24416e35, 0x2785f64d)` | `0xc1dbbb12, 0xfbede3b9, 0x93b1e2c8, 0x2067ac16` | `0x9297471c, 0xeb5f03e7, 0x8970ad1b` | `0x92565b17647cb906` | `0xcf914a65c5bfc357` |
| **`key42_addr0`** | `(0x0000002a, 0x00000000)` | `(0xc14755a0, 0x0ec3467d, 0xde211303)` | `0x9533450c, 0x288d3bd2, 0x7e833e19, 0xff96b720` | `0xb565ef5b, 0x50c62f0b, 0xf7c65da2` | `0x113204a23f2c5aaa` | `0xa2a2ab8dee50bccf` |
| **`key1_addr0`** | `(0x00000001, 0x00000000)` | `(0x5bf526f5, 0xafed9981, 0x196abc55)` | `0xd8d07063, 0xd78f6289, 0x180d4a60, 0xe7ddc2c3` | `0x9b622c40, 0x1bc8ae33, 0x274436e3` | `0x1ff90b90933eae2e` | `0xc8db90b4a0564280` |
| **`deadbeef`** | `(0xdeadbeef, 0x9e3779b9)` | `(0xf28e02c1, 0x1b0f916d, 0x9a78d4e5)` | `0x5612413d, 0xc09648c9, 0xd533833f, 0xee2fc833` | `0x59bec6de, 0xfcb0cbf3, 0xbc40ce77` | `0x0be53b12082495b7` | `0xa81bf05044ea4b69` |
| **`all_ones`** | `(0xffffffff, 0xffffffff)` | `(0x49554d20, 0x1833d849, 0xe8337575)` | `0x4ada0f26, 0x78b8b5de, 0xc07bf57c, 0xb049a08e` | `0xb8e535c0, 0x519cf7b7, 0x34b20631` | `0xf502aa641cebbf06` | `0x7fb358921bbf86c3` |

*(Note: For reference against Stage 1 ablation baselines prior to `Opsi_A3` promotion, unpatched `v2_base` produced word #0 `0x2df6d9b3` / $K=255$ FNV-1a `0xb2f0606b65e67d3b` for `(0,0)` and word #0 `0xf736e01a` / $K=255$ FNV-1a `0xcab7cc568b67bd99` for `(42,0)`.)*

---

## 4. Performance Evaluation and Clean Monte Carlo Shootout

All final benchmarks were executed on two pinned x86-64 platforms: **Machine A (Primary Client Workstation Node:** 11th Gen Intel Core i3-1115G4, Tiger Lake @ 3.00 GHz base / 4.10 GHz boost, AVX2 + AVX-512, Arch Linux `7.2.11-zen1`, **GCC 16.2.1**, `taskset -c 0`) and **Machine B (Server/Cloud Validation Node:** Xeon-class KVM hypervisor, 8 vCPU @ fixed `1.995 GHz`, AVX2 + AVX-512, Debian 13, **GCC 14.2.0**, `taskset -c 3`), compiled with `-O3 -march=native -mavx2 -fno-tree-vectorize -ffp-contract=off -std=gnu17 -lm`. Because different stages used distinct sampling protocols, each table explicitly labels its harness, trial count, and aggregation metric: **best-of-7** (`min` time) in the Stage 2 stateful engine ablation (`2026-11-04_v2-engine-opt`), **median-of-7** in the Stage 4 production sanity check (`2026-11-04_v2-final-battery`), and **median-of-9** interleaved rounds in the Stage 6 Cross-Machine `Suite D` (`2026-11-05_v2-vps-final-benchmark`) and Stage 3 Clean Monte Carlo (`2026-11-04_v2-clean-mc-harness`) harnesses.

### 4.1 Raw Throughput Across Burst Lengths $K$ and Stateful Engine Optimization (`Suite D` & Stage 2 Ablation)

Table 1 reports two complementary measurements on Machine A (`i3-1115G4`): (i) the **Stage 4 Production API Sanity Check** (`median-of-7` `MiB/s` for promoted stateless `ra_core_default_into` and 8-way AVX2 `ra_core_singleblock_x8_into`), and (ii) the **Stage 2 Stateful OTF Engine Ablation** (`best-of-7` / `median-of-7` `MiB/s` comparing coupled-pair `eng_v2_E2` against unrolled `eng_v2_old`, `eng_v1_baseline`, `Philox4x32-10`, and `ChaCha8`). Cross-machine 9-round median `Suite D` numbers on both Machine A and Machine B are reported in Section 4.5 (Table 5).

| Burst Length $K$ | `ra_core v2` Stateless (Stage 4 Med-of-7 `MiB/s`) | `eng_v2_E2` OTF (Stage 2 Best / Med `MiB/s`) | `eng_v2_old` OTF (Stage 2 Best / Med `MiB/s`) | `eng_v1` Baseline OTF (Stage 2 Best / Med `MiB/s`) | `E2` vs. `v2_old` / vs. `v1` (Best-of-7) | Philox4x32-10 (Stage 2 Best / Med `MiB/s`) | `ra_core v2` 8-Way AVX2 (Stage 4 Med `MiB/s`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 1$** | **$710.2$** | **$582.0$ / $578.4$** | $523.2$ / $514.0$ | $421.9$ / $412.1$ | **$+11.2\%$ ($1.11\times$) / $+38.0\%$ ($1.38\times$)** | $277.1$ / $272.5$ | $1{,}329.2$ |
| **$K = 2$** | **$764.4$** | **$710.7$ / $682.9$** | $666.6$ / $634.0$ | $609.1$ / $606.8$ | **$+6.6\%$ ($1.07\times$) / $+16.7\%$ ($1.17\times$)** | $534.7$ / $509.9$ | $2{,}478.3$ |
| **$K = 4$** | **$960.8$** | **$884.1$ / $834.0$** | $844.1$ / $819.2$ | $818.8$ / $814.2$ | **$+4.7\%$ ($1.05\times$) / $+8.0\%$ ($1.08\times$)** | $989.8$ / $971.9$ | $4{,}119.9$ |
| **$K = 8$** | **$1{,}292.1$** | **$1{,}012.6$ / $997.4$** | $915.2$ / $855.7$ | $939.2$ / $918.2$ | **$+10.6\%$ ($1.11\times$) / $+7.8\%$ ($1.08\times$)** | $1{,}046.8$ / $1{,}033.1$ | $5{,}648.4$ |
| **$K = 16$** | **$1{,}416.4$** | **$1{,}104.7$ / $1{,}094.3$** | $869.9$ / $860.4$ | $1{,}034.2$ / $1{,}012.6$ | **$+27.0\%$ ($1.27\times$) / $+6.8\%$ ($1.07\times$)** | $1{,}073.2$ / $1{,}062.6$ | $7{,}578.0$ |
| **$K = 64$** | **$1{,}476.4$** | **$1{,}169.9$ / $1{,}156.0$** | $927.4$ / $879.6$ | $1{,}252.7$ / $1{,}197.9$ | **$+26.1\%$ ($1.26\times$) / $-6.6\%$ ($0.93\times$)** | $1{,}082.7$ / $1{,}054.8$ | $9{,}851.4$ |
| **$K = 255$** | **$1{,}596.5$** | **$1{,}219.4$ / $1{,}162.9$** (`2.34 cpb`) | $918.9$ / $873.3$ (`3.11 cpb`) | $1{,}303.6$ / $1{,}293.7$ (`2.19 cpb`) | **$+32.7\%$ ($1.33\times$) / $-6.5\%$ ($0.94\times$)** | $1{,}077.7$ / $1{,}032.9$ (`2.65 cpb`) | **$10{,}687.9$ (`10.44 GiB/s`)** |
| **$K = 256$** | **$1{,}240.1$** | **$539.4$ / $437.4$** (`5.30 cpb`) | $486.6$ / $393.8$ (`5.87 cpb`) | $452.9$ / $304.7$ (`6.31 cpb`) | **$+10.8\%$ ($1.11\times$) / $+19.1\%$ ($1.19\times$)** | $974.2$ / $752.6$ | — |
| **$K = 1000$** | **$1{,}661.3$**$^{\mathrm{a}}$ | $839.2$ / $783.0$ (`3.40 cpb`) | $890.7$ / $813.2$ (`3.21 cpb`) | $732.1$ / $690.2$ (`3.90 cpb`) | $-5.8\%$ ($0.94\times$) / **$+14.6\%$ ($1.15\times$)** | $1{,}018.8$ / $944.2$ | — |

$^{\mathrm{a}}$Stage 4 production sanity check measured at $K = 1020$ (`1,661.3 MiB/s`); in the 9-round cross-machine `Suite D` shootout at $K = 1000$, stateless `Opsi_A3` reaches `1,390.47 MiB/s` on Machine A and `863.42 MiB/s` on Machine B.

Key observations from Table 1 and the cross-machine `Suite D` evaluation:
- **Recovery of Stateful OTF Throughput (`eng_v2_E2` vs. `eng_v2_old` and `eng_v1`):** Aligning `m_ring` setup at $K=2$ (`i=254`) and executing coupled 2-step OTF pairs from $K=3..254$ boosts $K=255$ stateful OTF throughput from $918.9\text{ MiB/s}$ in unrolled `eng_v2_old` to **$1{,}219.4\text{ MiB/s}$ best-of-7 / $1{,}162.9\text{ MiB/s}$ median-of-7 ($+32.7\%$, $1.33\times$ over `v2_old`)**, recovering $93.5\%$ of `eng_v1_baseline`'s simpler 1-step OTF loop at $K=255$ ($1{,}303.6\text{ MiB/s}$) while **outperforming `eng_v1_baseline` at $K=1..16$ (by $1.38\times$ at $K=1$), $K=256$ ($1.19\times$), and $K=1000$ ($1.15\times$)** and carrying full `Opsi_A3` 2D key-mixing.
- **Comparison Against `Philox4x32-10` and `ChaCha8` Across Regimes:** When $K$ is known upfront (`Opsi_A3` Direct Stateless), `ra_core v2` outperforms `Philox4x32-10` and `ChaCha8` across **all single-cycle and multi-cycle regimes ($K=1..1000$ on Machine A; $K=1..255$ and $K=510..1000$ on Machine B)**, reaching **$2.65\times\text{--}2.66\times$ over Philox at $K=1$** and **$1.20\times\text{--}1.34\times$ at $K=255$**. In incremental unknown-$K$ OTF mode (`eng_v2_E2`), `ra_core v2` outperforms `Philox4x32-10` at $K=1..2$ on both machines ($1.92\times\text{--}2.10\times$ at $K=1$) and at $K=16..255$ on Machine A ($1{,}104.7\text{--}1{,}219.4$ vs. $1{,}073.2\text{--}1{,}077.7\text{ MiB/s}$), whereas at $K \ge 256$ (where `eng_v2_E2` materializes its $M, L$ permutation arrays on the fly) and at $K=4..255$ on the fixed-clock server node (`Suite D` Machine B: $663.49$ vs. $766.49\text{ MiB/s}$ at $K=255$), counter-based `Philox4x32-10` remains faster than incremental OTF `eng_v2_E2`.

---

### 4.2 Dual-Mode Monte Carlo Evaluation Methodology (`Stage 3 Clean MC Harness`)

A recurring methodological challenge when comparing PRNG speeds on variable-length stochastic simulations (such as branching trees or Russian-roulette photon transport) is **stochastic work drift**: because each PRNG emits a different stream of uniform variates $U \in [0, 1)$, stochastic branching decisions ($U < p_{\text{branch}}$ or $U < w_{\text{survive}}$) diverge across generators, causing one PRNG to simulate $0.5\text{--}2\%$ more or fewer total tree nodes or scattering steps than another. Wall-clock runtime then confounds *per-draw generator speed* with *stochastic trajectory length fluctuations*.

Following the spirit of controlled simulation experimental design (such as Common Random Numbers in variance-reduction methodology), Stage 3 employed a **Dual-Mode Monte Carlo Evaluation Protocol**:
1. **Mode A — Coupled Stochastic Physics (`3A` / `E1`):** The candidate PRNG drives all physical decisions (branching counts, scattering angles, Russian-roulette survival). This validates **statistical accuracy** (confirming that every generator converges to the physical observable within statistical error) alongside end-to-end timing.
2. **Mode B — Fixed-Topology Same-Work Trace (`3B` / `E2`):** A reference topology schedule pre-computes the exact tree structure or survival step counts ($K_{\text{node}}$ or $K_{\text{photon}}$), while the candidate PRNG is freshly seeded per node/photon $(s, d)$ and executes the exact physical arithmetic. This guarantees **bit-identical draw counts ($0.000\%$ work drift)** across all generators, isolating true PRNG seeding and prefix generation cost.

---

### 4.3 Workload 2b-1 & 2b-2: 2D Lattice Bernoulli Sweep vs. True 2D Ising Checkerboard Metropolis at $T_c$

We separated 2D lattice simulations into two distinct benchmarks in `clean_mc_shootout.c` ($N = 9$ interleaved rounds):
- **Workload 2b-1 (`2D Lattice Bernoulli Sweep`, $2{,}097{,}152$ draws, $K=1$ stateless per site):** Pure threshold test measuring raw $K=1$ random-access overhead. `ra_core v2` (`Opsi_A3`) completes in **`15.351 ms` on Machine A** (**$1.342\times$ faster than Philox4x32-10** at `20.608 ms` and **$5.09\times$ faster than ChaCha8** at `78.115 ms`) and **`22.436 ms` on Machine B** (**$1.304\times$ faster than Philox** at `29.258 ms`).
- **Workload 2b-2 (`True 2D Ising Checkerboard Metropolis` at Critical Temperature $T_c = 2 / \ln(1 + \sqrt{2}) \approx 2.2691853$, $L \times L = 256 \times 256$ periodic lattice, $N_{\text{sweeps}} = 32$, `2,097,152` spin updates):** Evaluates a true interacting spin Hamiltonian $H = -J \sum_{\langle i,j \rangle} s_i s_j$ ($s_i \in \{-1, +1\}$) decomposed into **Black** ($(x+y) \bmod 2 = 0$) and **White** ($(x+y) \bmod 2 = 1$) checkerboard sublattices with precomputed Boltzmann lookup tables $\exp(-\Delta E / T_c)$ for $\Delta E \in \{4, 8\}$. Table 2 reports both **Mode 2b-2A (Per-Site Stateless $K=1$)** and **Mode 2b-2B (Row-Sublattice Batched $K=128$)** across Machine A and Machine B (where physical observables $\langle E \rangle / N$ and $\langle |M| \rangle$ are 100% bit-identical across both machines):

| Generator | Mode 2b-2A ($K=1$) Client / Server (`ms`) | 2b-2A Speedup vs. Philox (Client / Server) | Mode 2b-2A Energy $\langle E \rangle / N$ (`obs1`) | Mode 2b-2A Magnetization $\langle \|M\| \rangle$ (`obs2`) | Mode 2b-2B ($K=128$) Client / Server (`ms`) | 2b-2B Speedup vs. Philox (Client / Server) | Mode 2b-2B Energy $\langle E \rangle / N$ / $\langle \|M\| \rangle$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ra_core v2` (`Opsi_A3` Stateless)** | **`19.106` / `29.501`** | **$1.250\times$ / $1.276\times$** | **$-1.417175$** | **$0.582092$** | **`12.465`** / `22.573` | **$1.072\times$** / $0.949\times$ | **$-1.422180$ / $0.625549$** |
| **`ra_core v2` (`eng_v2_E2` Stateful)** | — | — | — | — | `13.883` / `29.698` | $0.962\times$ / $0.721\times$ | $-1.422180$ / $0.625549$ |
| **`ra_core v1` (`W0_D0` Baseline)** | `23.626` / `41.553` | $1.010\times$ / $0.906\times$ | $-1.421265$ | $0.625916$ | `14.544` / `21.905` | $0.918\times$ / $0.977\times$ | $-1.420898$ / $0.615570$ |
| **Philox4x32-10** | `23.873` / `37.644` | $1.000\times$ / $1.000\times$ | $-1.401489$ | $0.570770$ | `13.357` / **`21.411`** | $1.000\times$ / $1.000\times$ | $-1.404419$ / $0.563965$ |
| **ChaCha8** | `83.605` / `143.400` | $0.286\times$ / $0.263\times$ | $-1.408936$ | $0.616394$ | — | — | — |
| **Squares32** | `10.029` / `18.017` | $2.380\times$ / $2.089\times$ | $-1.430908$ | $0.607330$ | `9.914` / `16.031` | $1.347\times$ / $1.336\times$ | $-1.428345$ / $0.649567$ |
| **WyRand** (ad-hoc per-site seed) | `11.423` / `18.249` | $2.090\times$ / $2.063\times$ | **$-1.270264$** | **`0.288757` (ANOMALY)** | `9.320` / `14.715` | $1.433\times$ / $1.455\times$ | $-1.365356$ / $0.561890$ |
| **SplitMix64** (ad-hoc per-site seed) | `10.138` / `17.480` | $2.355\times$ / $2.154\times$ | $-1.389526$ | $0.564026$ | `10.219` / `15.458` | $1.307\times$ / $1.385\times$ | $-1.425110$ / $0.641815$ |
| **PCG32** (re-seeded per call) | `9.815` / `16.420` | $2.432\times$ / $2.293\times$ | $-1.429504$ | $0.616547$ | `10.723` / `15.556` | $1.246\times$ / $1.376\times$ | $-1.413269$ / $0.578949$ |
| **SFC64** (re-seeded per call) | `16.078` / `25.324` | $1.485\times$ / $1.486\times$ | $-1.423401$ | $0.652832$ | `9.606` / `17.711` | $1.390\times$ / $1.209\times$ | $-1.430725$ / $0.626923$ |
| **Xoshiro256\*\*** (re-seeded per call) | `10.929` / `18.080` | $2.184\times$ / $2.082\times$ | $-1.441589$ | $0.673187$ | `9.859` / `15.080` | $1.355\times$ / $1.420\times$ | $-1.403992$ / $0.582336$ |

Two findings emerge from Table 2:
1. **`ra_core v2` (`Opsi_A3`) Outperforms Structured Counter-Based Baselines at $K=1$:** In Mode 2b-2A ($K=1$ per site), `Opsi_A3` is **$1.250\times$ faster than `Philox4x32-10` on Machine A** (`19.106` vs. `23.873 ms`) and **$1.276\times$ faster on Machine B** (`29.501` vs. `37.644 ms`), while improving over `ra_core v1` by **$1.237\times\text{--}1.409\times$**. In Mode 2b-2B ($K=128$ row-sublattice batching), `Opsi_A3` is **$1.072\times$ faster than Philox on Machine A** (`12.465` vs. `13.357 ms`) and within $5\%$ on Machine B (`22.573` vs. `21.411 ms`), while `eng_v2_E2` produces **100% bit-identical observables and spin checksums (`0x63695284`)**. Lightweight non-cryptographic stateful generators (`PCG32`, `Squares32`, `Xoshiro256**`) execute faster when minimally seeded per site, as expected from their small state footprint.
2. **Diagnostic Lattice Anomaly Under Ad-Hoc Per-Site `WyRand` Seeding ($K=1$):** Under the tested per-site stateless coordinate mapping ($\text{seed} = \text{key} \oplus (\text{addr\_hi} \ll 32)$ followed by one `WyRand` increment and wide multiply), **WyRand** exhibits a marked trajectory anomaly on this 32-sweep realization: its magnetization drops to **$\langle |M| \rangle = 0.288757$** (`0.288811` in the preliminary harness) with energy $\langle E \rangle / N = -1.270264$, whereas `Opsi_A3`, `Philox4x32-10`, `ChaCha8`, and other generators cluster around $\langle |M| \rangle \in [0.564, 0.673]$ and $\langle E \rangle / N \in [-1.442, -1.390]$. While a single 32-sweep realization does not establish asymptotic equilibrium bias or a universal failure of `WyRand` under full `splitmix64` seeding, it provides a concrete diagnostic case showing how feeding un-premixed 2D $(\text{site}, \text{sweep})$ coordinates directly into a single-step multiplicative mixer can induce lattice trajectory divergence that `Opsi_A3`'s dual-`fmix32` pre-mixing avoids.

---

### 4.4 Workload 2a (Asian Option Pricing), Workload 3A/3B (Branching Tree), and Workload E1/E2 (Photon Transport)

#### Workload 2a: Parallel Path-Independent Asian Option Pricing ($K \in \{16, 64, 255\}$)
Simulating geometric Brownian motion paths ($S_0 = 100, K_{\text{strike}} = 100, r = 0.05, \sigma = 0.20, T = 1.0$) where each path ID is an independent random-access coordinate $(\text{key} = \text{path\_id}, \text{addr\_hi} = 0)$ in the Stage 3/6 Clean MC harness (`clean_mc_shootout.c`, $N=9$ rounds):

| Horizon $K$ (Total Draws) | `ra_core v2` (`Opsi_A3`) Client / Server (`ms`) | `eng_v2_E2` Stateful Client / Server (`ms`) | `ra_core v1` (`W0_D0`) Client / Server (`ms`) | Philox4x32-10 Client / Server (`ms`) | `Opsi_A3` vs. `v1` (Client / Server) | `Opsi_A3` vs. Philox (Client / Server) | Option Price ($\pm \text{SE}$, Client = Server) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 16$** (`1.60M` draws) | **`8.422`** / `13.965` | `9.716` / `17.562` | `11.149` / `16.796` | `9.219` / **`13.934`** | **$1.324\times$ / $1.203\times$** | **$1.095\times$** / $0.998\times$ | $6.018274 \pm 0.026314$ |
| **$K = 64$** (`1.28M` draws) | `15.588` / `24.383` | `16.342` / `26.860` | `19.644` / `28.193` | `16.723` / **`23.973`** | **$1.260\times$ / $1.156\times$** | **$1.073\times$** / $0.983\times$ | $5.847510 \pm 0.036514$ |
| **$K = 255$** (`5.10M` draws) | **`22.859`** / `37.705` | `24.646` / `40.483` | `30.326` / `43.961` | `25.787` / **`36.946`** | **$1.327\times$ / $1.166\times$** | **$1.128\times$** / $0.980\times$ | $5.786712 \pm 0.026067$ |

Across all three monitoring horizons, `ra_core v2` (`Opsi_A3`) is **$1.260\times\text{--}1.327\times$ faster than `ra_core v1` on Machine A** ($1.156\times\text{--}1.203\times$ on Machine B), **outperforms `Philox4x32-10` by $1.073\times\text{--}1.128\times$ on Machine A**, and runs at virtual parity ($0.980\times\text{--}0.998\times$) with `Philox4x32-10` on Machine B.

#### Workload 3A & 3B: Dual-Mode Branching Particle Cascade Tree
In **Mode 3B (Fixed-Topology Same-Work Trace)** of the Clean MC harness (`clean_mc_shootout.c`), a deterministic topology schedule `tree_schedule_ctrl(node_id, depth)` forces every generator to evaluate the exact same **`18,704` nodes** and **`396,628` uniform draws** ($0.000\%$ work drift):
- **`ra_core v2` (`Direct_A3` Stateless):** **`1.978 ms` on Machine A** (**$1.331\times$ faster than `ra_core v1`** at `2.634 ms`, **$1.056\times$ faster than `Philox4x32-10`** at `2.089 ms`, and **$1.170\times$ faster than `ChaCha8`** at `2.315 ms`). On Machine B (Server), `Direct_A3` completes in **`3.370 ms`** (**$1.485\times$ faster than `ra_core v1`** at `5.006 ms` and **$1.239\times$ faster than `ChaCha8`** at `4.175 ms`, while trailing `Philox4x32-10` at `2.949 ms`). *(Note: an earlier lightweight micro-tree prototype reported `0.464 ms` vs. `0.482 ms` prior to standardizing the 18,704-node / 396,628-draw Clean MC harness.)*
- **`eng_v2_E2` Stateful OTF Engine:** **`2.412 ms` on Machine A** (**$1.092\times$ faster than `v1` engine**) and **`4.529 ms` on Machine B** (**$1.105\times$ faster than `v1`**), producing the **100% bit-identical 64-bit checksum (`0x650730d8b86d062d`)** as `Direct_A3`.
- In **Mode 3A (Coupled Stochastic Physics, ~47k draws)**, `eng_v2_E2` completes in **`0.740 ms` on Machine A** (**$1.141\times$ faster than `Philox4x32-10`** at `0.845 ms`) and **`1.176 ms` on Machine B** (**$1.023\times$ faster than `Philox4x32-10`** at `1.203 ms`).

#### Workload E1 & E2: Dual-Mode Monte Carlo Photon Transport
We evaluated Photon Transport across two complementary harnesses: (i) the **Clean MC Slab Benchmark (`E1-Slab` / `E2-Slab`, $P_{\text{rej}} = 0.90$, exact `731,124` draws in Fixed Mode `E2`)** executed on both Machine A and Machine B, and (ii) the **Deep Turbid-Slab Benchmark ($\mu_a = 0.02, \mu_s = 0.98$, $10^5$ photons, exact `15,076,849` draws in Fixed Mode `E2`)** on Machine A:

| Harness & Mode | Generator | Machine A Time (`ms`) | Machine B Time (`ms`) | Speedup vs. `v1` OTF (A / B) | Speedup vs. Philox (A / B) | Exact Draws / Observables |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Clean MC `E1-Slab` ($P_{\text{rej}}=0.90$, Coupled)** | **`eng_v2_E2` Stateful OTF** | `7.955` | **`9.255`** | $0.985\times$ / **$1.056\times$** | **$1.077\times$ / $1.335\times$** | $731{,}124$ draws ($R_d, T_d$ exact) |
| Clean MC `E1-Slab` ($P_{\text{rej}}=0.90$, Coupled) | `ra_core v1` (`eng_v1`) | **`7.832`** | `9.771` | $1.000\times$ / $1.000\times$ | $1.094\times$ / $1.264\times$ | $736{,}230$ draws |
| Clean MC `E1-Slab` ($P_{\text{rej}}=0.90$, Coupled) | `Philox4x32-10` | `8.566` | `12.354` | $0.914\times$ / $0.791\times$ | $1.000\times$ / $1.000\times$ | $735{,}672$ draws |
| **Clean MC `E2-Slab` ($P_{\text{rej}}=0.90$, Fixed)** | **`eng_v2_E2` Stateful OTF** | **`5.282`** | **`7.064`** | **$1.169\times$ / $1.186\times$** | **$1.528\times$ / $1.242\times$** | **`731,124` exact draws** (`0x4cdedfa8a8`) |
| Clean MC `E2-Slab` ($P_{\text{rej}}=0.90$, Fixed) | `ra_core v1` (`eng_v1`) | `6.175` | `8.378` | $1.000\times$ / $1.000\times$ | $1.307\times$ / $1.047\times$ | `731,124` exact draws |
| Clean MC `E2-Slab` ($P_{\text{rej}}=0.90$, Fixed) | `Philox4x32-10` | `8.069` | `8.773` | $0.765\times$ / $0.955\times$ | $1.000\times$ / $1.000\times$ | `731,124` exact draws |
| Clean MC `E2-Slab` ($P_{\text{rej}}=0.90$, Fixed) | `ChaCha8` | `13.390` | `20.567` | $0.461\times$ / $0.407\times$ | $0.603\times$ / $0.427\times$ | `731,124` exact draws |
| **Deep Slab (`15.08M` Draws, Fixed)** | **`ra_core v2` (`Opsi_A3` Buffered / `eng_v2_E2`)** | **`112.69` / `127.63`** | — | **$1.268\times$ / $1.119\times$** | **$1.097\times$** / $0.969\times$ | **`15,076,849` exact draws** |
| Deep Slab (`15.08M` Draws, Fixed) | `ra_core v1` (`eng_v1`) / `Philox4x32-10` | `142.84` / `123.62` | — | $1.000\times$ / $1.155\times$ | $0.865\times$ / $1.000\times$ | `15,076,849` exact draws |

In the cross-machine `E2-Slab` fixed-trajectory benchmark (`731,124` exact draws across all generators), stateful `eng_v2_E2` outperforms `ra_core v1` by **$1.169\times\text{--}1.186\times$** and outperforms `Philox4x32-10` by **$1.528\times$ on Machine A (`5.282` vs. `8.069 ms`)** and **$1.242\times$ on Machine B (`7.064` vs. `8.773 ms`)**.

---

### 4.5 Cross-Architecture Reproducibility and Second-Machine Server Validation (`Stage 6`)

To verify that the bit-exact specification, cycle-normalized speedups, and Monte Carlo physical observables of `ra_core v2` are invariant across hardware microarchitectures, operating system kernels, and compiler toolchains, we executed the complete Stage 4 Gate 0 verification (`2,720,934` words), Suite D pure-generate shootout ($K \in [1, 1000]$), and Stage 3 Clean Monte Carlo battery on a second, independent server platform: a **Server/Cloud x86-64 Validation Node** (Common KVM processor / Xeon-class hypervisor, 8 vCPU @ fixed `1.995 GHz`, AVX2 + AVX-512, Debian GNU/Linux 13 *trixie*, Kernel `6.12.90+deb13-cloud-amd64`, **GCC 14.2.0**), pinned to an isolated core (`taskset -c 3`) and compared against the primary client workstation node (Intel Core i3-1115G4 @ `3.00–4.10 GHz`, Arch Linux `7.2.11-zen1`, **GCC 16.2.1**).

| Benchmark / Regime | Metric | Client Node (GCC 16.2.1) | Server Validation Node (GCC 14.2.0) | Cross-Machine Parity / Observation |
| :--- | :--- | :---: | :---: | :--- |
| **Gate 0 Parity & Golden KATs** | Verified Words (`Mismatches`) | `2,720,934` (`0`) | `2,720,934` (`0`) | **100% Bit-Exact (`0 mismatches`, all 5 Golden KATs identical)** |
| **Suite D: $K = 1$ Stateless** | `Opsi_A3` vs. `Philox4x32-10` | `716.65` vs. `270.44 MiB/s` (`3.98` vs. `10.56 cpb`) | **`461.09` vs. `173.15 MiB/s` (`4.12` vs. `10.99 cpb`)** | **$2.66\times$ speedup on Server (`CoV 4.15%`, cksum `0xef31cf72`)** |
| **Suite D: $K = 64$ Stateless** | `Opsi_A3` vs. `Philox4x32-10` | `1,438.82` vs. `1,078.68 MiB/s` (`1.99` vs. `2.65 cpb`) | **`960.00` vs. `803.48 MiB/s` (`1.98` vs. `2.37 cpb`)** | **$1.19\times$ speedup on Server (`1.98 cpb`, cksum `0x964a23b6`)** |
| **Suite D: $K = 255$ Stateless** | `Opsi_A3` vs. `Philox4x32-10` | `1,435.75` vs. `1,074.15 MiB/s` (`1.99` vs. `2.66 cpb`) | **`917.39` vs. `766.49 MiB/s` (`2.07` vs. `2.48 cpb`)** | **$1.20\times$ speedup on Server (8-way AVX2: `6,180.36 MiB/s` / `6.04 GiB/s`)** |
| **Workload 2b-1: Bernoulli ($K=1$)** | `Opsi_A3` vs. `Philox4x32-10` | `15.351` vs. `20.608 ms` (`21.92` vs. `29.43 cyc/d`) | **`22.436` vs. `29.258 ms` (`21.34` vs. `27.83 cyc/d`)** | **$1.304\times$ speedup on Server (`CoV 4.70%`, cksum `0x82121719`)** |
| **Workload 2b-2A: 2D Ising ($K=1$)** | `Opsi_A3` vs. `Philox4x32-10` | `19.106` vs. `23.873 ms` (`27.29` vs. `34.09 cyc/d`) | **`29.501` vs. `37.644 ms` (`28.06` vs. `35.81 cyc/d`)** | **$1.276\times$ speedup on Server ($\langle E \rangle/N = -1.417175$, $\langle \|M\| \rangle = 0.582092$)** |
| **Workload 2b-2A: WyRand Anomaly** | Magnetization $\langle \|M\| \rangle$ at $T_c$ | `0.288757` (`0.288811`) | **`0.288757` (`0.288811`)** | **Identical ad-hoc per-site seeding anomaly (`0xf7acbcfe`)** |
| **Workload 3B: Fixed-Topology Tree** | `Direct_A3` / `eng_v2_E2` vs. `v1` / `Philox` | `1.978` / `2.412` vs. `2.634` / `2.089 ms` | **`3.370` / `4.529` vs. `5.006` / `2.949 ms`** | **`396,628` exact draws, $1.331\times\text{--}1.485\times$ over `v1`, cksum `0x650730d8b86d062d`** |
| **Workload E2-Slab ($P_{\text{rej}}=0.90$)** | `eng_v2_E2` vs. `Philox4x32-10` | `5.282` vs. `8.069 ms` (`21.63` vs. `33.05 cyc/d`) | **`7.064` vs. `8.773 ms` (`19.27` vs. `23.93 cyc/d`)** | **$1.242\times$ speedup on Server (`731,124` draws, `CoV 3.37%`)** |

Three conclusions follow from the Stage 6 server validation:
1. **Zero Bit-Level Drift Across Compilers and Platforms:** Every single 32-bit word (`2,720,934` Gate 0 words), 64-bit FNV-1a KAT digest, Suite D stream checksum (`110/110`), and Monte Carlo trajectory checksum (`240/240`) matches bit-for-bit between `GCC 16.2.1` on the client workstation and `GCC 14.2.0` on the Debian 13 server node (`-fno-tree-vectorize -ffp-contract=off`).
2. **Cycle-Per-Byte (`cpb`) and Cycle-Per-Draw (`cyc/d`) Invariance:** Normalizing for the server node's fixed `1.995 GHz` clock versus the client's turbo clock, instruction-level efficiency is nearly identical across machines (`4.12` vs. `3.98 cpb` at $K=1$; `1.98` vs. `1.99 cpb` at $K=64$; `28.06` vs. `27.29 cyc/draw` in 2D Ising $K=1$).
3. **Low-Variance Server Timing (`CoV%`):** Absent mobile turbo boost transitions, the server node exhibits markedly tighter trial-to-trial Coefficient of Variation (`CoV = 1.00%–4.83%` across key regimes), confirming that the speedups of `ra_core v2` over `ra_core v1`, `Philox4x32-10`, and `ChaCha8` in stateless burst generation ($K=1..255, 510..1000$), $K=1$ lattice sweeps, and `E2-Slab` photon transport hold across CPU architectures.

---

### 4.6 Exploratory GPU CUDA Parity and Performance Characterization (`Blackwell sm_120`)

To evaluate whether the frozen `ra_core v2` specification (`Opsi_A3` + `W2_D2_xag_opt`) translates cleanly to massively parallel SIMT execution without bit drift, we implemented a direct CUDA C++17 port (`experiments/2026-11-05_gpu-colab-k1/` and `experiments/2026-11-05_gpu-batch1/`) and executed a sequential verification and characterization batch on an **NVIDIA RTX PRO 6000 Blackwell Server Edition GPU** (compute capability `12.0` / `sm_120`, `97,887 MiB` VRAM, CUDA 13.0 / `nvcc 13.0.88 -O3 -std=c++17 -arch=sm_120`), paired against the identical Random123-compatible `Philox4x32-10` function used in our CPU shootout:

| GPU Benchmark / Regime (`NVIDIA RTX PRO 6000 Blackwell`, `sm_120`) | `ra_core v2` GPU Kernel (`ms`) | `Philox4x32-10` GPU Kernel (`ms`) | `ra_core v2` GPU Total (`ms`) | `Philox4x32-10` GPU Total (`ms`) | Kernel Ratio (`Philox / RA`) | CPU$\leftrightarrow$GPU Bit-Exact Parity & Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Probe `raw_k1` ($16{,}777{,}216$ draws, 64 MiB)** | `0.0416` | `0.0415` | — | — | `0.998x` | **100% Parity (`1,024` addrs); tied (~1.61 TB/s bandwidth ceiling; 10 vs. 14 regs, 0 spill)** |
| **Probe `bernoulli_k1` ($16{,}777{,}216$ draws)** | `0.0719` | `0.0720` | — | — | `1.001x` | **100% Parity; tied within `0.5 us` CUDA event resolution (12 vs. 16 regs, 0 spill)** |
| **True 2D Ising $K=1$, 32 Sweeps ($L=64$)** | `0.132` | `0.133` | `0.157` | `0.159` | `1.010x` | **100% Exact Spin/Draw/Energy/Mag Parity vs. CPU (`CPU: 0.888 vs. 1.155 ms, 1.30x`)** |
| **True 2D Ising $K=1$, 32 Sweeps ($L=256$)** | `0.132` | `0.132` | `0.221` | `0.221` | `1.001x` | **100% Exact Parity (`cksum 0x8210691e`, $E/N=-1.417175$, $\|M\|=0.582092$; `CPU: 1.33x`)** |
| **True 2D Ising $K=1$, 32 Sweeps ($L=1024$)** | `0.263` | `0.264` | `1.144` | `1.145` | `1.004x` | **100% Exact Parity across all 5 timed runs (`CPU: 221.86 vs. 288.78 ms, 1.30x`)** |
| **Short Burst Consumer $K=2$ ($4{,}194{,}304$ addrs)** | `0.0143` | `0.0142` | `0.648` | `0.651` | `0.993x` | **100% Exact Parity (`1,024` addrs); tied on GPU** |
| **Short Burst Consumer $K=4$ ($4{,}194{,}304$ addrs)** | `0.0185` | `0.0144` | `0.652` | `0.649` | `0.777x` | **100% Exact Parity; Philox faster in kernel (1 block emits 4 words directly)** |
| **Short Burst Consumer $K=8$ ($4{,}194{,}304$ addrs)** | `0.0271` | `0.0226` | `0.663` | `0.654` | `0.833x` | **100% Exact Parity; Philox faster in kernel (2 blocks vs. 4 coupled pairs)** |

Three key findings and scope boundaries emerge from this GPU characterization:
1. **Bit-Exact CPU–GPU Simulation Parity:** Across all frozen KAT vectors ($K \in \{1, 2, 4, 8\}$), 1,024-address burst checks, and 32-sweep True 2D Ising Checkerboard Metropolis simulations at $L \in \{64, 256, 1024\}$, the CUDA kernels match the C/C++ CPU reference with **zero mismatches** on every final spin configuration, draw XOR audit, accepted flip count, energy ($\langle E \rangle / N = -1.417175$ at $L=256$), and magnetization ($\langle |M| \rangle = 0.582092$). `ptxas` confirms zero stack frame and zero register spills (`10–12` registers for `ra_core v2` vs. `14–16` for `Philox4x32-10` in the $K=1$ kernels).
2. **Hardware Regime Delineation (CPU ALU-Bound vs. GPU Bandwidth/Block-Bound):** While `ra_core v2` is consistently **$1.30\times\text{--}1.33\times$ faster than Philox on the host CPU** for 2D Ising and **$1.68\times\text{--}1.74\times$ faster on the host CPU** for $K=1$ output curves, their GPU kernel execution times at $K=1$, $K=2$, and 2D Ising ($L=64, 256, 1024$) are **effectively tied (`0.993x–1.010x`)**, as massive warp parallelism and memory/launch overhead mask per-draw ALU differences. Conversely, at $K=4$ and $K=8$ on the GPU, `Philox4x32-10` achieves `1.20x–1.29x` faster kernel times (`0.0144` vs. `0.0185 ms` at $K=4$) because a single 4-word Feistel block maps natively to four outputs without stepping a sequential window.
3. **Scope of the Exploratory GPU Batch:** This single-device Colab Blackwell run compares our direct Random123-compatible Philox kernel rather than vendor-tuned device libraries (such as `cuRAND`), and includes host-transfer audit records in `GPU total`. Consequently, we report the CUDA port as a **cross-architecture bit-exactness verification and hardware-regime boundary analysis**—demonstrating that `ra_core v2`'s deferred prefix computation provides its primary speed advantage on **CPU scalar/SIMD workloads**, while allowing identical simulation trajectories to be replayed or verified bit-for-bit on GPUs.

---

## 5. Parameter Sensitivity and State-Size Scaling ($N \in \{16, 32, 64, 128, 256\}$)

Because the `W2_D2_xag_opt` core and `END_CYCLE_M_W2` guard fold are parameterized solely by `RA_N` ($N = 2^k \ge 16$), `ra_core v2` supports compile-time state-size tuning (`-DRA_N=16..256`) to trade memory footprint against single-cycle burst capacity $K_{\max} = N - 1$:

| State Size $N$ (`RA_N`) | State Footprint ($M + L$ in Orbit Mode) | Single-Cycle Capacity $K_{\max} = N - 1$ | Sub-Orbit Boundary PractRand ($K \in [1, N-1]$) | Continuous Orbit PractRand (`key=42, addr_hi=0`) | Recommended Deployment Target |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$N = 16$** | $160\text{ B}$ | $15\text{ words}$ ($60\text{ B}$) | **15 / 15 PASS (`4 GiB CLEAN` at $K=14$)** | **1 GiB PASS** | Register-constrained GPU/SIMD warp lanes |
| **$N = 32$** | $288\text{ B}$ | $31\text{ words}$ ($124\text{ B}$) | **PASS (`1 GiB CLEAN`)** | **1 GiB PASS** | Embedded microcontrollers / L1-tight kernels |
| **$N = 64$** | $544\text{ B}$ | $63\text{ words}$ ($252\text{ B}$) | **PASS (`1 GiB CLEAN`)** | **1 GiB PASS** | Medium-horizon path MC ($K \le 63$) |
| **$N = 128$** | $1{,}056\text{ B}$ | $127\text{ words}$ ($508\text{ B}$) | **PASS (`1 GiB CLEAN`)** | **2 GiB PASS** | Sublattice row-batching ($K = 128$) |
| **$N = 256$ (Default)** | $2{,}080\text{ B}$ | $255\text{ words}$ ($1{,}020\text{ B}$) | **57 / 57 Regimes PASS (`0 FAIL`)** | **8 GiB PASS (`CLEAN`)** | General-purpose CPU, AVX2 SIMD, NumPy `BitGenerator` |

---

## 6. Conclusion

`ra_core v2` demonstrates how a permutation-coupled state-array PRNG can support **deferred computation of a stream prefix** alongside robust 2D coordinate seeding through three integrated mechanisms:
1. **`Opsi_A3` 2D Cross-Pre-Mix Key-Mixing**, which guarantees independent dual-`fmix32` avalanches and odd bijective index steps across 2D $(\text{key}, \text{addr\_hi})$ slices with zero in-loop instruction overhead, healing **16/16 adversarial seeding regimes** (`57/57 PASS` overall, combining 42 baseline and 15 adversarial regimes) and doubling toy-model Brent cycle lengths;
2. **`W2_D2_xag_opt` Causal 2-Tap Ramp-Up, 2-Step Unrolled Recurrence, and In-Place Coupling**, which restricts the initial causal dependency cone to 2 state words at $K=1$ and 3 words at $K=2$, breaks the 1-step serial scrambler dependency via lag-2 delayed feedback ($D2$), and replaces $O(N)$ inter-cycle reseeding with in-place $M[i+7] \mathrel{+}= L[d_{\text{use}}]$ accumulation and an $O(1)$ 7-word guard fold (`END_CYCLE_M_W2`); and
3. **`eng_v2_E2` Coupled Pair Stateful OTF Rolling**, which defers full array initialization inside the `3,520`-byte stateful engine until $K=256$, boosting incremental unknown-$K$ streaming at $K=255$ by **$1.33\times$ over unrolled `v2_old`** ($1{,}219.4\text{ MiB/s}$ best-of-7 / $1{,}162.9\text{ MiB/s}$ median-of-7 vs. $918.9\text{ MiB/s}$, recovering $93.5\%$ of `v1` baseline OTF throughput at $K=255$ while outperforming `v1` OTF by $1.38\times\text{--}1.74\times$ at $K=1$) and maintaining 100% bit-exact parity (`0 mismatches` across $2.72 \times 10^6$ verified words) with stateless batch generation.

Across both x86-64 CPU validation platforms, stateless `ra_core v2` (`Opsi_A3`) outperforms `Philox4x32-10` and `ChaCha8` in pure-generate bursts ($K=1..255$ and $K=510..1000$, reaching $10{,}687.9\text{ MiB/s}$ / $10.44\text{ GiB/s}$ client and $6{,}180.36\text{ MiB/s}$ / $6.04\text{ GiB/s}$ server in 8-way AVX2), 2D Lattice Bernoulli sweeps ($1.304\times\text{--}1.342\times$), and $K=1$ True 2D Ising Checkerboard Metropolis sweeps at $T_c$ ($1.250\times\text{--}1.276\times$), while stateful `eng_v2_E2` outperforms `Philox4x32-10` in $K=1$ incremental draws and `E2-Slab` fixed-trajectory Photon Transport Monte Carlo ($1.242\times\text{--}1.528\times$). On the NVIDIA RTX PRO 6000 Blackwell GPU, our CUDA port confirms **100% bit-exact CPU–GPU parity** across short bursts and 32-sweep critical 2D Ising simulations, running at parity with Philox at $K=1$ and Ising while clearly identifying wide-block GPU regimes ($K=4, 8$) and future reductions of the static `ra_core_engine_t` memory footprint as concrete directions for subsequent work.
