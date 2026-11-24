# GPU K1 probe on Colab

`gpu_k1_colab.ipynb` is a self-contained exploratory notebook for the NVIDIA
RTX PRO 6000 Blackwell Server Edition (compute capability 12.0) and CUDA 13.0.
Upload the notebook through Colab's **File → Upload notebook**, then run cells in
order. The CUDA source is embedded in the notebook and also maintained as
`gpu_k1_probe.cu` in this directory.

The correctness gate checks frozen `ra_core v2 Final` K1 vectors, the three
Philox4x32-10 vectors already used by the CPU shootout, and GPU/CPU parity for
the first 1024 addresses. The benchmark runs only after the separate `--bench`
cell is started. It measures one output per independent address in two modes:

- `raw_k1`: write one 32-bit output per address.
- `bernoulli_k1`: make one threshold decision per address and reduce to one
  count per CUDA block.

Both generators receive the same number of independent addresses and use the
same output or reduction kernel shape. Philox uses the CPU shootout's mapping:
counter `{0,0,0,0}` and key `{KEY, addr_hi ^ 0xFEEDC0DE}`. One of the four
Philox block words is consumed, matching the K1 regime. This does **not**
describe Philox's amortized K>1 performance or full simulation performance.

The `RESULT` rows contain all nine timing samples, median, minimum, and maximum
in milliseconds. The notebook offers downloads for the compile and run logs.
Treat one Colab session as exploratory. A publishable GPU result still needs
repeated sessions, an independent device or controlled runtime, and a real
simulation workload.

The host-only KAT can be checked locally without CUDA:

```bash
g++ -std=c++17 -O3 -DRA_HOST_ONLY -x c++ gpu_k1_probe.cu -o gpu_k1_host_kat
./gpu_k1_host_kat
```

The first Colab run and its interpretation are archived in `RESULTS.md` and
the `data/` logs.
