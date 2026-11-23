# AMD HIP performance

## Quick start on the RX 9060 XT

Use an Inkling HIP build that includes the FP8 tree verifier and an
`Inkling-2-Mini-SQ-15.strata` model file. The example runs from the checkout root on
Windows with the local `gfx1200` build. Replace the model path with yours.

```powershell
$env:STRATA_INKLING_HIP_S6_DECODE = '1'
$env:STRATA_INKLING_HIP_Q8_PROJECTIONS = '1'
$env:STRATA_INKLING_HIP_Q8_VECTOR_DECODE = '1'
$env:STRATA_INKLING_HIP_FP8_GQA_DECODE = '1'
$env:STRATA_INKLING_HIP_DSPARK_WINDOW = '1024'
$env:STRATA_INKLING_HIP_TREE_FP8 = '1'
$env:STRATA_INKLING_HIP_FP8_PREFILL = '1'
$env:STRATA_INKLING_HIP_FAST_KV_GROW = '1'
$env:STRATA_INKLING_HIP_MOE_SF4_LUT = '1'

.\build-hip\src\inkling2\strata-inkling.exe --serve --native 'D:\models\Inkling-2-Mini-SQ-15.strata' --max-context 32768 --kv fp8 --draft on
```

`--kv fp8` selects the cache format used by the FP8 attention paths. `--draft on`
enables DSpark, and the 1,024-token window limits the drafter's global attention.
The target model still verifies against its full configured context.
`STRATA_INKLING_HIP_FP8_PREFILL` uses native FP8 conversion while reading prompts
on RDNA4. With it enabled, `STRATA_INKLING_HIP_FAST_KV_GROW` lets the engine grow
small global KV caches in larger steps for a single request when there is enough
VRAM. `STRATA_INKLING_HIP_MOE_SF4_LUT` reuses decoded SF4 weight values in the
mixed-expert prefill kernels. `--serve` starts the engine's standard input/output protocol.
See [AMD HIP build instructions](AMD_HIP.md).

### RX 9060 XT benchmark

Measured on Windows with the RX 9060 XT (16 GB, `gfx1200`), TheRock 7.14.1 and
Inkling-2-Mini SQ-15. Both benchmark arms used the same executable and settings.
The baseline had the earlier optimizations and capped DSpark enabled. The new
version also enabled `STRATA_INKLING_HIP_TREE_FP8=1`.

| Metric at 32K prompt tokens | Baseline | New verifier | Change |
| --- | ---: | ---: | ---: |
| Decode with HIP graph replay | 29.74 tok/s | 35.59 tok/s | **+19.7%** |
| Verifier attention over matched cycles | 1,583 ms | 1,130 ms | **−28.6%** |

The decode comparison used two runs per setting in baseline/new/new/baseline
order. All four runs emitted the same 128 greedy tokens. The attention timings
came from a separate profiling run series with graph replay disabled, matching
16 verification cycles by context position and batch size. These figures cover
one prompt and GPU.

### Prefill after the merge

The HIP engine now reads the whole prompt in batches without a separate final
token for an unsupported checkpoint. The quick-start settings also speed up
FP8 attention and reduce reallocations of small global KV caches.

On the same RX 9060 XT, the optimized build read the 32K prompt in **33.30 s**
and generated at **41.95 tok/s**. The pre-merge build measured 37.03 s and
34.59 tok/s. These are medians of two runs per build from separate same-day
series, with the same 128 greedy tokens. In the four-run KV growth comparison,
median prefill was 41.62 s with faster growth off and 33.30 s with it on.

In a separate 32K test on the same card, the SF4 lookup table reduced median
prefill from 43.50 s to 41.28 s. Prefill plus 128 generated tokens took 47.31 s
with the lookup off and 45.21 s with it on. Each setting ran twice with graph
replay enabled, in baseline/new/new/baseline order. All four runs emitted the
same greedy tokens.

## RX 7900 XTX support and performance evidence

This opt-in Linux `gfx1100` backend supersedes the initial support in
[PR #94](https://github.com/Niko1221/Strata/pull/94). It retains HIP runtime and
wave32 integer-dot compatibility, native mmap layout validation, and MTP, then
adds HIP MMQ, optional calibrated dense hipBLASLt GEMM, native prefill batching,
and the host-memory / SSD paths used by the measured configuration. HIP blocking
expert uploads use one reusable pinned staging buffer, including prompt-cache
refills, to avoid repeated pageable-source registrations.

### Reproduce the configuration

Build instructions are in [AMD_HIP.md](AMD_HIP.md). The numbers
below were measured on a build with llama.cpp's MMQ kernels (Strata 0.2.15 and earlier; that build option was
removed in 0.2.16). To repeat them, build the v0.2.15 tag with `-DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ=ON` and use these
runtime variables:

```sh
export STRATA_PREFILL_MMQ=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1100-hipblaslt-100100.txt"
export STRATA_PREFILL_RING=96
export STRATA_IO_THREADS=32
```

The supplied table is calibrated for gfx1100 and hipBLASLt version 100100.
It is not a universal ROCm tuning table. The runtime guards architecture,
library version and actual shape/stride/workspace requirements, falling back
when a table entry is unavailable or incompatible. HIP MMQ is opt-in at runtime. Default CUDA selection is preserved.

Measured engine configuration: Orca Flash Next IQ3_XXS, native pack plus matching
GGUF/tokenizer/template, `--mmap-experts --resident-cpu-experts`, fixed ranked
`--expert-profile`, `--expert-cache auto`, `--prefill 8192`, `--spec 4`,
`--spec-min-p 0.5`, matching MTP runtime, `--max-context 262144`, `--kv int8`,
`--kv-resident 32768`, `--pool-workers 15`, `--adapt-every 0`, `--pcie-frac 0`,
and `--vram-reserve-mib 1024`. Keep PLE on SSD.

`--resident-cpu-experts` copies the complement of the static GPU cache to ordinary
RAM. It requires mmap and a fixed expert profile. The slots the prompt path may
borrow keep their experts in RAM too as far as the budget below allows (the rest
use the mapped fallback while borrowed and for their refill), and adaptive swaps
(`--adapt-every`) exchange experts between VRAM and the RAM copy without reading
the file. The copy is pageable; `--resident-experts` (CUDA, chosen by setup)
page-locks it and uses 4 GiB of headroom. Allocation needs sufficient available RAM;
on Linux this option requires readable standard cgroup-v2 mounts. For each
finite cgroup ancestor, the guard credits only `inactive_file` after subtracting
`file_dirty` and `file_writeback`, capped by current usage; it remains bounded by
the ancestor limit and host `MemAvailable`, with 8 GiB headroom. This accounts
for reclaimable clean file cache but cannot reserve memory against concurrent
system or process allocations.
The POSIX PLE path issues direct reads through a configurable worker pool.

Use a dedicated idle server, restart it between arms, and capture its engine log:

```sh
python3 terminal/tools/hip/bench_prefill.py \
  --model MODEL_NAME --url http://127.0.0.1:8080 \
  --engine-log /path/to/engine.log --label candidate --output candidate.json
```

The script uses the same synthetic source prompts and request order as the
reported measurements: a small warmup, then 140/280/140/280 functions, with one
follow-up after each fresh prompt. It enforces zero reused tokens for the fresh
prompts and records prefill, decode, wall time and stop reason. It requires the
unbuffered log to contain exactly one completed timing record per request. The
128-token output cap is intentional for throughput measurement, not task success.
Never use cancelled-request timing lines as throughput evidence.

### Final revision benchmark

Measured on 2026-09-29, source revision `9568e78da663e6b224ab96d57baf450717b36e73`.
HIP executable SHA-256: `705f5925a788bc227bad72abfb5466032e0879d752fea9afc3eb32051b35182b`.
The control and candidate below were both measured anew in this session; no
historical benchmark values are mixed into the table. The control binary SHA-256
is `4522d4937ca4a41ca294d31b60df8c2bc4df5b7a6caf65eaad10c036e08491f7`;
its source was an existing local runtime snapshot rather than a clean upstream commit.
Both arms use an RX 7900 XTX 24 GiB / gfx1100, Ryzen 9 7950X3D,
64 GiB installed RAM, Fedora-family Linux, ROCm 7.1 and a 272 W GPU cap.
Other model services are stopped for measurement and restored afterward.
Sampling: temperature 0, top-k 1, top-p 1, min-p 0, seed 42, reasoning disabled.

| Fresh request, execution order | Control prefill t/s | Candidate prefill t/s | Candidate output t/s | Candidate request wall time |
| --- | ---: | ---: | ---: | ---: |
| 4,210 tokens, first use | 240.0 | 447.5 | 55.5 | 11.729 s |
| 8,830 tokens, first use at this size | 484.8 | 873.7 | 55.7 | 12.420 s |
| 4,210 tokens, warmed | 457.2 | 901.1 | 56.7 | 6.937 s |
| 8,830 tokens, warmed | 478.9 | 966.5 | 59.2 | 11.314 s |

Each request generated exactly 128 tokens and finished at the intentional length
cap. All four fresh prompts had zero reused KV tokens. Fresh-prompt prefill was
1.80–2.02x the freshly measured existing AMD runtime; fresh-request wall time was
40–46% lower. This is a comparison with our existing HIP runtime, not unmodified
upstream, which does not provide this backend. The candidate also completed four
cached follow-ups in 2.95–3.26 seconds. Those follow-ups are not apples-to-apples
prefill comparisons: cache reuse differs with generated text and checkpoint
selection. Full sanitized measurements are in
[the fresh-run JSON](benchmarks/2026-09-29-gfx1100.json).

Before this revision, an intermediate package missing the HIP upload staging
buffer stalled on the 8,830-token request and was rejected. Restoring staging
allowed the complete sequence above to finish. That failed attempt is not
included in the throughput table. No watchdog limit was raised to obtain these
results.

Zero KV reuse is not equivalent to a cold filesystem cache. First-use speed
and warmed speed are reported separately. Individual observations do not
establish confidence intervals or a general rate at every context length.

### Correctness and quality limits

Both HIP and CUDA executables built from the measured source revision. CUDA was
compile-checked, not performance-tested by this run. The selected HIP CTest suite
passed **29/29** with the tuning table enabled, including expert-upload readback,
asynchronous handoff, QSA, MMQ, Lt GEMM, KV streaming and PLE reading. The excluded
`ple_parity` requires an external fixture; `platform_memory_test` requires a
larger locked-memory limit than the test account provides. Separately, three
real IQ3_XXS PLE matrix graph replays passed, and the POSIX direct-file test passed
queued reads, short EOF, wake, close/drain and reopen checks.

Development checks passed eight MMQ numerical comparisons, four actual Lt GEMM
comparisons, and native QSA/indexer/embedding parity including tail states,
chunk continuation and image overrides. Real quantized MMQ relative L2 error
was approximately 0.0026 against raw-FP32/dequantized-weight reference, reflecting
Q8 activation arithmetic; this is not bitwise numerical equivalence.

End-to-end coding quality is not established by these numerical checks or capped
throughput requests. Validate completed tasks with independent runtime tests; this
contribution makes no broad answer-quality or agentic-reliability claim.

The changes do not claim better model reasoning, verified full-context behavior,
end-to-end vision validation, Windows HIP, other AMD architectures, or mixed
AMD/NVIDIA execution. Existing CUDA multi-GPU code remains present but is not
validation of HIP multi-GPU support.

### Attribution and rejected experiments

The measurements above were taken before this backend was rebased onto engine 0.1.24. The PR's own batched
embedding gather and QSA indexer append (adapted from
[PR #108](https://github.com/Niko1221/Strata/pull/108), commit
`acd487233c0bbe2217a6881c5bb43f8a283b0de5`) were dropped in the rebase: 0.1.24 already does both on every
backend, bit-exact (C-2, C-4). #108's optional GDN parallel/split path is not included. Existing upstream MMQ orchestration is retained and enabled for
HIP with AMD architecture identification and the correct backend compilation.

Static 16K chunks, expanded FP16 expert tuning, and a 192-slot ring did not offer
a consistent end-to-end win in the evaluated workload. The table includes only
the selected 30 dense-shape rows. Microkernel speedups alone were not sufficient
to select a configuration. No increase in GPU power cap was used.
