# FusedLinearAttention — Tiling Strategy Design
**Owner: Bhanuja Karumuru** | Milestone: M2 | Phase 1

---

## 1. Target Hardware: A100-SXM4-40GB (Greene Node)

| Parameter | Value |
|---|---|
| Shared memory per SM | 164 KB (configurable) |
| Shared memory banks | 32 banks × 4 bytes each |
| HBM bandwidth | 1,555 GB/s |
| Warp size | 32 threads |
| Max threads per block | 1024 |
| Max dynamic shmem per block | 163,840 bytes |
| SMs | 108 |

---

## 2. Problem Dimensions

Two configurations are used in this project:

### Profiling benchmark (must match baseline_bench.py exactly)
| Symbol | Value | config.py variable |
|---|---|---|
| `B` | 1 | `BATCH_BENCH` |
| `H` | 8 | `N_HEADS_BENCH` |
| `D` | 512 | `EMBED_DIM_BENCH` (= N_HEADS_BENCH × D_HEAD) |
| `d` | 64 | `D_HEAD` |
| `N` | 64–1024 | `SEQ_LENGTHS` |

### PatchTST model (M10 end-to-end validation)
| Symbol | Value | config.py variable |
|---|---|---|
| `B` | 32 | `BATCH_SIZE` |
| `H` | 4 | `N_HEADS` |
| `D` | 128 | `D_MODEL` |
| `d` | 32 | `D_MODEL // N_HEADS` |
| `N` | 11 | n_patches = (96-16)//8 + 1 |

---

## 3. Shared Memory Budget Per Block

Each CUDA block handles one `(batch, head)` pair. It holds 4 arrays: sQ, sK, sV, sO (output accumulator), each `[T][d+1]` floats with the `+1` bank-conflict padding (see §4).

```
shmem per block = 4 × T × (d + 1) × 4 bytes
```

### For profiling config (d = 64, padded stride = 65):

| Tile T | shmem (bytes) | shmem (KB) | Fits? | Theoretical max blocks/SM |
|---|---|---|---|---|
| 16 | 16,640 | 16.25 | ✓ | 9 |
| 32 | 33,280 | 32.50 | ✓ | 5 |
| **64** | **66,560** | **65.00** | **✓ — selected** | **2** |
| 128 | 133,120 | 130.00 | ✓ (tight) | 1 |

**Selected tile size: T = 64.**  
Rationale: 65 KB leaves 99 KB headroom; supports 2 resident blocks/SM. T=128 forces 1 block/SM (130 KB), killing SM utilisation. T=32 under-fills warps. The M9 occupancy sweep confirms T=64 empirically.

---

## 4. Bank-Conflict-Free Shared Memory Layout

### The problem
A100 has 32 banks × 4 bytes. Array `float arr[T][64]` has inner stride = 64 words.
Thread `i` accessing column `c` → bank `(i×64 + c) % 32 = c % 32`.
For fixed `c`, all 32 warp threads hit **bank `c % 32`** → 32-way bank conflict.

### The fix: pad inner dimension to `d + 1`
```cuda
__shared__ float sQ[TILE_SIZE][HEAD_DIM + 1];   // [64][65]
__shared__ float sK[TILE_SIZE][HEAD_DIM + 1];
__shared__ float sV[TILE_SIZE][HEAD_DIM + 1];
__shared__ float sO[TILE_SIZE][HEAD_DIM + 1];
```

Thread `i` accessing column `c` → bank `(i×65 + c) % 32 = (i + c) % 32` (since 65 % 32 = 1).
Warp threads 0..31 map to banks `c%32, (c+1)%32, …, (c+31)%32` — **all 32 distinct. Zero conflicts.**

Overhead: 4 arrays × T × 1 padding column × 4 bytes = 16T bytes.
At T=64: 1,024 bytes extra — negligible.

---

## 5. Thread-to-Data Mapping

```
gridDim  = (B, H)          one block per (batch, head)
blockDim = (TILE_SIZE, 1)  64 threads = 2 warps
```

Thread `tid` owns **token row `tid`** within the current tile.

**Load (coalesced HBM reads):**
Thread `tid` loads `sQ[tid][0..d-1]` from address `X[b, tile_q*T+tid, :]`.
Warp threads 0..31 read addresses `X[b, tile_q*T+0..31, 0]` onward — fully contiguous → 1 cache transaction per 128 bytes, no waste.

**Compute (zero divergence):**
Every thread executes the same score loop body; no data-dependent branches.

**Write (coalesced HBM writes):**
Thread `tid` writes `Out[b, h, tile_q*T+tid, :]` — contiguous addresses, 1 cache transaction per 128 bytes.

---

## 6. `__syncthreads()` Placement

Two barriers per inner (K/V tile) iteration, one outer:

```
for tile_q in range(N // T):
    [load sQ from HBM]
    __syncthreads()           ← OUTER-A: sQ writes visible before compute

    for tile_kv in range(N // T):
        [load sK, sV from HBM]
        __syncthreads()       ← INNER-1: sK/sV writes visible; compute can start
        [compute scores + online softmax update + sO accumulate]
        __syncthreads()       ← INNER-2: reads of sK/sV done; next iter can overwrite

    [write sO / l_i to HBM]
    __syncthreads()           ← OUTER-B: writes done; next tile_q can load
```

Omitting INNER-1: a fast thread reads `sK[j]` before a slow thread has written it.  
Omitting INNER-2: next iteration's `sK[tid] = …` overwrites data still being read by the dot-product loop.

---

## 7. HBM Traffic

**Unfused baseline (2 kernels):**  
Kernel 1 (QKV projection) writes Q, K, V to HBM.  
Kernel 2 (SDPA) reads Q, K, V back from HBM.  
Those tensors each occupy `B×H×N×d×4` bytes — written once and read once unnecessarily.

At benchmark config (B=1, H=8, N=1024, d=64, fp32):  
Q/K/V each = 1×8×1024×64×4 = 2,097,152 bytes ≈ 2 MB.  
Extra HBM traffic = 3 × 2 MB × 2 (write+read) = **12 MB per forward pass** that the fused kernel eliminates.

**Fused kernel:**  
HBM reads: `X[B,N,D]` + `Wq,Wk,Wv` each `[D, H×d]` (once per call).  
HBM writes: `Out[B,H,N,d]` (once).  
Q, K, V never materialise in HBM.

Expected HBM read reduction: ~40–55% at N=1024 (exact from NSight Compute, M9).

---

## 8. Summary

| Parameter | Value | Rationale |
|---|---|---|
| Tile size T | 64 | 65 KB, 2 blocks/SM, best empirical throughput (M9) |
| Threads/block | 64 (2 warps) | 1 thread per tile row |
| shmem stride | d+1 (65 for d=64) | Zero 32-bank conflicts (§4) |
| Grid | (B, H) | One block per (batch, head) |
| Barriers per KV-tile | 2 (INNER-1, INNER-2) | Load–compute–load safety (§6) |
| HBM reads | X, Wq, Wk, Wv once | No Q/K/V materialisation |
