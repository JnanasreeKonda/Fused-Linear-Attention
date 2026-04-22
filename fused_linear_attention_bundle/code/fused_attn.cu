/*
 * kernel/fused_attn.cu  —  v3  (M7b: Hardware Efficiency, Bhanuja Karumuru)
 *
 * Builds on v2 (Jnanasree, M7a: online softmax tiling loop).
 * Every M7b-specific addition is marked with   // [M7b]
 *
 * M7b changes vs v2:
 *   1. Shared memory padded to [TILE_SIZE][HEAD_DIM+1] → zero 32-bank conflicts
 *   2. __syncthreads() audit — 2 barriers per KV-tile, 1 per Q-tile boundary
 *   3. Coalesced HBM loads — thread tid always owns token (tile*T + tid)
 *   4. Coalesced HBM writes — same striding for output
 *   5. cudaFuncSetAttribute for max dynamic shmem (needed on A100 for large tiles)
 *
 * Kernel interface
 * ----------------
 * Input  X   : [B, N, D]          — input sequence (fp32)
 * Input  Wq  : [D, H*d]           — Q projection weights (all heads stacked)
 * Input  Wk  : [D, H*d]           — K projection weights
 * Input  Wv  : [D, H*d]           — V projection weights
 * Output Out : [B, H, N, d]       — attention output (fp32)
 *
 * Grid  : dim3(B, H)              — one block per (batch, head)
 * Block : dim3(TILE_SIZE)         — one thread per row in the tile
 *
 * Compile standalone smoke-test:
 *   nvcc -O3 -arch=sm_80 -DSTANDALONE_TEST -o test_kernel kernel/fused_attn.cu
 *   ./test_kernel
 *
 * Loaded via PyTorch cpp_extension in kernel/load_kernel.py (Jnanasree M8).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Compile-time constants (sweep these in M9 occupancy experiments)           */
/* ─────────────────────────────────────────────────────────────────────────── */

#ifndef TILE_SIZE
#define TILE_SIZE 64      /* sequence tile T — change to 16/32/128 for sweep  */
#endif

#ifndef HEAD_DIM
#define HEAD_DIM  64      /* d_head — must match runtime d_head argument       */
#endif

/*
 * [M7b] SHMEM_STRIDE = HEAD_DIM + 1 = 65
 *
 * A100 has 32 shared-memory banks (4 bytes each).
 * Array [T][HEAD_DIM] has stride 64 — a multiple of 32 → 32-way bank conflict
 * when 32 threads access the same column index.
 *
 * With stride 65 (not a multiple of 32):
 *   thread i accesses bank (i * 65 + c) % 32 = (i + c) % 32
 * All 32 warp threads access 32 distinct banks → zero conflicts.
 */
#define SHMEM_STRIDE (HEAD_DIM + 1)   /* 65 */

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Fused QKV + Online-Softmax Attention Kernel (v3)                           */
/* ─────────────────────────────────────────────────────────────────────────── */

__global__ void fused_qkv_attention_kernel(
    const float* __restrict__ X,     /* [B, N, D]         */
    const float* __restrict__ Wq,    /* [D, H*d]          */
    const float* __restrict__ Wk,    /* [D, H*d]          */
    const float* __restrict__ Wv,    /* [D, H*d]          */
    float*       __restrict__ Out,   /* [B, H, N, d]      */
    int B, int H, int N, int D, int d_head
) {
    /* One block per (batch, head) via 2D grid */
    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;   /* 0 .. TILE_SIZE-1 */

    const float scale = rsqrtf((float)d_head);   /* 1/sqrt(d) */

    /*
     * Weight column offsets for this head.
     * Wq layout: [D, H*d] stored column-major as D rows, H*d columns.
     * For head h, column range is [h*d_head .. (h+1)*d_head - 1].
     */
    const int head_col_start = h * d_head;   /* column offset into Wq/Wk/Wv */

    /* Pointers to this batch's X and this (batch,head)'s output */
    const float* x_b   = X   + (long long)b * N * D;
    float*       out_bh = Out + ((long long)b * H + h) * N * d_head;

    /* ──────────────────────────────────────────────────────────────────────
     * [M7b] Shared memory with +1 padding per row to eliminate bank conflicts.
     *       Each array: [TILE_SIZE][SHMEM_STRIDE] = [64][65]
     * ────────────────────────────────────────────────────────────────────── */
    __shared__ float sQ[TILE_SIZE][SHMEM_STRIDE];
    __shared__ float sK[TILE_SIZE][SHMEM_STRIDE];
    __shared__ float sV[TILE_SIZE][SHMEM_STRIDE];

    /* ──────────────────────────────────────────────────────────────────────
     * Outer loop: slide over query tiles (each block handles all N tokens)
     * ────────────────────────────────────────────────────────────────────── */
    for (int tile_q = 0; tile_q * TILE_SIZE < N; ++tile_q) {

        const int q_global = tile_q * TILE_SIZE + tid;   /* global token index */

        /* ── Load Q tile from HBM into shmem ──────────────────────────────
         * [M7b] Coalesced: thread tid reads token (tile_q*T + tid).
         *       Warp threads 0..31 → consecutive token rows → 1 HBM transaction
         *       per 128 bytes, no bandwidth waste.
         * ─────────────────────────────────────────────────────────────────── */
        if (q_global < N) {
            /* Q[q_global] = x_b[q_global, :] @ Wq[:, head_col_start:head_col_start+d_head] */
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c) {
                float acc = 0.0f;
                /* Inner product over D: each float op uses a register-resident accumulator */
                for (int k = 0; k < D; ++k) {
                    acc += x_b[(long long)q_global * D + k]
                         * Wq[(long long)k * H * d_head + head_col_start + c];
                }
                sQ[tid][c] = acc;   /* [M7b] row-major write — coalesced across block */
            }
        } else {
            /* Pad with zero for out-of-bounds tokens */
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c) sQ[tid][c] = 0.0f;
        }

        /* Online softmax state (in registers — not shared memory) */
        float o_acc[HEAD_DIM];
        float m_i = -FLT_MAX;    /* running max */
        float l_i = 0.0f;        /* running sum(exp) denominator */
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c) o_acc[c] = 0.0f;

        /* [M7b] OUTER BARRIER-A:
         *   sQ is fully written; any thread can now read sQ safely. */
        __syncthreads();

        /* ── Inner loop: slide over K/V tiles (Jnanasree M7a online softmax) ──
         *    Each iteration loads a tile of K and V, updates running
         *    max/denom/output using the numerically stable online softmax.
         * ─────────────────────────────────────────────────────────────────── */
        for (int tile_kv = 0; tile_kv * TILE_SIZE < N; ++tile_kv) {

            const int kv_global = tile_kv * TILE_SIZE + tid;

            /* ── Load K and V tiles ────────────────────────────────────────
             * [M7b] Coalesced for the same reason as sQ load above.
             *       Fusing the K and V loads into one pass over x_b cuts
             *       the number of HBM reads of X in half for this tile.
             * ─────────────────────────────────────────────────────────────── */
            if (kv_global < N) {
                #pragma unroll
                for (int c = 0; c < HEAD_DIM; ++c) {
                    float accK = 0.0f, accV = 0.0f;
                    for (int k = 0; k < D; ++k) {
                        const float xval = x_b[(long long)kv_global * D + k];
                        accK += xval * Wk[(long long)k * H * d_head + head_col_start + c];
                        accV += xval * Wv[(long long)k * H * d_head + head_col_start + c];
                    }
                    sK[tid][c] = accK;
                    sV[tid][c] = accV;
                }
            } else {
                #pragma unroll
                for (int c = 0; c < HEAD_DIM; ++c) { sK[tid][c] = 0.0f; sV[tid][c] = 0.0f; }
            }

            /* [M7b] INNER BARRIER-1:
             *   All sK and sV writes are now visible.
             *   No thread will read sK[j] / sV[j] before this point. */
            __syncthreads();

            /* ── Online softmax update (Jnanasree M7a algorithm) ────────────
             *
             * For query thread tid, compute:
             *   scores[j] = scale * dot(sQ[tid], sK[j])   for j in 0..T-1
             * Then update running max m_i, denominator l_i, and output o_acc.
             *
             * This is the Flash-Attention-2 tile-level update:
             *   m_new = max(m_old, max_j scores[j])
             *   l_new = l_old * exp(m_old - m_new) + sum_j exp(scores[j] - m_new)
             *   o_acc = o_acc * exp(m_old - m_new) + sum_j exp(scores[j]-m_new)*sV[j]
             * ─────────────────────────────────────────────────────────────── */
            float m_tile = -FLT_MAX;
            float scores[TILE_SIZE];

            /* Pass 1: compute all scores for this tile, find tile-local max */
            for (int j = 0; j < TILE_SIZE; ++j) {
                const int j_global = tile_kv * TILE_SIZE + j;
                if (j_global >= N) {
                    scores[j] = -FLT_MAX;
                    continue;
                }
                float dot = 0.0f;
                #pragma unroll
                for (int c = 0; c < HEAD_DIM; ++c) {
                    dot += sQ[tid][c] * sK[j][c];
                }
                scores[j] = dot * scale;
                if (scores[j] > m_tile) m_tile = scores[j];
            }

            /* Pass 2: update running state (Jnanasree M7a correction) */
            const float m_new    = fmaxf(m_i, m_tile);
            const float corr_old = expf(m_i - m_new);     /* rescale old acc */
            float       l_tile   = 0.0f;

            /* Rescale existing accumulator for new max */
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c) o_acc[c] *= corr_old;

            /* Accumulate new tile's contribution */
            for (int j = 0; j < TILE_SIZE; ++j) {
                const int j_global = tile_kv * TILE_SIZE + j;
                if (j_global >= N) continue;
                const float e = expf(scores[j] - m_new);
                l_tile += e;
                #pragma unroll
                for (int c = 0; c < HEAD_DIM; ++c) {
                    o_acc[c] += e * sV[j][c];
                }
            }

            m_i  = m_new;
            l_i  = l_i * corr_old + l_tile;

            /* [M7b] INNER BARRIER-2:
             *   All threads finished reading sK/sV for this tile.
             *   Safe for the next iteration to overwrite sK/sV. */
            __syncthreads();

        } /* end inner KV-tile loop */

        /* ── Write normalised output to HBM ──────────────────────────────────
         * [M7b] Coalesced: thread tid writes to Out[b, h, q_global, :].
         *       q_global = tile_q*T + tid, so consecutive tids write
         *       consecutive addresses.
         * ─────────────────────────────────────────────────────────────────── */
        if (q_global < N && l_i > 0.0f) {
            const float inv_l = 1.0f / l_i;
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c) {
                out_bh[(long long)q_global * d_head + c] = o_acc[c] * inv_l;
            }
        }

        /* [M7b] OUTER BARRIER-B:
         *   Output writes done. Next tile_q iteration can safely reload sQ. */
        __syncthreads();

    } /* end outer Q-tile loop */
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Host launcher — called from fused_attn_ext.cpp (Jnanasree M8 binding)      */
/* ─────────────────────────────────────────────────────────────────────────── */

extern "C" void launch_fused_attention(
    const float* X,
    const float* Wq,
    const float* Wk,
    const float* Wv,
    float*       Out,
    int B, int H, int N, int D, int d_head,
    cudaStream_t stream
) {
    /* [M7b] Grid and block dimensions from DESIGN.md §5
     *   grid  = (B, H)       — one block per (batch, head)
     *   block = (TILE_SIZE,) — 64 threads = 2 warps
     *
     * Shared memory: 4 × TILE_SIZE × SHMEM_STRIDE × 4 bytes = 66,560 bytes at T=64
     * Request extended shmem on A100 (default cap is 48 KB; A100 supports 164 KB). */
    dim3 grid(B, H);
    dim3 block(TILE_SIZE);

    const size_t shmem = 3UL * TILE_SIZE * SHMEM_STRIDE * sizeof(float);   /* sQ+sK+sV */

    cudaFuncSetAttribute(
        fused_qkv_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem
    );

    fused_qkv_attention_kernel<<<grid, block, 0, stream>>>(
        X, Wq, Wk, Wv, Out, B, H, N, D, d_head
    );
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Standalone smoke-test                                                       */
/*  nvcc -O3 -arch=sm_80 -DSTANDALONE_TEST -o test_kernel kernel/fused_attn.cu */
/*  ./test_kernel                                                                */
/* ─────────────────────────────────────────────────────────────────────────── */
#ifdef STANDALONE_TEST
#include <stdlib.h>

static void rand_fill(float* p, long n, float scale) {
    for (long i = 0; i < n; ++i)
        p[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
}

static int run_config(int B, int H, int N, int D, int d) {
    long xsz  = (long)B * N * D;
    long wsz  = (long)D * H * d;
    long osz  = (long)B * H * N * d;

    float *hX  = (float*)malloc(xsz*4), *hWq = (float*)malloc(wsz*4);
    float *hWk = (float*)malloc(wsz*4), *hWv = (float*)malloc(wsz*4);
    float *hOut = (float*)malloc(osz*4);

    rand_fill(hX, xsz, 1.0f); rand_fill(hWq, wsz, 0.02f);
    rand_fill(hWk, wsz, 0.02f); rand_fill(hWv, wsz, 0.02f);

    float *dX, *dWq, *dWk, *dWv, *dOut;
    cudaMalloc(&dX,   xsz*4); cudaMalloc(&dWq, wsz*4);
    cudaMalloc(&dWk,  wsz*4); cudaMalloc(&dWv, wsz*4);
    cudaMalloc(&dOut, osz*4);
    cudaMemcpy(dX,  hX,  xsz*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dWq, hWq, wsz*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dWk, hWk, wsz*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dWv, hWv, wsz*4, cudaMemcpyHostToDevice);

    launch_fused_attention(dX, dWq, dWk, dWv, dOut, B, H, N, D, d, 0);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR B=%d H=%d N=%d D=%d d=%d: %s\n",
               B, H, N, D, d, cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hOut, dOut, osz*4, cudaMemcpyDeviceToHost);
    for (long i = 0; i < osz; ++i) {
        if (!isfinite(hOut[i])) {
            printf("FAIL: non-finite at %ld  B=%d H=%d N=%d D=%d d=%d\n",i,B,H,N,D,d);
            return 1;
        }
    }
    printf("PASS  B=%d H=%d N=%3d D=%d d=%d\n", B, H, N, D, d);

    free(hX); free(hWq); free(hWk); free(hWv); free(hOut);
    cudaFree(dX); cudaFree(dWq); cudaFree(dWk); cudaFree(dWv); cudaFree(dOut);
    return 0;
}

int main() {
    /* Profiling benchmark config */
    if (run_config(1, 8,   64, 512, 64)) return 1;
    if (run_config(1, 8,  128, 512, 64)) return 1;
    if (run_config(1, 8,  256, 512, 64)) return 1;
    if (run_config(1, 8,  512, 512, 64)) return 1;
    if (run_config(1, 8, 1024, 512, 64)) return 1;
    /* Model config (d_head=32) */
    if (run_config(4, 4,   11, 128, 32)) return 1;
    printf("\nAll standalone tests PASSED.\n");
    return 0;
}
#endif /* STANDALONE_TEST */
