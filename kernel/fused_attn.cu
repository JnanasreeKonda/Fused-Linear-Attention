/*
 * kernel/fused_attn.cu — canonical fused attention CUDA kernel.
 *
 * This is the consolidated version of the implementation that previously lived
 * in the draft bundle directory. The current kernel targets the benchmark
 * configuration TILE_SIZE=64 and HEAD_DIM=64 on A100.
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

#define SHMEM_STRIDE (HEAD_DIM + 1)

__global__ void fused_qkv_attention_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Wq,
    const float* __restrict__ Wk,
    const float* __restrict__ Wv,
    float* __restrict__ Out,
    int B, int H, int N, int D, int d_head
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;

    const float scale = rsqrtf(static_cast<float>(d_head));
    const int head_col_start = h * d_head;

    const float* x_b = X + static_cast<long long>(b) * N * D;
    float* out_bh = Out + (static_cast<long long>(b) * H + h) * N * d_head;

    __shared__ float sQ[TILE_SIZE][SHMEM_STRIDE];
    __shared__ float sK[TILE_SIZE][SHMEM_STRIDE];
    __shared__ float sV[TILE_SIZE][SHMEM_STRIDE];

    for (int tile_q = 0; tile_q * TILE_SIZE < N; ++tile_q) {
        const int q_global = tile_q * TILE_SIZE + tid;

        if (q_global < N) {
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c) {
                float acc = 0.0f;
                for (int k = 0; k < D; ++k) {
                    acc += x_b[static_cast<long long>(q_global) * D + k]
                         * Wq[static_cast<long long>(k) * H * d_head + head_col_start + c];
                }
                sQ[tid][c] = acc;
            }
        } else {
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c) {
                sQ[tid][c] = 0.0f;
            }
        }

        float o_acc[HEAD_DIM];
        float m_i = -FLT_MAX;
        float l_i = 0.0f;
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c) {
            o_acc[c] = 0.0f;
        }

        __syncthreads();

        for (int tile_kv = 0; tile_kv * TILE_SIZE < N; ++tile_kv) {
            const int kv_global = tile_kv * TILE_SIZE + tid;

            if (kv_global < N) {
                #pragma unroll
                for (int c = 0; c < HEAD_DIM; ++c) {
                    float accK = 0.0f;
                    float accV = 0.0f;
                    for (int k = 0; k < D; ++k) {
                        const float xval = x_b[static_cast<long long>(kv_global) * D + k];
                        accK += xval * Wk[static_cast<long long>(k) * H * d_head + head_col_start + c];
                        accV += xval * Wv[static_cast<long long>(k) * H * d_head + head_col_start + c];
                    }
                    sK[tid][c] = accK;
                    sV[tid][c] = accV;
                }
            } else {
                #pragma unroll
                for (int c = 0; c < HEAD_DIM; ++c) {
                    sK[tid][c] = 0.0f;
                    sV[tid][c] = 0.0f;
                }
            }

            __syncthreads();

            float m_tile = -FLT_MAX;
            float scores[TILE_SIZE];

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
                if (scores[j] > m_tile) {
                    m_tile = scores[j];
                }
            }

            const float m_new = fmaxf(m_i, m_tile);
            const float corr_old = expf(m_i - m_new);
            float l_tile = 0.0f;

            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c) {
                o_acc[c] *= corr_old;
            }

            for (int j = 0; j < TILE_SIZE; ++j) {
                const int j_global = tile_kv * TILE_SIZE + j;
                if (j_global >= N) {
                    continue;
                }

                const float e = expf(scores[j] - m_new);
                l_tile += e;
                #pragma unroll
                for (int c = 0; c < HEAD_DIM; ++c) {
                    o_acc[c] += e * sV[j][c];
                }
            }

            m_i = m_new;
            l_i = l_i * corr_old + l_tile;

            __syncthreads();
        }

        if (q_global < N && l_i > 0.0f) {
            const float inv_l = 1.0f / l_i;
            #pragma unroll
            for (int c = 0; c < HEAD_DIM; ++c) {
                out_bh[static_cast<long long>(q_global) * d_head + c] = o_acc[c] * inv_l;
            }
        }

        __syncthreads();
    }
}

extern "C" void launch_fused_attention(
    const float* X,
    const float* Wq,
    const float* Wk,
    const float* Wv,
    float* Out,
    int B, int H, int N, int D, int d_head,
    cudaStream_t stream
) {
    dim3 grid(B, H);
    dim3 block(TILE_SIZE);

    fused_qkv_attention_kernel<<<grid, block, 0, stream>>>(
        X, Wq, Wk, Wv, Out, B, H, N, D, d_head
    );
}
