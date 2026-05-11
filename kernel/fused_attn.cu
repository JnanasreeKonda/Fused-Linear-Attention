/*
 * kernel/fused_attn.cu — canonical fused attention CUDA kernel.
 *
 * This kernel now supports a mixed-precision path where X/Wq/Wk/Wv are stored
 * in fp16 while accumulators remain fp32. The math structure is still the same
 * fused tiled attention implementation, but the lower-precision storage path
 * cuts memory traffic and enables a modestly more vectorized projection loop.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

#define SHMEM_STRIDE (HEAD_DIM + 1)
#ifndef PROJ_K_TILE
#define PROJ_K_TILE 8
#endif
#define TILE_OFFSET(row, col) ((row) * SHMEM_STRIDE + (col))
#define X_TILE_OFFSET(row, col) ((row) * PROJ_K_TILE + (col))
#define W_TILE_OFFSET(row, col) ((row) * HEAD_DIM + (col))

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t value);

template <>
__device__ inline float scalar_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float scalar_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename scalar_t>
__device__ inline scalar_t scalar_zero();

template <>
__device__ inline float scalar_zero<float>() {
    return 0.0f;
}

template <>
__device__ inline __half scalar_zero<__half>() {
    return __float2half(0.0f);
}

template <>
__device__ inline __nv_bfloat16 scalar_zero<__nv_bfloat16>() {
    return __float2bfloat16(0.0f);
}

template <typename scalar_t>
__device__ inline void accumulate_proj_row(
    float* __restrict__ dst,
    const scalar_t* __restrict__ x_row,
    const scalar_t* __restrict__ w_tile
) {
    #pragma unroll
    for (int kk = 0; kk < PROJ_K_TILE; ++kk) {
        const float xval = scalar_to_float(x_row[kk]);
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c) {
            dst[c] += xval * scalar_to_float(w_tile[W_TILE_OFFSET(kk, c)]);
        }
    }
}

template <>
__device__ inline void accumulate_proj_row<__half>(
    float* __restrict__ dst,
    const __half* __restrict__ x_row,
    const __half* __restrict__ w_tile
) {
    #pragma unroll
    for (int kk = 0; kk < PROJ_K_TILE; ++kk) {
        const float xval = __half2float(x_row[kk]);
        const __half* w_row = w_tile + W_TILE_OFFSET(kk, 0);
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 2) {
            const __half2 packed = *reinterpret_cast<const __half2*>(w_row + c);
            const float2 vals = __half22float2(packed);
            dst[c] += xval * vals.x;
            dst[c + 1] += xval * vals.y;
        }
    }
}

template <>
__device__ inline void accumulate_proj_row<__nv_bfloat16>(
    float* __restrict__ dst,
    const __nv_bfloat16* __restrict__ x_row,
    const __nv_bfloat16* __restrict__ w_tile
) {
    #pragma unroll
    for (int kk = 0; kk < PROJ_K_TILE; ++kk) {
        const float xval = __bfloat162float(x_row[kk]);
        const __nv_bfloat16* w_row = w_tile + W_TILE_OFFSET(kk, 0);
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 2) {
            const float2 vals = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(w_row + c)
            );
            dst[c] += xval * vals.x;
            dst[c + 1] += xval * vals.y;
        }
    }
}

template <typename scalar_t>
__device__ inline void accumulate_two_proj_rows(
    float* __restrict__ dst0,
    float* __restrict__ dst1,
    const scalar_t* __restrict__ x_row,
    const scalar_t* __restrict__ w0_tile,
    const scalar_t* __restrict__ w1_tile
) {
    #pragma unroll
    for (int kk = 0; kk < PROJ_K_TILE; ++kk) {
        const float xval = scalar_to_float(x_row[kk]);
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c) {
            dst0[c] += xval * scalar_to_float(w0_tile[W_TILE_OFFSET(kk, c)]);
            dst1[c] += xval * scalar_to_float(w1_tile[W_TILE_OFFSET(kk, c)]);
        }
    }
}

template <>
__device__ inline void accumulate_two_proj_rows<__half>(
    float* __restrict__ dst0,
    float* __restrict__ dst1,
    const __half* __restrict__ x_row,
    const __half* __restrict__ w0_tile,
    const __half* __restrict__ w1_tile
) {
    #pragma unroll
    for (int kk = 0; kk < PROJ_K_TILE; ++kk) {
        const float xval = __half2float(x_row[kk]);
        const __half* w0_row = w0_tile + W_TILE_OFFSET(kk, 0);
        const __half* w1_row = w1_tile + W_TILE_OFFSET(kk, 0);
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 2) {
            const float2 v0 = __half22float2(*reinterpret_cast<const __half2*>(w0_row + c));
            const float2 v1 = __half22float2(*reinterpret_cast<const __half2*>(w1_row + c));
            dst0[c] += xval * v0.x;
            dst0[c + 1] += xval * v0.y;
            dst1[c] += xval * v1.x;
            dst1[c + 1] += xval * v1.y;
        }
    }
}

template <>
__device__ inline void accumulate_two_proj_rows<__nv_bfloat16>(
    float* __restrict__ dst0,
    float* __restrict__ dst1,
    const __nv_bfloat16* __restrict__ x_row,
    const __nv_bfloat16* __restrict__ w0_tile,
    const __nv_bfloat16* __restrict__ w1_tile
) {
    #pragma unroll
    for (int kk = 0; kk < PROJ_K_TILE; ++kk) {
        const float xval = __bfloat162float(x_row[kk]);
        const __nv_bfloat16* w0_row = w0_tile + W_TILE_OFFSET(kk, 0);
        const __nv_bfloat16* w1_row = w1_tile + W_TILE_OFFSET(kk, 0);
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; c += 2) {
            const float2 v0 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(w0_row + c)
            );
            const float2 v1 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(w1_row + c)
            );
            dst0[c] += xval * v0.x;
            dst0[c + 1] += xval * v0.y;
            dst1[c] += xval * v1.x;
            dst1[c + 1] += xval * v1.y;
        }
    }
}

template <typename scalar_t>
__launch_bounds__(TILE_SIZE, 2)
__global__ void fused_qkv_attention_kernel(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ Wq,
    const scalar_t* __restrict__ Wk,
    const scalar_t* __restrict__ Wv,
    float* __restrict__ Out,
    int B, int H, int N, int D, int d_head
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tile_q = blockIdx.z;
    const int tid = threadIdx.x;

    const float scale = rsqrtf(static_cast<float>(d_head));
    const int head_col_start = h * d_head;

    const scalar_t* x_b = X + static_cast<long long>(b) * N * D;
    float* out_bh = Out + (static_cast<long long>(b) * H + h) * N * d_head;

    extern __shared__ unsigned char shared_mem[];
    float* sK = reinterpret_cast<float*>(shared_mem);
    float* sV = sK + TILE_SIZE * SHMEM_STRIDE;
    scalar_t* sX = reinterpret_cast<scalar_t*>(sV + TILE_SIZE * SHMEM_STRIDE);
    scalar_t* sW0 = sX + TILE_SIZE * PROJ_K_TILE;
    scalar_t* sW1 = sW0 + PROJ_K_TILE * HEAD_DIM;

    const int q_global = tile_q * TILE_SIZE + tid;
    float q_acc[HEAD_DIM];

    float o_acc[HEAD_DIM];
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM; ++c) {
        q_acc[c] = 0.0f;
        o_acc[c] = 0.0f;
    }

    __syncthreads();

    for (int k_base = 0; k_base < D; k_base += PROJ_K_TILE) {
        #pragma unroll
        for (int kk = 0; kk < PROJ_K_TILE; ++kk) {
            const int gk = k_base + kk;
            sX[X_TILE_OFFSET(tid, kk)] = (q_global < N && gk < D)
                ? x_b[static_cast<long long>(q_global) * D + gk]
                : scalar_zero<scalar_t>();
        }

        for (int idx = tid; idx < PROJ_K_TILE * HEAD_DIM; idx += TILE_SIZE) {
            const int kk = idx / HEAD_DIM;
            const int c = idx % HEAD_DIM;
            const int gk = k_base + kk;
            sW0[W_TILE_OFFSET(kk, c)] = (gk < D)
                ? Wq[static_cast<long long>(gk) * H * d_head + head_col_start + c]
                : scalar_zero<scalar_t>();
        }

        __syncthreads();
        accumulate_proj_row<scalar_t>(q_acc, &sX[X_TILE_OFFSET(tid, 0)], sW0);
        __syncthreads();
    }

    for (int tile_kv = 0; tile_kv * TILE_SIZE < N; ++tile_kv) {
        const int kv_global = tile_kv * TILE_SIZE + tid;

        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c) {
            sK[TILE_OFFSET(tid, c)] = 0.0f;
            sV[TILE_OFFSET(tid, c)] = 0.0f;
        }

        __syncthreads();

        for (int k_base = 0; k_base < D; k_base += PROJ_K_TILE) {
            #pragma unroll
            for (int kk = 0; kk < PROJ_K_TILE; ++kk) {
                const int gk = k_base + kk;
                sX[X_TILE_OFFSET(tid, kk)] = (kv_global < N && gk < D)
                    ? x_b[static_cast<long long>(kv_global) * D + gk]
                    : scalar_zero<scalar_t>();
            }

            for (int idx = tid; idx < PROJ_K_TILE * HEAD_DIM; idx += TILE_SIZE) {
                const int kk = idx / HEAD_DIM;
                const int c = idx % HEAD_DIM;
                const int gk = k_base + kk;
                sW0[W_TILE_OFFSET(kk, c)] = (gk < D)
                    ? Wk[static_cast<long long>(gk) * H * d_head + head_col_start + c]
                    : scalar_zero<scalar_t>();
                sW1[W_TILE_OFFSET(kk, c)] = (gk < D)
                    ? Wv[static_cast<long long>(gk) * H * d_head + head_col_start + c]
                    : scalar_zero<scalar_t>();
            }

            __syncthreads();
            accumulate_two_proj_rows<scalar_t>(
                &sK[TILE_OFFSET(tid, 0)],
                &sV[TILE_OFFSET(tid, 0)],
                &sX[X_TILE_OFFSET(tid, 0)],
                sW0,
                sW1
            );
            __syncthreads();
        }

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
                dot += q_acc[c] * sK[TILE_OFFSET(j, c)];
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
                o_acc[c] += e * sV[TILE_OFFSET(j, c)];
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
}

template <typename scalar_t>
void launch_fused_attention_impl(
    const scalar_t* X,
    const scalar_t* Wq,
    const scalar_t* Wk,
    const scalar_t* Wv,
    float* Out,
    int B, int H, int N, int D, int d_head,
    cudaStream_t stream
) {
    const int n_q_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(B, H, n_q_tiles);
    dim3 block(TILE_SIZE);
    const size_t shmem_bytes =
        (2 * TILE_SIZE * SHMEM_STRIDE) * sizeof(float) +
        (TILE_SIZE * PROJ_K_TILE + 2 * PROJ_K_TILE * HEAD_DIM) * sizeof(scalar_t);

    cudaFuncSetAttribute(
        fused_qkv_attention_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    );
    cudaFuncSetAttribute(
        fused_qkv_attention_kernel<scalar_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100
    );

    fused_qkv_attention_kernel<scalar_t><<<grid, block, shmem_bytes, stream>>>(
        X, Wq, Wk, Wv, Out, B, H, N, D, d_head
    );
}

extern "C" void launch_fused_attention_fp32(
    const void* X,
    const void* Wq,
    const void* Wk,
    const void* Wv,
    void* Out,
    int B, int H, int N, int D, int d_head,
    cudaStream_t stream
) {
    launch_fused_attention_impl(
        static_cast<const float*>(X),
        static_cast<const float*>(Wq),
        static_cast<const float*>(Wk),
        static_cast<const float*>(Wv),
        static_cast<float*>(Out),
        B,
        H,
        N,
        D,
        d_head,
        stream
    );
}

extern "C" void launch_fused_attention_fp16(
    const void* X,
    const void* Wq,
    const void* Wk,
    const void* Wv,
    void* Out,
    int B, int H, int N, int D, int d_head,
    cudaStream_t stream
) {
    launch_fused_attention_impl(
        static_cast<const __half*>(X),
        static_cast<const __half*>(Wq),
        static_cast<const __half*>(Wk),
        static_cast<const __half*>(Wv),
        static_cast<float*>(Out),
        B,
        H,
        N,
        D,
        d_head,
        stream
    );
}

extern "C" void launch_fused_attention_bf16(
    const void* X,
    const void* Wq,
    const void* Wk,
    const void* Wv,
    void* Out,
    int B, int H, int N, int D, int d_head,
    cudaStream_t stream
) {
    launch_fused_attention_impl(
        static_cast<const __nv_bfloat16*>(X),
        static_cast<const __nv_bfloat16*>(Wq),
        static_cast<const __nv_bfloat16*>(Wk),
        static_cast<const __nv_bfloat16*>(Wv),
        static_cast<float*>(Out),
        B,
        H,
        N,
        D,
        d_head,
        stream
    );
}
