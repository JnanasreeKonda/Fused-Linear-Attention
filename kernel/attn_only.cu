/*
 * kernel/attn_only.cu — experimental attention-only CUDA kernel.
 *
 * This kernel consumes precomputed Q/K/V tensors and computes scaled
 * dot-product attention tile by tile. It exists to test a practical midpoint:
 * keep projections on the optimized library side, while using a custom kernel
 * for the attention stage itself.
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

#define SHMEM_PAD 4
#define SHMEM_STRIDE (HEAD_DIM + SHMEM_PAD)
#define TILE_OFFSET(row, col) ((row) * SHMEM_STRIDE + (col))

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
__device__ inline void load_row_to_float(
    float* __restrict__ dst,
    const scalar_t* __restrict__ src,
    bool valid_row
) {
    #pragma unroll
    for (int c = 0; c < HEAD_DIM; ++c) {
        dst[c] = valid_row ? scalar_to_float(src[c]) : 0.0f;
    }
}

template <>
__device__ inline void load_row_to_float<float>(
    float* __restrict__ dst,
    const float* __restrict__ src,
    bool valid_row
) {
    if (!valid_row) {
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c) {
            dst[c] = 0.0f;
        }
        return;
    }
    #pragma unroll
    for (int c = 0; c < HEAD_DIM; c += 4) {
        *reinterpret_cast<float4*>(dst + c) =
            *reinterpret_cast<const float4*>(src + c);
    }
}

template <>
__device__ inline void load_row_to_float<__half>(
    float* __restrict__ dst,
    const __half* __restrict__ src,
    bool valid_row
) {
    if (!valid_row) {
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c) {
            dst[c] = 0.0f;
        }
        return;
    }
    #pragma unroll
    for (int c = 0; c < HEAD_DIM; c += 2) {
        const float2 vals = __half22float2(*reinterpret_cast<const __half2*>(src + c));
        dst[c] = vals.x;
        dst[c + 1] = vals.y;
    }
}

template <>
__device__ inline void load_row_to_float<__nv_bfloat16>(
    float* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    bool valid_row
) {
    if (!valid_row) {
        #pragma unroll
        for (int c = 0; c < HEAD_DIM; ++c) {
            dst[c] = 0.0f;
        }
        return;
    }
    #pragma unroll
    for (int c = 0; c < HEAD_DIM; c += 2) {
        const float2 vals = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(src + c)
        );
        dst[c] = vals.x;
        dst[c + 1] = vals.y;
    }
}

__device__ inline float dot_shared_rows(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs
) {
    float acc = 0.0f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM; c += 4) {
        const float4 lv = *reinterpret_cast<const float4*>(lhs + c);
        const float4 rv = *reinterpret_cast<const float4*>(rhs + c);
        acc += lv.x * rv.x + lv.y * rv.y + lv.z * rv.z + lv.w * rv.w;
    }
    return acc;
}

__device__ inline void axpy_shared_row(
    float* __restrict__ dst,
    float alpha,
    const float* __restrict__ src
) {
    #pragma unroll
    for (int c = 0; c < HEAD_DIM; c += 4) {
        const float4 sv = *reinterpret_cast<const float4*>(src + c);
        dst[c] += alpha * sv.x;
        dst[c + 1] += alpha * sv.y;
        dst[c + 2] += alpha * sv.z;
        dst[c + 3] += alpha * sv.w;
    }
}

template <typename scalar_t>
__launch_bounds__(TILE_SIZE, 4)
__global__ void attn_only_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int N, int d_head
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tile_q = blockIdx.z;
    const int tid = threadIdx.x;
    const int q_global = tile_q * TILE_SIZE + tid;

    const float scale = rsqrtf(static_cast<float>(d_head));

    const scalar_t* q_bh = Q + ((static_cast<long long>(b) * H + h) * N * d_head);
    const scalar_t* k_bh = K + ((static_cast<long long>(b) * H + h) * N * d_head);
    const scalar_t* v_bh = V + ((static_cast<long long>(b) * H + h) * N * d_head);
    float* out_bh = Out + ((static_cast<long long>(b) * H + h) * N * d_head);

    extern __shared__ float shared_mem[];
    float* sQ = shared_mem;
    float* sK = sQ + TILE_SIZE * SHMEM_STRIDE;
    float* sV = sK + TILE_SIZE * SHMEM_STRIDE;

    load_row_to_float(
        &sQ[TILE_OFFSET(tid, 0)],
        q_bh + static_cast<long long>(q_global) * d_head,
        q_global < N
    );

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

        load_row_to_float(
            &sK[TILE_OFFSET(tid, 0)],
            k_bh + static_cast<long long>(kv_global) * d_head,
            kv_global < N
        );
        load_row_to_float(
            &sV[TILE_OFFSET(tid, 0)],
            v_bh + static_cast<long long>(kv_global) * d_head,
            kv_global < N
        );

        __syncthreads();

        float m_tile = -FLT_MAX;
        float scores[TILE_SIZE];

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            const int j_global = tile_kv * TILE_SIZE + j;
            if (j_global >= N || q_global >= N) {
                scores[j] = -FLT_MAX;
                continue;
            }

            scores[j] = dot_shared_rows(
                &sQ[TILE_OFFSET(tid, 0)],
                &sK[TILE_OFFSET(j, 0)]
            ) * scale;
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

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            const int j_global = tile_kv * TILE_SIZE + j;
            if (j_global >= N || q_global >= N) {
                continue;
            }

            const float e = expf(scores[j] - m_new);
            l_tile += e;
            axpy_shared_row(o_acc, e, &sV[TILE_OFFSET(j, 0)]);
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
void launch_attn_only_impl(
    const scalar_t* Q,
    const scalar_t* K,
    const scalar_t* V,
    float* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
) {
    const int n_q_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(B, H, n_q_tiles);
    dim3 block(TILE_SIZE);
    const size_t shmem_bytes = (3 * TILE_SIZE * SHMEM_STRIDE) * sizeof(float);

    cudaFuncSetAttribute(
        attn_only_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    );
    cudaFuncSetAttribute(
        attn_only_kernel<scalar_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100
    );

    attn_only_kernel<scalar_t><<<grid, block, shmem_bytes, stream>>>(
        Q, K, V, Out, B, H, N, d_head
    );
}

extern "C" void launch_attn_only_fp32(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
) {
    launch_attn_only_impl(
        static_cast<const float*>(Q),
        static_cast<const float*>(K),
        static_cast<const float*>(V),
        static_cast<float*>(Out),
        B, H, N, d_head, stream
    );
}

extern "C" void launch_attn_only_fp16(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
) {
    launch_attn_only_impl(
        static_cast<const __half*>(Q),
        static_cast<const __half*>(K),
        static_cast<const __half*>(V),
        static_cast<float*>(Out),
        B, H, N, d_head, stream
    );
}

extern "C" void launch_attn_only_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
) {
    launch_attn_only_impl(
        static_cast<const __nv_bfloat16*>(Q),
        static_cast<const __nv_bfloat16*>(K),
        static_cast<const __nv_bfloat16*>(V),
        static_cast<float*>(Out),
        B, H, N, d_head, stream
    );
}
