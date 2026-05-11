/*
 * kernel/attn_only_warp4.cu — experimental warp-cooperative attention kernel.
 *
 * Each query row is handled by a small group of threads, reducing per-thread
 * state and making the Q·K / softmax / V accumulation path more cooperative.
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

#ifndef Q_GROUP_SIZE
#define Q_GROUP_SIZE 4
#endif

#define DIMS_PER_LANE (HEAD_DIM / Q_GROUP_SIZE)
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

template <typename scalar_t>
__launch_bounds__(TILE_SIZE * Q_GROUP_SIZE, 1)
__global__ void attn_only_warp4_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    float* __restrict__ Out,
    int B, int H, int N, int d_head
) {
    const int tid = threadIdx.x;
    const int q_local = tid / Q_GROUP_SIZE;
    const int lane = tid % Q_GROUP_SIZE;
    const int q_global = blockIdx.z * TILE_SIZE + q_local;
    const unsigned mask = 0xffffffffu;

    const scalar_t* q_bh = Q + ((static_cast<long long>(blockIdx.x) * H + blockIdx.y) * N * d_head);
    const scalar_t* k_bh = K + ((static_cast<long long>(blockIdx.x) * H + blockIdx.y) * N * d_head);
    const scalar_t* v_bh = V + ((static_cast<long long>(blockIdx.x) * H + blockIdx.y) * N * d_head);
    float* out_bh = Out + ((static_cast<long long>(blockIdx.x) * H + blockIdx.y) * N * d_head);

    const int dim_base = lane * DIMS_PER_LANE;
    const float scale = rsqrtf(static_cast<float>(d_head));

    float q_reg[DIMS_PER_LANE];
    if (q_global < N) {
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            q_reg[i] = scalar_to_float(
                q_bh[static_cast<long long>(q_global) * d_head + dim_base + i]
            );
        }
    } else {
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            q_reg[i] = 0.0f;
        }
    }

    float o_acc[DIMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < DIMS_PER_LANE; ++i) {
        o_acc[i] = 0.0f;
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    extern __shared__ float shared_mem[];
    float* sK = shared_mem;
    float* sV = sK + TILE_SIZE * SHMEM_STRIDE;

    for (int tile_kv = 0; tile_kv * TILE_SIZE < N; ++tile_kv) {
        const int kv_row = q_local;
        const int kv_global = tile_kv * TILE_SIZE + kv_row;

        if (lane == 0) {
            load_row_to_float(
                &sK[TILE_OFFSET(kv_row, 0)],
                k_bh + static_cast<long long>(kv_global) * d_head,
                kv_global < N
            );
            load_row_to_float(
                &sV[TILE_OFFSET(kv_row, 0)],
                v_bh + static_cast<long long>(kv_global) * d_head,
                kv_global < N
            );
        }

        __syncthreads();

        float m_tile = -FLT_MAX;
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            const int j_global = tile_kv * TILE_SIZE + j;
            float partial = 0.0f;
            if (q_global < N && j_global < N) {
                #pragma unroll
                for (int i = 0; i < DIMS_PER_LANE; ++i) {
                    partial += q_reg[i] * sK[TILE_OFFSET(j, dim_base + i)];
                }
            }
            #pragma unroll
            for (int offset = Q_GROUP_SIZE / 2; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(mask, partial, offset, Q_GROUP_SIZE);
            }
            if (lane == 0 && j_global < N && q_global < N) {
                const float score = partial * scale;
                if (score > m_tile) {
                    m_tile = score;
                }
            }
        }

        float m_new = (lane == 0) ? fmaxf(m_i, m_tile) : 0.0f;
        m_new = __shfl_sync(mask, m_new, 0, Q_GROUP_SIZE);
        const float corr_old = expf(m_i - m_new);

        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            o_acc[i] *= corr_old;
        }

        float l_tile = 0.0f;
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            const int j_global = tile_kv * TILE_SIZE + j;
            float partial = 0.0f;
            if (q_global < N && j_global < N) {
                #pragma unroll
                for (int i = 0; i < DIMS_PER_LANE; ++i) {
                    partial += q_reg[i] * sK[TILE_OFFSET(j, dim_base + i)];
                }
            }
            #pragma unroll
            for (int offset = Q_GROUP_SIZE / 2; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(mask, partial, offset, Q_GROUP_SIZE);
            }

            float e = 0.0f;
            if (lane == 0 && q_global < N && j_global < N) {
                e = expf(partial * scale - m_new);
                l_tile += e;
            }
            e = __shfl_sync(mask, e, 0, Q_GROUP_SIZE);

            if (q_global < N && j_global < N) {
                #pragma unroll
                for (int i = 0; i < DIMS_PER_LANE; ++i) {
                    o_acc[i] += e * sV[TILE_OFFSET(j, dim_base + i)];
                }
            }
        }

        if (lane == 0) {
            l_i = l_i * corr_old + l_tile;
            m_i = m_new;
        }
        m_i = __shfl_sync(mask, m_i, 0, Q_GROUP_SIZE);
        l_i = __shfl_sync(mask, l_i, 0, Q_GROUP_SIZE);

        __syncthreads();
    }

    if (q_global < N && l_i > 0.0f) {
        const float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            out_bh[static_cast<long long>(q_global) * d_head + dim_base + i] =
                o_acc[i] * inv_l;
        }
    }
}

template <typename scalar_t>
void launch_attn_only_warp4_impl(
    const scalar_t* Q,
    const scalar_t* K,
    const scalar_t* V,
    float* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
) {
    const int n_q_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(B, H, n_q_tiles);
    dim3 block(TILE_SIZE * Q_GROUP_SIZE);
    const size_t shmem_bytes = (2 * TILE_SIZE * SHMEM_STRIDE) * sizeof(float);

    cudaFuncSetAttribute(
        attn_only_warp4_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    );
    cudaFuncSetAttribute(
        attn_only_warp4_kernel<scalar_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100
    );

    attn_only_warp4_kernel<scalar_t><<<grid, block, shmem_bytes, stream>>>(
        Q, K, V, Out, B, H, N, d_head
    );
}

extern "C" void launch_attn_only_warp4_fp32(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
) {
    launch_attn_only_warp4_impl(
        static_cast<const float*>(Q),
        static_cast<const float*>(K),
        static_cast<const float*>(V),
        static_cast<float*>(Out),
        B, H, N, d_head, stream
    );
}

extern "C" void launch_attn_only_warp4_fp16(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
) {
    launch_attn_only_warp4_impl(
        static_cast<const __half*>(Q),
        static_cast<const __half*>(K),
        static_cast<const __half*>(V),
        static_cast<float*>(Out),
        B, H, N, d_head, stream
    );
}

extern "C" void launch_attn_only_warp4_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
) {
    launch_attn_only_warp4_impl(
        static_cast<const __nv_bfloat16*>(Q),
        static_cast<const __nv_bfloat16*>(K),
        static_cast<const __nv_bfloat16*>(V),
        static_cast<float*>(Out),
        B, H, N, d_head, stream
    );
}
