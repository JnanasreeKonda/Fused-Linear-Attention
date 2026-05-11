/*
 * kernel/wmma_proj.cu — experimental WMMA projection microkernel.
 *
 * This is an isolated Tensor Core prototype for the dominant bottleneck in the
 * fused attention kernel: the projection stage X @ W. It intentionally does
 * not try to solve the full fused attention problem. Instead it answers the
 * narrower question: can a warp-cooperative Tensor Core projection beat our
 * scalar loop projection for the benchmark geometry on H100-class GPUs?
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void wmma_projection_kernel_fp16(
    const half* __restrict__ X,
    const half* __restrict__ W,
    float* __restrict__ Out,
    int M,
    int K,
    int N
) {
    const int row_tile = blockIdx.x;
    const int col_tile = blockIdx.y;

    const int row_start = row_tile * WMMA_M;
    const int col_start = col_tile * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k_start = 0; k_start < K; k_start += WMMA_K) {
        const half* a_ptr = X + static_cast<long long>(row_start) * K + k_start;
        const half* b_ptr = W + static_cast<long long>(k_start) * N + col_start;
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float* out_ptr = Out + static_cast<long long>(row_start) * N + col_start;
    wmma::store_matrix_sync(out_ptr, acc_frag, N, wmma::mem_row_major);
}

}  // namespace

extern "C" void launch_wmma_projection_fp16(
    const void* X,
    const void* W,
    void* Out,
    int M,
    int K,
    int N,
    cudaStream_t stream
) {
    dim3 grid(M / WMMA_M, N / WMMA_N);
    dim3 block(32);
    wmma_projection_kernel_fp16<<<grid, block, 0, stream>>>(
        static_cast<const half*>(X),
        static_cast<const half*>(W),
        static_cast<float*>(Out),
        M,
        K,
        N
    );
}
