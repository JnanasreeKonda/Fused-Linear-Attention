/*
 * kernel/wmma_proj_ext.cpp — PyTorch binding for the WMMA projection prototype.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace py = pybind11;

extern "C" void launch_wmma_projection_fp16(
    const void* X,
    const void* W,
    void* Out,
    int M,
    int K,
    int N,
    cudaStream_t stream
);

extern "C" void launch_wmma_projection_bf16(
    const void* X,
    const void* W,
    void* Out,
    int M,
    int K,
    int N,
    cudaStream_t stream
);

torch::Tensor wmma_projection_forward(
    torch::Tensor X,
    torch::Tensor W
) {
    TORCH_CHECK(X.device().is_cuda(), "X must be CUDA");
    TORCH_CHECK(W.device().is_cuda(), "W must be CUDA");
    TORCH_CHECK(
        X.scalar_type() == torch::kFloat16 || X.scalar_type() == torch::kBFloat16,
        "X must be float16 or bfloat16"
    );
    TORCH_CHECK(W.scalar_type() == X.scalar_type(), "W must match X dtype");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(X.dim() == 2, "X must be 2-D [M, K]");
    TORCH_CHECK(W.dim() == 2, "W must be 2-D [K, N]");
    TORCH_CHECK(X.size(1) == W.size(0), "Inner dimensions must match");
    TORCH_CHECK(X.size(0) % 16 == 0, "M must be a multiple of 16");
    TORCH_CHECK(X.size(1) % 16 == 0, "K must be a multiple of 16");
    TORCH_CHECK(W.size(1) % 16 == 0, "N must be a multiple of 16");

    const int M = static_cast<int>(X.size(0));
    const int K = static_cast<int>(X.size(1));
    const int N = static_cast<int>(W.size(1));

    auto out = torch::zeros(
        {M, N},
        torch::TensorOptions().dtype(torch::kFloat32).device(X.device())
    );

    cudaStream_t stream = c10::cuda::getDefaultCUDAStream();
    if (X.scalar_type() == torch::kFloat16) {
        launch_wmma_projection_fp16(
            X.data_ptr(),
            W.data_ptr(),
            out.data_ptr(),
            M,
            K,
            N,
            stream
        );
    } else {
        launch_wmma_projection_bf16(
            X.data_ptr(),
            W.data_ptr(),
            out.data_ptr(),
            M,
            K,
            N,
            stream
        );
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Experimental WMMA projection prototype";
    m.def("forward", &wmma_projection_forward, "WMMA projection forward");
}
