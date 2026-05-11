/*
 * kernel/attn_only_warp4_ext.cpp — PyTorch binding for the warp-cooperative attention kernel.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace py = pybind11;

extern "C" void launch_attn_only_warp4_fp32(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
);

extern "C" void launch_attn_only_warp4_fp16(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
);

extern "C" void launch_attn_only_warp4_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
);

torch::Tensor attn_only_warp4_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int64_t B,
    int64_t H,
    int64_t N,
    int64_t d_head
) {
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");

    const auto dtype = Q.scalar_type();
    TORCH_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
        "Q must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(K.scalar_type() == dtype, "K must match Q dtype");
    TORCH_CHECK(V.scalar_type() == dtype, "V must match Q dtype");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Q/K/V must be contiguous");

    auto out = torch::zeros(
        {B, H, N, d_head},
        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device())
    );

    cudaStream_t stream = c10::cuda::getDefaultCUDAStream();
    if (dtype == torch::kFloat32) {
        launch_attn_only_warp4_fp32(Q.data_ptr(), K.data_ptr(), V.data_ptr(), out.data_ptr(), B, H, N, d_head, stream);
    } else if (dtype == torch::kFloat16) {
        launch_attn_only_warp4_fp16(Q.data_ptr(), K.data_ptr(), V.data_ptr(), out.data_ptr(), B, H, N, d_head, stream);
    } else {
        launch_attn_only_warp4_bf16(Q.data_ptr(), K.data_ptr(), V.data_ptr(), out.data_ptr(), B, H, N, d_head, stream);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Warp-cooperative attention-only CUDA extension";
    m.def(
        "forward",
        &attn_only_warp4_forward,
        "Warp-cooperative attention-only kernel",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("B"),
        py::arg("H"),
        py::arg("N"),
        py::arg("d_head")
    );
}
