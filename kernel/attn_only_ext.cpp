/*
 * kernel/attn_only_ext.cpp — PyTorch binding for the experimental attention-only kernel.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace py = pybind11;

extern "C" void launch_attn_only_fp32(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
);

extern "C" void launch_attn_only_fp16(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
);

extern "C" void launch_attn_only_bf16(
    const void* Q,
    const void* K,
    const void* V,
    void* Out,
    int B, int H, int N, int d_head,
    cudaStream_t stream
);

torch::Tensor attn_only_forward(
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

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Q/K/V must be 4-D [B, H, N, d_head]");
    TORCH_CHECK(Q.size(0) == B && Q.size(1) == H && Q.size(2) == N && Q.size(3) == d_head, "Q shape mismatch");
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(2) == N && K.size(3) == d_head, "K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == N && V.size(3) == d_head, "V shape mismatch");

    auto out = torch::zeros(
        {B, H, N, d_head},
        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device())
    );

    cudaStream_t stream = c10::cuda::getDefaultCUDAStream();
    if (dtype == torch::kFloat32) {
        launch_attn_only_fp32(Q.data_ptr(), K.data_ptr(), V.data_ptr(), out.data_ptr(), B, H, N, d_head, stream);
    } else if (dtype == torch::kFloat16) {
        launch_attn_only_fp16(Q.data_ptr(), K.data_ptr(), V.data_ptr(), out.data_ptr(), B, H, N, d_head, stream);
    } else {
        launch_attn_only_bf16(Q.data_ptr(), K.data_ptr(), V.data_ptr(), out.data_ptr(), B, H, N, d_head, stream);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Attention-only CUDA extension";
    m.def(
        "forward",
        &attn_only_forward,
        "Attention-only scaled dot-product attention",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("B"),
        py::arg("H"),
        py::arg("N"),
        py::arg("d_head")
    );
}
