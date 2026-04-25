/*
 * kernel/fused_attn_ext.cpp — PyTorch C++ extension binding.
 * Owner: Jnanasree (M8) — provided here so Bhanuja's load_kernel.py compiles.
 *
 * Exposes one function to Python:
 *
 *   fused_linear_attention.forward(X, Wq, Wk, Wv, B, H, N, D, d_head)
 *     -> torch.Tensor [B, H, N, d_head]
 *
 * Compiled by load_kernel.py via torch.utils.cpp_extension.load().
 * No manual cmake/setup.py needed.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

/* Declared in fused_attn.cu */
extern "C" void launch_fused_attention(
    const float* X,
    const float* Wq,
    const float* Wk,
    const float* Wv,
    float*       Out,
    int B, int H, int N, int D, int d_head,
    cudaStream_t stream
);

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Tensor-level wrapper                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

torch::Tensor fused_attention_forward(
    torch::Tensor X,      /* [B, N, D]       fp32, CUDA, contiguous */
    torch::Tensor Wq,     /* [D, H*d_head]   fp32, CUDA, contiguous */
    torch::Tensor Wk,     /* [D, H*d_head]   fp32, CUDA, contiguous */
    torch::Tensor Wv,     /* [D, H*d_head]   fp32, CUDA, contiguous */
    int64_t B,
    int64_t H,
    int64_t N,
    int64_t D,
    int64_t d_head
) {
    /* ── Input validation ────────────────────────────────────────────────── */
    TORCH_CHECK(X.device().is_cuda(),  "X must be a CUDA tensor");
    TORCH_CHECK(Wq.device().is_cuda(), "Wq must be a CUDA tensor");
    TORCH_CHECK(Wk.device().is_cuda(), "Wk must be a CUDA tensor");
    TORCH_CHECK(Wv.device().is_cuda(), "Wv must be a CUDA tensor");

    TORCH_CHECK(X.dtype()  == torch::kFloat32, "X must be float32");
    TORCH_CHECK(Wq.dtype() == torch::kFloat32, "Wq must be float32");
    TORCH_CHECK(Wk.dtype() == torch::kFloat32, "Wk must be float32");
    TORCH_CHECK(Wv.dtype() == torch::kFloat32, "Wv must be float32");

    TORCH_CHECK(X.is_contiguous(),  "X must be contiguous — call .contiguous() first");
    TORCH_CHECK(Wq.is_contiguous(), "Wq must be contiguous");
    TORCH_CHECK(Wk.is_contiguous(), "Wk must be contiguous");
    TORCH_CHECK(Wv.is_contiguous(), "Wv must be contiguous");

    TORCH_CHECK(X.dim()  == 3, "X must be 3-D [B, N, D]");
    TORCH_CHECK(Wq.dim() == 2, "Wq must be 2-D [D, H*d_head]");
    TORCH_CHECK(Wk.dim() == 2, "Wk must be 2-D [D, H*d_head]");
    TORCH_CHECK(Wv.dim() == 2, "Wv must be 2-D [D, H*d_head]");

    TORCH_CHECK(X.size(0)  == B && X.size(1) == N && X.size(2) == D,
                "X shape must be [B, N, D]");
    TORCH_CHECK(Wq.size(0) == D && Wq.size(1) == H * d_head,
                "Wq shape must be [D, H*d_head]");
    TORCH_CHECK(Wk.size(0) == D && Wk.size(1) == H * d_head,
                "Wk shape must be [D, H*d_head]");
    TORCH_CHECK(Wv.size(0) == D && Wv.size(1) == H * d_head,
                "Wv shape must be [D, H*d_head]");

    /* ── Allocate output tensor ──────────────────────────────────────────── */
    auto Out = torch::zeros(
        {B, H, N, d_head},
        torch::TensorOptions().dtype(torch::kFloat32).device(X.device())
    );

    /* ── Launch CUDA kernel ──────────────────────────────────────────────── */
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_fused_attention(
        X.data_ptr<float>(),
        Wq.data_ptr<float>(),
        Wk.data_ptr<float>(),
        Wv.data_ptr<float>(),
        Out.data_ptr<float>(),
        (int)B, (int)H, (int)N, (int)D, (int)d_head,
        stream
    );

    return Out;   /* [B, H, N, d_head] */
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  PYBIND11 module registration                                                */
/* ─────────────────────────────────────────────────────────────────────────── */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FusedLinearAttention CUDA extension — QKV projection fused with attention";
    m.def(
        "forward",
        &fused_attention_forward,
        "Fused QKV projection + scaled dot-product attention (single CUDA kernel).\n\n"
        "Args:\n"
        "    X   (Tensor [B, N, D]):       input sequence, fp32, CUDA, contiguous\n"
        "    Wq  (Tensor [D, H*d_head]):   Q projection weights\n"
        "    Wk  (Tensor [D, H*d_head]):   K projection weights\n"
        "    Wv  (Tensor [D, H*d_head]):   V projection weights\n"
        "    B   (int): batch size\n"
        "    H   (int): number of heads\n"
        "    N   (int): sequence length\n"
        "    D   (int): embed / model dimension\n"
        "    d_head (int): head dimension (D // H)\n\n"
        "Returns:\n"
        "    Tensor [B, H, N, d_head]: attention output",
        py::arg("X"), py::arg("Wq"), py::arg("Wk"), py::arg("Wv"),
        py::arg("B"), py::arg("H"), py::arg("N"), py::arg("D"), py::arg("d_head")
    );
}
