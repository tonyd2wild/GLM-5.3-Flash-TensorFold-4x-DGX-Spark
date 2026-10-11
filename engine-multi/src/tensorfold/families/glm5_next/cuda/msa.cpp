// Mia's sparse attention for GLM-5.3-Flash's prompt rows (msa.cu): the binding. By Mia's AI Lab (Apache-2.0).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void msa_prompt_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, double);
int msa_max_heads();

void prompt(const at::Tensor& qa, const at::Tensor& cache, const at::Tensor& tokens, const at::Tensor& counts,
            at::Tensor out, double scale) {
    auto ok = [](const at::Tensor& x, at::ScalarType t) {
        return x.is_cuda() && x.scalar_type() == t && x.is_contiguous();
    };
    TORCH_CHECK(ok(qa, at::kBFloat16) && ok(out, at::kBFloat16) && qa.dim() == 3 && out.sizes() == qa.sizes(),
                "msa: contiguous bf16 queries and output [R, H, 512]");
    TORCH_CHECK(ok(cache, at::kByte) && cache.dim() == 2, "msa: a contiguous FP8 latent cache (uint8 kv8 rows)");
    TORCH_CHECK(ok(tokens, at::kInt) && ok(counts, at::kInt) && tokens.dim() == 2 &&
                    tokens.size(0) >= qa.size(0) && counts.size(0) >= qa.size(0),
                "msa: contiguous int32 token lists and counts for every row");
    c10::cuda::CUDAGuard guard(qa.device());
    msa_prompt_cuda(qa, cache, tokens, counts, out, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prompt", &prompt);
    m.def("max_heads", &msa_max_heads);
}
