"""GLM-5.3-Flash images on CUDA: the vision tower in plain torch, its image processor, and prompt expansion.

Written from vLLM's GLM5Next vision tower and processor (``vllm/models/glm5next/nvidia/multimodal.py`` and
``vllm/transformers_utils/processors/glm5next.py``, Apache-2.0, Copyright contributors to the vLLM project): a
GLM-4V/GLM-OCR tower without absolute position embeddings, q/k RMSNorm (eps 1e-5), 2-D rotary positions on half the
head width, clamped SwiGLU MLPs, a 2x2 downsampling convolution and a clamped SwiGLU merger into the text width.
The language model has no position encoding, so an image is its feature rows written over its placeholder tokens.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PREFIX = "model.visual."


@dataclass
class VisionConfig:
    depth: int
    hidden: int
    heads: int
    mlp: int
    out_hidden: int
    merger_width: int
    patch: int
    temporal: int
    merge: int
    eps: float
    limit: float
    image_token: int
    image_start: int
    image_end: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    min_tokens: int
    max_tokens: int
    expand: int

    @classmethod
    def read(cls, model_dir: str | Path) -> "VisionConfig":
        d = Path(model_dir)
        raw = json.loads((d / "config.json").read_text())
        v = raw["vision_config"]
        proc = {}
        if (d / "processor_config.json").exists():
            proc = json.loads((d / "processor_config.json").read_text()).get("image_processor", {})
        limit = v.get("swiglu_limit") or (raw.get("text_config") or {}).get("swiglu_limit") or 10.0
        return cls(depth=int(v["depth"]), hidden=int(v["hidden_size"]), heads=int(v["num_heads"]),
                   mlp=int(v["intermediate_size"]), out_hidden=int(v["out_hidden_size"]),
                   merger_width=int(v["projection_intermediate_size"]), patch=int(v["patch_size"]),
                   temporal=int(v["temporal_patch_size"]), merge=int(v["spatial_merge_size"]),
                   eps=float(v.get("rms_norm_eps", 1e-5)), limit=float(limit),
                   image_token=int(raw["image_token_id"]), image_start=int(raw["image_start_token_id"]),
                   image_end=int(raw["image_end_token_id"]),
                   mean=tuple(proc.get("image_mean", (0.48145466, 0.4578275, 0.40821073))),
                   std=tuple(proc.get("image_std", (0.26862954, 0.26130258, 0.27577711))),
                   min_tokens=int(proc.get("min_image_tokens", 16)), max_tokens=int(proc.get("max_image_tokens", 8000)),
                   expand=int(proc.get("patch_expand_factor", 1)))


# -- processor --------------------------------------------------------------------------------------------------

def _ceil(value: int, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _fit(t: int, h: int, w: int, factor: int, max_pixels: int) -> tuple[int, int]:
    """Largest proportional canvas, each side rounded up to ``factor``, within ``max_pixels`` (binary search)."""

    lo, hi, best = 1, h, (factor, factor)
    while lo <= hi:
        ch = (lo + hi) // 2
        cw = max(1, math.floor(w * ch / h))
        ah, aw = _ceil(ch, factor), _ceil(cw, factor)
        if t * ah * aw <= max_pixels:
            best, lo = (ah, aw), ch + 1
        else:
            hi = ch - 1
    return best


def canvas(cfg: VisionConfig, h: int, w: int) -> tuple[int, int]:
    """The aligned canvas an h x w image is placed on: sides rounded up, refit under the token budget."""

    factor = cfg.patch * cfg.merge * cfg.expand
    unit = cfg.temporal * (cfg.patch * cfg.merge) ** 2
    lo, hi = cfg.min_tokens * unit, cfg.max_tokens * unit
    t = cfg.temporal
    ah, aw = _ceil(h, factor), _ceil(w, factor)
    if t * ah * aw > hi:
        return _fit(t, h, w, factor, hi)
    if t * ah * aw < lo:
        beta = math.sqrt(lo / (t * h * w))
        ah, aw = _ceil(max(1, math.ceil(h * beta)), factor), _ceil(max(1, math.ceil(w * beta)), factor)
        if t * ah * aw > hi:
            return _fit(t, h, w, factor, hi)
    return ah, aw


def preprocess(cfg: VisionConfig, image) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """A PIL image (or encoded bytes) -> (patches [L, C*T*P*P] fp32, grid (t, h, w) in patches).

    The image keeps its aspect ratio: it is shrunk to fit the canvas (enlarged only below the token floor), then the
    right and bottom are padded with zeros, as the checkpoint's processor (``resize_mode: pad``) does.
    """

    from PIL import Image

    if hasattr(image, "pixels") and hasattr(image, "width"):          # the server's decoded ImageInput (RGB bytes)
        image = Image.frombytes("RGB", (image.width, image.height), image.pixels)
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    image = image.convert("RGB")
    w, h = image.size
    ch, cw = canvas(cfg, h, w)
    unit = cfg.temporal * (cfg.patch * cfg.merge) ** 2
    upscale = cfg.temporal * h * w < cfg.min_tokens * unit
    scale = min(ch / h, cw / w)
    if not upscale:
        scale = min(1.0, scale)
    nh, nw = max(1, min(ch, math.floor(h * scale))), max(1, min(cw, math.floor(w * scale)))
    if (nh, nw) != (h, w):
        image = image.resize((nw, nh), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)     # [C, nh, nw]
    mean = torch.tensor(cfg.mean).view(3, 1, 1)
    std = torch.tensor(cfg.std).view(3, 1, 1)
    x = (x - mean) / std
    x = F.pad(x, (0, cw - nw, 0, ch - nh), value=0.0)                                      # pad right and bottom
    p, m, t = cfg.patch, cfg.merge, cfg.temporal
    gh, gw = ch // p, cw // p
    x = x.unsqueeze(0).repeat(t, 1, 1, 1)                                                  # [T, C, H, W]: one frame
    x = x.view(1, t, 3, gh // m, m, p, gw // m, m, p)                                      # grid_t = 1
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)                                               # (gt, gh, gw, mh, mw, C, tp, ph, pw)
    return x.reshape(gh * gw, 3 * t * p * p).contiguous(), (1, gh, gw)


def image_tokens(cfg: VisionConfig, grid: tuple[int, int, int]) -> int:
    t, h, w = grid
    return t * h * w // (cfg.merge * cfg.merge)


def expand(ids: list[int], cfg: VisionConfig, counts: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    """Each image placeholder token -> its image's token count; returns the ids and each image's (start, length)."""

    out, spans, k = [], [], 0
    for tok in ids:
        if tok == cfg.image_token:
            if k >= len(counts):
                raise ValueError("the prompt has more image placeholders than images")
            spans.append((len(out), counts[k]))
            out.extend([cfg.image_token] * counts[k])
            k += 1
        else:
            out.append(tok)
    if k != len(counts):
        raise ValueError(f"{len(counts)} images for {k} image placeholders")
    return out, spans


# -- tower ------------------------------------------------------------------------------------------------------

def checkpoint_reader(model_dir: str | Path):
    """``get(name)`` for the vision tensors, read whole from the checkpoint (the rank reader drops ``model.visual.*``)."""

    from safetensors import safe_open

    d = Path(model_dir)
    index = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    opened: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        f = index[name]
        if f not in opened:
            opened[f] = safe_open(str(d / f), framework="pt")
        return opened[f].get_tensor(name)

    return get



def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate) * up


class Tower:
    """The vision tower's weights on one device, run in bf16 (norms and softmax in fp32)."""

    def __init__(self, cfg: VisionConfig, get, device: torch.device) -> None:
        self.cfg, self.dev = cfg, device
        bf = torch.bfloat16

        def t(name: str) -> torch.Tensor:
            return get(PREFIX + name).to(device=device, dtype=bf).contiguous()

        self.patch_w, self.patch_b = t("patch_embed.proj.weight"), t("patch_embed.proj.bias")
        self.blocks = []
        for i in range(cfg.depth):
            b = f"blocks.{i}."
            self.blocks.append({k: t(b + k) for k in (
                "norm1.weight", "norm2.weight", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
                "attn.q_norm.weight", "attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.gate_proj.bias",
                "mlp.up_proj.weight", "mlp.up_proj.bias", "mlp.down_proj.weight", "mlp.down_proj.bias")})
        self.post_norm = t("post_layernorm.weight")
        self.down_w, self.down_b = t("downsample.weight"), t("downsample.bias")
        self.m_proj = t("merger.proj.weight")
        self.m_norm_w, self.m_norm_b = t("merger.post_projection_norm.weight"), t("merger.post_projection_norm.bias")
        self.m_gate, self.m_up, self.m_down = t("merger.gate_proj.weight"), t("merger.up_proj.weight"), t("merger.down_proj.weight")
        head = cfg.hidden // cfg.heads
        rot = head // 2                                                                    # partial rotary factor 0.5
        self.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rot, 2, dtype=torch.float32, device=device) / rot))

    def nbytes(self) -> int:
        total = sum(v.numel() * 2 for b in self.blocks for v in b.values())
        for v in (self.patch_w, self.down_w, self.m_proj, self.m_gate, self.m_up, self.m_down):
            total += v.numel() * 2
        return total

    def _rotary(self, grid: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        t, h, w = grid
        m = self.cfg.merge
        hp = torch.arange(h, device=self.dev).unsqueeze(1).expand(-1, w)
        wp = torch.arange(w, device=self.dev).unsqueeze(0).expand(h, -1)
        hp = hp.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wp = wp.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        pos = torch.stack([hp, wp], dim=-1).repeat(t, 1)                                    # [L, 2]
        freqs = pos.float().unsqueeze(-1) * self.inv_freq                                   # [L, 2, rot/2]
        freqs = freqs.flatten(1)                                                            # [L, rot]: h then w
        return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)

    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Neox-style rotation over the whole head: pairs (i, i + head/2) by [h | w] frequencies."""

        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)                                       # [L, 1, head/2]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    @torch.inference_mode()
    def __call__(self, patches: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """One image's patches [L, C*T*P*P] -> its feature rows [L / merge^2, out_hidden] bf16."""

        cfg = self.cfg
        L = patches.shape[0]
        head = cfg.hidden // cfg.heads
        x = patches.to(self.dev, torch.bfloat16).view(L, 3, cfg.temporal, cfg.patch, cfg.patch)
        x = F.conv3d(x, self.patch_w, self.patch_b, stride=(cfg.temporal, cfg.patch, cfg.patch)).view(L, cfg.hidden)
        cos, sin = self._rotary(grid)
        for b in self.blocks:
            h = _rms(x, b["norm1.weight"], cfg.eps)
            qkv = F.linear(h, b["attn.qkv.weight"], b["attn.qkv.bias"]).view(L, 3, cfg.heads, head)
            q, k, v = qkv.unbind(1)
            q = _rms(q, b["attn.q_norm.weight"], 1e-5)
            k = _rms(k, b["attn.k_norm.weight"], 1e-5)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            x = x + F.linear(o.transpose(0, 1).reshape(L, cfg.hidden), b["attn.proj.weight"], b["attn.proj.bias"])
            h = _rms(x, b["norm2.weight"], cfg.eps)
            g = F.linear(h, b["mlp.gate_proj.weight"], b["mlp.gate_proj.bias"])
            u = F.linear(h, b["mlp.up_proj.weight"], b["mlp.up_proj.bias"])
            x = x + F.linear(_swiglu(g, u, cfg.limit), b["mlp.down_proj.weight"], b["mlp.down_proj.bias"])
        x = _rms(x, self.post_norm, cfg.eps)
        x = x.view(-1, cfg.merge, cfg.merge, cfg.hidden).permute(0, 3, 1, 2)
        x = F.conv2d(x, self.down_w, self.down_b, stride=cfg.merge).view(-1, cfg.out_hidden)
        x = F.linear(x, self.m_proj)
        x = F.gelu(F.layer_norm(x, (cfg.out_hidden,), self.m_norm_w, self.m_norm_b))
        x = F.linear(_swiglu(F.linear(x, self.m_gate), F.linear(x, self.m_up), cfg.limit), self.m_down)
        return x.contiguous()


# -- the server's frontend --------------------------------------------------------------------------------------

@dataclass
class Prepared:
    """A prompt with images: expanded token ids, each image's patches and grid, and its rows' (start, length)."""

    token_ids: list[int]
    patches: list[torch.Tensor]
    grids: list[tuple[int, int, int]]
    spans: list[tuple[int, int]]


class Frontend:
    """What ``tensorfold.server.prompts.prepare_images`` calls: render -> tokens, images -> patches, placeholders -> rows."""

    allow_urls = False
    videos = False

    def __init__(self, model_dir: str | Path, tokenizer=None) -> None:
        self.config = VisionConfig.read(model_dir)
        if tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tokenizer = tokenizer

    def _encode(self, text: str) -> list[int]:
        enc = self.tokenizer.encode(text, add_special_tokens=False) if hasattr(self.tokenizer, "encode") else None
        ids = getattr(enc, "ids", enc)
        return [int(t) for t in ids]

    def prepare(self, text: str, images: list, *, max_prompt_tokens: int | None = None,
                max_visual_tokens: int | None = None, **_: object) -> Prepared:
        ids = self._encode(text)
        patches, grids = [], []
        for image in images:
            p, g = preprocess(self.config, image)
            patches.append(p)
            grids.append(g)
        counts = [image_tokens(self.config, g) for g in grids]
        if max_visual_tokens is not None and sum(counts) > max_visual_tokens:
            raise ValueError(f"the images need {sum(counts)} visual tokens, more than this server's "
                             f"{max_visual_tokens}")
        ids, spans = expand(ids, self.config, counts)
        if max_prompt_tokens is not None and len(ids) >= max_prompt_tokens:
            raise ValueError(f"the prompt with its images is {len(ids)} tokens, past the {max_prompt_tokens}-token "
                             "context window")
        return Prepared(ids, patches, grids, spans)

    def continued(self, prepared: Prepared, ids: list[int]) -> Prepared:
        """The same images under a longer prompt (no positions to extend: the rows stay where they were)."""

        n = len(prepared.token_ids)
        if list(ids[:n]) != list(prepared.token_ids):
            raise ValueError("a continuation must keep its image prompt as a prefix")
        return Prepared(list(ids), prepared.patches, prepared.grids, prepared.spans)
