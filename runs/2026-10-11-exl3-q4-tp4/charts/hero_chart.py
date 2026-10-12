#!/usr/bin/env python3
"""hero_chart.py TF.json VLLM.json OUTDIR -- one 1200x675 comparison image (light + dark SVG): mixed prompts C1-C6,
cold prefill and single-stream decode, TensorFold against vLLM, every pair labelled with its difference."""
import html, json, os, sys

tf, vl, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
os.makedirs(out, exist_ok=True)
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  tf="#2a78d6", vl="#eb6834", up="#0f7a3d", down="#b3261e"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 tf="#3987e5", vl="#d95926", up="#4cc38a", down="#f07167"),
}
FONT = 'font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"'


def T(x, y, s, t, size=12, fill=None, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" fill="{fill or t["ink2"]}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(s))}</text>')


def nice(v):
    for step in (25, 50, 100, 250, 500, 1000):
        n = -(-v // step)
        if n <= 5:
            return step * n, int(n)
    return v, 4


def bars(o, t, x0, y0, w, h, title, cats, a, b, fmt):
    """grouped bars: a = TensorFold, b = vLLM; the difference under each category."""
    o.append(T(x0, y0 - 14, title, t, 15, t["ink"], weight="600"))
    top = y0 + 8; ch = h - 62
    ymax, n = nice(max(a + b) * 1.12)
    for g in range(n + 1):
        gy = top + ch - ch * g / n
        o.append(f'<line x1="{x0+40}" x2="{x0+w}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="{t["grid"] if g else t["axis"]}" stroke-width="1"/>')
        o.append(T(x0 + 34, gy + 4, f"{int(ymax * g / n):,}", t, 10, t["muted"], "end"))
    slot = (w - 44) / len(cats); bw = min(26, slot / 3.2); gap = max(2, min(12, slot - 2 * bw - 8))
    for i, c in enumerate(cats):
        cx = x0 + 44 + slot * (i + 0.5)
        for k, (v, col) in enumerate(((a[i], t["tf"]), (b[i], t["vl"]))):
            x = cx - bw - gap / 2 + k * (bw + gap); bh = ch * v / ymax; y = top + ch - bh
            o.append(f'<path d="M{x:.1f},{top+ch:.1f} V{y+3:.1f} Q{x:.1f},{y:.1f} {x+3:.1f},{y:.1f} H{x+bw-3:.1f} '
                     f'Q{x+bw:.1f},{y:.1f} {x+bw:.1f},{y+3:.1f} V{top+ch:.1f} Z" fill="{col}"/>')
            o.append(T(x + bw / 2, y - 5, fmt(v), t, 10, t["ink"] if k == 0 else t["ink2"], "middle", "600" if k == 0 else "400"))
        o.append(T(cx, top + ch + 17, c, t, 11, t["ink2"], "middle"))
        d = 100 * (a[i] / b[i] - 1)
        o.append(T(cx, top + ch + 34, f"{d:+.0f}%", t, 12, t["up"] if d >= 0.5 else (t["down"] if d <= -0.5 else t["ink2"]), "middle", "600"))


def med(d, key="median"):
    return d[key]


lv = [1, 2, 3, 4, 5, 6]
mix_tf = [med(tf["sweep"][f"c{c}"]["agg_tok_s"]) for c in lv]
mix_vl = [med(vl["sweep"][f"c{c}"]["agg_tok_s"]) for c in lv]
pre_k = list(tf["prefill"])
pre_tf = [tf["prefill"][k]["prefill_tok_s"] for k in pre_k]
pre_vl = [vl["prefill"][k]["prefill_tok_s"] for k in pre_k]
pre_c = [f"{tf['prefill'][k]['prompt_tokens'] // 1000}K" for k in pre_k]
order = [("code", "code"), ("json", "json"), ("sql", "sql"), ("prose", "prose"), ("narrative", "story"),
         ("summary", "summary"), ("count300", "count 300"), ("count100", "count 100"), ("math", "math")]
s_c = [lab for _, lab in order]
s_tf = [med(tf["single"][k]["decode_tok_s"]) for k, _ in order]
s_vl = [med(vl["single"][k]["decode_tok_s"]) for k, _ in order]

for mode, t in THEME.items():
    W, H = 1200, 675
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="GLM-5.3-Flash on 4x DGX Spark: TensorFold against vLLM">',
         f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>']
    o.append(T(32, 42, "GLM-5.3-Flash on 4× DGX Spark: TensorFold vs vLLM", t, 24, t["ink"], weight="700"))
    o.append(f'<rect x="32" y="58" width="13" height="13" rx="3" fill="{t["tf"]}"/>')
    o.append(T(51, 69, "TensorFold (EXL3, q4 dense, 8 streams)", t, 13, t["ink2"]))
    o.append(f'<rect x="318" y="58" width="13" height="13" rx="3" fill="{t["vl"]}"/>')
    o.append(T(337, 69, "vLLM (NVFP4)", t, 13, t["ink2"]))
    o.append(T(W - 32, 69, "same fleet, same harness, temperature 0, 500K context", t, 12, t["muted"], "end"))
    bars(o, t, 32, 122, 600, 250, "Mixed real prompts, C1–C6: aggregate tok/s", [f"C{c}" for c in lv], mix_tf, mix_vl,
         lambda v: f"{v:.0f}")
    bars(o, t, 672, 122, 496, 250, "Cold prefill: tok/s", pre_c, pre_tf, pre_vl, lambda v: f"{v/1000:.2f}k")
    bars(o, t, 32, 410, 1136, 236, "One request at a time: decode tok/s", s_c, s_tf, s_vl, lambda v: f"{v:.0f}")
    o.append(T(32, H - 12, "TP4 over RoCE, FP8 KV, reasoning_effort low. vLLM: this fleet's knapcio stack on nvidia/GLM-5.3-Flash-NVFP4.  "
               "github.com/tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark", t, 11, t["muted"]))
    o.append("</svg>")
    open(os.path.join(out, f"tf-vs-vllm-{mode}.svg"), "w").write("\n".join(o))
    print("wrote", f"tf-vs-vllm-{mode}.svg")
