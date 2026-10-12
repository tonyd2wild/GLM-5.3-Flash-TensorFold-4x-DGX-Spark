#!/usr/bin/env python3
"""build_report_exl3.py EXL3.json NVFP4.json VLLM.json OUTDIR -- tables and chart specs (for make_charts.py) of the
multi-stream engine on the EXL3 checkpoint, against the same engine on our NVFP4 derisked pack and the vLLM knapcio stack."""
import json, os, sys

mu = json.load(open(sys.argv[1]))
sg = json.load(open(sys.argv[2]))
vl = json.load(open(sys.argv[3]))
out = sys.argv[4]
os.makedirs(out, exist_ok=True)
TYPES = ["prose", "code", "math", "json", "count100", "structured"]
LABEL = {"count100": "count to 100", "count300": "count to 300", "tooluse": "tool call", "structured": "structured (YAML)"}
SHORT = {"count100": "count 100", "count300": "count 300", "tooluse": "tool call", "structured": "YAML"}
levels = [1, 2, 3, 4, 5, 6]


def med(d, key="median"):
    return d.get(key) if isinstance(d, dict) else None


def f1(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "–"


def single_dec(run, k):
    r = (run.get("single") or {}).get(k)
    if not r or "decode_tok_s" not in r:
        return None
    return med(r["e2e_tok_s"] if k == "tooluse" else r["decode_tok_s"])   # a tool call arrives as one chunk


md = ["### Single stream (decode tok/s, median of 3; time to first token)\n",
      "| prompt | EXL3 | TTFT | vs vLLM | NVFP4 derisked | vLLM knapcio stack |",
      "|---|---|---|---|---|---|"]
order = ["count100", "count300", "code", "json", "sql", "tooluse", "math", "structured", "summary", "prose", "narrative"]
s_cats, s_mu, s_vl = [], [], []
for k in order:
    m = single_dec(mu, k)
    if m is None:
        continue
    v, s = single_dec(vl, k), single_dec(sg, k)
    ttft = med(mu["single"][k]["ttft_s"])
    rel = f"{100 * m / v:.0f} %" if v else "–"
    note = " ¹" if k == "tooluse" else ""
    md.append(f"| {LABEL.get(k, k)}{note} | **{m:.1f}** | {ttft:.2f} s | {rel} | {f1(s)} | {f1(v)} |")
    if v:
        s_cats.append(SHORT.get(k, k)); s_mu.append(m); s_vl.append(v)
md.append("\n¹ end to end: the server sends a tool call as one chunk when it completes.\n")

md.append("### C1–C6 by prompt type (aggregate tok/s, median of 3 rounds / per-stream decode / TTFT p50 · p90)\n")
md.append("| type | " + " | ".join(f"C{c}" for c in levels) + " |")
md.append("|---|" + "---|" * len(levels))
ty_mu, ty_sg = {}, {}
for ty in TYPES:
    cells, a_mu, a_sg = [], [], []
    for c in levels:
        x = (mu.get("typesweep") or {}).get(ty, {}).get(f"c{c}", {})
        y = (sg.get("typesweep") or {}).get(ty, {}).get(f"c{c}", {})
        a, p = med(x.get("agg_tok_s", {})), med(x.get("per_stream_decode", {}))
        t50, t90 = med(x.get("ttft_s", {})), med(x.get("ttft_s", {}), "p90")
        a_mu.append(a or 0); a_sg.append(med(y.get("agg_tok_s", {})) or 0)
        cells.append(f"**{a:.1f}** / {p:.1f} / {t50:.2f}·{t90:.2f} s" if a is not None else "–")
    md.append(f"| {LABEL.get(ty, ty)} | " + " | ".join(cells) + " |")
    ty_mu[SHORT.get(ty, ty)] = a_mu; ty_sg[SHORT.get(ty, ty)] = a_sg

md.append("\n### C1–C6, mixed real prompts (aggregate tok/s, median / peak of 3 rounds)\n")
md.append("| streams | EXL3 | TTFT p90 | NVFP4 derisked | vLLM knapcio stack |")
md.append("|---|---|---|---|---|")
mix_mu, mix_vl = [], []
for c in levels:
    x, y, z = (mu.get("sweep") or {}).get(f"c{c}", {}), vl["sweep"].get(f"c{c}", {}), (sg.get("sweep") or {}).get(f"c{c}", {})
    a, pk, t90 = med(x.get("agg_tok_s", {})), med(x.get("agg_tok_s", {}), "peak"), med(x.get("ttft_s", {}), "p90")
    ra, rp = med(y.get("agg_tok_s", {})), med(y.get("agg_tok_s", {}), "peak")
    sa, sp = med(z.get("agg_tok_s", {})), med(z.get("agg_tok_s", {}), "peak")
    mix_mu.append(a or 0); mix_vl.append(ra or 0)
    md.append(f"| C{c} | **{f1(a)}** / {f1(pk)} | {f1(t90)} s | {f1(sa)} / {f1(sp)} | {f1(ra)} / {f1(rp)} |")

md.append("\n### Cold prefill (salted prompts, one output token)\n")
md.append("| prompt | EXL3 | TTFT | NVFP4 derisked | vLLM knapcio stack |")
md.append("|---|---|---|---|---|")
p_cats, p_mu, p_vl = [], [], []


def near(run, n):
    for rr in (run.get("prefill") or {}).values():
        if rr.get("prompt_tokens") and abs(rr["prompt_tokens"] - n) / n < 0.2:
            return rr
    return None


for size, r in (mu.get("prefill") or {}).items():
    if "prefill_tok_s" not in r:
        md.append(f"| ~{int(size):,} filler items | failed: {r.get('error')} | | | |"); continue
    v, s = near(vl, r["prompt_tokens"]), near(sg, r["prompt_tokens"])
    md.append(f"| {r['prompt_tokens']:,} tokens | **{r['prefill_tok_s']:,.0f} tok/s** | {r['ttft_s']:.1f} s | "
              f"{s['prefill_tok_s']:,.0f} tok/s | {v['prefill_tok_s']:,.0f} tok/s |" if v and s else
              f"| {r['prompt_tokens']:,} tokens | **{r['prefill_tok_s']:,.0f} tok/s** | {r['ttft_s']:.1f} s | – | – |")
    if v:
        p_cats.append(f"{r['prompt_tokens'] // 1000}K"); p_mu.append(r["prefill_tok_s"]); p_vl.append(v["prefill_tok_s"])

v = mu.get("vision") or {}
md.append(f"\n**Vision check:** {'PASS' if v.get('ok') else 'FAIL'} in {v.get('seconds')} s: \"{v.get('reply', '')}\"\n")
open(os.path.join(out, "tables.md"), "w").write("\n".join(md) + "\n")

NOTE = "4x DGX Spark TP4, EXL3 tr3 4bpw (brandonmusic) over NFS, dense FP8, FP8 KV, 500K context, effort low, temperature 0, 2026-10-11"
charts = {
    "tfx-single-vs-vllm": {"kind": "prefill", "title": "Single stream: decode tok/s by prompt (median of 3)",
                           "cats": s_cats, "new": s_mu, "old": s_vl,
                           "new_label": "TensorFold, EXL3", "old_label": "vLLM knapcio stack (reference)",
                           "note": NOTE},
    "tfx-sweep-vs-vllm": {"kind": "sweep", "title": "Mixed real prompts, C1-C6: aggregate tok/s (median of 3 rounds)",
                          "x": [f"C{c}" for c in levels],
                          "series": [("TensorFold, EXL3", mix_mu), ("vLLM knapcio stack (reference)", mix_vl)],
                          "note": NOTE},
    "tfx-types-vs-nvfp4": {"kind": "decode", "title": "C1-C6 by prompt type: aggregate tok/s, EXL3 vs NVFP4 derisked (same engine)",
                            "types": list(ty_mu), "levels": levels, "new": ty_mu, "old": ty_sg,
                            "new_label": "EXL3 tr3 4bpw", "old_label": "NVFP4 derisked lossless8",
                            "note": NOTE},
}
if p_cats:
    charts["tfx-prefill-vs-vllm"] = {"kind": "prefill", "title": "Cold prefill, tok/s", "cats": p_cats, "new": p_mu,
                                     "old": p_vl, "new_label": "TensorFold, EXL3",
                                     "old_label": "vLLM knapcio stack (reference)", "note": "salted prompts, one output token"}
json.dump(charts, open(os.path.join(out, "charts.json"), "w"), indent=1)
print("\n".join(md))
