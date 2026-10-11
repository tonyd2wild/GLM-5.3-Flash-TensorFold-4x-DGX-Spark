#!/usr/bin/env python3
"""build_report.py TF.json VLLM.json OUTDIR -- chart specs (for make_charts.py) and markdown tables of the TensorFold baseline."""
import json, os, sys

tf = json.load(open(sys.argv[1]))
vl = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else None
out = sys.argv[3] if len(sys.argv) > 3 else "."
os.makedirs(out, exist_ok=True)
TYPES = ["prose", "code", "math", "json", "count100", "structured"]
LABEL = {"count100": "count to 100", "count300": "count to 300", "tooluse": "tool call", "structured": "structured (YAML)"}
levels = [1, 2, 3, 4, 5, 6]


def med(d, key="median"):
    return d.get(key) if isinstance(d, dict) else None


md = []
md.append("### Single stream (decode tok/s, median of 3; time to first token)\n")
md.append("| prompt | TensorFold TP4 | TTFT | vLLM knapcio stack (reference) |")
md.append("|---|---|---|---|")
order = ["count100", "count300", "code", "json", "sql", "tooluse", "math", "structured", "summary", "prose", "narrative"]
suite_cats, suite_vals = [], []
for k in order:
    r = tf.get("single", {}).get(k)
    if not r or "decode_tok_s" not in r:
        continue
    dec = med(r["decode_tok_s"])
    if k == "tooluse":           # the tool call arrives as one chunk at the end: its decode rate is end to end
        dec = med(r["e2e_tok_s"])
    ref = vl["single"].get(k, {}).get("decode_tok_s", {}).get("median") if vl else None
    note = " ¹" if k == "tooluse" else ""
    md.append(f"| {LABEL.get(k, k)}{note} | **{dec:.1f}** | {med(r['ttft_s']):.2f} s | {ref if ref is not None else '–'} |")
    suite_cats.append(LABEL.get(k, k)); suite_vals.append(dec)
md.append("\n¹ end to end: the server sends a tool call as one chunk when it completes.\n")

md.append("### C1–C6 by prompt type (aggregate tok/s, median of 3 rounds / per-stream decode / TTFT p50 · p90)\n")
ts = tf.get("typesweep", {})
md.append("| type | " + " | ".join(f"C{c}" for c in levels) + " |")
md.append("|---|" + "---|" * len(levels))
sweep_series = []
for ty in TYPES:
    row = ts.get(ty, {})
    cells = []
    aggs = []
    for c in levels:
        x = row.get(f"c{c}", {})
        a = med(x.get("agg_tok_s", {})); p = med(x.get("per_stream_decode", {}))
        t50 = med(x.get("ttft_s", {})); t90 = med(x.get("ttft_s", {}), "p90")
        aggs.append(a or 0)
        cells.append(f"{a:.1f} / {p:.1f} / {t50:.1f}·{t90:.1f}s" if a is not None else "–")
    md.append(f"| {LABEL.get(ty, ty)} | " + " | ".join(cells) + " |")
    sweep_series.append((LABEL.get(ty, ty), aggs))

md.append("\n### C1–C6, mixed real prompts (aggregate tok/s, median / peak of 3 rounds)\n")
md.append("| streams | TensorFold TP4 | TTFT p90 | vLLM knapcio stack (reference) |")
md.append("|---|---|---|---|")
mix_tf, mix_vl = [], []
for c in levels:
    x = tf.get("sweep", {}).get(f"c{c}", {})
    y = vl["sweep"].get(f"c{c}", {}) if vl else {}
    a, pk, t90 = med(x.get("agg_tok_s", {})), med(x.get("agg_tok_s", {}), "peak"), med(x.get("ttft_s", {}), "p90")
    ra, rp = med(y.get("agg_tok_s", {})), med(y.get("agg_tok_s", {}), "peak")
    mix_tf.append(a or 0); mix_vl.append(ra or 0)
    md.append(f"| C{c} | {a} / **{pk}** | {t90} s | {ra} / {rp} |")

md.append("\n### Cold prefill (salted prompts, one output token)\n")
md.append("| prompt | TensorFold TP4 | TTFT | vLLM knapcio stack (reference) |")
md.append("|---|---|---|---|")
pre_cats, pre_tf, pre_vl = [], [], []
for size, r in tf.get("prefill", {}).items():
    if "prefill_tok_s" not in r:
        md.append(f"| ~{int(size):,} | failed: {r.get('error')} | | |"); continue
    ref = None
    if vl:
        for rs, rr in vl.get("prefill", {}).items():
            if abs(rr.get("prompt_tokens", 0) - r["prompt_tokens"]) / r["prompt_tokens"] < 0.2:
                ref = rr
    ref_s = f"{ref['prefill_tok_s']:,.0f} tok/s" if ref else "–"
    md.append(f"| {r['prompt_tokens']:,} tokens | **{r['prefill_tok_s']:,.0f} tok/s** | {r['ttft_s']:.1f} s | {ref_s} |")
    if ref:
        pre_cats.append(f"{r['prompt_tokens'] // 1000}K"); pre_tf.append(r["prefill_tok_s"]); pre_vl.append(ref["prefill_tok_s"])

v = tf.get("vision") or {}
md.append(f"\n**Vision check:** {'PASS' if v.get('ok') else 'FAIL'} in {v.get('seconds')} s: \"{v.get('reply', '')}\"\n")
open(os.path.join(out, "tables.md"), "w").write("\n".join(md) + "\n")

charts = {
    "tf-suite": {"kind": "suite", "title": "GLM-5.3-Flash on TensorFold TP4: single-stream decode by prompt",
                 "sub": "tok/s, median of 3, temperature 0, reasoning_effort low, MTP + DFlash2 drafts",
                 "cats": suite_cats, "vals": suite_vals,
                 "note": "4x DGX Spark, derisked lossless8 weights over NFS from Reddie, 500K context, 2026-10-11"},
}
if vl:
    charts["tf-sweep-vs-vllm"] = {"kind": "sweep", "title": "Mixed real prompts, C1-C6: aggregate tok/s (median of 3 rounds)",
                                  "x": [f"C{c}" for c in levels],
                                  "series": [("TensorFold TP4 (one request at a time)", mix_tf), ("vLLM knapcio stack (reference)", mix_vl)],
                                  "note": "TensorFold's GLM CUDA engine serves one request at a time today; vLLM batches"}
    if pre_cats:
        charts["tf-prefill-vs-vllm"] = {"kind": "prefill", "title": "Cold prefill, tok/s",
                                        "cats": pre_cats, "new": pre_tf, "old": pre_vl,
                                        "new_label": "TensorFold TP4", "old_label": "vLLM knapcio stack (reference)",
                                        "note": "salted prompts, one output token"}
json.dump(charts, open(os.path.join(out, "charts.json"), "w"), indent=1)
print("\n".join(md))
