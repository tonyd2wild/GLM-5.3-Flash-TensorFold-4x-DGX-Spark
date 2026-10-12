# TP2 lane: two Sparks, EXL3 with 4-bit dense layers (2026-10-12)

The same checkpoint and engine as the TP4 lane ([`../2026-10-11-exl3-q4-tp4/`](../2026-10-11-exl3-q4-tp4/TABLES.md)) on two
Sparks: Reddie rank 0 (HTTP, weights local) and Spark4 rank 1 (weights over NFS from Reddie), RoCE all-gathers on the
switch. MiaAI-Lab's two-Spark defaults: 4 streams, a 12.5 GiB kept-state cap, a 14.5 GiB reserve; the four-Spark-only
prompt micro-batches, 4,096-row chunks and decode side stream are off. Launch: `TP=2 deploy/serve_exl3.sh`.

| | |
|---|---|
| max context | 500,000 tokens a request (prompt + reply) |
| streams | 4 at once |
| KV pool | 2,033,664 tokens shared (FP8): all 4 streams can hold 500K at once (2,000,064) |
| memory | startup estimate 84.6 of 99.2 GiB a Spark |

Run on a private port (`:8001`); 00:27-01:17 UTC, no preemptions, no failures. The harness runs C1-C6 on every lane: at
C5 and C6 this lane's fifth and sixth requests wait for a stream, which shows as the 18-21 s first token there. Raw:
[`results/`](results/). Charts: [`charts/`](charts/), TP2 against TP4.

**Quality check** ([`bench/qcheck.py`](../../bench/qcheck.py)): 10 / 10
([`results/qcheck-exl3-q4dense-tp2.json`](results/qcheck-exl3-q4dense-tp2.json)).

### Single stream (decode tok/s, median of 3; time to first token)

| prompt | TP2 lane | TTFT | vs TP4 | vLLM knapcio stack (TP4) | TP4 lane |
|---|---|---|---|---|---|
| count to 100 | **109.9** | 0.17 s | 61 % | 181.7 | 180.2 |
| count to 300 | **107.5** | 0.17 s | 62 % | 165.7 | 172.8 |
| code | **82.0** | 0.28 s | 65 % | 120.5 | 127.0 |
| json | **84.1** | 0.17 s | 63 % | 120.5 | 132.7 |
| sql | **72.1** | 0.16 s | 64 % | 109.1 | 113.5 |
| tool call ¹ | **60.3** | 0.36 s | 65 % | 67.9 | 93.2 |
| math | **78.8** | 0.11 s | 65 % | 141.8 | 121.2 |
| structured (YAML) | **73.7** | 0.18 s | 63 % | – | 117.6 |
| summary | **51.5** | 0.26 s | 66 % | 70.6 | 78.5 |
| prose | **44.0** | 0.25 s | 67 % | 61.0 | 65.4 |
| narrative | **44.0** | 0.25 s | 68 % | 60.8 | 64.9 |

¹ end to end: the server sends a tool call as one chunk when it completes.

### C1–C6 by prompt type (aggregate tok/s, median of 3 rounds / per-stream decode / TTFT p50 · p90)

| type | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| prose | **42.6** / 43.6 / 0.40·0.40 s | **61.4** / 31.8 / 0.65·0.78 s | **73.7** / 25.4 / 0.49·0.74 s | **82.4** / 21.7 / 0.76·0.77 s | **70.7** / 21.5 / 0.76·31.09 s | **73.5** / 21.9 / 0.76·33.36 s |
| code | **79.2** / 83.3 / 0.45·0.45 s | **105.2** / 54.9 / 0.51·0.75 s | **127.7** / 44.3 / 0.79·0.79 s | **139.6** / 37.4 / 0.75·0.84 s | **123.0** / 38.3 / 0.59·19.62 s | **130.2** / 38.4 / 0.84·20.24 s |
| math | **71.4** / 75.9 / 0.42·0.43 s | **104.9** / 54.9 / 0.57·0.73 s | **124.3** / 44.6 / 0.56·0.77 s | **147.1** / 39.3 / 0.84·0.87 s | **119.2** / 37.3 / 0.60·15.33 s | **127.9** / 39.0 / 0.60·15.21 s |
| json | **80.3** / 84.5 / 0.50·0.50 s | **113.3** / 59.7 / 0.61·0.89 s | **136.1** / 47.6 / 0.78·0.99 s | **160.2** / 41.9 / 0.69·0.82 s | **138.5** / 44.5 / 0.96·19.04 s | **149.1** / 50.1 / 0.91·19.90 s |
| count to 100 | **101.8** / 112.5 / 0.21·0.21 s | **162.5** / 90.6 / 0.28·0.48 s | **180.2** / 71.0 / 0.31·0.56 s | **238.8** / 67.9 / 0.34·0.57 s | **193.9** / 71.6 / 0.33·3.53 s | **213.2** / 72.1 / 0.34·3.71 s |
| structured (YAML) | **61.0** / 73.6 / 0.50·0.51 s | **102.8** / 62.6 / 0.60·0.61 s | **120.7** / 47.5 / 0.65·1.02 s | **163.7** / 49.2 / 0.69·0.70 s | **114.4** / 43.9 / 0.82·5.34 s | **133.6** / 48.7 / 0.71·5.16 s |

### C1–C6, mixed real prompts (aggregate tok/s, median / peak of 3 rounds)

| streams | TP2 lane | TTFT p90 | vLLM knapcio stack (TP4) | TP4 lane |
|---|---|---|---|---|
| C1 | **77.1** / 80.2 | 0.5 s | 115.4 / 116.5 | 124.0 / 124.2 |
| C2 | **68.3** / 97.0 | 0.9 s | 103.4 / 152.6 | 106.0 / 158.2 |
| C3 | **73.3** / 106.9 | 1.0 s | 104.5 / 155.8 | 117.0 / 174.7 |
| C4 | **108.2** / 109.5 | 0.8 s | 159.9 / 162.1 | 175.8 / 176.6 |
| C5 | **85.8** / 116.3 | 18.0 s | 134.5 / 176.1 | 143.4 / 189.0 |
| C6 | **90.7** / 91.3 | 20.9 s | 141.2 / 146.5 | 151.2 / 166.2 |

### Cold prefill (salted prompts, one output token)

| prompt | TP2 lane | TTFT | vLLM knapcio stack (TP4) | TP4 lane |
|---|---|---|---|---|
| 5,942 tokens | **1,721 tok/s** | 3.5 s | 2,378 tok/s | 2,447 tok/s |
| 29,867 tokens | **1,827 tok/s** | 16.3 s | 2,798 tok/s | 2,780 tok/s |
| 113,910 tokens | **1,781 tok/s** | 64.0 s | 2,794 tok/s | 2,808 tok/s |
| 498,633 tokens | **1,467 tok/s** | 339.8 s | 2,462 tok/s | 2,611 tok/s |

**Vision check:** PASS in 1.58 s: "The image shows red on the left half and blue on the right half."

