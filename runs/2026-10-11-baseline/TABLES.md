### Single stream (decode tok/s, median of 3; time to first token)

| prompt | TensorFold TP4 | TTFT | vLLM knapcio stack (reference) |
|---|---|---|---|
| count to 100 | **110.6** | 0.11 s | 181.72 |
| count to 300 | **106.3** | 0.11 s | 165.65 |
| code | **80.6** | 0.04 s | 120.52 |
| json | **81.4** | 0.10 s | 120.54 |
| sql | **64.9** | 0.04 s | 109.14 |
| tool call ¹ | **82.7** | 0.41 s | 174.02 |
| math | **78.2** | 0.11 s | 141.78 |
| structured (YAML) | **74.9** | 0.08 s | – |
| summary | **55.7** | 0.11 s | 70.59 |
| prose | **46.8** | 0.09 s | 60.98 |
| narrative | **45.7** | 0.10 s | 60.83 |

¹ end to end: the server sends a tool call as one chunk when it completes.

### C1–C6 by prompt type (aggregate tok/s, median of 3 rounds / per-stream decode / TTFT p50 · p90)

| type | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| prose | 45.6 / 46.1 / 0.2·0.2s | 45.2 / 46.0 / 7.7·15.7s | 46.0 / 45.9 / 13.3·30.2s | 46.0 / 46.1 / 22.8·45.5s | 45.3 / 45.6 / 30.1·61.6s | 45.0 / 45.8 / 38.1·74.9s |
| code | 77.3 / 79.5 / 0.3·0.3s | 78.9 / 81.2 / 4.8·9.5s | 79.1 / 82.0 / 9.0·18.0s | 79.2 / 82.7 / 13.1·26.3s | 80.2 / 81.6 / 18.2·35.0s | 78.9 / 80.6 / 22.4·44.8s |
| math | 81.5 / 84.5 / 0.3·0.3s | 77.5 / 81.6 / 3.1·8.1s | 79.4 / 82.3 / 6.4·13.3s | 78.2 / 81.6 / 9.3·18.3s | 78.4 / 80.9 / 11.9·24.1s | 79.0 / 83.0 / 15.3·29.1s |
| json | 81.0 / 83.5 / 0.3·0.3s | 80.3 / 81.2 / 4.6·10.3s | 81.5 / 83.1 / 8.8·18.7s | 79.7 / 82.3 / 14.2·29.1s | 80.3 / 82.2 / 20.1·40.0s | 78.8 / 81.9 / 24.3·46.8s |
| count to 100 | 96.1 / 106.3 / 0.2·0.2s | 95.0 / 105.3 / 1.3·2.4s | 95.2 / 104.9 / 2.3·4.6s | 95.5 / 105.8 / 3.5·6.6s | 95.5 / 104.6 / 4.6·8.8s | 95.4 / 105.5 / 5.4·10.8s |
| structured (YAML) | 71.6 / 78.9 / 0.3·0.3s | 68.5 / 76.9 / 1.5·3.1s | 64.8 / 71.0 / 3.3·6.2s | 67.4 / 74.3 / 4.6·8.9s | 68.4 / 74.7 / 6.0·11.5s | 67.4 / 75.8 / 7.1·13.6s |

### C1–C6, mixed real prompts (aggregate tok/s, median / peak of 3 rounds)

| streams | TensorFold TP4 | TTFT p90 | vLLM knapcio stack (reference) |
|---|---|---|---|
| C1 | 74.62 / **75.41** | 0.24 s | 115.39 / 116.53 |
| C2 | 67.16 / **77.21** | 9.63 s | 103.36 / 152.62 |
| C3 | 56.06 / **75.31** | 27.37 s | 104.51 / 155.8 |
| C4 | 74.46 / **74.7** | 31.02 s | 159.89 / 162.1 |
| C5 | 57.31 / **75.93** | 42.14 s | 134.51 / 176.08 |
| C6 | 61.06 / **66.12** | 47.81 s | 141.17 / 146.47 |

### Cold prefill (salted prompts, one output token)

| prompt | TensorFold TP4 | TTFT | vLLM knapcio stack (reference) |
|---|---|---|---|
| 5,942 tokens | **864 tok/s** | 6.9 s | 2,378 tok/s |
| 29,867 tokens | **894 tok/s** | 33.4 s | 2,798 tok/s |
| 113,910 tokens | **887 tok/s** | 128.4 s | 2,794 tok/s |
| 498,635 tokens | **772 tok/s** | 645.6 s | 2,462 tok/s |
| ~1.8M tokens (past the 500K window) | refused (HTTP 400), as it should be | | |

**Vision check:** PASS in 1.45 s: "The image shows red on the left half and blue on the right half, separated by a vertical line."

