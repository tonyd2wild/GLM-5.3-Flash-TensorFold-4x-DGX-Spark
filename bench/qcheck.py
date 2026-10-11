#!/usr/bin/env python3
"""qcheck.py URL LABEL OUT.json -- greedy replies to short prompts with known answers; checks each reply ends (finish_reason stop) and is right."""
import json, sys, time, urllib.request

URL, LABEL, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
CASES = [
    ("math", "A train leaves at 9:40 and arrives at 13:15. How many minutes is the trip? Answer with the number only.", ["215"]),
    ("math2", "What is 17 * 23? Answer with the number only.", ["391"]),
    ("json", 'Return only JSON: {"name": string, "age": number} for: "Maria is 34 years old."', ["maria", "34"]),
    ("code", "Write a Python function is_prime(n) that returns True if n is prime. Code only.", ["def is_prime"]),
    ("fr_code", "Écris une fonction Python qui inverse une chaîne de caractères.", ["def ", "[::-1]"]),
    ("fr_short", "Quelle est la capitale du Canada ? Réponds en une phrase.", ["ottawa"]),
    ("es_short", "¿Cuántos días tiene una semana? Responde en una frase.", ["siete", "7"]),
    ("zh_short", "日本的首都是哪里？请用一句话回答。", ["东京"]),
    ("fact", "Which planet is closest to the Sun? One word.", ["mercury"]),
    ("reason", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Answer yes or no, then one sentence why.", ["yes"]),
]
res = {"label": LABEL, "url": URL, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}
ok_all = 0
for name, prompt, want in CASES:
    body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                       "max_tokens": 2048, "reasoning_effort": "low"}).encode()
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", body,
                                                                    {"Content-Type": "application/json"}), timeout=300))
        ch = r["choices"][0]; text = ch["message"].get("content") or ""
        fin = ch.get("finish_reason"); low = text.lower()
        hit = any(w.lower() in low for w in want) if name not in ("fr_code",) else all(w in text for w in want[:1])
        ok = fin == "stop" and hit
        ok_all += ok
        res["cases"][name] = {"ok": ok, "finish": fin, "hit": hit, "tokens": r["usage"]["completion_tokens"],
                              "seconds": round(time.time() - t, 2), "reply": text[:600]}
        print(f"  {name:9s} {'PASS' if ok else 'FAIL'}  finish={fin} hit={hit} toks={r['usage']['completion_tokens']}  {text[:70]!r}", flush=True)
    except Exception as e:
        res["cases"][name] = {"ok": False, "error": str(e)}; print(f"  {name:9s} ERROR {e}", flush=True)
res["passed"] = ok_all; res["total"] = len(CASES)
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"  {LABEL}: {ok_all}/{len(CASES)} pass -> {OUT}")
