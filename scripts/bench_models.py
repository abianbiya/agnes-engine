#!/usr/bin/env python3
"""Benchmark Ollama models on the real RAG prompt.

Answers one question: how long does an actual Agnes answer take per model?
Reports prefill/decode split so you can tell a slow model from a slow load.

    python scripts/bench_models.py                    # all installed models
    python scripts/bench_models.py gemma3:12b llama3.1:8b
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chat.prompts import RAG_CHAT_TEMPLATE  # noqa: E402
from src.config.settings import get_settings  # noqa: E402

NS = 1e9
BASE = get_settings().llm.ollama_base_url
# A question whose honest answer is long - that is where the seconds go.
QUESTION = "Apa saja fakultas yang ada di UNNES dan jelaskan singkat masing-masing?"
CONTEXT = (Path(__file__).parent / "bench_context.txt").read_text()

# Embedding models cannot answer prompts.
SKIP = ("bge-m3", "nomic-embed-text")


def post(path: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def installed() -> list[str]:
    models = json.load(urllib.request.urlopen(f"{BASE}/api/tags", timeout=30))["models"]
    return [m["name"] for m in models if not m["name"].startswith(SKIP)]


def bench(model: str, prompt: str) -> dict | None:
    try:
        post("/api/generate", {"model": model, "prompt": "hi", "stream": False})  # warm
        t = time.time()
        r = post("/api/generate", {"model": model, "prompt": prompt, "stream": False})
        r["wall"] = time.time() - t
        post("/api/generate", {"model": model, "keep_alive": 0})  # release VRAM
        return r
    except Exception as e:  # noqa: BLE001
        print(f"  {model}: FAILED - {e}")
        return None


def main() -> None:
    models = sys.argv[1:] or installed()
    prompt = RAG_CHAT_TEMPLATE.format(
        context=CONTEXT, chat_history="", question=QUESTION
    )
    print(f"{BASE} | prompt ~{len(prompt) // 4} tok | {len(models)} models\n")

    rows = []
    for m in models:
        print(f"benchmarking {m} ...", flush=True)
        r = bench(m, prompt)
        if not r:
            continue
        eval_s = r.get("eval_duration", 0) / NS
        rows.append(
            {
                "model": m,
                "wall": r["wall"],
                "prefill": r.get("prompt_eval_duration", 0) / NS,
                "decode": eval_s,
                "out_tok": r.get("eval_count", 0),
                "tok_s": r.get("eval_count", 0) / eval_s if eval_s else 0,
                "answer": r["response"].strip(),
            }
        )

    rows.sort(key=lambda x: x["wall"])
    print(f"\n{'model':<24}{'total':>8}{'prefill':>9}{'decode':>8}{'out':>6}{'tok/s':>8}")
    print("-" * 63)
    for r in rows:
        print(
            f"{r['model']:<24}{r['wall']:>7.1f}s{r['prefill']:>8.1f}s"
            f"{r['decode']:>7.1f}s{r['out_tok']:>6}{r['tok_s']:>8.1f}"
        )

    print("\n\n=== answers (judge Bahasa Indonesia quality yourself) ===")
    for r in rows:
        print(f"\n--- {r['model']} ({r['wall']:.1f}s, {r['out_tok']} tok) ---")
        print(r["answer"][:700])

    Path("/tmp/bench_models.json").write_text(json.dumps(rows, indent=2))
    print("\nfull output: /tmp/bench_models.json")


if __name__ == "__main__":
    main()
