"""Measure §7 caching speedup and §9 security pipeline on the local Mac
using MockLLM. The point is to capture REAL numbers from the actual
shipped code, not to fabricate Colab results.

What this validates honestly:
  - The cache implementation (LRU keyed on (question, technique, ...))
    actually returns hits faster than misses, with a measurable ratio.
  - The security test harness correctly routes attacks through the agent,
    splices poisoned evidence when applicable, and applies the regex
    success criteria. (It does NOT validate Phi-3.5's defenses — that
    requires GPU.)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.agent import Agent  # noqa: E402
from src.llm import MockLLM  # noqa: E402
from src.security import run_all, ATTACKS  # noqa: E402

CORPUS = REPO / "data" / "corpus" / "corpus.jsonl"

print("=" * 72)
print("§7 PROMPT CACHE — real measurement (MockLLM with 50ms simulated latency)")
print("=" * 72)

agent_cache = Agent(CORPUS, enable_web=False, cache_size=128)
mock = MockLLM(latency_ms=50)

QUESTION = "What does the bible say about Babylon as a symbol?"
TECHNIQUE = "zero_shot"

t0 = time.perf_counter()
a1 = agent_cache.answer(QUESTION, model=mock, technique=TECHNIQUE)
t_miss = time.perf_counter() - t0

t0 = time.perf_counter()
a2 = agent_cache.answer(QUESTION, model=mock, technique=TECHNIQUE)
t_hit = time.perf_counter() - t0

speedup = t_miss / max(t_hit, 1e-9)
stats = agent_cache.cache_stats()

print(f"  cache miss latency:  {t_miss*1000:>8.2f} ms")
print(f"  cache hit  latency:  {t_hit*1000:>8.4f} ms")
print(f"  speedup:             {speedup:>8.0f}x")
print(f"  hits / misses:       {stats['hits']} / {stats['misses']}")
print(f"  hit rate:            {stats['hit_rate']*100:.0f}%")

cache_result = {
    "miss_ms": round(t_miss * 1000, 4),
    "hit_ms": round(t_hit * 1000, 6),
    "speedup_x": int(speedup),
    "hits": stats["hits"],
    "misses": stats["misses"],
    "hit_rate": stats["hit_rate"],
    "note": (
        "Measured locally with MockLLM (deterministic stand-in, 50ms "
        "simulated generation latency). The cache implementation under test "
        "is the same one used in Colab (src/agent.py LRU). The absolute "
        "miss latency is small here because MockLLM doesn't actually call "
        "a transformer; the SPEEDUP RATIO is what generalizes — a GPU "
        "miss at ~5s would yield a similar ~5,000-50,000x speedup on the "
        "cache hit, which is sub-millisecond."
    ),
}

print()
print("=" * 72)
print("§9 SECURITY TESTS — pipeline validation against 5 attacks (MockLLM)")
print("=" * 72)

agent_sec = Agent(CORPUS, enable_web=False)
mock_sec = MockLLM(latency_ms=10)
results = run_all(agent_sec, mock_sec)
defended = sum(1 for r in results if not r.succeeded)

print(f"  attacks defended: {defended}/{len(results)}")
print()
for r in results:
    verdict = "DEFENDED" if not r.succeeded else "SUCCEEDED (BAD)"
    print(f"  [{r.attack.name:<32s}] {verdict:<15s} ({r.latency_s*1000:.1f}ms)")
print()
print("  CAVEAT: MockLLM returns a fixed stub answer regardless of input,")
print("  so it cannot be jailbroken by definition. This result validates")
print("  that the agent.gather() + poisoned-evidence splicing + regex")
print("  judging pipeline works end-to-end. Actual Phi-3.5 defense rates")
print("  require GPU execution of notebook §9.")

security_result = {
    "n_attacks": len(results),
    "defended": defended,
    "attacks": [
        {
            "name": r.attack.name,
            "category": r.attack.category,
            "defended": not r.succeeded,
            "latency_ms": round(r.latency_s * 1000, 2),
        }
        for r in results
    ],
    "note": (
        "MockLLM returns a constant stub answer, so the regex success "
        "patterns cannot match. Result: 5/5 'defended' is a pipeline "
        "validation, not a real model defense measurement. The attacks "
        "and judging logic are in src/security.py and are wired through "
        "agent.gather() with poisoned-evidence splicing for attack #3."
    ),
}

print()
out = REPO / "data" / "local_caching_security.json"
out.write_text(json.dumps(
    {"cache": cache_result, "security": security_result}, indent=2,
))
print(f"saved {out.relative_to(REPO)}")
