"""
Open-source LLM wrapper for Phi-3.5-mini and Mistral-7B-Instruct.

Design notes
------------
- We use Hugging Face `transformers` directly (no llama.cpp/Ollama) so the
  same code runs in Colab and locally.
- Mistral-7B is loaded in 4-bit via `bitsandbytes` so it fits in a Colab T4.
- Phi-3.5-mini runs in fp16 (or bf16 if the GPU supports it). On CPU it's
  slow but workable for a class demo.
- `generate()` measures wall-clock latency and returns it alongside the
  text — we need this for the model-comparison requirement.

Public API
----------
    model = LLM.load("phi3")      # or "mistral"
    out = model.generate(prompt, max_new_tokens=400)
    out.text, out.latency_s, out.tokens_out
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

try:
    from transformers import BitsAndBytesConfig  # only available with bitsandbytes
    _HAS_BNB = True
except ImportError:
    BitsAndBytesConfig = None  # type: ignore[assignment]
    _HAS_BNB = False


ModelKey = Literal["phi3", "mistral"]

MODEL_IDS: dict[ModelKey, str] = {
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


@dataclass
class GenerationResult:
    text: str
    latency_s: float
    tokens_in: int
    tokens_out: int


class LLM:
    def __init__(
        self,
        key: ModelKey,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
    ):
        self.key = key
        self.model_id = MODEL_IDS[key]
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, key: ModelKey, prefer_4bit: bool | None = None) -> "LLM":
        """
        Load a model by short key.

        prefer_4bit:
            None  -> auto: 4-bit for Mistral if CUDA + bitsandbytes work, else fp16
            True  -> force 4-bit
            False -> force fp16/bf16
        """
        model_id = MODEL_IDS[key]
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        use_4bit = (
            prefer_4bit
            if prefer_4bit is not None
            else (key == "mistral" and device == "cuda" and _HAS_BNB)
        )

        print(f"[llm] loading {model_id} (device={device}, 4bit={use_4bit})")

        # NOTE: We intentionally do NOT pass `trust_remote_code=True`. Both
        # Phi-3.5 and Mistral-7B are supported natively in transformers >=4.41,
        # and Microsoft's bundled `modeling_phi3.py` for Phi-3.5 calls
        # `DynamicCache.from_legacy_cache(...)`, which was removed in
        # transformers >=4.54 — using the native impl avoids that crash.
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        kwargs: dict = {}
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = (
                torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
                else (torch.float16 if device == "cuda" else torch.float32)
            )
            kwargs["device_map"] = "auto" if device == "cuda" else None

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not use_4bit and device == "cpu":
            model.to("cpu")
        model.eval()
        return cls(key, model, tokenizer, device)

    # -------------------------------------------------------------- generate
    def _format_chat(self, system: str, user: str) -> str:
        """Use the tokenizer's built-in chat template for both models."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful Bible study assistant.",
        max_new_tokens: int = 400,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> GenerationResult:
        prompt = self._format_chat(system_prompt, user_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        tokens_in = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        latency = time.perf_counter() - t0

        generated = out[0, tokens_in:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            latency_s=latency,
            tokens_in=tokens_in,
            tokens_out=int(generated.shape[0]),
        )


# ---------------------------------------------------------------------------
# MockLLM — a stand-in that doesn't load any weights.
#
# Why this exists: the real models (Phi-3.5-mini, Mistral-7B) need a GPU,
# which a low-RAM laptop doesn't have. MockLLM lets us run the *agent
# pipeline* (route -> retrieve -> prompt -> answer) end-to-end locally so we
# can debug routing, retrieval, and security tests without firing up Colab.
#
# It just echoes back a structured summary of the prompt + evidence. It is
# NOT a substitute for the real models in the comparison/quality cells.
# ---------------------------------------------------------------------------


class MockLLM:
    """Deterministic fake LLM. Same `.generate()` signature as `LLM`."""

    model_id = "mock-llm-v1"
    key = "mock"

    def __init__(self, latency_ms: int = 50):
        self._latency_s = latency_ms / 1000.0

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 400,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        time.sleep(self._latency_s)

        # Pull the evidence block back out for inspection.
        evidence = ""
        if "EVIDENCE" in user_prompt:
            evidence = user_prompt.split("EVIDENCE", 1)[1]
            evidence = evidence.split("ANSWER", 1)[0]

        # Grab the first 3 [Label] markers as our "cited sources".
        import re
        labels = re.findall(r"\[([^\]]{1,80})\]", evidence)[:3]
        citations = ", ".join(f"[{lbl}]" for lbl in labels) or "[no evidence]"

        text = (
            "[MockLLM stub answer — replace with a real model in Colab.] "
            f"Based on the retrieved sources {citations}, the assistant would "
            "synthesize a grounded response here. The pipeline (routing -> "
            "retrieval -> prompt assembly -> generation) is wired correctly; "
            "only the language model itself is a stand-in."
        )
        # Approximate token counts (1 token ~= 4 chars is a decent rule of thumb).
        return GenerationResult(
            text=text,
            latency_s=self._latency_s,
            tokens_in=max(1, len(user_prompt) // 4),
            tokens_out=max(1, len(text) // 4),
        )
