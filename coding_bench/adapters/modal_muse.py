"""Muse Glimmer 30B, self hosted on a Modal GPU.

Modal earns its keep here because the weights have to run somewhere with an
H100. The two Groq models are plain HTTP clients, no container.

Serving choice: transformers, not vLLM. As of vLLM 0.27.1 the muse_glimmer
architecture is not in its model registry, while transformers has shipped it
natively since v5.15.0. Revisit once vLLM registers it, because throughput on a
600 note run is the whole cost of this adapter.

The model is multimodal, and we only ever send it text. Its own card recommends
temperature 1.0, but a benchmark that cannot be rerun to the same number is not
a benchmark, so this defaults to greedy decoding and records the deviation.

Deploy once, then run evaluations against it:

    modal deploy coding_bench/adapters/modal_muse.py
"""

from __future__ import annotations

import os
import time

import modal

# coding_bench is deliberately not imported at module level. Modal imports this
# file inside the GPU container, where the image carries only torch and
# transformers, and the container side of this adapter has no need for the
# package: MuseGlimmer returns plain dicts. Only MuseClient, which runs on the
# caller, needs the shared types, so it imports them when it is used.

MODEL_ID = "meta-models/Muse-Glimmer-30B"
MODEL_REVISION = "main"

# Bump when the image or the generation contract changes, so run records point
# at the exact server that produced them.
APP_NAME = "coding-bench-muse-v1"

CACHE_DIR = "/cache"

app = modal.App(APP_NAME)

model_cache = modal.Volume.from_name("muse-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.0",
        # Muse Glimmer is multimodal, so AutoProcessor builds an image
        # processor and refuses to load without torchvision, even for the
        # text-only prompting this benchmark does.
        "torchvision==0.24.0",
        "transformers>=5.15.0",
        "accelerate>=1.0.0",
        "huggingface_hub[hf_transfer]>=0.30.0",
        "pillow",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.cls(
    image=image,
    gpu="H100",
    volumes={CACHE_DIR: model_cache},
    timeout=3600,
    # Idle H100 time is pure cost. The container reloads in about 70 seconds,
    # which is cheap next to holding a GPU open between runs.
    scaledown_window=90,
)
class MuseGlimmer:
    """One warm copy of the model, reused across every note in a run."""

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading {MODEL_ID}", flush=True)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        print(f"Loaded in {time.perf_counter() - started:.0f}s", flush=True)

    @modal.method()
    def warm(self) -> dict:
        return {"status": "warm", "model": MODEL_ID}

    @modal.method()
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        reasoning_strength: str = "high",
    ) -> dict:
        """Generate once. Returns a stop reason the caller can act on."""
        import torch

        if reasoning_strength:
            system = f"Reasoning strength: {reasoning_strength}\n\n{system}"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        sampling = (
            {"do_sample": False}
            if temperature == 0.0
            else {"do_sample": True, "temperature": temperature, "top_p": 0.95, "top_k": 64}
        )

        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **sampling,
            )
        elapsed = time.perf_counter() - started

        new_tokens = generated[0][prompt_tokens:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True)

        # Hitting the token limit without an end of sequence token means the
        # answer was cut off. That has to reach the caller as a truncation, not
        # as a short answer, so the two are never scored the same way.
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
        ended_cleanly = bool(len(new_tokens)) and int(new_tokens[-1]) in eos_ids
        hit_limit = int(len(new_tokens)) >= max_tokens
        stop_reason = "length" if (hit_limit and not ended_cleanly) else "stop"

        return {
            "text": text,
            "stop_reason": stop_reason,
            "latency_s": elapsed,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(len(new_tokens)),
            },
        }


class MuseClient:
    """Client side handle that satisfies the LLMClient protocol.

    Works from a laptop and from inside another Modal function, which is what
    eval_remote.py needs so that note text and weights stay in the same place.
    """

    model_id = MODEL_ID

    def __init__(
        self,
        temperature: float = 0.0,
        reasoning_strength: str = "high",
        app_name: str = APP_NAME,
    ):
        self.temperature = temperature
        self.reasoning_strength = reasoning_strength
        self._remote = modal.Cls.from_name(app_name, "MuseGlimmer")()

    def complete(self, system: str, user: str, max_tokens: int):
        from coding_bench.approaches.base import Completion, Truncated

        result = self._remote.complete.remote(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=self.temperature,
            reasoning_strength=self.reasoning_strength,
        )
        if result["stop_reason"] == "length":
            raise Truncated(
                self.model_id, "length", produced_tokens=result["usage"]["completion_tokens"]
            )
        return Completion(
            text=result["text"],
            stop_reason=result["stop_reason"],
            latency_s=result["latency_s"],
            usage=result["usage"],
        )


@app.local_entrypoint()
def smoke(prompt: str = "Reply with the JSON {\"ok\": true} and nothing else."):
    """Check the deployment answers before spending a run on it."""
    model = MuseGlimmer()
    print(model.warm.remote())
    result = model.complete.remote(system="You answer with JSON only.", user=prompt, max_tokens=256)
    print(f"stop_reason={result['stop_reason']} latency={result['latency_s']:.1f}s")
    print(result["text"][:500])
