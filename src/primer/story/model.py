"""The real story implementation: a small pretrained instruct model behind `LLMStoryWriter`.

Weights load lazily, never at import time: the serving layer imports every area, and a test that
imports this module must not pull a checkpoint off disk or import torch. `torch` and `transformers`
come from the `story` extra (`pip install -e ".[story]"`), which CI does not install.

For this submission the models are small pretrained instruct checkpoints run on the member's Mac
(MPS), pinned by revision in `configs/story_grid.yaml`; the 4B candidates and QLoRA are later work
(plan Decisions, 2026-11-15).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from primer.story.interface import Story, StoryRequest
from primer.story.prompts import Message
from primer.story.writer import GenerationSettings, LLMStoryWriter


@dataclass(frozen=True)
class ModelSpec:
    """One candidate, pinned: a rerun downloads exactly the weights a number was measured on."""

    name: str
    repo: str
    revision: str
    license: str


def pick_device(torch: Any) -> str:
    """MPS on Apple silicon, CUDA where present, otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class HFChatBackend:
    """A Hugging Face causal LM with a chat template, loaded on first use."""

    def __init__(self, spec: ModelSpec, device: str | None = None) -> None:
        self.spec = spec
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device or pick_device(torch)
        # fp16 on MPS/CUDA halves memory (1.5B is ~3 GB in fp16); fp32 on CPU, where fp16 is slow.
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.spec.repo, revision=self.spec.revision)
        # Typed as Any like `self._model`: with transformers installed, its stubs mistype `.to(device)`,
        # and CI (which does not install it) never sees them.
        model: Any = AutoModelForCausalLM.from_pretrained(self.spec.repo, revision=self.spec.revision, dtype=dtype)
        model.to(device).eval()
        self.device, self._tokenizer, self._model = device, tokenizer, model

    def generate(self, messages: list[Message], settings: GenerationSettings) -> str:
        if self._model is None:
            self._load()
        import torch

        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        torch.manual_seed(settings.seed)
        sample = settings.temperature > 0
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=sample,
                temperature=settings.temperature if sample else None,
                top_p=settings.top_p if sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True))


class StoryWriterModel:
    """The production story implementation. Satisfies ``interface.StoryWriter``."""

    def __init__(self, spec: ModelSpec, settings: GenerationSettings | None = None, max_attempts: int = 3) -> None:
        self._writer = LLMStoryWriter(
            backend=HFChatBackend(spec), settings=settings or GenerationSettings(), max_attempts=max_attempts
        )

    def write(self, request: StoryRequest) -> Story:
        return self._writer.write(request)
