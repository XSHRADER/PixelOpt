"""Loading the models, and sharing one 8 GB GPU between them.

Four models are involved and they do not fit together. The two small ones --
Florence-2 for names, SAM 2.1 for outlines -- share the GPU while the user is
looking around. Before the fill model runs they are moved to main memory, and
so is anything else: the fill model wants nearly the whole card. Ollama, a
separate process, is handled the same way from the other side (see
reason.suggest_fill and Hub.rest).

Nothing here is imported until it is needed. `availability()` only asks
whether the libraries could be imported, so a machine without the vision
extra pays nothing and gets a clear answer.
"""

from __future__ import annotations

import gc
import importlib.util
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .detect import Detector
from .inpaint import work_size
from .segment import Segmenter
from .types import Box, Found, Scene

FLORENCE = "florence-community/Florence-2-large"
SAM = "facebook/sam2.1-hiera-large"
SDXL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SD15 = "stable-diffusion-v1-5/stable-diffusion-inpainting"

LIBRARIES = ("torch", "torchvision", "transformers", "diffusers", "accelerate")

# Tried in this order unless one is asked for; the other is the fallback when
# the first runs out of memory. Measured on an 8 GB RTX 4060: SDXL takes about
# 25 s a fill and peaks at 5.4 GB; SD 1.5 takes about 6 s and peaks at 2.5 GB.
FILL_MODELS = (SDXL, SD15)
LABELS = {SDXL: "SDXL inpainting", SD15: "Stable Diffusion 1.5 inpainting"}
# The resolution each model was trained at; it paints best there.
NATIVE_SIZE = {SDXL: 1024, SD15: 512}
STEPS = {SDXL: 25, SD15: 30}
GUIDANCE = {SDXL: 8.0, SD15: 7.5}
# SDXL's inpainting checkpoint asks for a strength just under 1.
STRENGTH = {SDXL: 0.99, SD15: 1.0}
NEGATIVE = "blurry, low quality, distorted, watermark, text, frame"


@dataclass(frozen=True)
class Availability:
    """What this machine can do for the Objects page."""

    installed: bool
    cuda: bool
    gpu: str
    vram_gb: float
    missing: Tuple[str, ...]


def availability() -> Availability:
    """Whether the vision libraries are installed and a CUDA GPU is usable.

    Costs nothing when the libraries are absent: it looks them up without
    importing them.
    """
    missing = tuple(name for name in LIBRARIES if importlib.util.find_spec(name) is None)
    if missing:
        return Availability(False, False, "", 0.0, missing)
    import torch

    if not torch.cuda.is_available():
        return Availability(True, False, "", 0.0, ())
    card = torch.cuda.get_device_properties(0)
    return Availability(True, True, str(card.name), card.total_memory / 2 ** 30, ())


def fill_order(prefer: Optional[str] = None) -> Tuple[str, ...]:
    """The fill models to try: the preferred one first, the rest as fallbacks."""
    if prefer not in FILL_MODELS:
        return FILL_MODELS
    return (prefer,) + tuple(name for name in FILL_MODELS if name != prefer)


class Hub:
    """Every model, loaded once, with the GPU given to whichever is working."""

    def __init__(self) -> None:
        self.status = availability()
        self.device = "cuda" if self.status.cuda else "cpu"
        self.painted_with = ""
        self._detector = None
        self._segmenter = None
        self._pipelines: Dict[str, object] = {}
        # Streamlit reruns can overlap; two models racing for one GPU cannot.
        self._lock = threading.RLock()

    # ------------------------------------------------------------- small models

    def _detector_ready(self) -> Detector:
        if self._detector is None:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = Florence2ForConditionalGeneration.from_pretrained(FLORENCE, dtype=dtype)
            self._detector = Detector(model.eval(), AutoProcessor.from_pretrained(FLORENCE))
        self._detector.model.to(self.device)
        return self._detector

    def _segmenter_ready(self) -> Segmenter:
        if self._segmenter is None:
            from transformers import Sam2Model, Sam2Processor

            model = Sam2Model.from_pretrained(SAM)
            self._segmenter = Segmenter(model.eval(), Sam2Processor.from_pretrained(SAM))
        self._segmenter.model.to(self.device)
        return self._segmenter

    def describe(self, image: Image.Image) -> Scene:
        with self._lock:
            return self._detector_ready().describe(image)

    def find(self, image: Image.Image, name: str) -> List[Found]:
        with self._lock:
            return self._detector_ready().find(image, name)

    def mask_for(self, image: Image.Image, box: Box) -> np.ndarray:
        with self._lock:
            return self._segmenter_ready().mask_for(image, box)

    def rest(self) -> None:
        """Clear the GPU for something else: the fill model, or Ollama."""
        with self._lock:
            loaded = [part.model for part in (self._detector, self._segmenter) if part is not None]
            if not loaded or self.device != "cuda":
                return
            import torch

            for model in loaded:
                model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- fill model

    def _pipeline(self, name: str):
        if name not in self._pipelines:
            import torch
            from diffusers import AutoPipelineForInpainting

            pipeline = AutoPipelineForInpainting.from_pretrained(
                name, dtype=torch.float16, variant="fp16", use_safetensors=True
            )
            # Keeps each part of the pipeline on the GPU only while it runs.
            pipeline.enable_model_cpu_offload()
            pipeline.set_progress_bar_config(disable=True)
            self._pipelines[name] = pipeline
        return self._pipelines[name]

    def _paint_with(self, name: str, picture: Image.Image, hole: Image.Image,
                    prompt: str, seed: int) -> Image.Image:
        import torch

        size = work_size(picture.width, picture.height, NATIVE_SIZE[name])
        source = picture if picture.size == size else picture.resize(size, Image.LANCZOS)
        target = hole if hole.size == size else hole.resize(size, Image.NEAREST)
        output = self._pipeline(name)(
            prompt=prompt, negative_prompt=NEGATIVE, image=source, mask_image=target,
            width=size[0], height=size[1], num_inference_steps=STEPS[name],
            guidance_scale=GUIDANCE[name], strength=STRENGTH[name],
            # A CPU generator gives the same picture for the same seed whatever
            # the pipeline's parts are doing on the GPU.
            generator=torch.Generator("cpu").manual_seed(int(seed)),
        )
        if any(getattr(output, "nsfw_content_detected", None) or []):
            raise RuntimeError("The fill model's content filter blocked this result. "
                               "Try another description or another guess.")
        painted = output.images[0]
        return painted if painted.size == picture.size else painted.resize(picture.size, Image.LANCZOS)

    def paint(self, picture: Image.Image, hole: Image.Image, prompt: str, seed: int,
              prefer: Optional[str] = None) -> Image.Image:
        """The Painter for inpaint.generative.

        Paints with `prefer` if given, otherwise SDXL, and falls back to the
        other model when the first will not fit in the GPU's memory.
        `painted_with` says which one did the work.
        """
        if self.device != "cuda":
            raise RuntimeError("The generated fill needs a CUDA GPU.")
        import torch

        with self._lock:
            self.rest()
            for name in fill_order(prefer):
                try:
                    painted = self._paint_with(name, picture, hole, prompt, seed)
                except torch.OutOfMemoryError:
                    self._pipelines.pop(name, None)
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                self.painted_with = name
                return painted
        raise RuntimeError("Neither fill model fits in this GPU's memory.")
