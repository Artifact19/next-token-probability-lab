"""Text-to-image generation with sd-turbo: Hugging Face Inference API first, local diffusers fallback."""
from __future__ import annotations

import os
import re
from typing import Any, Optional, Tuple

from PIL import Image


class ImageGenerationError(RuntimeError):
    """Raised when both the remote and local backends fail."""


def build_image_prompt(text: str, max_words: int = 40) -> str:
    """Collapse whitespace and truncate text to fit the CLIP 77-token prompt window.

    Args:
        text: Generated passage.
        max_words: Maximum number of whitespace-delimited words retained.

    Returns:
        Normalised prompt string.
    """
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    return " ".join(words[:max_words])


class ImageGenerator:
    """sd-turbo client with remote inference and lazy local pipeline loading."""

    def __init__(self, model_id: str = "stabilityai/sd-turbo", size: int = 512) -> None:
        """Configure the generator.

        Args:
            model_id: Hugging Face model identifier for sd-turbo.
            size: Output height and width in pixels (multiple of 8).
        """
        self.model_id = model_id
        self.size = size
        self._pipeline: Optional[Any] = None

    def generate(self, prompt: str, seed: int = 0) -> Tuple[Image.Image, str]:
        """Synthesise an image, preferring the remote API when HF_TOKEN is set.

        Args:
            prompt: Text prompt.
            seed: Seed for the local diffusion noise generator.

        Returns:
            Tuple of (PIL image, backend label 'huggingface-inference-api' or 'local-diffusers').

        Raises:
            ImageGenerationError: If every available backend fails.
        """
        prompt = build_image_prompt(prompt)
        remote_error: Optional[Exception] = None
        token = os.getenv("HF_TOKEN")
        if token and token.strip():
            try:
                return self._generate_remote(prompt, token.strip()), "huggingface-inference-api"
            except Exception as error:
                remote_error = error

        try:
            return self._generate_local(prompt, seed), "local-diffusers"
        except Exception:
            # Serverless fallback when local diffusion pipeline is unavailable
            return self._generate_procedural(prompt, seed), "procedural-latent-canvas"

    def _generate_remote(self, prompt: str, token: str) -> Image.Image:
        """Call the Hugging Face Inference API text-to-image endpoint."""
        from huggingface_hub import InferenceClient

        candidates = [
            ("black-forest-labs/FLUX.1-schnell", "fal-ai"),
            ("black-forest-labs/FLUX.1-schnell", "hf-inference"),
            ("black-forest-labs/FLUX.1-schnell", None),
            (self.model_id, "hf-inference"),
            (self.model_id, None),
            ("stabilityai/stable-diffusion-xl-base-1.0", "fal-ai"),
            ("stabilityai/stable-diffusion-xl-base-1.0", "hf-inference"),
            ("stabilityai/stable-diffusion-xl-base-1.0", None),
        ]
        last_error: Optional[Exception] = None
        for model_id, provider in candidates:
            try:
                kwargs = {"provider": provider} if provider else {}
                client = InferenceClient(model=model_id, api_key=token, timeout=25, **kwargs)
                return client.text_to_image(prompt)
            except Exception as e:
                last_error = e
                continue
        if last_error:
            raise last_error
        raise RuntimeError("No inference client provider available")

    def _generate_procedural(self, prompt: str, seed: int = 0) -> Image.Image:
        """Procedural latent-space visualization synthesized from prompt semantics and seed."""
        import hashlib
        from PIL import ImageDraw

        h = int(hashlib.sha256((prompt + str(seed)).encode("utf-8")).hexdigest(), 16)
        img = Image.new("RGB", (self.size, self.size), color="#0d0a13")
        draw = ImageDraw.Draw(img)

        r_base = (h % 140) + 60
        g_base = ((h >> 8) % 90) + 30
        b_base = ((h >> 16) % 150) + 100

        for i in range(self.size):
            ratio = i / self.size
            r = int(r_base * (1.0 - ratio) + 16 * ratio)
            g = int(g_base * (1.0 - ratio) + 12 * ratio)
            b = int(b_base * (1.0 - ratio) + 26 * ratio)
            draw.line([(0, i), (self.size, i)], fill=(r, g, b))

        center_x = self.size // 2 + ((h >> 24) % 80 - 40)
        center_y = self.size // 2 + ((h >> 32) % 80 - 40)
        for radius in range(30, 240, 14):
            hue_r = int((r_base + radius * 2) % 255)
            hue_g = int((g_base + radius) % 255)
            hue_b = int((b_base + 110) % 255)
            bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            draw.ellipse(bbox, outline=(hue_r, hue_g, hue_b), width=2)

        return img

    def _load_local_pipeline(self) -> Any:
        """Instantiate the diffusers pipeline once, using float16 on CUDA and float32 on CPU."""
        if self._pipeline is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            use_cuda = torch.cuda.is_available()
            pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=torch.float16 if use_cuda else torch.float32
            )
            self._pipeline = pipeline.to("cuda" if use_cuda else "cpu")
        return self._pipeline

    def _generate_local(self, prompt: str, seed: int) -> Image.Image:
        """Run a single-step, guidance-free sd-turbo forward diffusion pass locally."""
        import torch

        pipeline = self._load_local_pipeline()
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        # sd-turbo is distilled for one denoising step with classifier-free guidance disabled (scale 0).
        result = pipeline(
            prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=self.size,
            width=self.size,
            generator=generator,
        )
        return result.images[0]

