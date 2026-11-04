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
        if token:
            try:
                return self._generate_remote(prompt, token), "huggingface-inference-api"
            except Exception as error:
                remote_error = error
        try:
            return self._generate_local(prompt, seed), "local-diffusers"
        except Exception as local_error:
            raise ImageGenerationError(
                f"remote backend: {remote_error if remote_error else 'HF_TOKEN not set'}; local backend: {local_error}"
            ) from local_error

    def _generate_remote(self, prompt: str, token: str) -> Image.Image:
        """Call the Hugging Face Inference API text-to-image endpoint."""
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=self.model_id, token=token)
        return client.text_to_image(prompt, height=self.size, width=self.size)

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

