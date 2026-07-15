# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E smoke test for Stable Diffusion 3.5 medium with online FP8 quantization.

This test intentionally uses the upstream BF16 checkpoint plus
``quantization="fp8"`` so it can be run on a single CUDA GPU without requiring
an extra pre-quantized SD3 checkpoint.
"""

import gc
import os as _os

import numpy as np
import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

MODEL = _os.environ.get("SD35_MODEL", "stabilityai/stable-diffusion-3.5-medium")
HEIGHT = 256
WIDTH = 256
NUM_STEPS = 2


def _sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=4.5,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
    )


def _first_request_images(outputs) -> list:
    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput) and hasattr(req_out, "images")
    return req_out.images


def _generate_image_with_peak_memory(**omni_kwargs) -> tuple[list, float]:
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    try:
        with OmniRunner(MODEL, enforce_eager=True, **omni_kwargs) as runner:
            current_omni_platform.reset_peak_memory_stats()
            outputs = runner.omni.generate(
                "a cozy reading corner with a chair, lamp, and books",
                _sampling_params(),
            )
    finally:
        peak_used_mb = monitor.peak_used_mb
        monitor.stop()

    images = _first_request_images(outputs)
    gc.collect()
    current_omni_platform.empty_cache()
    return images, peak_used_mb


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"})
def test_sd35_fp8_load_and_generate():
    """Load SD3.5-medium with online FP8 quantization and generate one image."""
    with OmniRunner(MODEL, enforce_eager=True, quantization="fp8") as runner:
        outputs = runner.omni.generate(
            "a cozy reading corner with a chair, lamp, and books",
            _sampling_params(),
        )
        images = _first_request_images(outputs)
        assert len(images) >= 1, "Expected at least one generated image"
        img = images[0]
        assert img.width == WIDTH and img.height == HEIGHT
        arr = np.array(img)
        assert arr.std() > 1.0, "Generated image appears blank (std ≈ 0)"


@pytest.mark.diffusion
@pytest.mark.slow
@hardware_test(res={"cuda": "L4"})
def test_sd35_fp8_uses_less_memory_than_baseline():
    baseline_images, baseline_peak = _generate_image_with_peak_memory()
    cleanup_dist_env_and_memory()
    quant_images, quant_peak = _generate_image_with_peak_memory(quantization="fp8")

    assert len(baseline_images) >= 1
    assert len(quant_images) >= 1

    print(f"SD3.5 baseline peak memory: {baseline_peak:.0f} MB")
    print(f"SD3.5 fp8 peak memory:      {quant_peak:.0f} MB")
    assert quant_peak < baseline_peak, (
        f"Expected FP8 peak memory ({quant_peak:.0f} MB) to be lower than baseline "
        f"({baseline_peak:.0f} MB)"
    )
