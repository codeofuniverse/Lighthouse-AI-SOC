"""GPU detection and device-param helpers shared across training scripts."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def detect_gpu() -> tuple[bool, str]:
    """Probe for a usable NVIDIA GPU.

    Tries three methods in order:
      1. nvidia-smi (always reliable when CUDA driver is present)
      2. torch.cuda (if PyTorch is installed)
      3. cupy (if CuPy is installed)

    Returns:
        (gpu_available, device_name) — device_name is "" when no GPU found.
    """
    # --- Method 1: nvidia-smi ---
    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                name = result.stdout.strip().splitlines()[0].strip()
                logger.info("GPU detected via nvidia-smi: %s", name)
                return True, name
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass

    # --- Method 2: PyTorch ---
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("GPU detected via torch.cuda: %s", name)
            return True, name
    except ImportError:
        pass

    # --- Method 3: CuPy ---
    try:
        import cupy  # type: ignore[import]
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        logger.info("GPU detected via cupy: %s", name)
        return True, name
    except (ImportError, Exception):
        pass

    logger.info("No GPU detected — training on CPU")
    return False, ""


def xgb_gpu_params(base: dict[str, Any], gpu: bool) -> dict[str, Any]:
    """Return XGBoost params with GPU settings applied when available.

    Adds ``device='cuda'`` and ``tree_method='hist'`` when *gpu* is True.
    Falls back gracefully so the same dict works on CPU.
    """
    params = dict(base)
    if gpu:
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    else:
        params.setdefault("tree_method", "hist")
    return params


def lgb_gpu_params(base: dict[str, Any], gpu: bool) -> dict[str, Any]:
    """Return LightGBM params with GPU settings applied when available."""
    params = dict(base)
    params["device_type"] = "gpu" if gpu else "cpu"
    return params


def xgb_fit_with_fallback(model_cls: Any, params: dict[str, Any], gpu: bool, **fit_kwargs: Any) -> Any:
    """Instantiate and fit an XGBoost model, falling back to CPU on GPU error.

    Args:
        model_cls: e.g. ``xgb.XGBClassifier``
        params: Params dict (already GPU-patched via ``xgb_gpu_params``).
        gpu: Whether GPU was requested.
        **fit_kwargs: Forwarded to ``model.fit()``.

    Returns:
        Fitted model instance.
    """

    fit_kwargs = dict(fit_kwargs)
    early_stopping_rounds = fit_kwargs.pop("early_stopping_rounds", None)
    callbacks = fit_kwargs.pop("callbacks", None)
    if early_stopping_rounds is not None or callbacks is not None:
        logger.warning(
            "XGBoost fit() in this environment does not support early stopping/callbacks; "
            "training will continue without them"
        )

    model = model_cls(**params)
    try:
        model.fit(**fit_kwargs)
        return model
    except Exception as exc:
        if not gpu:
            raise
        logger.warning("XGBoost GPU training failed (%s) — retrying on CPU", exc)
        cpu_params = {k: v for k, v in params.items() if k not in ("device",)}
        cpu_params["tree_method"] = "hist"
        model = model_cls(**cpu_params)
        model.fit(**fit_kwargs)
        return model


def lgb_fit_with_fallback(model_cls: Any, params: dict[str, Any], gpu: bool, **fit_kwargs: Any) -> Any:
    """Instantiate and fit a LightGBM model, falling back to CPU on GPU error."""
    model = model_cls(**params)
    try:
        model.fit(**fit_kwargs)
        return model
    except Exception as exc:
        if not gpu:
            raise
        logger.warning("LightGBM GPU training failed (%s) — retrying on CPU", exc)
        cpu_params = dict(params)
        cpu_params["device_type"] = "cpu"
        model = model_cls(**cpu_params)
        model.fit(**fit_kwargs)
        return model
