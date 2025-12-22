"""Receipt image segmentation using Meta SAM3.

This module is designed to be a small, standalone library that can be called
from the ingestion pipeline.

Segmentation is optional: callers should handle ImportError/runtime errors and
fall back to normal ingestion if SAM3 isn't installed or configured.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class _TorchCudaRedirect:
    """Temporarily redirect torch factory calls from device='cuda' -> 'cpu'.

    Some SAM3 modules hardcode CUDA device strings during model construction.
    On Apple Silicon / CPU-only environments, that raises:
      AssertionError: Torch not compiled with CUDA enabled
    For our use-case, we can safely construct on CPU and later move to MPS/CPU.
    """

    def __init__(self):
        self._orig: Dict[str, Any] = {}

    def __enter__(self):
        import torch

        def wrap(fn):
            def _wrapped(*args, **kwargs):
                dev = kwargs.get("device")
                if dev == "cuda" and not torch.cuda.is_available():
                    kwargs["device"] = "cpu"
                return fn(*args, **kwargs)

            return _wrapped

        for name in ("zeros", "ones", "empty", "rand", "randn", "arange", "linspace", "tensor"):
            if hasattr(torch, name):
                self._orig[name] = getattr(torch, name)
                setattr(torch, name, wrap(getattr(torch, name)))

        return self

    def __exit__(self, exc_type, exc, tb):
        import torch

        for name, fn in self._orig.items():
            setattr(torch, name, fn)
        self._orig.clear()
        return False


class _TorchPinMemoryNoop:
    """Temporarily make Tensor.pin_memory() a no-op.

    Some upstream SAM3 code uses pin_memory() even when targeting MPS, which can
    raise device mismatch errors on Apple Silicon.
    """

    def __init__(self):
        self._orig = None

    def __enter__(self):
        import torch

        self._orig = torch.Tensor.pin_memory

        def _pin_memory_noop(self, *args, **kwargs):  # type: ignore[no-redef]
            return self

        torch.Tensor.pin_memory = _pin_memory_noop  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        import torch

        if self._orig is not None:
            torch.Tensor.pin_memory = self._orig  # type: ignore[assignment]
        self._orig = None
        return False


@dataclass(frozen=True)
class SegmentationConfig:
    """Configuration for receipt segmentation."""

    enable: bool
    output_dir: Path
    device: str  # auto|mps|cpu
    checkpoint_path: Optional[Path]
    text_prompt: str
    confidence_threshold: float

    # Heuristics / limits
    max_masks: int
    max_segments: int
    min_area_ratio: float
    min_width_px: int
    min_height_px: int
    min_fill_ratio: float
    iou_dedup_threshold: float
    bbox_padding_px: int


def _select_torch_device(device: str) -> Tuple[str, "torch.device"]:
    import torch

    dev = (device or "auto").lower()
    if dev not in {"auto", "mps", "cpu"}:
        logger.warning(f"Unknown segmentation.device '{device}', falling back to 'auto'")
        dev = "auto"

    if dev == "cpu":
        return ("cpu", torch.device("cpu"))

    if dev == "mps":
        if torch.backends.mps.is_available():
            return ("mps", torch.device("mps"))
        logger.warning("Requested MPS device but torch.backends.mps.is_available() is False; falling back to CPU")
        return ("cpu", torch.device("cpu"))

    # auto
    if torch.backends.mps.is_available():
        return ("mps", torch.device("mps"))
    return ("cpu", torch.device("cpu"))


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) inclusive-exclusive bbox for a boolean mask."""
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def _bbox_area(b: Tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = b
    return max(0, x1 - x0) * max(0, y1 - y0)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _dedup_bboxes(bboxes: List[Tuple[int, int, int, int]], iou_threshold: float) -> List[Tuple[int, int, int, int]]:
    """Deduplicate bboxes via greedy NMS-style suppression."""
    if not bboxes:
        return []

    boxes = sorted(bboxes, key=_bbox_area, reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        if all(_iou(b, k) < iou_threshold for k in kept):
            kept.append(b)
    return kept


class Sam3ReceiptSegmenter:
    """Receipt segmenter built on SAM3.

    Notes:
    - SAM3 is imported lazily.
    - The exact SAM3 API may vary; we attempt multiple common entrypoints.
    """

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._model: Any = None
        self._processor: Any = None
        self._device_label: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return

        if not self.cfg.checkpoint_path:
            raise RuntimeError("segmentation.checkpoint_path is not set")

        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"PyTorch is required for segmentation: {e}")

        device_label, device = _select_torch_device(self.cfg.device)
        self._device_label = device_label

        checkpoint_path = self._resolve_checkpoint_path(self.cfg.checkpoint_path)

        # Lazy import sam3
        #
        # SAM3 currently imports some optional dependencies unconditionally (e.g. `triton`, `decord`)
        # that are not required for our image-only receipt segmentation use-case. On Apple Silicon
        # these packages are often unavailable. We stub them to allow SAM3 to import.
        for _ in range(8):  # bounded retries to avoid infinite loops
            try:
                importlib.import_module("sam3")
                break
            except ModuleNotFoundError as e:
                if e.name == "triton":
                    self._install_sam3_edt_stub()
                    importlib.invalidate_caches()
                    continue
                if e.name == "decord":
                    self._install_decord_stub()
                    importlib.invalidate_caches()
                    continue
                if e.name == "pycocotools":
                    self._install_pycocotools_stub()
                    importlib.invalidate_caches()
                    continue
                if e.name == "psutil":
                    self._install_psutil_stub()
                    importlib.invalidate_caches()
                    continue
                raise RuntimeError(
                    "SAM3 package is not installed/available. Install Meta SAM3 from https://github.com/facebookresearch/sam3\n"
                    f"Underlying error: {e}"
                )
            except Exception as e:
                raise RuntimeError(
                    "SAM3 package is not installed/available. Install Meta SAM3 from https://github.com/facebookresearch/sam3\n"
                    f"Underlying error: {e}"
                )
        else:
            raise RuntimeError("Unable to import SAM3 after stubbing optional dependencies")

        # Try to build a model (API may differ across revisions)
        model = None
        builder_attempts: List[str] = []

        for module_name, fn_name in [
            ("sam3.model_builder", "build_sam3"),
            ("sam3.model_builder", "build_sam3_image_model"),
            ("sam3.model_builder", "build_sam3_model"),
        ]:
            try:
                mod = importlib.import_module(module_name)
                fn = getattr(mod, fn_name, None)
                if fn is None:
                    continue

                sig = inspect.signature(fn)
                kwargs: Dict[str, Any] = {}
                # Common checkpoint arg names
                for ck_key in ("checkpoint", "checkpoint_path", "ckpt_path", "model_path"):
                    if ck_key in sig.parameters:
                        kwargs[ck_key] = str(checkpoint_path)
                        break
                # Common device arg names
                for dev_key in ("device", "device_type"):
                    if dev_key in sig.parameters:
                        # Keep internal builder on CPU; we move to MPS/CPU ourselves below.
                        kwargs[dev_key] = "cpu"
                        break

                builder_attempts.append(f"{module_name}.{fn_name}({', '.join(kwargs.keys())})")
                with _TorchCudaRedirect():
                    model = fn(**kwargs)
                break
            except Exception as e:
                builder_attempts.append(f"{module_name}.{fn_name} -> {e}")
                continue

        if model is None:
            attempts = "\n".join(f"- {a}" for a in builder_attempts) or "- (no builder functions found)"
            raise RuntimeError(
                "Unable to construct a SAM3 model using known builder entrypoints.\n"
                "Tried:\n"
                f"{attempts}\n\n"
                "Please verify your installed SAM3 version and adjust the builder integration accordingly."
            )

        # Move to device
        try:
            model = model.to(device)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to move SAM3 model to device '{device_label}': {e}")

        # Try to construct a processor/predictor.
        processor = None
        processor_attempts: List[str] = []
        for module_name, cls_name in [
            ("sam3.model.sam3_image_processor", "Sam3Processor"),
            ("sam3.sam3_image_processor", "Sam3Processor"),
            ("sam3.sam3_image_predictor", "Sam3ImagePredictor"),
            ("sam3.image_predictor", "Sam3ImagePredictor"),
        ]:
            try:
                mod = importlib.import_module(module_name)
                cls = getattr(mod, cls_name, None)
                if cls is None:
                    continue
                processor_attempts.append(f"{module_name}.{cls_name}")
                kwargs: Dict[str, Any] = {}
                try:
                    sig = inspect.signature(cls)
                    if "device" in sig.parameters:
                        kwargs["device"] = self._device_label or device_label
                    if "confidence_threshold" in sig.parameters:
                        kwargs["confidence_threshold"] = float(self.cfg.confidence_threshold)
                except Exception:
                    # If signature inspection fails, fall back to default constructor.
                    kwargs = {}

                with _TorchCudaRedirect():
                    processor = cls(model, **kwargs) if kwargs else cls(model)
                break
            except Exception as e:
                processor_attempts.append(f"{module_name}.{cls_name} -> {e}")
                continue

        if processor is None:
            attempts = "\n".join(f"- {a}" for a in processor_attempts) or "- (no processor classes found)"
            raise RuntimeError(
                "Unable to construct a SAM3 image processor/predictor using known entrypoints.\n"
                "Tried:\n"
                f"{attempts}"
            )

        self._model = model
        self._processor = processor

        logger.info(f"SAM3 segmenter initialized on device={device_label}")

    @staticmethod
    def _install_sam3_edt_stub() -> None:
        """Install a stub for `sam3.model.edt` to avoid importing Triton on non-CUDA systems.

        SAM3 currently imports `triton` in `sam3.model.edt` unconditionally. Unfortunately, stubbing
        the *global* `triton` module can break PyTorch/torchvision internals (torch._inductor tries
        to import real Triton backends). Instead, we stub **only** `sam3.model.edt` so the SAM3
        package can import, without introducing a fake `triton` into the runtime.
        """
        # If a previous run injected a fake triton, remove it to avoid torch._inductor issues.
        for k in ("triton", "triton.language", "triton.backends", "triton.backends.compiler"):
            if k in sys.modules:
                del sys.modules[k]

        if "sam3.model.edt" in sys.modules:
            return

        import types

        edt = types.ModuleType("sam3.model.edt")

        def edt_triton(*_args, **_kwargs):
            raise RuntimeError(
                "SAM3 attempted to call edt_triton, but Triton is not available in this environment. "
                "This should not be needed for image-only receipt segmentation."
            )

        edt.edt_triton = edt_triton  # type: ignore[attr-defined]
        sys.modules["sam3.model.edt"] = edt

    @staticmethod
    def _install_decord_stub() -> None:
        """Install a minimal stub for `decord` to let SAM3 import on environments without it.

        SAM3 imports `decord` from its training/video dataset utilities. Receipt segmentation does not
        rely on those features.
        """
        if "decord" in sys.modules:
            return

        import types

        decord = types.ModuleType("decord")

        # SAM3 uses: `from decord import cpu, VideoReader`
        def cpu(*_args, **_kwargs):  # type: ignore[override]
            raise RuntimeError("decord is not available in this environment")

        class VideoReader:  # noqa: N801 - match expected name
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("decord is not available in this environment")

        decord.cpu = cpu  # type: ignore[attr-defined]
        decord.VideoReader = VideoReader  # type: ignore[attr-defined]
        sys.modules["decord"] = decord

    @staticmethod
    def _install_pycocotools_stub() -> None:
        """Install a minimal stub for `pycocotools` to let SAM3 import without COCO tooling.

        SAM3 imports COCO JSON loaders in its training dataset utilities at import-time. Receipt
        segmentation does not rely on those features.
        """
        if "pycocotools" in sys.modules:
            return

        import types

        pycocotools = types.ModuleType("pycocotools")
        mask = types.ModuleType("pycocotools.mask")

        def _not_available(*_args, **_kwargs):
            raise RuntimeError("pycocotools is not available in this environment")

        # Commonly referenced APIs
        mask.decode = _not_available  # type: ignore[attr-defined]
        mask.encode = _not_available  # type: ignore[attr-defined]
        mask.area = _not_available  # type: ignore[attr-defined]
        mask.toBbox = _not_available  # type: ignore[attr-defined]
        mask.frPyObjects = _not_available  # type: ignore[attr-defined]
        mask.merge = _not_available  # type: ignore[attr-defined]
        mask.iou = _not_available  # type: ignore[attr-defined]

        pycocotools.mask = mask  # type: ignore[attr-defined]
        sys.modules["pycocotools"] = pycocotools
        sys.modules["pycocotools.mask"] = mask

    @staticmethod
    def _install_psutil_stub() -> None:
        """Install a minimal stub for `psutil` to let SAM3 import on minimal environments.

        SAM3 imports `psutil` for video predictor utilities. Receipt segmentation does not require it.
        """
        if "psutil" in sys.modules:
            return

        import types

        psutil = types.ModuleType("psutil")

        def _not_available(*_args, **_kwargs):
            raise RuntimeError("psutil is not available in this environment")

        # Provide common APIs as placeholders
        psutil.virtual_memory = _not_available  # type: ignore[attr-defined]
        psutil.cpu_count = _not_available  # type: ignore[attr-defined]
        psutil.Process = _not_available  # type: ignore[attr-defined]

        sys.modules["psutil"] = psutil

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: Path) -> Path:
        """Resolve a SAM3 checkpoint path.

        The user may provide either:
        - a direct checkpoint file (e.g. sam3.pt), or
        - a directory containing the checkpoint (e.g. HF cache directory).
        """
        p = checkpoint_path.expanduser()
        if p.is_file():
            return p
        if p.is_dir():
            # Common HF filename
            candidate = p / "sam3.pt"
            if candidate.exists():
                return candidate
            # Otherwise, pick the first *.pt in the directory (best-effort)
            pts = sorted(p.glob("*.pt"))
            if pts:
                return pts[0]
        # Fall back to original (SAM3 may handle downloading if this is a model id in some forks)
        return p

    def _generate_masks(self, pil_img: Image.Image) -> List[np.ndarray]:
        self._ensure_loaded()

        # We attempt common patterns:
        # - processor.set_image(image) -> state; processor.generate(state=state) -> {"masks": ...}
        # - predictor.set_image(image); predictor.generate() or predictor.predict(...)

        proc = self._processor

        # Convert to RGB to avoid mode issues (SAM3 processor expects 3 channels).
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Try SAM3 processor API: set_image + set_text_prompt -> state["masks"]
        if hasattr(proc, "set_image"):
            try:
                with _TorchCudaRedirect(), _TorchPinMemoryNoop():
                    state = proc.set_image(pil_img)
                if hasattr(proc, "set_text_prompt"):
                    with _TorchCudaRedirect(), _TorchPinMemoryNoop():
                        state = proc.set_text_prompt(self.cfg.text_prompt, state)
                    if isinstance(state, dict) and "masks" in state:
                        return self._coerce_masks(state["masks"])
            except Exception as e:
                # MPS sometimes fails due to unsupported ops; fall back to CPU once.
                if self._device_label == "mps":
                    logger.warning(f"SAM3 generation on MPS failed, retrying on CPU: {e}")
                    self._force_cpu_reload()
                    return self._generate_masks(pil_img)
                raise

        # Try generate without state
        if hasattr(proc, "generate"):
            try:
                with _TorchCudaRedirect(), _TorchPinMemoryNoop():
                    out = proc.generate(image=pil_img)  # some APIs take image directly
                masks = out.get("masks") if isinstance(out, dict) else None
                if masks is not None:
                    return self._coerce_masks(masks)
            except Exception:
                pass

        # Try predictor-like API
        for fn_name in ("predict", "predict_masks", "predictor"):
            if hasattr(proc, fn_name):
                try:
                    fn = getattr(proc, fn_name)
                    with _TorchCudaRedirect(), _TorchPinMemoryNoop():
                        out = fn(pil_img)
                    if isinstance(out, dict) and "masks" in out:
                        return self._coerce_masks(out["masks"])
                except Exception:
                    pass

        raise RuntimeError("Unable to generate masks with the installed SAM3 API")

    def _force_cpu_reload(self) -> None:
        # Clear any loaded state and reload on CPU.
        self._model = None
        self._processor = None
        self._device_label = None
        self.cfg = SegmentationConfig(
            **{**self.cfg.__dict__, "device": "cpu"}  # type: ignore[arg-type]
        )

    @staticmethod
    def _coerce_masks(masks: Any) -> List[np.ndarray]:
        """Coerce SAM3 masks output into a list of boolean HxW numpy arrays."""
        # Common representations: list[np.ndarray], np.ndarray [N,H,W], torch.Tensor [N,H,W]
        try:
            import torch

            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()
        except Exception:
            pass

        if isinstance(masks, list):
            out: List[np.ndarray] = []
            for m in masks:
                if hasattr(m, "detach"):
                    try:
                        import torch

                        if isinstance(m, torch.Tensor):
                            m = m.detach().cpu().numpy()
                    except Exception:
                        pass
                m = np.asarray(m)
                if m.ndim == 3 and m.shape[0] == 1:
                    m = m[0]
                out.append(m.astype(bool))
            return out

        arr = np.asarray(masks)
        if arr.ndim == 2:
            return [arr.astype(bool)]
        if arr.ndim == 3:
            return [arr[i].astype(bool) for i in range(arr.shape[0])]
        if arr.ndim == 4:
            # Common shape: [N, 1, H, W]
            if arr.shape[1] == 1:
                arr = arr[:, 0, :, :]
                return [arr[i].astype(bool) for i in range(arr.shape[0])]
        raise RuntimeError(f"Unexpected mask array shape: {arr.shape}")

    def segment_receipts(self, input_image_path: Path) -> List[Path]:
        """Segment an input image into receipt crops and save them as PNGs.

        Returns:
            List of paths to created PNG files. Empty list if no segments found.
        """
        if not input_image_path.exists():
            raise FileNotFoundError(str(input_image_path))

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(input_image_path)
        w, h = img.size
        masks = self._generate_masks(img)

        if self.cfg.max_masks > 0:
            masks = masks[: self.cfg.max_masks]

        candidates: List[Tuple[Tuple[int, int, int, int], float]] = []

        for m in masks:
            if m.shape[0] != h or m.shape[1] != w:
                # Some models output resized masks; ignore for now.
                # (If needed, we can add resize-to-original later.)
                continue

            bbox = _bbox_from_mask(m)
            if bbox is None:
                continue

            x0, y0, x1, y1 = bbox
            bw = x1 - x0
            bh = y1 - y0

            if bw < self.cfg.min_width_px or bh < self.cfg.min_height_px:
                continue

            bbox_area = _bbox_area(bbox)
            if bbox_area <= 0:
                continue

            mask_area = int(m.sum())
            area_ratio = mask_area / float(w * h)
            if area_ratio < self.cfg.min_area_ratio:
                continue

            fill_ratio = mask_area / float(bbox_area)
            if fill_ratio < self.cfg.min_fill_ratio:
                continue

            # score proxy: prefer bigger, fuller regions
            score = area_ratio * 0.7 + fill_ratio * 0.3
            candidates.append((bbox, score))

        if not candidates:
            return []

        # Deduplicate
        candidates.sort(key=lambda t: t[1], reverse=True)
        bboxes = [b for b, _ in candidates]
        bboxes = _dedup_bboxes(bboxes, self.cfg.iou_dedup_threshold)

        # Sort for stable naming (top-to-bottom, then left-to-right)
        bboxes.sort(key=lambda b: (b[1], b[0]))

        if self.cfg.max_segments > 0:
            bboxes = bboxes[: self.cfg.max_segments]

        created: List[Path] = []
        stem = input_image_path.stem

        for idx, b in enumerate(bboxes, start=1):
            pad = self.cfg.bbox_padding_px
            x0, y0, x1, y1 = b
            b2 = _clamp_bbox((x0 - pad, y0 - pad, x1 + pad, y1 + pad), w=w, h=h)
            crop = img.crop(b2)

            out_name = f"{stem}__seg_{idx:03d}.png"
            out_path = self.cfg.output_dir / out_name
            crop.save(out_path, format="PNG")
            created.append(out_path)

        return created


def segment_receipts_to_pngs(
    input_image_path: Path,
    cfg: SegmentationConfig,
) -> List[Path]:
    """Convenience function to segment a single image."""
    return Sam3ReceiptSegmenter(cfg).segment_receipts(input_image_path)
