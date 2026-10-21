"""
Hybrid SAM3 -> SAM2.1 multi-camera segmentation.

Stage 1 (SAM3 cross-camera initialization)
------------------------------------------
Take the first image from every camera folder, treat them as one ordered video,
and use SAM3 + the text prompt to obtain one first-frame object mask per camera.

Stage 2 (SAM2.1 multi-GPU per-camera propagation)
--------------------------------------------------
For each camera folder, use the COMPLETE Stage-1 binary mask as a SAM2.1 mask
prompt on frame 0, then propagate that object through the remaining frames.
Camera folders are distributed across all selected GPUs, optionally with
multiple independent SAM2.1 workers per GPU.

The Stage-2 path is throughput-oriented for H100/H200:
  * SAM2.1 Hiera Small by default;
  * official VOS-optimized/full-model torch.compile path when --compile is set;
  * BF16 autocast;
  * GPU-resident frames/state by default;
  * asynchronous frame loading;
  * GPU-side mask threshold/union;
  * background PNG writes with compression=0;
  * persistent TorchInductor/Triton caches across runs.

Saved masks are single-channel uint8 PNG:
    background = 0
    foreground = 255
"""

import argparse
import gc
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_COMPILE_CACHE_ROOT = Path(__file__).resolve().parent / ".compile_cache"
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_COMPILE_CACHE_ROOT / "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_COMPILE_CACHE_ROOT / "triton"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3 import build_sam3_predictor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SAM2_MODEL_SPECS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run two-stage SAM 3 video propagation on a recursive image folder: "
            "(1) propagate across the first image of every camera folder, then "
            "(2) use each folder's first mask to propagate within that folder."
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Input image root. Camera folders are discovered recursively.",
    )
    parser.add_argument(
        "--text_input",
        type=str,
        required=True,
        help='Text prompt for stage 1, e.g. "bottle".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output mask root. Default: sibling 'masks' directory next to the "
            "input 'images' directory."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Local SAM 3 checkpoint. Default: <sam3_repo>/checkpoints/sam3.pt "
            "where <sam3_repo> is the directory containing this script."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="SAM 3 threshold for detector / output masks. Default: 0.5",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='CUDA device, e.g. "cuda" or "cuda:0". Default: cuda',
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip writing masks that already exist.",
    )
    parser.add_argument(
        "--empty_if_no_detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save an all-zero mask when no object matches the prompt / tracking. "
            "Default: enabled."
        ),
    )
    parser.add_argument(
        "--offload_video_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Offload loaded frame tensors to CPU. Disabled by default for maximum H100/H200 "
            "throughput; enable only if GPU memory becomes a bottleneck."
        ),
    )
    parser.add_argument(
        "--offload_state_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload SAM 3 inference state / memory to CPU. Default: disabled.",
    )
    parser.add_argument(
        "--async_loading_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load image frames asynchronously so CPU decode/resize overlaps GPU inference. "
            "This uses the same SAM 3 preprocessing and does not change accuracy. Default: enabled."
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Use SAM2.1 VOS-optimized full-model torch.compile for Stage 2. "
              "This is the fastest official SAM2 video path after warm-up."),
    )
    parser.add_argument(
        "--compile_stage1",
        "--compile-stage1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compile the short 31-frame Stage 1 as well. Disabled by default because its "
            "compile startup cost usually exceeds the Stage-1 runtime."
        ),
    )
    parser.add_argument(
        "--stage2_num_points",
        type=int,
        default=5,
        help=argparse.SUPPRESS,  # kept only for backward CLI compatibility; Stage 2 now uses full masks
    )
    parser.add_argument(
        "--fallback_to_text_if_mask_empty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help=(
            'GPUs used by Stage 2, e.g. "auto" or "0,1,2,3". '
            'auto/all uses every CUDA-visible GPU with free VRAM strictly above '
            '--min_free_vram_mb. Default: auto.'
        ),
    )
    parser.add_argument(
        "--min_free_vram_mb",
        "--min-free-vram-mb",
        type=int,
        default=32768,
        help="Selected GPUs must have STRICTLY more free VRAM than this many MiB. Default: 32768 (32 GiB).",
    )
    parser.add_argument(
        "--stage1_gpu",
        type=int,
        default=None,
        help=(
            "GPU used for Stage 1 cross-camera initialization. "
            "Default: the first GPU selected by --gpus."
        ),
    )
    parser.add_argument(
        "--folders_per_gpu",
        type=int,
        default=1,
        help=(
            "GPU assignment chunk size. Default: 1 for best load balancing across cameras."
        ),
    )
    parser.add_argument(
        "--sam2_repo",
        "--sam2-repo",
        type=Path,
        default=None,
        help="SAM2 repository root. Default: <this_script_dir>/sam2.",
    )
    parser.add_argument(
        "--sam2_model",
        "--sam2-model",
        choices=tuple(SAM2_MODEL_SPECS),
        default="large",
        help="SAM2.1 model used for Stage 2. Default: large.",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        "--sam2-checkpoint",
        type=Path,
        default=None,
        help="SAM2.1 checkpoint; inferred from --sam2-model by default.",
    )
    parser.add_argument(
        "--sam2_config",
        "--sam2-config",
        type=str,
        default=None,
        help="SAM2.1 Hydra config; inferred from --sam2-model by default.",
    )
    parser.add_argument(
        "--sam2_score_thresh",
        "--sam2-score-thresh",
        type=float,
        default=0.0,
        help="SAM2 mask-logit threshold. Official VOS default: 0.0.",
    )
    parser.add_argument(
        "--sam2_postprocess",
        "--sam2-postprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SAM2 postprocessing. Disabled by default for maximum throughput.",
    )
    parser.add_argument(
        "--sam2_compile_mode",
        "--sam2-compile-mode",
        choices=("max-autotune-no-cudagraphs", "max-autotune"),
        default="max-autotune-no-cudagraphs",
        help=(
            "Compile mode used inside the SAM2.1 VOS-optimized predictor. "
            "Default disables CUDA Graph replay while retaining max-autotune, "
            "which avoids PyTorch CUDAGraph output-overwrite failures on recent "
            "PyTorch while preserving most compile speedups."
        ),
    )
    parser.add_argument(
        "--workers_per_gpu",
        "--workers-per-gpu",
        type=int,
        default=1,
        help=(
            "Independent SAM2.1 predictor processes per GPU for Stage 2. Default: 1. "
            "Benchmark 1 vs 2 on your H200s and keep the faster wall-clock setting."
        ),
    )
    parser.add_argument(
        "--mask_writer_threads",
        "--mask-writer-threads",
        type=int,
        default=4,
        help=(
            "Background PNG writer threads per Stage-2 worker. Default: 4. These overlap "
            "PNG encoding/disk I/O with GPU inference."
        ),
    )
    parser.add_argument(
        "--png_compression",
        "--png-compression",
        type=int,
        default=0,
        choices=range(10),
        help="PNG compression for masks. 0 is fastest and still lossless. Default: 0.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show per-frame tqdm bars. Disabled by default to minimize hot-loop overhead.",
    )
    return parser.parse_args()


def resolve_paths(args):
    folder = args.folder.expanduser().resolve()

    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    if args.output is None:
        if folder.name.lower() == "images":
            output = folder.parent / "masks"
        else:
            output = folder.parent / f"{folder.name}_masks"
    else:
        output = args.output.expanduser().resolve()

    if args.checkpoint is None:
        repo_root = Path(__file__).resolve().parent
        checkpoint = repo_root / "checkpoints" / "sam3.pt"
    else:
        checkpoint = args.checkpoint.expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "SAM 3 checkpoint not found:\n"
            f"  {checkpoint}\n\n"
            "Pass it explicitly with:\n"
            "  --checkpoint /path/to/sam3.pt"
        )

    output.mkdir(parents=True, exist_ok=True)
    return folder, output, checkpoint


def resolve_sam2_paths(args):
    script_root = Path(__file__).resolve().parent
    sam2_repo = (
        args.sam2_repo.expanduser().resolve()
        if args.sam2_repo is not None
        else (script_root / "sam2").resolve()
    )
    package_build = sam2_repo / "sam2" / "build_sam.py"
    if not package_build.is_file():
        raise FileNotFoundError(
            "SAM2 package source not found. Expected:\n"
            f"  {package_build}\n"
            "Pass --sam2-repo /path/to/facebookresearch/sam2"
        )

    default_cfg, default_ckpt_name = SAM2_MODEL_SPECS[args.sam2_model]
    sam2_config = args.sam2_config or default_cfg
    sam2_checkpoint = (
        args.sam2_checkpoint.expanduser().resolve()
        if args.sam2_checkpoint is not None
        else (sam2_repo / "checkpoints" / default_ckpt_name).resolve()
    )
    if not sam2_checkpoint.is_file():
        raise FileNotFoundError(
            "SAM2.1 checkpoint not found:\n"
            f"  {sam2_checkpoint}"
        )
    return sam2_repo, sam2_config, sam2_checkpoint


def import_sam2_builder(sam2_repo: Path):
    # Running this hybrid script from the parent of the cloned SAM2 repo can shadow
    # the nested `sam2` package. Put the repo itself first on sys.path.
    repo_str = str(Path(sam2_repo).resolve())
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)

    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor


def make_sam2_predictor(gpu_index, args):
    build_sam2_video_predictor = import_sam2_builder(args.sam2_repo)

    if not args.progress:
        os.environ["TQDM_DISABLE"] = "1"
        try:
            import sam2.sam2_video_predictor as _sam2_vp
            _sam2_vp.tqdm = lambda iterable, *a, **k: iterable
        except Exception:
            pass

    # SAM2VideoPredictorVOS internally hard-codes torch.compile(mode="max-autotune").
    # On recent PyTorch, that mode enables CUDA Graphs. SAM2's temporal memory keeps
    # tensors alive across compiled component invocations, and CUDA Graph replay can
    # overwrite those live outputs ("tensor output of CUDAGraphs ... overwritten").
    #
    # Keep the important max-autotune Inductor/Triton optimizations, but disable only
    # CUDA Graph replay by temporarily remapping SAM2's compile calls during model
    # construction to "max-autotune-no-cudagraphs".
    real_torch_compile = torch.compile

    def compile_for_sam2(*c_args, **c_kwargs):
        requested = c_kwargs.get("mode")
        if (
            bool(args.compile)
            and args.sam2_compile_mode == "max-autotune-no-cudagraphs"
            and requested == "max-autotune"
        ):
            c_kwargs["mode"] = "max-autotune-no-cudagraphs"
        return real_torch_compile(*c_args, **c_kwargs)

    if bool(args.compile):
        torch.compile = compile_for_sam2

    try:
        predictor = build_sam2_video_predictor(
            config_file=args.sam2_config,
            ckpt_path=str(args.sam2_checkpoint),
            device=f"cuda:{int(gpu_index)}",
            mode="eval",
            apply_postprocessing=bool(args.sam2_postprocess),
            vos_optimized=bool(args.compile),
        )
    finally:
        torch.compile = real_torch_compile

    return predictor


def sam2_logits_to_mask(mask_logits, frame_size, score_thresh=0.0):
    """Threshold/reduce on GPU and transfer only one HxW uint8 mask."""
    width, height = frame_size
    if mask_logits is None:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(mask_logits, torch.Tensor):
        x = mask_logits.detach()
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        elif x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise RuntimeError(f"Unexpected SAM2 mask-logit shape: {tuple(x.shape)}")

        if x.shape[0] == 1:
            union_t = x[0] > float(score_thresh)
        else:
            union_t = torch.any(x > float(score_thresh), dim=0)

        union = union_t.to(dtype=torch.uint8).mul_(255).cpu().numpy()
    else:
        x = np.asarray(mask_logits)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        elif x.ndim == 2:
            x = x[None]
        if x.ndim != 3:
            raise RuntimeError(f"Unexpected SAM2 mask-logit shape: {tuple(x.shape)}")
        union = np.any(x > float(score_thresh), axis=0).astype(np.uint8) * 255

    if union.shape != (height, width):
        union = np.asarray(
            Image.fromarray(union, mode="L").resize(
                (width, height), resample=Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
    return union


def setup_cuda(device_str):
    device = torch.device(device_str)
    if device.type != "cuda":
        raise RuntimeError(
            "The official SAM 3 video predictor is CUDA-oriented. "
            f"Received --device {device_str!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"--device {device_str} requested, but CUDA is not available.")

    if device.index is None:
        gpu_index = torch.cuda.current_device()
    else:
        gpu_index = device.index
        torch.cuda.set_device(gpu_index)

    actual_device = torch.device(f"cuda:{gpu_index}")
    gpu = torch.cuda.get_device_name(actual_device)
    capability = torch.cuda.get_device_capability(actual_device)
    print(f"[GPU] {gpu}, cuda:{gpu_index}, compute capability {capability[0]}.{capability[1]}")
    if capability[0] < 8:
        print(
            "[WARNING] The official SAM 3 video predictor internally uses BF16 in its public API. "
            "On pre-Ampere GPUs (e.g. RTX 2080 Ti), this may be less reliable.",
            file=sys.stderr,
        )
    # Throughput-oriented CUDA settings. SAM3 video inference is predominantly BF16,
    # while these settings accelerate any remaining eligible FP32 matmuls/convolutions.
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    return gpu_index



def _visible_gpu_free_mb():
    """Return [(visible_cuda_index, free_MiB), ...], respecting CUDA_VISIBLE_DEVICES."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
    n_visible = torch.cuda.device_count()
    if n_visible <= 0:
        raise RuntimeError("No CUDA GPUs are visible to this process.")

    rows = []
    for idx in range(n_visible):
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(idx)
            rows.append((idx, int(free_bytes // (1024 * 1024))))
        except Exception as exc:
            print(f"[GPU auto] cuda:{idx}: free-memory query failed -> skip ({exc})", file=sys.stderr)
    return rows


def parse_gpu_indices(gpus_spec, min_free_vram_mb=32768):
    if int(min_free_vram_mb) < 0:
        raise ValueError("--min_free_vram_mb must be >= 0")

    mem_rows = _visible_gpu_free_mb()
    n_visible = torch.cuda.device_count()
    free_by_gpu = dict(mem_rows)
    spec = str(gpus_spec).strip().lower()

    if spec in {"auto", "all", "*"}:
        eligible = [(idx, free_mb) for idx, free_mb in mem_rows if free_mb > int(min_free_vram_mb)]
        eligible.sort(key=lambda x: (-x[1], x[0]))
        for idx, free_mb in mem_rows:
            state = "eligible" if free_mb > int(min_free_vram_mb) else "skip"
            print(f"[GPU auto] cuda:{idx}: {free_mb} MiB free -> {state}")
        gpu_indices = [idx for idx, _ in eligible]
        if not gpu_indices:
            raise RuntimeError(
                f"No CUDA-visible GPU has free VRAM > {int(min_free_vram_mb)} MiB."
            )
        print(f"[GPU auto] Stage-2 selected all qualifying GPUs: {gpu_indices}")
        return gpu_indices

    gpu_indices = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if not 0 <= idx < n_visible:
            raise ValueError(
                f"GPU index {idx} from --gpus is out of range; "
                f"{n_visible} CUDA device(s) are visible (valid: 0..{n_visible - 1})."
            )
        if idx not in gpu_indices:
            gpu_indices.append(idx)

    if not gpu_indices:
        raise ValueError("--gpus did not contain any valid GPU index.")
    low = [(idx, free_by_gpu.get(idx, -1)) for idx in gpu_indices if free_by_gpu.get(idx, -1) <= int(min_free_vram_mb)]
    if low:
        raise RuntimeError(
            f"Explicit --gpus contains GPU(s) without free VRAM > {int(min_free_vram_mb)} MiB: {low}"
        )
    return gpu_indices


def natural_path_key(path):
    """Sort camera folders numerically: 1,2,3,...,10 instead of 1,10,...,2."""
    key = []
    for part in Path(path).parts:
        chunks = re.split(r"(\d+)", part.lower())
        key.append(tuple((0, int(x)) if x.isdigit() else (1, x) for x in chunks if x))
    return tuple(key)


def assign_sequences_to_gpus(sequences, gpu_indices, folders_per_gpu):
    """Assign contiguous folder chunks to GPU workers in round-robin order."""
    if folders_per_gpu <= 0:
        raise ValueError("--folders_per_gpu must be >= 1")
    assignments = {gpu: [] for gpu in gpu_indices}
    chunks = [
        sequences[i : i + folders_per_gpu]
        for i in range(0, len(sequences), folders_per_gpu)
    ]
    for chunk_idx, chunk in enumerate(chunks):
        gpu = gpu_indices[chunk_idx % len(gpu_indices)]
        assignments[gpu].extend(chunk)
    return assignments

def sort_image_paths(paths):
    try:
        return sorted(paths, key=lambda p: int(p.stem))
    except Exception:
        return sorted(paths, key=lambda p: p.name)


def discover_cam_sequences(root):
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        return []

    by_parent = {}
    for p in images:
        by_parent.setdefault(p.parent, []).append(p)

    sequences = []
    for parent, img_list in by_parent.items():
        sorted_imgs = sort_image_paths(img_list)
        rel_dir = parent.relative_to(root)
        # SAM 3 video inference assumes a constant original frame size within
        # a sequence. Read it once here instead of reopening every image during
        # propagation (large I/O saving for long camera sequences).
        with Image.open(sorted_imgs[0]) as im:
            frame_size = im.size

        sequences.append(
            {
                "dir": parent,
                "rel_dir": rel_dir,
                "images": sorted_imgs,
                "frame_size": frame_size,
            }
        )

    return sorted(sequences, key=lambda x: natural_path_key(x["rel_dir"]))


def mask_output_path(output_root, folder_root, image_path):
    rel = image_path.relative_to(folder_root)
    return (output_root / rel).with_suffix(".png")


def save_mask(mask, out_path, compress_level=0):
    # Output camera directories are created once up front, so avoid mkdir/stat calls
    # in the per-frame hot path. PNG compression 0 is still fully lossless and is
    # substantially cheaper to encode than compressed PNG.
    Image.fromarray(mask, mode="L").save(
        out_path,
        compress_level=int(compress_level),
    )


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def outputs_to_union_mask(outputs, frame_size):
    """Collapse instance masks with the reduction on GPU whenever possible.

    The old implementation copied the entire NxHxW tensor to CPU and then ran
    np.any(). This version reduces NxHxW -> HxW on CUDA first and transfers only
    the final binary mask, reducing PCIe traffic and CPU work.
    """
    width, height = frame_size
    masks = None if outputs is None else outputs.get("out_binary_masks")

    if masks is None:
        return 0, np.zeros((height, width), dtype=np.uint8)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach()
        if masks.numel() == 0:
            return 0, np.zeros((height, width), dtype=np.uint8)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise RuntimeError(f"Unexpected SAM 3 video mask shape: {tuple(masks.shape)}")

        n_instances = int(masks.shape[0])
        union_t = torch.any(masks.bool(), dim=0)
        union = union_t.to(dtype=torch.uint8).mul_(255).cpu().numpy()
    else:
        masks = np.asarray(masks)
        if masks.size == 0:
            return 0, np.zeros((height, width), dtype=np.uint8)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise RuntimeError(f"Unexpected SAM 3 video mask shape: {tuple(masks.shape)}")
        n_instances = int(masks.shape[0])
        union = np.any(masks.astype(bool, copy=False), axis=0).astype(np.uint8) * 255

    if union.shape != (height, width):
        union = np.asarray(
            Image.fromarray(union, mode="L").resize(
                (width, height), resample=Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
    return n_instances, union


def symlink_or_copy(src, dst):
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def build_temp_sequence(image_paths):
    tmp = tempfile.TemporaryDirectory(prefix="sam3_seq_")
    seq_dir = Path(tmp.name)
    for i, p in enumerate(image_paths):
        ext = p.suffix.lower() if p.suffix.lower() in IMAGE_EXTS else ".jpg"
        dst = seq_dir / f"{i:06d}{ext}"
        symlink_or_copy(str(p), str(dst))
    return tmp, seq_dir


def load_image_size(image_path):
    with Image.open(image_path) as im:
        return im.size


def close_session_fast(predictor, session_id):
    """Close a SAM 3 session without per-session gc.collect()/empty_cache().

    SAM 3's close_session() already clears the session/state dictionaries and
    releases their tensor references. Disabling run_gc_collect keeps PyTorch's
    CUDA caching allocator warm for the next camera sequence and avoids the
    forced CUDA synchronization from empty_cache(). This changes allocation
    behavior only; model outputs are unchanged.
    """
    return predictor.handle_request(
        {
            "type": "close_session",
            "session_id": session_id,
            "run_gc_collect": False,
        }
    )


def pick_foreground_seed_point(mask):
    """Pick one foreground pixel used only to register obj_id / tracker state.

    SAM 3's high-level video request wrapper exposes points but not masks.  The
    underlying tracker *does* expose add_new_mask().  We therefore use one
    temporary positive point only to let the official instance-interactivity
    path allocate/register obj_id=1 and create its tracker state.  Immediately
    afterwards add_new_mask() replaces the point input on the same frame, so the
    conditioning prompt used for propagation is the full Stage-1 mask.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    # Pick the foreground pixel nearest the mask centroid.  This is guaranteed
    # to lie inside the Stage-1 mask and is more stable than the bbox center for
    # non-convex objects.
    cx = float(xs.mean())
    cy = float(ys.mean())
    dist2 = (xs.astype(np.float64) - cx) ** 2 + (ys.astype(np.float64) - cy) ** 2
    k = int(np.argmin(dist2))
    return [[float(xs[k]), float(ys[k])]]


def add_stage1_mask_prompt(predictor, session_id, frame_idx, mask_u8, obj_id=1):
    """Inject a full binary mask into the SAM 3 tracker for one live session.

    The public Sam3BasePredictor.add_prompt()/handle_request() route currently
    exposes text/box/point prompts, while the underlying Sam3TrackerPredictor
    exposes add_new_mask().  This helper bridges the two APIs for the single-GPU
    predictor used by this script.

    Steps:
      1. Use one temporary positive point to go through SAM 3's official
         add_tracker_new_points() path.  This allocates obj_id and creates the
         per-object tracker inference state + metadata.
      2. Call tracker.add_new_mask() on that state with the *entire* Stage-1
         binary mask.  add_new_mask() removes the point input on this frame.
      3. Run tracker preflight again so the mask, not the temporary point, is
         encoded into conditioning memory for propagation.

    Returns a response shaped like handle_request(add_prompt), so the rest of
    this script can keep the same bookkeeping code.
    """
    mask_bool_np = np.asarray(mask_u8) > 0
    if mask_bool_np.ndim != 2 or not mask_bool_np.any():
        return None

    # This script constructs build_sam3_predictor(..., gpus_to_use=[gpu_index]),
    # so the predictor is intentionally single-GPU.  Direct access to the local
    # model/session state would need explicit worker dispatch in true multi-GPU.
    if int(getattr(predictor, "world_size", 1)) != 1:
        raise RuntimeError(
            "Direct Stage-2 mask prompting currently requires a single-GPU SAM 3 predictor. "
            "This script already builds it with exactly one GPU via gpus_to_use=[gpu_index]."
        )

    if not hasattr(predictor, "model"):
        raise RuntimeError("SAM 3 predictor has no local .model attribute; cannot access tracker.add_new_mask().")

    model = predictor.model
    tracker = getattr(model, "tracker", None)
    if tracker is None or not hasattr(tracker, "add_new_mask"):
        raise RuntimeError(
            "This SAM 3 installation does not expose model.tracker.add_new_mask(). "
            "Please update to a SAM 3 version with Sam3TrackerPredictor.add_new_mask()."
        )
    if not hasattr(model, "add_tracker_new_points"):
        raise RuntimeError(
            "This SAM 3 video model does not expose add_tracker_new_points(), which is "
            "needed to register a new tracker object before injecting its mask."
        )

    # Get the high-level video inference state created by start_session().
    session = predictor._get_session(session_id)
    inference_state = session["state"]

    seed_points = pick_foreground_seed_point(mask_u8)
    if seed_points is None:
        return None

    # Bootstrap ONLY: register obj_id and create SAM 3's tracker state/metadata.
    # rel_coordinates=False because seed_points are pixel coordinates.
    model.add_tracker_new_points(
        inference_state=inference_state,
        frame_idx=int(frame_idx),
        obj_id=int(obj_id),
        points=torch.tensor(seed_points, dtype=torch.float32),
        labels=torch.ones(len(seed_points), dtype=torch.int32),
        rel_coordinates=False,
        use_prev_mem_frame=False,
    )

    tracker_states = model._get_tracker_inference_states_by_obj_ids(
        inference_state, [int(obj_id)]
    )
    if len(tracker_states) != 1:
        raise RuntimeError(
            f"Expected exactly one tracker state for obj_id={obj_id}, got {len(tracker_states)}."
        )
    tracker_state = tracker_states[0]

    # REAL Stage-2 prompt: full Stage-1 binary mask.  add_new_mask() internally
    # removes point_inputs_per_frame[frame_idx], so the temporary bootstrap point
    # is no longer the conditioning input on this frame.
    mask_tensor = torch.from_numpy(mask_bool_np).to(dtype=torch.bool)
    _, obj_ids, _low_res_masks, video_res_masks = tracker.add_new_mask(
        inference_state=tracker_state,
        frame_idx=int(frame_idx),
        obj_id=int(obj_id),
        mask=mask_tensor,
    )

    # Consolidate the mask prompt and run the memory encoder now, before temporal
    # propagation.  This makes the Stage-1 mask the tracker conditioning memory.
    tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)

    # Keep the high-level frame cache consistent with the exact Stage-1 mask.
    if getattr(model, "rank", 0) == 0:
        cached = inference_state.setdefault("cached_frame_outputs", {}).setdefault(
            int(frame_idx), {}
        )
        cached[int(obj_id)] = mask_tensor.to(model.device).unsqueeze(0)

    # Return the exact input mask in the same output shape used by the public API.
    return {
        "frame_index": int(frame_idx),
        "outputs": {
            "out_obj_ids": np.asarray([int(obj_id)], dtype=np.int64),
            "out_binary_masks": mask_bool_np[None, ...],
        },
    }

def make_predictor(checkpoint, gpu_index, args, compile_enabled=None):
    # This script uses the original SAM 3 video predictor/tracker internals
    # (including model.tracker.add_new_mask()). SAM 3.1 multiplex checkpoints
    # require the dedicated multiplex predictor path and are NOT drop-in weights
    # for build_sam3_predictor(version="sam3"). Fail early with a concise error
    # instead of printing thousands of missing/unexpected state_dict keys.
    ckpt_name = Path(checkpoint).name.lower()
    if "sam3.1" in ckpt_name or "multiplex" in ckpt_name:
        raise RuntimeError(
            "This two-stage script currently targets the original SAM 3 video predictor. "
            f"Checkpoint {checkpoint} is a SAM 3.1 multiplex checkpoint and is not "
            "compatible with build_sam3_predictor(version='sam3'). Use checkpoints/sam3.pt "
            "for this script, or port the whole Stage-1/Stage-2 tracker code to the SAM 3.1 "
            "multiplex predictor API."
        )

    if compile_enabled is None:
        compile_enabled = bool(args.compile)

    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint),
        version="sam3",
        compile=bool(compile_enabled),
        async_loading_frames=args.async_loading_frames,
        gpus_to_use=[gpu_index],
    )
    if hasattr(predictor, "model") and hasattr(predictor.model, "score_threshold_detection"):
        predictor.model.score_threshold_detection = float(args.threshold)
    return predictor


@torch.inference_mode()
def run_stage1(predictor, sequences, folder_root, output_root, args):
    """
    Stage 1: build a virtual video from the first image of each camera folder and
    propagate from the first camera's first image using the text prompt.
    """
    first_images = [seq["images"][0] for seq in sequences]
    tmp_ctx, seq_dir = build_temp_sequence(first_images)
    session_id = None

    stage1_masks = {}
    stats = {"written": 0, "skipped": 0, "empty": 0, "instances": 0}

    try:
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(seq_dir),
                "offload_video_to_cpu": args.offload_video_to_cpu,
                "offload_state_to_cpu": args.offload_state_to_cpu,
            }
        )
        session_id = response["session_id"]

        prompt_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": args.text_input,
                "output_prob_thresh": args.threshold,
            }
        )

        first_size = sequences[0]["frame_size"]
        n_instances, union = outputs_to_union_mask(prompt_response.get("outputs", {}), first_size)
        stage1_masks[0] = union

        stream = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "output_prob_thresh": args.threshold,
            }
        )

        for response in tqdm(stream, total=len(first_images), desc="Stage 1: cross-camera first-frame propagation", unit="frame"):
            frame_idx = response.get("frame_index")
            if frame_idx is None:
                continue
            frame_idx = int(frame_idx)
            if not (0 <= frame_idx < len(first_images)):
                continue
            image_path = first_images[frame_idx]
            frame_size = sequences[frame_idx]["frame_size"]
            n_instances, union = outputs_to_union_mask(response.get("outputs", {}), frame_size)
            stage1_masks[frame_idx] = union

        for i, seq in enumerate(sequences):
            image_path = seq["images"][0]
            out_path = mask_output_path(output_root, folder_root, image_path)
            frame_size = seq["frame_size"]
            union = stage1_masks.get(i)
            if union is None:
                union = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)
                n_instances = 0
            else:
                n_instances = int((union > 0).any())

            if args.skip_existing and out_path.is_file():
                stats["skipped"] += 1
            elif n_instances > 0 or args.empty_if_no_detection:
                save_mask(union, out_path, args.png_compression)
                stats["written"] += 1

            if union.max() == 0:
                stats["empty"] += 1
            stage1_masks[str(seq["rel_dir"])] = union

        return stage1_masks, stats

    finally:
        if session_id is not None:
            try:
                close_session_fast(predictor, session_id)
            except Exception:
                pass
        tmp_ctx.cleanup()


@torch.inference_mode()
def run_stage2_sam2_for_sequence(
    predictor, seq, folder_root, output_root, first_mask, args, writer_pool=None
):
    images = seq["images"]
    written = skipped = empty = total_instances = 0
    write_futures = []

    def submit_mask(mask, path):
        if writer_pool is None:
            save_mask(mask, path, args.png_compression)
        else:
            write_futures.append(
                writer_pool.submit(save_mask, mask, path, args.png_compression)
            )

    if args.skip_existing:
        out_paths = [mask_output_path(output_root, folder_root, p) for p in images]
        if all(p.is_file() for p in out_paths):
            return {
                "written": 0,
                "skipped": len(images),
                "empty": 0,
                "instances": 0,
                "used_mask_prompt": True,
            }

    first_mask = np.asarray(first_mask, dtype=np.uint8)
    has_prompt = bool(first_mask.max() > 0)

    # SAM2.1 has no text prompt. If SAM3 Stage 1 missed this camera, do not waste
    # GPU time loading and propagating the whole sequence.
    if not has_prompt:
        if args.empty_if_no_detection:
            zero = np.zeros(
                (seq["frame_size"][1], seq["frame_size"][0]), dtype=np.uint8
            )
            for image_path in images:
                out_path = mask_output_path(output_root, folder_root, image_path)
                if args.skip_existing and out_path.is_file():
                    skipped += 1
                    continue
                submit_mask(zero, out_path)
                written += 1
                empty += 1
            for future in write_futures:
                future.result()
        return {
            "written": written,
            "skipped": skipped,
            "empty": empty,
            "instances": 0,
            "used_mask_prompt": False,
        }

    # Preserve the exact SAM3 first-frame mask.
    first_out_path = mask_output_path(output_root, folder_root, images[0])
    if not first_out_path.is_file():
        submit_mask(first_mask, first_out_path)
        written += 1

    inference_state = None
    seen = {0}

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(
                video_path=str(seq["dir"]),
                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                async_loading_frames=bool(args.async_loading_frames),
            )

            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                mask=(first_mask > 0),
            )

            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0,
                reverse=False,
            ):
                frame_idx = int(frame_idx)
                if not (0 <= frame_idx < len(images)):
                    continue
                seen.add(frame_idx)

                if frame_idx == 0:
                    continue

                out_path = mask_output_path(
                    output_root, folder_root, images[frame_idx]
                )

                if args.skip_existing and out_path.is_file():
                    skipped += 1
                    continue

                union = sam2_logits_to_mask(
                    mask_logits,
                    seq["frame_size"],
                    score_thresh=args.sam2_score_thresh,
                )
                is_empty = not bool(union.max())
                if is_empty:
                    empty += 1
                else:
                    total_instances += 1

                if (not is_empty) or args.empty_if_no_detection:
                    submit_mask(union, out_path)
                    written += 1

            if args.empty_if_no_detection:
                zero = None
                for frame_idx, image_path in enumerate(images):
                    if frame_idx in seen:
                        continue
                    out_path = mask_output_path(
                        output_root, folder_root, image_path
                    )
                    if args.skip_existing and out_path.is_file():
                        skipped += 1
                        continue
                    if zero is None:
                        zero = np.zeros(
                            (seq["frame_size"][1], seq["frame_size"][0]),
                            dtype=np.uint8,
                        )
                    submit_mask(zero, out_path)
                    written += 1
                    empty += 1

    finally:
        if inference_state is not None:
            try:
                predictor.reset_state(inference_state)
            except Exception:
                pass
            del inference_state

    for future in write_futures:
        future.result()

    return {
        "written": written,
        "skipped": skipped,
        "empty": empty,
        "instances": total_instances,
        "used_mask_prompt": True,
    }


def stage2_sam2_gpu_worker(
    gpu_index,
    worker_slot,
    sequences,
    folder_root,
    output_root,
    args,
    result_queue,
):
    predictor = None
    writer_pool = None
    worker_totals = {
        "written": 0,
        "skipped": 0,
        "empty": 0,
        "instances": 0,
        "folders": 0,
    }

    try:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        torch.cuda.set_device(gpu_index)
        args.worker_gpu = gpu_index
        args.tqdm_position = worker_slot
        setup_cuda(f"cuda:{gpu_index}")

        print(
            f"[SAM2.1 worker {worker_slot + 1}] cuda:{gpu_index} "
            f"({torch.cuda.get_device_name(gpu_index)}) "
            f"model={args.sam2_model}, vos_optimized={bool(args.compile)}, "
            f"compile_mode={args.sam2_compile_mode}, folders={[str(seq['rel_dir']) for seq in sequences]}",
            flush=True,
        )

        predictor = make_sam2_predictor(gpu_index, args)
        writer_pool = ThreadPoolExecutor(
            max_workers=max(1, int(args.mask_writer_threads)),
            thread_name_prefix=f"mask-writer-gpu{gpu_index}",
        )

        for seq in sequences:
            first_mask = seq.get("stage1_mask")
            if first_mask is None:
                first_path = mask_output_path(
                    output_root, folder_root, seq["images"][0]
                )
                if first_path.is_file():
                    with Image.open(first_path) as im:
                        first_mask = np.asarray(im.convert("L"), dtype=np.uint8)
                else:
                    first_mask = np.zeros(
                        (seq["frame_size"][1], seq["frame_size"][0]),
                        dtype=np.uint8,
                    )

            stats = run_stage2_sam2_for_sequence(
                predictor=predictor,
                seq=seq,
                folder_root=folder_root,
                output_root=output_root,
                first_mask=first_mask,
                args=args,
                writer_pool=writer_pool,
            )

            mode = "SAM2-mask" if stats["used_mask_prompt"] else "empty-stage1"
            print(
                f"[cuda:{gpu_index}] {seq['rel_dir']}: prompt={mode}, "
                f"written={stats['written']}, skipped={stats['skipped']}, "
                f"empty={stats['empty']}",
                flush=True,
            )

            for key in ("written", "skipped", "empty", "instances"):
                worker_totals[key] += stats[key]
            worker_totals["folders"] += 1

        result_queue.put({
            "ok": True,
            "gpu": gpu_index,
            "worker_slot": worker_slot,
            "stats": worker_totals,
        })

    except Exception as exc:
        import traceback
        result_queue.put({
            "ok": False,
            "gpu": gpu_index,
            "worker_slot": worker_slot,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise

    finally:
        if writer_pool is not None:
            try:
                writer_pool.shutdown(wait=True)
            except Exception:
                pass
        if predictor is not None:
            del predictor
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    args = parse_args()

    if args.workers_per_gpu < 1:
        raise ValueError("--workers_per_gpu must be >= 1")
    if args.mask_writer_threads < 1:
        raise ValueError("--mask_writer_threads must be >= 1")

    folder, output_root, checkpoint = resolve_paths(args)
    sam2_repo, sam2_config, sam2_checkpoint = resolve_sam2_paths(args)
    args.sam2_repo = sam2_repo
    args.sam2_config = sam2_config
    args.sam2_checkpoint = sam2_checkpoint

    initial_gpu_indices = parse_gpu_indices(args.gpus, args.min_free_vram_mb)
    stage1_gpu = (
        initial_gpu_indices[0]
        if args.stage1_gpu is None
        else int(args.stage1_gpu)
    )
    if stage1_gpu not in initial_gpu_indices:
        raise ValueError(
            f"--stage1_gpu={stage1_gpu} must also be included in "
            f"--gpus={args.gpus!r}."
        )

    sequences = discover_cam_sequences(folder)
    if not sequences:
        raise RuntimeError(f"No supported images found under: {folder}")

    total_images = sum(len(seq["images"]) for seq in sequences)
    for seq in sequences:
        (output_root / seq["rel_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"[Input]          {folder}")
    print(f"[Output]         {output_root}")
    print(f"[Prompt]         {args.text_input!r}")
    print(f"[SAM3 ckpt]      {checkpoint}")
    print(f"[SAM2 model]     {args.sam2_model}")
    print(f"[SAM2 config]    {sam2_config}")
    print(f"[SAM2 ckpt]      {sam2_checkpoint}")
    print(f"[SAM2 compile]   {bool(args.compile)} · mode={args.sam2_compile_mode}")
    print(f"[SAM2 postproc]  {bool(args.sam2_postprocess)}")
    print(f"[SAM2 threshold] {args.sam2_score_thresh}")
    print(f"[Cam folders]    {len(sequences)}")
    print(f"[Images total]   {total_images}")
    print(f"[Async loader]   {args.async_loading_frames}")
    print(f"[Video on CPU]   {args.offload_video_to_cpu}")
    print(f"[State on CPU]   {args.offload_state_to_cpu}")
    print(f"[Stage 1 GPU]    cuda:{stage1_gpu}")
    print(f"[Workers/GPU]    {args.workers_per_gpu}")
    print(
        f"[PNG writers]    {args.mask_writer_threads}/worker, "
        f"compression={args.png_compression}"
    )
    print(
        f"[Compile cache]  {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}"
    )

    total_written = total_skipped = total_empty = total_instances = 0

    print("\n=== Stage 1: SAM3 cross-camera first-frame initialization ===")
    setup_cuda(f"cuda:{stage1_gpu}")
    predictor = make_predictor(
        checkpoint,
        stage1_gpu,
        args,
        compile_enabled=args.compile_stage1,
    )
    try:
        stage1_masks, stage1_stats = run_stage1(
            predictor, sequences, folder, output_root, args
        )
        total_written += stage1_stats["written"]
        total_skipped += stage1_stats["skipped"]
        total_empty += stage1_stats["empty"]
        total_instances += stage1_stats["instances"]

        # Avoid re-decoding first-mask PNGs in Stage 2.
        for seq in sequences:
            mask = stage1_masks.get(str(seq["rel_dir"]))
            if mask is not None:
                seq["stage1_mask"] = np.asarray(mask, dtype=np.uint8)
    finally:
        if hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception as exc:
                print(
                    f"[WARNING] Stage-1 predictor shutdown raised: {exc}",
                    file=sys.stderr,
                )
        del predictor
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n=== Stage 2: SAM2.1 multi-GPU per-camera propagation ===")
    gpu_indices = parse_gpu_indices(args.gpus, args.min_free_vram_mb)
    assignments = assign_sequences_to_gpus(
        sequences, gpu_indices, args.folders_per_gpu
    )
    print(f"[Stage 2 GPUs]   {gpu_indices}")

    active = []
    for gpu in gpu_indices:
        seqs = assignments[gpu]
        if not seqs:
            continue
        n_slots = min(int(args.workers_per_gpu), len(seqs))
        buckets = [seqs[i::n_slots] for i in range(n_slots)]
        for local_slot, bucket in enumerate(buckets):
            if bucket:
                active.append((gpu, local_slot, bucket))

    print("[Stage 2 assignment]")
    for worker_slot, (gpu, local_slot, seqs) in enumerate(active):
        print(
            f"  worker {worker_slot + 1}: cuda:{gpu} slot={local_slot + 1} "
            f"-> {[str(seq['rel_dir']) for seq in seqs]}"
        )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for worker_slot, (gpu, local_slot, seqs) in enumerate(active):
        worker_id = f"cuda{gpu}-slot{local_slot}"
        p = ctx.Process(
            target=stage2_sam2_gpu_worker,
            args=(gpu, worker_slot, seqs, folder, output_root, args, result_queue),
            name=f"sam2-stage2-{worker_id}",
        )
        p.start()
        processes.append((p, gpu, worker_id))

    results = []
    while len(results) < len(processes):
        try:
            results.append(result_queue.get(timeout=1.0))
        except queue.Empty:
            if all(not p.is_alive() for p, _, _ in processes):
                break

    for p, _, _ in processes:
        p.join()

    while len(results) < len(processes):
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break

    failures = [r for r in results if not r.get("ok", False)]
    reported_slots = {r.get("worker_slot") for r in results}

    for worker_slot, (p, gpu, worker_id) in enumerate(processes):
        if p.exitcode != 0 and worker_slot not in reported_slots:
            failures.append({
                "gpu": gpu,
                "worker_slot": worker_slot,
                "error": (
                    f"{worker_id} exited with code {p.exitcode} "
                    "without reporting details"
                ),
                "traceback": "",
            })

    for r in results:
        if not r.get("ok", False):
            continue
        stats = r["stats"]
        total_written += stats["written"]
        total_skipped += stats["skipped"]
        total_empty += stats["empty"]
        total_instances += stats["instances"]

    if failures:
        print(
            "\n[ERROR] One or more SAM2.1 Stage-2 GPU workers failed:",
            file=sys.stderr,
        )
        for r in failures:
            print(
                f"\n--- cuda:{r.get('gpu')} worker_slot={r.get('worker_slot')} ---\n"
                f"{r.get('error', 'unknown error')}\n"
                f"{r.get('traceback', '')}",
                file=sys.stderr,
            )
        raise RuntimeError(
            f"{len(failures)} SAM2.1 Stage-2 GPU worker(s) failed."
        )

    if len(results) != len(processes):
        raise RuntimeError(
            f"Expected {len(processes)} Stage-2 worker result(s), "
            f"got {len(results)}."
        )

    print("\nDone.")
    print(f"  mask files written:       {total_written}")
    print(f"  existing masks skipped:   {total_skipped}")
    print(f"  empty masks / misses:     {total_empty}")
    print(f"  non-empty tracked frames: {total_instances}")
    print(f"  masks saved to:           {output_root}")


if __name__ == "__main__":
    main()
