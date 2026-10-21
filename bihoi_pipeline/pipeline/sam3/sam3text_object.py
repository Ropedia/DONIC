#!/usr/bin/env python3
"""
SAM 3 text-prompt VIDEO-mode segmentation for an image folder.

Unlike image mode, this script keeps one SAM 3 video inference session alive:
    text prompt on one frame
        -> SAM 3 detector initializes object(s)
        -> tracker propagates through the ordered images
        -> temporal memory / inference state is preserved across frames

Expected layout:
    ../<object_name>/images/
        1.jpg
        2.jpg
        3.jpg
        ...

Masks are written to:
    ../<object_name>/mask/
        1.png
        2.png
        3.png
        ...

Example:
    python sam3text_object_video.py \
        --object_name bottle \
        --text_input bottle

The image folder is treated as a video sequence. If all filename stems are
integers, frames are ordered numerically (1, 2, ..., 10); otherwise they are
ordered lexicographically.

If multiple tracked instances match the text prompt, their masks are unioned
into one single-channel uint8 mask:
    background = 0
    foreground = 255
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3 import build_sam3_predictor


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM 3 VIDEO-mode text-prompt segmentation on images directly "
            "under ../<object_name>/images/, preserving tracker memory/state "
            "across the ordered images."
        )
    )
    parser.add_argument(
        "--object_name",
        type=str,
        required=True,
        help=(
            "Object folder name. Images are read from "
            "../<object_name>/images/ and masks are saved to "
            "../<object_name>/mask/."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".."),
        help="Parent directory containing <object_name>. Default: ..",
    )
    parser.add_argument(
        "--text_input",
        type=str,
        required=True,
        help='Text prompt, e.g. "bottle".',
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Local SAM 3 checkpoint. Default: <script_dir>/checkpoints/sam3.pt. "
            "A local checkpoint is always supplied to the predictor, so this "
            "script does not need Hugging Face checkpoint download."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold used for SAM 3 output masks and, when available, "
            "the video detector score threshold. Default: 0.5"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            'CUDA device, e.g. "cuda" or "cuda:0". '
            "The official SAM 3 video predictor is CUDA-oriented. Default: cuda"
        ),
    )
    parser.add_argument(
        "--prompt_frame",
        type=int,
        default=0,
        help=(
            "0-based ordered-frame index on which the text prompt is first "
            "applied. Default: 0 (the first image)."
        ),
    )
    parser.add_argument(
        "--propagation_direction",
        choices=["forward", "backward", "both"],
        default="forward",
        help=(
            "Tracking direction from --prompt_frame. For prompt_frame=0, "
            "'forward' is normally what you want. Default: forward"
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Do not overwrite mask PNGs that already exist. Note: SAM 3 still "
            "propagates through those frames internally so memory/state remains "
            "continuous."
        ),
    )
    parser.add_argument(
        "--empty_if_no_detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save an all-zero mask for a frame when SAM 3 returns no tracked "
            "object. Default: enabled."
        ),
    )
    parser.add_argument(
        "--offload_video_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep loaded video/image-frame tensors on CPU when possible to save "
            "VRAM. Tracker state/memory still works normally. Default: enabled."
        ),
    )
    parser.add_argument(
        "--offload_state_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Offload tracker inference state/memory to CPU to save more VRAM, "
            "at some speed cost. Default: disabled."
        ),
    )
    parser.add_argument(
        "--async_loading_frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use SAM 3 asynchronous frame loading. For a short image folder, "
            "disabled is simpler and deterministic. Default: disabled."
        ),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable predictor compilation if supported. Default: disabled.",
    )
    return parser.parse_args()


def resolve_paths(args):
    object_root = (args.root.expanduser() / args.object_name).resolve()
    image_folder = object_root / "images1"
    output = object_root / "mask"

    if not image_folder.is_dir():
        raise FileNotFoundError(
            "Input image folder does not exist:\n"
            f"  {image_folder}"
        )

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
    return image_folder, output, checkpoint


def find_images(folder):
    """Match SAM 3's image-folder video ordering as closely as possible."""
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not images:
        return []

    # Official SAM 3 video folder loader sorts integer stems numerically and
    # falls back to lexicographic sorting otherwise.
    try:
        return sorted(images, key=lambda p: int(p.stem))
    except ValueError:
        return sorted(images, key=lambda p: p.name)


def setup_cuda(device_str):
    device = torch.device(device_str)

    if device.type != "cuda":
        raise RuntimeError(
            "The official SAM 3 video predictor currently expects CUDA. "
            f"Received --device {device_str!r}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {device_str} requested, but CUDA is not available."
        )

    if device.index is None:
        gpu_index = torch.cuda.current_device()
    else:
        gpu_index = device.index
        torch.cuda.set_device(gpu_index)

    actual_device = torch.device(f"cuda:{gpu_index}")
    gpu = torch.cuda.get_device_name(actual_device)
    capability = torch.cuda.get_device_capability(actual_device)

    print(
        f"[GPU]        {gpu}, cuda:{gpu_index}, "
        f"compute capability {capability[0]}.{capability[1]}"
    )

    if capability[0] < 8:
        print(
            "[WARNING]    The current official SAM 3 video stack uses BF16 "
            "autocast internally. Pre-Ampere GPUs such as RTX 2080 Ti may "
            "not support this path reliably.",
            file=sys.stderr,
        )

    return gpu_index


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def union_video_masks(outputs, frame_size):
    """
    Convert SAM 3 video output to one union uint8 mask.

    Expected outputs key:
        out_binary_masks: [N, H, W] or [N, 1, H, W]
    """
    width, height = frame_size

    if outputs is None:
        masks = None
    else:
        masks = outputs.get("out_binary_masks")

    masks = to_numpy(masks)

    if masks is None or masks.size == 0:
        return 0, np.zeros((height, width), dtype=np.uint8)

    # A single [H, W] mask -> [1, H, W].
    if masks.ndim == 2:
        masks = masks[None, ...]

    # [N, 1, H, W] -> [N, H, W].
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim != 3:
        raise RuntimeError(
            "Unexpected SAM 3 video mask shape: "
            f"{tuple(masks.shape)}"
        )

    union = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255
    n_instances = int(masks.shape[0])

    # Normally SAM 3 already returns original-resolution masks. Keep a safe
    # nearest-neighbor resize in case a different repo revision returns another
    # resolution.
    if union.shape != (height, width):
        union = np.asarray(
            Image.fromarray(union, mode="L").resize(
                (width, height),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        )

    return n_instances, union


def save_mask(mask, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path)


def expected_frame_indices(num_frames, prompt_frame, direction):
    if direction == "forward":
        return range(prompt_frame, num_frames)
    if direction == "backward":
        return range(0, prompt_frame + 1)
    return range(0, num_frames)


def main():
    args = parse_args()
    folder, output_root, checkpoint = resolve_paths(args)
    gpu_index = setup_cuda(args.device)

    images = find_images(folder)
    if not images:
        raise RuntimeError(f"No supported images found under: {folder}")

    if not (0 <= args.prompt_frame < len(images)):
        raise ValueError(
            f"--prompt_frame must be in [0, {len(images) - 1}], "
            f"got {args.prompt_frame}."
        )

    with Image.open(images[0]) as im:
        frame_size = im.size

    print(f"[Input]      {folder}")
    print(f"[Output]     {output_root}")
    print(f"[Prompt]     {args.text_input!r}")
    print(f"[Prompt frm] {args.prompt_frame} -> {images[args.prompt_frame].name}")
    print(f"[Direction]  {args.propagation_direction}")
    print(f"[Checkpoint] {checkpoint}")
    print(f"[Threshold]  {args.threshold}")
    print(f"[Images]     {len(images)}")
    print(f"[Frame size] {frame_size[0]} x {frame_size[1]}")
    print(f"[Offload vid] {args.offload_video_to_cpu}")
    print(f"[Offload mem] {args.offload_state_to_cpu}")

    # SAM 3 video mode:
    #   build predictor once
    #   -> start one persistent session
    #   -> inject text on one frame
    #   -> propagate through the sequence using tracker memory/state
    #
    # Supplying checkpoint_path prevents checkpoint auto-download.
    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint),
        version="sam3",
        compile=args.compile,
        async_loading_frames=args.async_loading_frames,
        gpus_to_use=[gpu_index],
    )

    # The base SAM 3 video builder defaults this detector threshold to 0.5.
    # Preserve the old script's --threshold intent when the attribute is exposed.
    if hasattr(predictor, "model") and hasattr(
        predictor.model, "score_threshold_detection"
    ):
        predictor.model.score_threshold_detection = float(args.threshold)

    session_id = None
    processed = 0
    skipped = 0
    empty = 0
    total_instances = 0
    seen_frames = set()

    try:
        # 1) Create persistent inference state for the entire image sequence.
        response = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(folder),
                "offload_video_to_cpu": args.offload_video_to_cpu,
                "offload_state_to_cpu": args.offload_state_to_cpu,
            }
        )
        session_id = response["session_id"]
        print(f"[Session]    {session_id}")

        # 2) Run text grounding once on the prompt frame. The detected object(s)
        # become tracked objects in this same session.
        prompt_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": args.prompt_frame,
                "text": args.text_input,
                "output_prob_thresh": args.threshold,
            }
        )

        # Save prompt-frame output immediately as a fallback. Propagation usually
        # emits it again, in which case the propagated result simply overwrites it.
        prompt_outputs = prompt_response.get("outputs", {})
        n_instances, union = union_video_masks(prompt_outputs, frame_size)
        prompt_out = output_root / f"{images[args.prompt_frame].stem}.png"

        if args.skip_existing and prompt_out.is_file():
            skipped += 1
        elif n_instances > 0 or args.empty_if_no_detection:
            save_mask(union, prompt_out)
            processed += 1

        if n_instances == 0:
            empty += 1
        total_instances += n_instances
        seen_frames.add(args.prompt_frame)

        print(
            f"[Init]       prompt frame has {n_instances} tracked instance(s)"
        )

        # 3) Propagate using the same inference state / temporal memory.
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": args.propagation_direction,
            "start_frame_index": args.prompt_frame,
            "output_prob_thresh": args.threshold,
        }

        stream = predictor.handle_stream_request(request)

        for response in tqdm(
            stream,
            desc="SAM3 video propagation",
            unit="frame",
            total=len(
                list(
                    expected_frame_indices(
                        len(images),
                        args.prompt_frame,
                        args.propagation_direction,
                    )
                )
            ),
        ):
            frame_idx = response.get("frame_index")
            if frame_idx is None:
                continue

            frame_idx = int(frame_idx)
            if not (0 <= frame_idx < len(images)):
                print(
                    f"\n[WARNING] predictor returned out-of-range frame "
                    f"{frame_idx}; ignoring.",
                    file=sys.stderr,
                )
                continue

            image_path = images[frame_idx]
            out_path = output_root / f"{image_path.stem}.png"
            outputs = response.get("outputs", {})

            n_instances, union = union_video_masks(outputs, frame_size)

            # Avoid double-counting prompt frame if propagation emits it again.
            first_time_seen = frame_idx not in seen_frames
            seen_frames.add(frame_idx)

            if args.skip_existing and out_path.is_file():
                if first_time_seen:
                    skipped += 1
            elif n_instances > 0 or args.empty_if_no_detection:
                save_mask(union, out_path)
                if first_time_seen:
                    processed += 1

            if first_time_seen:
                if n_instances == 0:
                    empty += 1
                total_instances += n_instances

        # If SAM 3 emits no result for some expected frame, optionally create an
        # empty mask so downstream code still has one mask per input image.
        for frame_idx in expected_frame_indices(
            len(images),
            args.prompt_frame,
            args.propagation_direction,
        ):
            if frame_idx in seen_frames:
                continue

            image_path = images[frame_idx]
            out_path = output_root / f"{image_path.stem}.png"

            if args.skip_existing and out_path.is_file():
                skipped += 1
                continue

            if args.empty_if_no_detection:
                empty_mask = np.zeros(
                    (frame_size[1], frame_size[0]),
                    dtype=np.uint8,
                )
                save_mask(empty_mask, out_path)
                processed += 1
                empty += 1

    except Exception as exc:
        print(
            f"\n[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise

    finally:
        # Release the persistent video session and its memory.
        if session_id is not None:
            try:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )
            except Exception as exc:
                print(
                    f"[WARNING] failed to close SAM 3 session cleanly: {exc}",
                    file=sys.stderr,
                )

        if hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception as exc:
                print(
                    f"[WARNING] predictor shutdown raised: {exc}",
                    file=sys.stderr,
                )

    print("\nDone.")
    print(f"  frames written:           {processed}")
    print(f"  existing masks skipped:   {skipped}")
    print(f"  empty/no tracked object:  {empty}")
    print(f"  summed active instances:  {total_instances}")
    print(f"  masks saved to:           {output_root}")


if __name__ == "__main__":
    main()
