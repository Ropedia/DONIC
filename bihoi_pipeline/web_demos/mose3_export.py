"""Lossless prediction export and bounded-memory browser video encoding."""
import time
import zipfile
import os


def save_predictions(path, payload):
    import numpy as np
    # Dense float predictions compress poorly. Standard NPZ also permits stored
    # members: avoid costly DEFLATE while preserving every shape, dtype and value.
    # Level 1 is available when disk space matters more than export latency.
    level = int(os.environ.get("DEMO_NPZ_COMPRESSION", "0"))
    if level not in (0, 1):
        raise ValueError("DEMO_NPZ_COMPRESSION 必须为 0（无压缩）或 1（快速压缩）")
    temporary = path.with_suffix(".npz.tmp")
    started = time.perf_counter()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED if level else zipfile.ZIP_STORED,
                             compresslevel=1 if level else None, allowZip64=True) as archive:
            for name, array in payload.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    print(f"  predictions.npz 已保存（{elapsed:.1f} 秒）", flush=True)
    return elapsed


def save_video(path, images, fps, rendered=None, quality=5):
    import imageio.v2 as imageio
    import numpy as np
    # Concatenate one frame at a time, avoiding an extra full video-sized copy.
    with imageio.get_writer(path, fps=fps, quality=quality, macro_block_size=1,
                            ffmpeg_params=["-preset", "veryfast", "-threads", "4", "-movflags", "+faststart"]) as writer:
        for i, frame in enumerate(images):
            writer.append_data(frame if rendered is None else np.concatenate([frame, rendered[i]], axis=1))
