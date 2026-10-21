"""Keep sparse MoSE3 visualization videos as long as their source video."""
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import zipfile


VIDEO_NAMES = ("video.mp4", "tracks.mp4", "se3.mp4", "rigidity.mp4")


def probe_video(path):
    probe = shutil.which("ffprobe")
    if probe:
        result = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=duration,nb_frames:format=duration",
                                 "-of", "json", str(path)], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        duration = stream.get("duration", data.get("format", {}).get("duration"))
        if duration not in (None, "N/A"):
            duration = float(duration)
            if math.isfinite(duration) and duration > 0:
                count = stream.get("nb_frames")
                return duration, int(count) if count not in (None, "N/A") else None
    import imageio_ffmpeg
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("无法读取原视频的有效时长")
    return float(duration), count


def retime_video(path, duration, frame_count):
    """Preserve every sampled frame; give it an equal share of the source duration."""
    import imageio_ffmpeg
    path = Path(path)
    rate = (Fraction(frame_count, 1) / Fraction(str(duration))).limit_denominator(1_000_000)
    current_duration, current_count = probe_video(path)
    if current_count is not None and current_count != frame_count:
        raise ValueError(f"{path.name} 帧数与推理结果不一致：{current_count} != {frame_count}")
    if abs(current_duration - duration) <= 0.0001:
        return float(rate)
    target = path.with_name(path.stem + ".timing.mp4")
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                        "-map", "0:v:0", "-an",
                        "-vf", f"settb=AVTB,setpts=N*{rate.denominator}/({rate.numerator}*TB)",
                        "-r", f"{rate.numerator}/{rate.denominator}", "-vsync", "cfr",
                        "-frames:v", str(frame_count), "-c:v", "libx264", "-preset", "fast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-video_track_timescale", "1000000",
                        "-movflags", "+faststart", str(target)], capture_output=True, text=True, check=True)
        actual_duration, actual_count = probe_video(target)
        if abs(actual_duration - duration) > .005 or actual_count not in (None, frame_count):
            raise ValueError(f"{path.name} 时长校验失败：{actual_duration} 秒，目标 {duration} 秒")
        backup = path.parent / "original_timing" / path.name
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        target.replace(path)
    finally:
        target.unlink(missing_ok=True)
    return float(rate)


def align_result_videos(config, frame_count):
    source_duration, _ = probe_video(config["video"])
    fixed_rate = config.get("sampling_fps")
    duration = frame_count / fixed_rate if fixed_rate else source_duration
    output = Path(config["output"])
    for name in VIDEO_NAMES:
        if not (output / name).is_file():
            raise ValueError(f"缺少结果视频 {name}")
        rate = retime_video(output / name, duration, frame_count)
    timing = {"source_duration_seconds":source_duration, "output_duration_seconds":duration,
              "playback_fps":rate, "video_timing_version":2 if fixed_rate else 1,
              "playback_timing":"fixed 10 FPS sampling and playback" if fixed_rate else "sampled frames displayed uniformly over source video duration"}
    clip_path = output / "config.json"
    if clip_path.is_file():
        clip = json.loads(clip_path.read_text())
        clip.update(timing)
        clip["fps"] = rate
        temporary = clip_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(clip, indent=2))
        temporary.replace(clip_path)
    return timing


def repair_saved_result(directory):
    directory = Path(directory)
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("model") != "MoSE3":
        raise ValueError("仅支持 MoSE3 结果")
    config = json.loads((directory / "input.json").read_text())
    config["output"] = str(directory)
    result.update(align_result_videos(config, result["frames"]))
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.replace(result_path)
    archive_path = directory / "results.timing.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in directory.iterdir():
            if path.is_file() and path.name not in ("results.zip", "results.timing.zip", "input.json", "run.log"):
                archive.write(path, path.name)
    archive_path.replace(directory / "results.zip")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Existing MoSE3 job directory")
    args = parser.parse_args()
    result = repair_saved_result(args.directory)
    print(json.dumps({key:result[key] for key in ("frames", "source_duration_seconds", "output_duration_seconds", "playback_fps")}))
