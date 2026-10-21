"""Package completed results without recompressing large binary artifacts."""
from pathlib import Path
import zipfile


def result_files(directory):
    return [p for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.name not in ("input.json", "results.zip")
            and not p.name.endswith(".tmp")]


def build_result_archive(directory):
    directory = Path(directory)
    target = directory / "results.zip"
    files = [p for p in result_files(directory) if p.name != "run.log"]
    if target.is_file() and all(p.stat().st_mtime_ns <= target.stat().st_mtime_ns for p in files):
        return target
    temporary = directory / "results.zip.tmp"
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for path in files:
                compressed = path.suffix in (".npz", ".mp4", ".png", ".jpg", ".jpeg")
                archive.write(path, path.name, compress_type=zipfile.ZIP_STORED if compressed else zipfile.ZIP_DEFLATED,
                              compresslevel=None if compressed else 1)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
