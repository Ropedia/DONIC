#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr

try:
    from .settings import PIPELINE_ROOT, PROJECT_ROOT, CALIBRATION_DIR, MODEL_ROOT, environment, python_for
except ImportError:
    from settings import PIPELINE_ROOT, PROJECT_ROOT, CALIBRATION_DIR, MODEL_ROOT, environment, python_for

# Support both direct execution and importing the WebUI as a namespace package.
try:
    from .tracking_run import METHODS as TRACKING_METHODS, read_state as tracking_state
except ImportError:
    from tracking_run import METHODS as TRACKING_METHODS, read_state as tracking_state

APP_TITLE = "Hand–Object Interaction Pipeline v2"
EASYMOCAP_ROOT = PIPELINE_ROOT
# Contact-aware MANO refinement, export and final rendering share the pipeline root.
STAGE5_EASYMOCAP_ROOT = PIPELINE_ROOT
DATA_KIND = "rigid"
DEFAULT_PROMPT = "the single foreground object in the center of the image"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# Pipeline runs are executed by a detached runner process (this same file with
# --run-dir) inside a dedicated tmux server, so they survive browser reloads and
# even a restart of this web UI. Each run keeps meta.json / state.json / log.txt
# under RUNS_DIR and the page simply polls those files.
RUNS_DIR = Path(os.getenv("WEBUI_RUNS_DIR", str(Path(__file__).resolve().parent / ".webui_runs")))
TMUX_SOCKET = os.getenv("WEBUI_TMUX_SOCKET", "hoi_webui")
LOG_TAIL_CHARS = 120000
STATUS_ICON = {"queued": "⏳", "running": "▶️", "done": "✅", "failed": "❌", "stopped": "⏹️", "lost": "⚠️"}
ACTIVE_STATUSES = ("queued", "running")
_LAST_RUN_CHOICES = {"choices": None}
_LAST_STEP_CHOICES: Dict[Optional[str], Optional[list]] = {}

# GPU scheduling policy (default VRAM budget per GPU step, overridable per run in
# Advanced parameters): a GPU can take a step only when live free VRAM minus the
# reservations of just-launched steps is STRICTLY greater than this. Multi-GPU
# steps take every qualifying GPU; single-GPU steps take the one with the most
# headroom. The runner exports GPU_PICK / GPU_LIST / GPU_VRAM_MB to each step.
GPU_MIN_FREE_VRAM_MB = int(os.getenv("WEBUI_VRAM_PER_STEP_MB", "32768"))
GPU_BUSY_RC = 75  # EX_TEMPFAIL: a step found its assigned GPU(s) taken at launch -> scheduler requeues it
# GPUs the scheduler may use, e.g. "4,5,6,7" or "4-7"; empty / "all" = every GPU. Set per machine in the launcher.
DEFAULT_ALLOWED_GPUS = os.getenv("WEBUI_ALLOWED_GPUS", "all")


def _parse_gpu_list(text) -> List[int]:
    """'4,5,6,7' / '4-7' / '0,2-3' -> [4,5,6,7]; '' or 'all' -> [] (= no restriction)."""
    text = str(text or "").strip().lower()
    if text in ("", "all", "auto", "*"):
        return []
    out: List[int] = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


# Takes: one capture session (= dataset folder) can hold several recordings of the same
# object, videos/take01, videos/take02, ...  take01 is processed in the dataset folder
# itself (the layout every script and every already-processed dataset uses); every further
# take gets its own working folder <dataset>/takes/<take>/ with the same internal layout.
# Resources that do not depend on the recording are shared with take01 through symlinks:
# calib/ and the reconstructed object mesh (meshes/), so Stage 2 reconstruction runs once.
PRIMARY_TAKE = "take01"
TAKES_SUBDIR = "takes"
TAKE_RE = re.compile(r"^take\d+$")
# Steps of a take01 chain that later takes of the same dataset wait for (shared outputs).
MESH_RECON_STEP_TITLE = "MV-SAM3D metric-scale object reconstruction"


@dataclass
class Job:
    dataset: str
    hand_prior_name: str
    text_prompt: str = DEFAULT_PROMPT
    hand_side: str = "bimanual"
    take: str = PRIMARY_TAKE

    @property
    def is_primary_take(self) -> bool:
        return self.take == PRIMARY_TAKE

    @property
    def dataset_logical_name(self) -> str:
        """Logical name of the dataset folder (take01 = the dataset itself)."""
        return f"{DATA_KIND}/{self.hand_prior_name}/{self.dataset}"

    @property
    def logical_data_name(self) -> str:
        if self.is_primary_take:
            return self.dataset_logical_name
        return f"{self.dataset_logical_name}/{TAKES_SUBDIR}/{self.take}"

    @property
    def display_name(self) -> str:
        return self.dataset if self.is_primary_take else f"{self.dataset}/{self.take}"


@dataclass
class Step:
    title: str
    cwd: Path
    command: str
    conda_env: Optional[str] = None
    gpu: Optional[str] = None  # None = CPU only, "single" = uses $GPU_PICK, "multi" = uses $GPU_LIST
    # Optional cross-chain dependency {"chain": logical_data_name, "title": step title}: the
    # scheduler does not start this step before that step of the other chain is done.
    after: Optional[dict] = None


def _validate_name(value: str, field: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{field} is required")
    if value in (".", "..") or not NAME_RE.fullmatch(value):
        raise ValueError(
            f"Invalid {field}: {value!r}. Allowed characters: letters, digits, _, -, ."
        )
    return value


def _q(x) -> str:
    return shlex.quote(str(x))


def _data_root_path(root_str: str) -> Path:
    return Path(os.path.expanduser(str(root_str))).resolve()


def _ensure_pipeline_data_name(data_root: Path, job: Job, root: Path = EASYMOCAP_ROOT) -> str:
    """Return a path usable by EasyMocap-side scripts while code root stays fixed.

    If the data root lives inside ``root`` (EASYMOCAP_ROOT by default; Stage 5
    passes STAGE5_EASYMOCAP_ROOT), use a normal relative path. Otherwise create a
    stable symlink under ``root``/.webui_data_roots/ so scripts that internally
    prepend the EasyMocap root or `../` still work.
    """
    ds_abs = _dataset_root(data_root, job).resolve()
    try:
        return str(ds_abs.relative_to(root))
    except ValueError:
        import hashlib
        alias_root = root / ".webui_data_roots"
        alias_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(data_root).encode("utf-8")).hexdigest()[:12]
        link = alias_root / digest
        if link.is_symlink():
            current = link.resolve()
            if current != data_root.resolve():
                link.unlink()
        elif link.exists():
            raise ValueError(f"Data-root alias path exists but is not a symlink: {link}")
        if not link.exists():
            link.symlink_to(data_root.resolve(), target_is_directory=True)
        rel = Path(".webui_data_roots") / digest / DATA_KIND / job.hand_prior_name / job.dataset
        if not job.is_primary_take:
            rel = rel / TAKES_SUBDIR / job.take
        return str(rel)


def _prior_root(root: Path, hand_prior_name: str) -> Path:
    return root / DATA_KIND / hand_prior_name


def _ensure_pipeline_prior_name(data_root: Path, job: Job) -> str:
    """Return a hand-prior path usable by EasyMocap-side scripts.

    If the data root is inside EASYMOCAP_ROOT, return the normal relative path
    such as ``rigid/collector01``. Otherwise reuse the same .webui_data_roots symlink
    strategy as _ensure_pipeline_data_name(), so scripts that resolve the hand
    prior relative to EasyMocap can still find calib/restored_flat_real_scale.pt.
    """
    prior_abs = _prior_root(data_root, job.hand_prior_name).resolve()
    try:
        return str(prior_abs.relative_to(EASYMOCAP_ROOT))
    except ValueError:
        import hashlib

        alias_root = EASYMOCAP_ROOT / ".webui_data_roots"
        alias_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(data_root.resolve()).encode("utf-8")).hexdigest()[:12]
        link = alias_root / digest

        if link.is_symlink():
            current = link.resolve()
            if current != data_root.resolve():
                link.unlink()
        elif link.exists():
            raise ValueError(f"Data-root alias path exists but is not a symlink: {link}")

        if not link.exists():
            link.symlink_to(data_root.resolve(), target_is_directory=True)

        return str(Path(".webui_data_roots") / digest / DATA_KIND / job.hand_prior_name)


def _dataset_folder(root: Path, job: Job) -> Path:
    """The dataset folder (shared by all takes)."""
    return root / DATA_KIND / job.hand_prior_name / job.dataset


def _take_root(ds_folder: Path, take: str) -> Path:
    """Working folder of one take: the dataset folder for take01, <ds>/takes/<take>/ otherwise."""
    return ds_folder if take == PRIMARY_TAKE else ds_folder / TAKES_SUBDIR / take


def _dataset_root(root: Path, job: Job) -> Path:
    """Working folder of the job (dataset folder for take01, <ds>/takes/<take>/ for later takes)."""
    return _take_root(_dataset_folder(root, job), job.take)


def _has_video_files(folder: Path) -> bool:
    try:
        return folder.is_dir() and any(p.is_file() and p.suffix.lower() in VIDEO_EXTS_SET for p in folder.iterdir())
    except OSError:
        return False


def _dataset_takes(ds_folder: Path) -> List[str]:
    """Takes recorded for a dataset: videos/take01, take02, ... that contain video files.

    Returns [take01] when there is no take folder at all (already-processed datasets whose raw
    videos were removed still have their outputs in the dataset folder).
    """
    vdir = ds_folder / "videos"
    takes = []
    if vdir.is_dir():
        try:
            takes = sorted((p.name for p in vdir.iterdir() if p.is_dir() and TAKE_RE.match(p.name) and _has_video_files(p)),
                           key=_natural_key)
        except OSError:
            takes = []
    if PRIMARY_TAKE not in takes and (_has_video_files(vdir / PRIMARY_TAKE) or not takes):
        takes.insert(0, PRIMARY_TAKE)
    return takes


def _take_video_dir(root: Path, job: Job) -> Path:
    """Raw videos of this take (always under the dataset folder, never under takes/)."""
    return _dataset_folder(root, job) / "videos" / job.take


def _ensure_link(link: Path, target: Path) -> Optional[str]:
    """Create/refresh symlink `link` -> `target` (relative). Returns a note when something changed."""
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return None
        link.unlink()
    elif link.exists():
        return None  # a real folder of that name: the take has its own copy, leave it alone
    rel = os.path.relpath(target.resolve(), link.parent.resolve())
    link.symlink_to(rel, target_is_directory=True)
    return f"linked {link.name}/ -> {rel}"


def _prepare_take_dir(root: Path, job: Job) -> List[str]:
    """Create <ds>/takes/<take>/ for a secondary take and link the resources shared with take01."""
    notes: List[str] = []
    if job.is_primary_take:
        return notes
    ds_folder = _dataset_folder(root, job)
    take_root = _dataset_root(root, job)
    take_root.mkdir(parents=True, exist_ok=True)
    for shared in ("calib", "meshes"):
        src = ds_folder / shared
        if src.is_dir():
            n = _ensure_link(take_root / shared, src)
            if n:
                notes.append(n)
    return notes


def _rows(value):
    if value is None:
        return []
    if hasattr(value, "values") and hasattr(value, "columns"):
        return value.values.tolist()
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except Exception:
        return []


def _prompt_map(prompt_table) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for row in _rows(prompt_table):
        if row is None:
            continue
        if isinstance(row, dict):
            ds = row.get("dataset") or row.get("Dataset")
            prompt = row.get("text_prompt") or row.get("Text Prompt") or ""
        else:
            row = list(row) if isinstance(row, (tuple, list)) else [row]
            if not row:
                continue
            ds = row[0]
            prompt = row[1] if len(row) > 1 and row[1] is not None else ""
        if ds is None:
            continue
        try:
            ds = _validate_name(str(ds), "dataset")
        except ValueError:
            continue
        result[ds] = str(prompt or "").strip()
    return result


def _build_jobs(root_str, hand_prior_name, selected_datasets, prompt_table, hand_side,
                all_takes: bool = True) -> List[Job]:
    """One Job per (dataset, take). With all_takes every videos/take* of a dataset becomes a
    job (take01 first, so later takes can reuse its object mesh); otherwise only take01."""
    root = _data_root_path(root_str)
    if not root.is_dir():
        raise ValueError(f"Data root does not exist: {root}")
    if not (root / DATA_KIND).is_dir():
        raise ValueError(f"Data root must contain a `{DATA_KIND}/` folder: {root}")

    hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
    hs = _validate_name(str(hand_side or "bimanual"), "hand_side")
    if hs not in {"left", "right", "bimanual"}:
        raise ValueError("hand_side must be left, right, or bimanual")

    if isinstance(selected_datasets, str):
        selected_datasets = [selected_datasets]
    selected_datasets = list(selected_datasets or [])
    if not selected_datasets:
        raise ValueError("Select at least one dataset.")

    pmap = _prompt_map(prompt_table)
    jobs: List[Job] = []
    seen = set()
    for raw_ds in selected_datasets:
        ds = _validate_name(str(raw_ds), "dataset")
        if ds in seen:
            continue
        ds_root = root / DATA_KIND / hp / ds
        if not ds_root.is_dir():
            raise ValueError(f"Dataset directory does not exist: {ds_root}")
        prompt = pmap.get(ds, "").strip() or DEFAULT_PROMPT
        takes = _dataset_takes(ds_root) if all_takes else [PRIMARY_TAKE]
        for take in takes:
            jobs.append(Job(ds, hp, prompt, hs, take))
        seen.add(ds)
    return jobs


def _list_dirs(base: Path) -> List[str]:
    if not base.is_dir():
        return []
    return sorted(
        p.name for p in base.iterdir()
        if p.is_dir() and NAME_RE.fullmatch(p.name)
    )


def refresh_hand_priors(root_str: str, current_prior=None):
    try:
        root = _data_root_path(root_str)
        rigid_root = root / DATA_KIND
        choices = _list_dirs(rigid_root)
        current = str(current_prior or "")
        value = current if current in choices else (choices[0] if choices else None)
        status = f"Data Root: `{root}`  \nFound **{len(choices)}** hand-prior folder(s) under `{rigid_root}`."
        return status, gr.Dropdown(choices=choices, value=value)
    except Exception as exc:
        return f"⚠️ Could not scan `{DATA_KIND}`: {exc}", gr.Dropdown(choices=[], value=None)


# ---------------------------------------------------------------------------
# Dataset status markers (artifacts on disk + run records + manual rating)
# ---------------------------------------------------------------------------

STAGE_KEYS = ["preprocess", "masks", "recon", "meshscale", "tracking", "hands", "final"]


def _stage_order(stage: str) -> int:
    """Sort key for stage names; unknown stages (older/newer runs) go last instead of raising."""
    return STAGE_KEYS.index(stage) if stage in STAGE_KEYS else len(STAGE_KEYS)
STAGE_GLYPHS = "①②③④⑤"
RATING_CHOICES = ["unrated", "★ good", "? maybe", "✗ bad"]
RATING_MARK = {"★ good": "★", "? maybe": "?", "✗ bad": "✗"}


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


def _stage_artifacts(ds_root: Path) -> List[bool]:
    """Which pipeline stages already left their outputs on disk (independent of run records)."""
    images = ds_root / "images"
    try:
        s1 = images.is_dir() and any(p.is_dir() for p in images.iterdir()) and (ds_root / "masks").is_dir()
    except OSError:
        s1 = False
    s2 = (ds_root / "meshes" / "compressed.ply").is_file()
    s3 = (ds_root / "mesh_overlay_allcams_fast_green").is_dir() or (ds_root / "debug_depth" / "geometry_global_opt").is_dir()
    s4 = any((ds_root / "output" / m / "smpl").is_dir() for m in ("manol", "manor"))
    s5 = (ds_root / "standardized_hoi_30fps").is_dir() or (ds_root / "outputmano" / "contact_mano").is_dir()
    return [bool(s1), bool(s2), bool(s3), bool(s4), bool(s5)]


def _latest_run_by_dataset(runs: Optional[List[dict]] = None) -> Dict[str, dict]:
    """logical_data_name -> summary of the most recent run that touched it."""
    out: Dict[str, dict] = {}
    for m in (runs if runs is not None else _list_runs()):  # newest first
        step_states = m["state"].get("steps", {})
        per_ds: Dict[str, List[tuple]] = {}
        for i, s in enumerate(m.get("steps", [])):
            per_ds.setdefault(s["logical_data_name"], []).append((i, s, step_states.get(str(i), {})))
        for name, items in per_ds.items():
            if name in out:
                continue
            statuses = [ss.get("status", "pending") for _, _, ss in items]
            failed = [(s, ss) for _, s, ss in items if ss.get("status") == "failed"]
            running = [s for _, s, ss in items if ss.get("status") == "running"]
            if m["state"].get("status") in ACTIVE_STATUSES:
                mark, detail = "▶", (running[0]["title"] if running else "queued")
            elif failed:
                mark, detail = "❌", f"{failed[0][0]['stage']}: {failed[0][0]['title']}"
            elif all(st == "done" for st in statuses):
                mark, detail = "✅", "all steps done"
            elif m["state"].get("status") == "stopped":
                mark, detail = "⏹", "stopped"
            else:
                mark, detail = "⚠️", m["state"].get("status", "?")
            out[name] = {"run_id": m["run_id"], "mark": mark, "detail": detail,
                         "when": time.strftime("%m-%d %H:%M", time.localtime(float(m.get("created", 0)))),
                         "stages": sorted({s["stage"] for _, s, ss in items if ss.get("status") == "done"}, key=_stage_order)}
    return out


def _screening_path(root: Path, hp: str) -> Path:
    return _prior_root(root, hp) / ".webui_screening.json"


def _load_screening(root: Path, hp: str) -> Dict[str, dict]:
    return _read_json(_screening_path(root, hp), {}) or {}


def _save_screening(root: Path, hp: str, data: Dict[str, dict]) -> None:
    _write_json_atomic(_screening_path(root, hp), data)


def _nested_dir(ds_root: Path) -> Optional[Path]:
    """<ds>/<ds>/ created by unzipping an archive with a top-level folder, while <ds> itself
    has no videos/ or images/. Returns the inner directory in that case."""
    inner = ds_root / ds_root.name
    if inner.is_dir() and not (ds_root / "videos").exists() and not (ds_root / "images").exists():
        return inner
    return None


def _merge_up(src: Path, dst: Path, moved: List[str], kept: List[str], rel: str = "") -> None:
    """Move everything under src into dst. Same-named directories are merged recursively;
    a file that already exists in dst is kept there and the inner copy stays behind."""
    for item in sorted(src.iterdir()):
        name = f"{rel}{item.name}"
        target = dst / item.name
        if not target.exists():
            shutil.move(str(item), str(target))
            moved.append(name)
        elif item.is_dir() and target.is_dir() and not item.is_symlink():
            _merge_up(item, target, moved, kept, rel=name + "/")
            try:
                item.rmdir()
            except OSError:
                pass
        else:
            kept.append(name)


def _ensure_flat(ds_root: Path) -> Optional[str]:
    """Flatten <ds>/<ds>/… in place when present. Returns a one-line note, or None if nothing was done."""
    inner = _nested_dir(ds_root)
    if inner is None:
        return None
    moved: List[str] = []
    kept: List[str] = []
    _merge_up(inner, ds_root, moved, kept)
    try:
        inner.rmdir()
        left = ""
    except OSError:
        left = f"; `{inner.name}/` left in place with: {', '.join(kept[:6])}{'…' if len(kept) > 6 else ''}"
    return f"auto-flattened nested folder `{ds_root.name}/{ds_root.name}/` ({len(moved)} item(s) moved up{left})"


def flatten_nested_datasets(root_str: str, hand_prior_name, selected_datasets=None):
    """Flatten every nested dataset under the hand prior (the automatic per-dataset fallback does
    the same thing when a nested dataset is previewed or submitted)."""
    lines = []
    try:
        for rec in _dataset_records(root_str, hand_prior_name):
            note = _ensure_flat(rec["path"])
            if note:
                lines.append(f"- **{rec['name']}**: {note}")
        msg = ("### ✅ Flattened\n" + "\n".join(lines)) if lines else "### Nothing to do\nNo nested dataset folders found."
    except Exception as exc:
        msg = f"### ❌ Flatten failed\n{exc}"
    _, ds_update = refresh_datasets(root_str, hand_prior_name, selected_datasets)
    return msg, ds_update, _screen_dataset_update(root_str, hand_prior_name, None)


def _dir_has_files(folder: Path, suffixes: Optional[Sequence[str]] = None) -> bool:
    try:
        return folder.is_dir() and any(
            p.is_file() and (suffixes is None or p.suffix.lower() in suffixes) for p in folder.iterdir())
    except OSError:
        return False


def _stage_complete(ds_root: Path, stage: str, tracking_method=None) -> bool:
    """True when the stage's final output already exists in this take's folder, i.e. re-running
    it would only reproduce what is on disk. Used to skip takes/stages that were run before.
    'masks' (re-prompt) and 'meshscale' (optional check) are explicit re-runs: never complete."""
    if stage == "preprocess":
        images, masks = ds_root / "images", ds_root / "masks"
        try:
            cams = [p for p in images.iterdir() if p.is_dir()] if images.is_dir() else []
        except OSError:
            cams = []
        return bool(cams) and all(_dir_has_files(c) for c in cams) and all(_dir_has_files(masks / c.name) for c in cams)
    if stage == "recon":
        depth = ds_root / "depth"
        try:
            depth_ok = depth.is_dir() and any(_dir_has_files(p) for p in depth.iterdir() if p.is_dir())
        except OSError:
            depth_ok = False
        return (ds_root / "meshes" / "compressed.ply").is_file() and depth_ok
    if stage == "tracking":
        state = tracking_state(ds_root)
        if state.get("status") not in ("legacy", "complete"):
            return False
        if tracking_method is not None and state.get("method") != tracking_method:
            return False
        return (_dir_has_files(ds_root / "debug_depth" / "geometry_global_opt" / "ob_in_world", (".txt",))
                and (ds_root / "mesh_overlay_allcams_fast_green" / "mesh_overlay_allcams_fast_green.mp4").is_file())
    if stage == "hands":
        return any(_dir_has_files(ds_root / "output" / m / "smpl", (".json",)) for m in ("manol", "manor"))
    if stage == "final":
        video = ds_root / "outputmano" / "grid" / "final_hand_object_interaction.mp4"
        if not video.is_file() or not (ds_root / EXPORT_STANDARD_DIR).is_dir():
            return False
        state = tracking_state(ds_root)
        # A separately launched Stage 3 also invalidates the previous final export.
        return (state.get("status") in ("legacy", "complete")
                and video.stat().st_mtime >= float(state.get("completed", 0)))
    return False


def _dataset_records(root_str: str, hand_prior_name) -> List[dict]:
    """One record per dataset folder: name, stage flags, latest run, rating, notes."""
    root = _data_root_path(root_str)
    hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
    base = _prior_root(root, hp)
    latest = _latest_run_by_dataset()
    screening = _load_screening(root, hp)
    records = []
    for ds in _list_dirs(base):
        ds_root = base / ds
        logical = f"{DATA_KIND}/{hp}/{ds}"
        rec = screening.get(ds, {})
        takes = []
        for take in _dataset_takes(ds_root):
            if take == PRIMARY_TAKE:
                continue
            troot = _take_root(ds_root, take)
            t_stages = _stage_artifacts(troot) if troot.is_dir() else [False] * 5
            # The mesh is a symlink to take01's; count Stage 2 for the take only once its own depth exists.
            t_stages[1] = t_stages[1] and (troot / "depth").is_dir()
            takes.append({
                "name": take,
                "path": troot,
                "stages": t_stages,
                "run": latest.get(f"{logical}/{TAKES_SUBDIR}/{take}"),
            })
        records.append({
            "name": ds,
            "path": ds_root,
            "stages": _stage_artifacts(ds_root),
            "run": latest.get(logical),
            "rating": rec.get("rating", "unrated"),
            "notes": rec.get("notes", ""),
            "nested": _nested_dir(ds_root) is not None,
            "takes": takes,  # secondary takes only; take01 == the dataset itself
        })
    return records


def selection_summary(root_str: str, hand_prior_name, selected_datasets=None) -> str:
    """One-line count of the current selection, with a breakdown that matters before submitting."""
    try:
        recs = _dataset_records(root_str, hand_prior_name)
    except Exception:
        recs = []
    by_name = {r["name"]: r for r in recs}
    sel = [str(x) for x in (selected_datasets or []) if str(x) in by_name]
    if not recs:
        return "**Selected: 0**"
    if not sel:
        return f"**Selected: 0 / {len(recs)}** datasets"
    never_run = sum(1 for n in sel if not by_name[n]["run"])
    running = sum(1 for n in sel if by_name[n]["run"] and by_name[n]["run"]["mark"] == "▶")
    complete = sum(1 for n in sel if all(by_name[n]["stages"]))
    good = sum(1 for n in sel if by_name[n]["rating"] == "★ good")
    bad = sum(1 for n in sel if by_name[n]["rating"] == "✗ bad")
    nested = sum(1 for n in sel if by_name[n].get("nested"))
    extra_takes = sum(len(by_name[n].get("takes", [])) for n in sel)
    parts = [f"{never_run} never run", f"{complete} fully processed"]
    if extra_takes:
        parts.append(f"{extra_takes} extra take(s) beyond take01")
    if running:
        parts.append(f"{running} running now")
    if good:
        parts.append(f"{good} ★ good")
    if bad:
        parts.append(f"⚠️ {bad} rated ✗ bad")
    if nested:
        parts.append(f"{nested} nested (auto-flattened on submit)")
    return f"**Selected: {len(sel)} / {len(recs)}** datasets — " + " · ".join(parts)


def _dataset_label(rec: dict) -> str:
    stages = "".join(STAGE_GLYPHS[i] if ok else "·" for i, ok in enumerate(rec["stages"]))
    run = rec.get("run")
    run_txt = f"{run['mark']} {run['when']}" if run else "— never run"
    if run and run["mark"] == "❌":
        run_txt += f" ({run['detail'].split(':')[0]})"
    rating = RATING_MARK.get(rec.get("rating", ""), "")
    nested = "   ⚠ nested folder" if rec.get("nested") else ""
    takes_txt = ""
    if rec.get("takes"):
        parts = []
        for t in rec["takes"]:
            t_stages = "".join(STAGE_GLYPHS[i] if ok else "·" for i, ok in enumerate(t["stages"]))
            t_run = t["run"]["mark"] if t["run"] else "—"
            parts.append(f"{t['name']} [{t_stages}] {t_run}")
        takes_txt = f"   +{len(rec['takes'])} take(s): " + ", ".join(parts)
    return f"{rec['name']}   [{stages}]   {run_txt}" + (f"   {rating}" if rating else "") + nested + takes_txt


def _dataset_choices(root_str: str, hand_prior_name) -> List[tuple]:
    return [(_dataset_label(r), r["name"]) for r in _dataset_records(root_str, hand_prior_name)]


def refresh_datasets(root_str: str, hand_prior_name, current_selected=None):
    try:
        choices = _dataset_choices(root_str, hand_prior_name)
        names = [c[1] for c in choices]
        current = [str(x) for x in (current_selected or []) if str(x) in names]
        base = _prior_root(_data_root_path(root_str), _validate_name(str(hand_prior_name or ""), "hand_prior_name"))
        n_run = sum(1 for r in _dataset_records(root_str, hand_prior_name) if r["run"])
        status = (f"Found **{len(choices)}** dataset(s) under `{base}` — {n_run} with run records.  \n"
                  f"Legend: `[①②③④⑤]` = stage outputs present on disk · ✅/❌/▶/⏹ = latest run · ★/?/✗ = your screening rating.")
        return status, gr.CheckboxGroup(choices=choices, value=current)
    except Exception as exc:
        return f"⚠️ Could not scan datasets: {exc}", gr.CheckboxGroup(choices=[], value=[])


def sync_prompt_rows(selected_datasets, current_table):
    pmap = _prompt_map(current_table)
    selected = [str(x) for x in (selected_datasets or [])]
    return [[ds, pmap.get(ds, "")] for ds in selected]


def select_all_datasets(root_str, hand_prior_name):
    try:
        choices = _dataset_choices(root_str, hand_prior_name)
        names = [c[1] for c in choices]
        return gr.CheckboxGroup(choices=choices, value=names), [[ds, ""] for ds in names]
    except Exception:
        return gr.CheckboxGroup(choices=[], value=[]), []


def select_good_datasets(root_str, hand_prior_name, current_table=None):
    """Select every dataset rated ★ good, unrun ones included; keeps existing prompts."""
    try:
        recs = _dataset_records(root_str, hand_prior_name)
        choices = [(_dataset_label(r), r["name"]) for r in recs]
        good = [r["name"] for r in recs if r["rating"] == "★ good"]
        pmap = _prompt_map(current_table)
        return gr.CheckboxGroup(choices=choices, value=good), [[ds, pmap.get(ds, "")] for ds in good]
    except Exception:
        return gr.CheckboxGroup(choices=[], value=[]), []


def clear_all_datasets(root_str, hand_prior_name):
    try:
        choices = _dataset_choices(root_str, hand_prior_name)
    except Exception:
        choices = []
    return gr.CheckboxGroup(choices=choices, value=[]), []


# ---------------------------------------------------------------------------
# Quick manual screening: 31-camera contact sheets per dataset
# ---------------------------------------------------------------------------

SHEET_TIMES = [0.0, 0.3, 0.5, 0.7]          # first frame + three mid-sequence samples (fractions of the length)
SHEET_COLS, SHEET_THUMB = 8, (320, 180)
VIDEO_EXTS_SET = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _preview_dir(ds_root: Path) -> Path:
    import hashlib
    return RUNS_DIR / "previews" / hashlib.sha1(str(ds_root.resolve()).encode()).hexdigest()[:12]


def _ffprobe_duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _extract_video_frame(video: Path, t_sec: float, out: Path) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t_sec):.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", f"scale={SHEET_THUMB[0]}:{SHEET_THUMB[1]}:force_original_aspect_ratio=decrease", "-q:v", "4", str(out)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=60).returncode == 0 and out.exists()
    except Exception:
        return False


def _compose_sheet(tiles: List[tuple], title: str, out: Path) -> None:
    """tiles: [(cam_label, image_path or None)] -> one JPEG grid with a title bar."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = font_small = ImageFont.load_default()
    tw, th = SHEET_THUMB
    rows = max(1, (len(tiles) + SHEET_COLS - 1) // SHEET_COLS)
    head = 34
    sheet = Image.new("RGB", (SHEET_COLS * tw, head + rows * th), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), title, fill=(240, 240, 240), font=font)
    for k, (label, img_path) in enumerate(tiles):
        x, y = (k % SHEET_COLS) * tw, head + (k // SHEET_COLS) * th
        if img_path and Path(img_path).exists():
            try:
                im = Image.open(img_path).convert("RGB")
                im.thumbnail((tw, th))
                sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
            except Exception:
                draw.text((x + 8, y + th // 2), "decode error", fill=(255, 80, 80))
        else:
            draw.text((x + 8, y + th // 2), "missing", fill=(255, 80, 80))
        lw = int(draw.textlength(str(label), font=font_small)) + 10
        draw.rectangle([x, y, x + lw, y + 22], fill=(0, 0, 0))
        draw.text((x + 5, y + 2), str(label), fill=(255, 220, 0), font=font_small)
    sheet.save(out, quality=82)


def _build_contact_sheets(ds_root: Path, force: bool = False) -> Tuple[List[Tuple[str, str]], str]:
    """Return ([(sheet_path, caption)], note). Cached under RUNS_DIR/previews."""
    from concurrent.futures import ThreadPoolExecutor
    pdir = _preview_dir(ds_root)
    meta_path = pdir / "meta.json"
    meta = _read_json(meta_path)
    if meta and not force and all((pdir / f).exists() for f in meta.get("files", [])):
        return [(str(pdir / f), c) for f, c in zip(meta["files"], meta["captions"])], f"cached ({meta.get('source')})"

    pdir.mkdir(parents=True, exist_ok=True)
    for old in pdir.glob("*.jpg"):
        old.unlink()
    t0 = time.time()
    images = ds_root / "images"
    cams = sorted((p for p in images.iterdir() if p.is_dir()), key=lambda p: _natural_key(p.name)) if images.is_dir() else []
    sheets: List[Tuple[str, str]] = []
    if cams:
        # Preprocessed dataset: use the extracted 10 fps frames directly (no decoding needed).
        per_cam = {c.name: sorted((p for p in c.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=lambda p: _natural_key(p.name)) for c in cams}
        n = min(len(v) for v in per_cam.values()) if per_cam else 0
        if n == 0:
            return [], "images/ has no frames"
        for k, frac in enumerate(SHEET_TIMES):
            idx = min(n - 1, int(round(frac * (n - 1))))
            tiles = [(cam, per_cam[cam][idx] if idx < len(per_cam[cam]) else None) for cam in per_cam]
            out = pdir / f"sheet_{k}.jpg"
            _compose_sheet(tiles, f"{ds_root.name} — extracted frame {idx}/{n - 1} ({'first frame' if k == 0 else f'{int(frac * 100)}%'}) · {len(tiles)} cams", out)
            sheets.append((str(out), f"frame {idx} ({'first' if k == 0 else f'{int(frac * 100)}%'})"))
        source, n_cams = "images/", len(per_cam)
    else:
        vdir = ds_root / "videos"
        videos = sorted((p for p in vdir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS_SET and p.stem.isdigit()),
                        key=lambda p: _natural_key(p.stem)) if vdir.is_dir() else []
        if not videos and (vdir / "take01").is_dir():
            videos = sorted((p for p in (vdir / "take01").iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS_SET),
                            key=lambda p: _natural_key(p.stem))
        if not videos:
            return [], "no images/ and no videos found"
        with ThreadPoolExecutor(max_workers=16) as pool:
            durations = [d for d in pool.map(_ffprobe_duration, videos) if d]
        dur = min(durations) if durations else 0.0
        jobs = []
        for k, frac in enumerate(SHEET_TIMES):
            t = min(frac * dur, max(0.0, dur - 0.2)) if k else 0.0
            for j, v in enumerate(videos):
                jobs.append((v, t, pdir / f"raw_{k}_{j:02d}.jpg"))
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda a: _extract_video_frame(*a), jobs))
        for k, frac in enumerate(SHEET_TIMES):
            t = min(frac * dur, max(0.0, dur - 0.2)) if k else 0.0
            tiles = [(v.stem if v.stem.isdigit() else v.stem.replace("node", "n"), pdir / f"raw_{k}_{j:02d}.jpg") for j, v in enumerate(videos)]
            out = pdir / f"sheet_{k}.jpg"
            _compose_sheet(tiles, f"{ds_root.name} — raw video t={t:.1f}s of {dur:.1f}s ({'first frame' if k == 0 else f'{int(frac * 100)}%'}) · {len(tiles)} cams", out)
            sheets.append((str(out), f"t={t:.1f}s ({'first' if k == 0 else f'{int(frac * 100)}%'})"))
        for f in pdir.glob("raw_*.jpg"):
            f.unlink()
        source, n_cams = "raw videos", len(videos)
    _write_json_atomic(meta_path, {"files": [Path(p).name for p, _ in sheets], "captions": [c for _, c in sheets],
                                   "source": source, "cams": n_cams, "created": time.time()})
    return sheets, f"generated from {source} in {time.time() - t0:.1f}s"


def _screen_dataset_update(root_str, hand_prior_name, current=None):
    try:
        choices = _dataset_choices(root_str, hand_prior_name)
    except Exception:
        choices = []
    names = [c[1] for c in choices]
    cur = str(current or "")
    return gr.Dropdown(choices=choices, value=cur if cur in names else (names[0] if names else None))


def show_dataset_sheets(root_str, hand_prior_name, dataset, force=False):
    """(status_md, gallery items, rating, notes) for one dataset."""
    try:
        root = _data_root_path(root_str)
        hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
        ds = _validate_name(str(dataset or ""), "dataset")
        ds_root = _prior_root(root, hp) / ds
        if not ds_root.is_dir():
            raise ValueError(f"Dataset directory does not exist: {ds_root}")
        flat_note = _ensure_flat(ds_root)
        rec = next((r for r in _dataset_records(root_str, hp) if r["name"] == ds), None)
        sheets, note = _build_contact_sheets(ds_root, force=bool(force))
        n_cams = (_read_json(_preview_dir(ds_root) / "meta.json") or {}).get("cams")
        cam_txt = (f"**Cameras:** {n_cams}" + ("" if n_cams == 31 else " ⚠️ full rig has 31 — missing cameras are dropped from the calibration automatically")) if n_cams else ""
        stages = " ".join(f"S{i + 1}{'✓' if ok else '·'}" for i, ok in enumerate(rec["stages"])) if rec else ""
        run = rec.get("run") if rec else None
        run_txt = (f"{run['mark']} run `{run['run_id']}` ({run['when']}): {run['detail']}" if run else "no run record")
        status = (f"**{ds}** — sheets {note}  \n{cam_txt}  \n**Stages on disk:** {stages}  \n**Latest run:** {run_txt}")
        if flat_note:
            status = f"ℹ️ {flat_note}  \n" + status
        return status, sheets, (rec["rating"] if rec else "unrated"), (rec["notes"] if rec else "")
    except Exception as exc:
        return f"⚠️ {exc}", [], "unrated", ""


def _neighbor_dataset(root_str, hand_prior_name, current, delta: int):
    names = [c[1] for c in _dataset_choices(root_str, hand_prior_name)]
    if not names:
        return None
    try:
        i = names.index(str(current))
    except ValueError:
        return names[0]
    return names[(i + delta) % len(names)]


def screen_step(root_str, hand_prior_name, current, delta):
    ds = _neighbor_dataset(root_str, hand_prior_name, current, int(delta))
    status, sheets, rating, notes = show_dataset_sheets(root_str, hand_prior_name, ds)
    return gr.Dropdown(value=ds), status, sheets, rating, notes


def save_screening_rating(root_str, hand_prior_name, dataset, rating, notes, selected_datasets=None):
    """Persist rating/notes for one dataset and refresh both dataset lists so the markers update."""
    try:
        root = _data_root_path(root_str)
        hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
        ds = _validate_name(str(dataset or ""), "dataset")
        data = _load_screening(root, hp)
        data[ds] = {"rating": str(rating or "unrated"), "notes": str(notes or ""), "updated": time.time()}
        _save_screening(root, hp, data)
        msg = f"Saved: **{ds}** → {rating}" + (f" — {notes}" if notes else "")
    except Exception as exc:
        msg = f"⚠️ Could not save rating: {exc}"
    _, ds_update = refresh_datasets(root_str, hand_prior_name, selected_datasets)
    return msg, ds_update, _screen_dataset_update(root_str, hand_prior_name, dataset)


def generate_all_sheets(root_str, hand_prior_name, force=False):
    """Generator: build contact sheets for every dataset under the prior, streaming progress."""
    try:
        recs = _dataset_records(root_str, hand_prior_name)
    except Exception as exc:
        yield f"⚠️ {exc}"
        return
    t0 = time.time()
    done = []
    for k, r in enumerate(recs, 1):
        _, note = _build_contact_sheets(r["path"], force=bool(force))
        done.append(f"{r['name']}: {note}")
        yield f"Generating sheets {k}/{len(recs)} ({time.time() - t0:.0f}s)…\n\n" + "\n".join(f"- {x}" for x in done[-8:])
    yield f"✅ Sheets ready for {len(recs)} dataset(s) in {time.time() - t0:.0f}s. Use ◀ / ▶ above to flip through them."


def _safe_dataset_base(root_str: str, hand_prior_name: str, dataset: str) -> Path:
    root = _data_root_path(root_str)
    hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
    ds = _validate_name(str(dataset or ""), "dataset")
    base = (root / DATA_KIND / hp / ds).resolve()
    if not base.is_dir():
        raise ValueError(f"Dataset directory does not exist: {base}")
    return base


def _safe_dataset_subdir(base: Path, relative_subdir: str) -> Path:
    rel = str(relative_subdir or ".").strip() or "."
    if Path(rel).is_absolute():
        raise ValueError("Current folder must be relative to the dataset root.")
    current = (base / rel).resolve()
    if current != base and base not in current.parents:
        raise ValueError("Path escapes the selected dataset root.")
    if not current.is_dir():
        raise ValueError(f"Folder does not exist: {current}")
    return current


def _file_manager_dataset_update(root_str: str, hand_prior_name, current_value=None):
    try:
        root = _data_root_path(root_str)
        hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
        choices = _list_dirs(_prior_root(root, hp))
        cur = str(current_value or "")
        value = cur if cur in choices else (choices[0] if choices else None)
        return gr.Dropdown(choices=choices, value=value)
    except Exception:
        return gr.Dropdown(choices=[], value=None)


def _scan_dataset_directory(root_str: str, hand_prior_name: str, dataset: str, relative_subdir: str, limit: int = 500):
    base = _safe_dataset_base(root_str, hand_prior_name, dataset)
    current = _safe_dataset_subdir(base, relative_subdir)
    paths = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    total = len(paths)
    shown = paths[:limit]
    entry_names = [p.name for p in shown]
    folder_names = [p.name for p in shown if p.is_dir() and not p.is_symlink()]
    rel = "." if current == base else str(current.relative_to(base))
    suffix = f" Showing the first {limit}." if total > limit else ""
    status = (
        f"**Dataset:** `{base.name}`  \n"
        f"**Current folder:** `{rel}`  \n"
        f"Found **{total}** direct entr{'y' if total == 1 else 'ies'}.{suffix}"
    )
    return (
        status,
        gr.CheckboxGroup(choices=entry_names, value=[]),
        gr.Dropdown(choices=folder_names, value=None),
    )


def scan_dataset_directory(root_str: str, hand_prior_name: str, dataset: str, relative_subdir: str):
    try:
        return _scan_dataset_directory(root_str, hand_prior_name, dataset, relative_subdir)
    except Exception as exc:
        return (
            f"⚠️ File manager scan failed: {exc}",
            gr.CheckboxGroup(choices=[], value=[]),
            gr.Dropdown(choices=[], value=None),
        )


def file_manager_open_folder(root_str: str, hand_prior_name: str, dataset: str, relative_subdir: str, folder_name: str):
    try:
        base = _safe_dataset_base(root_str, hand_prior_name, dataset)
        current = _safe_dataset_subdir(base, relative_subdir)
        folder = str(folder_name or "")
        if not folder:
            raise ValueError("Choose a folder to open.")
        allowed = {p.name for p in current.iterdir() if p.is_dir() and not p.is_symlink()}
        if folder not in allowed:
            raise ValueError("Selected folder is not a direct child of the current folder.")
        target = (current / folder).resolve()
        if target != base and base not in target.parents:
            raise ValueError("Folder escapes the selected dataset root.")
        return "." if target == base else str(target.relative_to(base))
    except Exception as exc:
        gr.Warning(str(exc))
        return relative_subdir or "."


def file_manager_up(root_str: str, hand_prior_name: str, dataset: str, relative_subdir: str):
    try:
        base = _safe_dataset_base(root_str, hand_prior_name, dataset)
        current = _safe_dataset_subdir(base, relative_subdir)
        if current == base:
            return "."
        parent = current.parent
        return "." if parent == base else str(parent.relative_to(base))
    except Exception as exc:
        gr.Warning(str(exc))
        return "."


def delete_dataset_entries(root_str: str, hand_prior_name: str, dataset: str, relative_subdir: str, selected_entries, confirmation: str):
    try:
        active = _active_runs()
        if active:
            raise ValueError(
                f"A pipeline run is still active ({active[0]['run_id']}). Stop it before deleting dataset files."
            )

        if str(confirmation or "").strip() != "DELETE":
            raise ValueError('Type exactly "DELETE" in the confirmation box before deleting.')

        selected = [str(x) for x in (selected_entries or [])]
        if not selected:
            raise ValueError("Select at least one file or folder to delete.")

        base = _safe_dataset_base(root_str, hand_prior_name, dataset)
        current = _safe_dataset_subdir(base, relative_subdir)
        allowed = {p.name: p for p in current.iterdir()}

        deleted = []
        for name in selected:
            if name not in allowed:
                raise ValueError(f"Refusing to delete unknown entry: {name!r}")
            target = allowed[name]
            # Never permit the dataset root itself to be a deletion target.
            if target == base:
                raise ValueError("Refusing to delete the dataset root.")
            if target.is_symlink():
                target.unlink()
                kind = "symlink"
            elif target.is_dir():
                shutil.rmtree(target)
                kind = "folder"
            else:
                target.unlink()
                kind = "file"
            deleted.append(f"- {kind}: `{name}`")

        status, entries_update, folders_update = _scan_dataset_directory(
            root_str, hand_prior_name, dataset, relative_subdir
        )
        delete_log = "### ✅ Deleted permanently\n" + "\n".join(deleted) + "\n\n" + status
        return delete_log, entries_update, folders_update, ""
    except Exception as exc:
        return (
            f"### ❌ Nothing deleted\n{exc}",
            gr.CheckboxGroup(),
            gr.Dropdown(),
            confirmation,
        )


# ---------------------------------------------------------------------------
# Export of final results (Stage 5 artifacts) to a hand-off folder / zip.
# ---------------------------------------------------------------------------
EXPORT_STANDARD_DIR = "standardized_hoi_30fps"
EXPORT_ZIP_STORED_EXTS = {".mp4", ".npz", ".zip", ".png", ".jpg"}


def _default_export_dir(root_str: str, hand_prior_name: str) -> Path:
    hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
    return _data_root_path(root_str) / "exports" / hp


def _resolve_export_dir(root_str: str, hand_prior_name: str, export_dir: str) -> Path:
    text = str(export_dir or "").strip()
    if not text:
        return _default_export_dir(root_str, hand_prior_name)
    out = Path(os.path.expanduser(text))
    if not out.is_absolute():
        out = _data_root_path(root_str) / out
    return out.resolve()


def _export_targets(base: Path) -> Tuple[List[Path], Optional[Path]]:
    """Return (mp4 files under outputmano/, standardized_hoi_30fps dir or None)."""
    outputmano = base / "outputmano"
    mp4s = sorted(p for p in outputmano.rglob("*.mp4") if p.is_file()) if outputmano.is_dir() else []
    std = base / EXPORT_STANDARD_DIR
    return mp4s, (std if std.is_dir() else None)


def _copy_if_changed(src: Path, dst: Path) -> bool:
    """Copy src -> dst unless dst already has the same size and mtime. Returns True when copied."""
    try:
        s, d = src.stat(), dst.stat()
        if s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime):
            return False
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_tree_if_changed(src: Path, dst: Path) -> Tuple[int, int]:
    """Mirror src into dst (stale files in dst are removed). Returns (files_total, files_copied)."""
    total = copied = 0
    keep = set()
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        keep.add(rel)
        total += 1
        if _copy_if_changed(p, dst / rel):
            copied += 1
    if dst.is_dir():
        for p in sorted(dst.rglob("*")):
            if p.is_file() and p.relative_to(dst) not in keep:
                p.unlink()
    return total, copied


def _export_dataset_update(root_str, hand_prior_name, selected_datasets, current_value=None):
    """Export list mirrors the main dataset selection (labels show which stages exist)."""
    try:
        choices = _dataset_choices(root_str, hand_prior_name)
    except Exception:
        choices = []
    names = {c[1] for c in choices}
    value = [str(x) for x in (selected_datasets or []) if str(x) in names]
    return gr.CheckboxGroup(choices=choices, value=value)


def export_datasets(root_str, hand_prior_name, selected_datasets, export_dir, make_zip=True):
    """Copy outputmano/**/*.mp4 + standardized_hoi_30fps/ of each selected dataset to the export dir."""
    try:
        hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
        names = [_validate_name(str(x), "dataset") for x in (selected_datasets or [])]
        if not names:
            raise ValueError("Select at least one dataset to export.")
        out_root = _resolve_export_dir(root_str, hp, export_dir)

        busy = set()
        for m in _active_runs():
            if m.get("hand_prior_name") == hp:
                busy.update(m.get("datasets", []))

        lines: List[str] = []
        exported: List[Path] = []
        n_mp4 = n_std_files = n_copied = 0
        # One export entry per (dataset, take): take01 -> <export>/<ds>/, later takes -> <export>/<ds>/takes/<take>/.
        targets: List[Tuple[str, Path, Path]] = []
        for ds in names:
            ds_base = _safe_dataset_base(root_str, hp, ds)
            if out_root == ds_base or ds_base in out_root.parents:
                raise ValueError(f"Export folder must not be inside dataset `{ds}`: {out_root}")
            for take in _dataset_takes(ds_base):
                troot = _take_root(ds_base, take)
                if take == PRIMARY_TAKE:
                    targets.append((ds, troot, out_root / ds))
                elif troot.is_dir():
                    targets.append((f"{ds}/{take}", troot, out_root / ds / TAKES_SUBDIR / take))
        for ds, base, dst in targets:
            mp4s, std = _export_targets(base)
            if not mp4s and std is None:
                lines.append(f"- ⚠️ `{ds}` — nothing to export (no `outputmano/*.mp4`, no `{EXPORT_STANDARD_DIR}/`; run Stage 5 first)")
                continue
            parts: List[str] = []
            outputmano = base / "outputmano"
            counts: Dict[str, int] = {}
            for p in mp4s:
                counts[p.name] = counts.get(p.name, 0) + 1
            for p in mp4s:
                # Flat layout unless two mp4s share a basename; then keep the outputmano/ sub-path in the name.
                name = p.name if counts[p.name] == 1 else "__".join(p.relative_to(outputmano).parts)
                if _copy_if_changed(p, dst / name):
                    n_copied += 1
                n_mp4 += 1
            if mp4s:
                parts.append(f"{len(mp4s)} mp4")
            else:
                parts.append("no mp4 ⚠️")
            if std is not None:
                total, copied = _copy_tree_if_changed(std, dst / EXPORT_STANDARD_DIR)
                n_std_files += total
                n_copied += copied
                parts.append(f"`{EXPORT_STANDARD_DIR}/` ({total} files)")
            else:
                parts.append(f"no `{EXPORT_STANDARD_DIR}/` ⚠️")
            warn = "  ⏳ *pipeline run still active for this dataset — files may be incomplete*" if ds.split("/")[0] in busy else ""
            lines.append(f"- ✅ `{ds}` → `{dst}` — " + ", ".join(parts) + warn)
            exported.append(dst)

        if not exported:
            raise ValueError("None of the selected datasets has Stage 5 outputs yet.\n" + "\n".join(lines))

        zip_path = None
        if make_zip:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            zip_path = out_root / f"hoi_export_{hp}_{len(exported)}ds_{stamp}.zip"
            import zipfile
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # <ds>/takes/<take>/ lives inside <ds>/: collect files once.
                files = sorted({p for dst in exported for p in dst.rglob("*") if p.is_file()})
                for p in files:
                    ctype = zipfile.ZIP_STORED if p.suffix.lower() in EXPORT_ZIP_STORED_EXTS else zipfile.ZIP_DEFLATED
                    zf.write(p, str(p.relative_to(out_root)), compress_type=ctype)
            zip_mb = zip_path.stat().st_size / 1e6

        head = (
            f"### ✅ Exported {len(exported)} dataset(s) to `{out_root}`\n"
            f"{n_mp4} mp4 file(s), {n_std_files} standardized file(s); {n_copied} file(s) newly copied, rest already up to date.\n"
        )
        if zip_path is not None:
            head += f"Zip for download: `{zip_path}` ({zip_mb:.1f} MB)\n"
        return head + "\n" + "\n".join(lines), (str(zip_path) if zip_path is not None else None)
    except Exception as exc:
        return f"### ❌ Export failed\n{exc}", None


def maybe_auto_export(auto_export, root_str, hand_prior_name, selected_datasets, export_dir, make_zip):
    """Runs after the export list is re-synced from the main selection; no-op unless auto-export is on."""
    if not auto_export or not selected_datasets:
        return gr.skip(), gr.skip()
    return export_datasets(root_str, hand_prior_name, selected_datasets, export_dir, make_zip)


def _conda_bootstrap() -> str:
    return r'''
# Make conda available to non-interactive bash and to nested pipeline scripts.
if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
elif [ -n "${CONDA_EXE:-}" ]; then
  _cb="$("${CONDA_EXE}" info --base 2>/dev/null || true)"
  [ -n "$_cb" ] && [ -f "$_cb/etc/profile.d/conda.sh" ] && source "$_cb/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  _cb="$(conda info --base 2>/dev/null || true)"
  [ -n "$_cb" ] && [ -f "$_cb/etc/profile.d/conda.sh" ] && source "$_cb/etc/profile.d/conda.sh"
fi
'''


def _shell_for_step(step: Step) -> str:
    lines = ["set -o pipefail", "export PYTHONUNBUFFERED=1", _conda_bootstrap()]
    if step.conda_env:
        lines += [
            "if ! command -v conda >/dev/null 2>&1; then echo '[webui] conda not found'; exit 127; fi",
            f"conda activate {_q(environment(step.conda_env))}",
        ]
    # GPUs were chosen by the scheduler a moment ago; other users may have taken the memory since.
    # Re-check right before launch and exit GPU_BUSY_RC so the scheduler requeues instead of failing.
    if step.gpu == "single":
        lines += [
            "if [ -z \"${GPU_PICK:-}\" ]; then echo '[GPU sched] ERROR: GPU_PICK not set by scheduler' >&2; exit 1; fi",
            "_free=\"$(nvidia-smi --id=\"$GPU_PICK\" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')\"",
            f"if [ -n \"$_free\" ] && [ \"$_free\" -le \"${{GPU_VRAM_MB:-0}}\" ]; then echo \"[GPU sched][{step.title}] GPU $GPU_PICK now has only $_free MiB free (< budget $GPU_VRAM_MB) — requeue\" >&2; exit {GPU_BUSY_RC}; fi",
            f"echo \"[GPU sched][{step.title}] physical GPU $GPU_PICK ($_free MiB free)\"",
        ]
    elif step.gpu == "multi":
        lines += [
            "if [ -z \"${GPU_LIST:-}\" ]; then echo '[GPU sched] ERROR: GPU_LIST not set by scheduler' >&2; exit 1; fi",
            "_keep=\"\"; for _g in ${GPU_LIST//,/ }; do _free=\"$(nvidia-smi --id=\"$_g\" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')\"; "
            "if [ -z \"$_free\" ] || [ \"$_free\" -gt \"${GPU_VRAM_MB:-0}\" ]; then _keep=\"${_keep:+$_keep,}$_g\"; else echo \"[GPU sched] dropping GPU $_g: only $_free MiB free now\" >&2; fi; done",
            f"if [ -z \"$_keep\" ]; then echo \"[GPU sched][{step.title}] none of GPUs $GPU_LIST still has > $GPU_VRAM_MB MiB free — requeue\" >&2; exit {GPU_BUSY_RC}; fi",
            "export GPU_LIST=\"$_keep\"",
            # Logical ids 0..n-1 for scripts run under CUDA_VISIBLE_DEVICES=$GPU_LIST: their torch-based
            # GPU scans then never touch (or create 500 MiB contexts on) other users' full GPUs.
            "export GPU_LIST_LOGICAL=\"$(seq -s, 0 $(( $(echo \"$GPU_LIST\" | tr ',' '\\n' | wc -l) - 1 )))\"",
            f"echo \"[GPU sched][{step.title}] physical GPUs $GPU_LIST (logical $GPU_LIST_LOGICAL)\"",
        ]
    lines += [f"cd {_q(step.cwd)}", step.command]
    return "\n".join(lines)


def _equalize_step(job: Job, data_root: Path, dirs: Sequence[str] = ("images", "masks")) -> Step:
    """Trim images/<cam> and masks/<cam> to the common frame count.

    Cameras whose source video is a little shorter than --end-sec yield fewer
    trailing frames; multi-view stages (DA3 --all_da3) require identical counts.
    Idempotent and cheap, so it is also run at the start of the recon stage.
    In the preprocess stage it runs before the masks exist: pass dirs=("images",)
    there, otherwise leftover empty masks/<cam>/ folders fail the step.
    """
    ds_root = _dataset_root(data_root, job)
    script = Path(__file__).resolve().parent / "equalize_camera_frames.py"
    return Step("Equalize per-camera frame counts", ds_root,
                f"{_q(sys.executable)} {_q(script)} --data_dir {_q(ds_root)} --dirs {' '.join(dirs)}", None)


def _calib_sync_step(job: Job, data_root: Path) -> Step:
    """Drop cameras that have no images from the dataset's intri/extri copy (30-camera captures etc.).

    Idempotent; keeps the full rig calibration as *.yml.full. Runs after prepare and at the
    start of every later stage so scripts that enumerate cameras from the calibration
    (EasyMocap, FoundationPose renderers, ...) always see the cameras that really exist.
    """
    ds_root = _dataset_root(data_root, job)
    script = Path(__file__).resolve().parent / "sync_calibration.py"
    return Step("Sync calibration to available cameras", ds_root,
                f"python {_q(script)} --data_dir {_q(ds_root)}", "easymocap")


def _preprocess_steps(
    job: Job,
    code_root: Path,
    data_root: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
    width: int,
    height: int,
    ffmpeg_threads: int,
    auto_end_sec: bool = True,
) -> List[Step]:

    data_name = _ensure_pipeline_data_name(data_root, job)
    ds_root = _dataset_root(data_root, job)

    # ---------------------------------------------------------
    # FAST frame extraction:
    # - original resolution
    # - NO resize
    # - one FFmpeg process per camera
    # - up to 31 cameras in parallel
    # ---------------------------------------------------------
    extract_workers = 31
    extract_ffmpeg_threads = 1

    if auto_end_sec:
        # End time follows the SHORTEST video of this dataset.
        # End sec acts only as an upper cap.
        probe = Path(__file__).resolve().parent / "shortest_video_end.py"

        cut_title = "Extract frames (31-camera parallel, auto shortest end)"

        cut_cmd = (
            f"END_SEC=\"$({_q(sys.executable)} {_q(probe)} "
            f"--video_dir {_q(ds_root / 'videos')} "
            f"--fps {float(fps)} "
            f"--start-sec {float(start_sec)} "
            f"--max-end-sec {float(end_sec)})\" && "
            f"echo \"[auto end] using --end-sec $END_SEC\" && "

            f"python rename_videos_extract_frames_ffmpeg_pts_compat.py "
            f"--data-name {_q(data_name)} "
            f"--start-sec {float(start_sec)} "
            f"--end-sec \"$END_SEC\" "
            f"--fps {float(fps)} "
            f"--workers {extract_workers} "
            f"--ffmpeg-threads {extract_ffmpeg_threads} "
            f"--clear-output "
            f"--execute"
        )

    else:
        cut_title = "Extract frames (31-camera parallel)"

        cut_cmd = (
            f"python rename_videos_extract_frames_ffmpeg_pts_compat.py "
            f"--data-name {_q(data_name)} "
            f"--start-sec {float(start_sec)} "
            f"--end-sec {float(end_sec)} "
            f"--fps {float(fps)} "
            f"--workers {extract_workers} "
            f"--ffmpeg-threads {extract_ffmpeg_threads} "
            f"--clear-output "
            f"--execute"
        )

    return [
        # -----------------------------------------------------
        # 1. Rename/copy raw videos
        # -----------------------------------------------------
        Step(
            f"Rename 31 camera videos ({job.take})",
            code_root,
            f"python rename_videos_31cams.py "
            f"--input_dir {_q(_take_video_dir(data_root, job))} "
            f"--output_dir {_q(ds_root / 'videos')} "
            f"--overwrite "
            f"--allow-missing",
            "easymocap",
        ),

        # -----------------------------------------------------
        # 2. One original-resolution reference image per camera
        # -----------------------------------------------------
        Step(
            "Extract reference frames (t=0)",
            ds_root,
            _ref_frames_cmd(ds_root, 0.0),
            None,
        ),

        # -----------------------------------------------------
        # 3. FAST extraction
        # IMPORTANT:
        #   NO resizing here.
        #   images/<cam>/ contains original-resolution frames.
        # -----------------------------------------------------
        Step(
            cut_title,
            code_root,
            cut_cmd,
            "easymocap",
        ),

        # -----------------------------------------------------
        # 4. Make sure all cameras have the same frame count
        # -----------------------------------------------------
        _equalize_step(
            job,
            data_root,
            dirs=("images",),
        ),

        # -----------------------------------------------------
        # 5. Resize AFTER extraction
        # -----------------------------------------------------
        Step(
            "Resize + prepare image folders",
            code_root,
            f"python prepare_bowl_0820_ck_ffmpeg_fast.py "
            f"--data-name {_q(data_name)} "
            f"--calib-dir {_q(CALIBRATION_DIR)} "
            f"--width {int(width)} "
            f"--height {int(height)} "
            f"--ffmpeg-threads 1 "
            f"--workers 31 "
            f"--only images",
            "easymocap",
        ),

        # -----------------------------------------------------
        # 6. Calibration
        # -----------------------------------------------------
        _calib_sync_step(job, data_root),

        # -----------------------------------------------------
        # 7+. SAM3
        # -----------------------------------------------------
        *_sam3_mask_steps(job, code_root, data_root),
    ]

def _ref_frames_cmd(ds_root: Path, start_sec: float = 0.0) -> str:
    """images1/<cam>.jpg: one full-resolution reference frame per camera (t=0 by default —
    before the hand grabs the object — which is what the object mesh should be built from)."""
    script = Path(__file__).resolve().parent / "extract_reference_frames.py"
    return (f"{_q(sys.executable)} {_q(script)} --video_dir {_q(ds_root / 'videos')} "
            f"--output_dir {_q(ds_root / 'images1')} --time-sec {float(start_sec)}")


def _mesh_scale_step(job: Job, data_root: Path) -> Step:
    """Optional fallback (stage 'meshscale', button 2b): metric scale from calibrated silhouettes.

    Not part of the regular chain. MV-SAM3D derives its scale from DA3 depth inside the t=0
    masks, but depth/<cam>/000000 is the frame at start_sec; if the object moved in between
    (start_sec > 0 while the hand already grabs it) the mesh comes out several times too big.
    The default start_sec is therefore 0. This step rescales the mesh from mask/ + calibration
    when the two estimates disagree by more than 15% (original kept as compressed_prescale.ply).
    """
    ds_root = _dataset_root(data_root, job)
    script = Path(__file__).resolve().parent / "mesh_scale_from_silhouettes.py"
    return Step("Check mesh scale against silhouettes", ds_root,
                f"python {_q(script)} --data_dir {_q(ds_root)} --tolerance 0.15", "easymocap")


def _sam3_mask_steps(job: Job, code_root: Path, data_root: Path) -> List[Step]:
    """
    SAM3 first-frame object mask -> sanity check -> accelerated per-frame masks.

    Stage 1:
        Generate first-frame masks using the text prompt.

    Stage 2:
        Validate that enough camera views detected the object.

    Stage 3:
        Run hybrid SAM3 -> SAM2.1 propagation with:
          - SAM3 cross-camera first-frame initialization
          - SAM2.1 Hiera Small per-camera propagation
          - official VOS-optimized full-model torch.compile path
          - BF16, GPU-resident video/state, async loading
          - multi-GPU/multi-worker assignment
          - asynchronous lossless PNG writing
    """
    data_name = _ensure_pipeline_data_name(data_root, job)
    ds_root = _dataset_root(data_root, job)
    helper = Path(__file__).resolve().parent / "mask_views.py"

    return [
        Step(
            "SAM3 reconstruction mask",
            code_root / "sam3",
            f"CUDA_VISIBLE_DEVICES=\"$GPU_PICK\" "
            f"python sam3text_object.py "
            f"--object_name {_q(data_name)} "
            f"--text_input {_q(job.text_prompt)}",
            "sam3",
            gpu="single",
        ),

        # Fail fast before the expensive per-frame propagation when
        # the text prompt detected the target in too few camera views.
        Step(
            "Check first-frame masks",
            ds_root,
            f"{_q(sys.executable)} {_q(helper)} check "
            f"--data_dir {_q(ds_root)}",
            None,
        ),

        Step(
            "SAM3 init + SAM2.1 per-frame masks (ultrafast multi-GPU)",
            code_root / "sam3",
            f"CUDA_VISIBLE_DEVICES=\"$GPU_LIST\" "
            f"python sam3text_multi_gpu.py "
            f"--folder {_q(ds_root / 'images')} "
            f"--text_input {_q(job.text_prompt)} "
            f"--gpus \"$GPU_LIST_LOGICAL\" "
            f"--min_free_vram_mb \"$GPU_VRAM_MB\" "
            f"--sam2-repo {_q(code_root / 'sam3' / 'sam2')} "
            f"--sam2-model large "
            f"--sam2-score-thresh 0.0 "
            f"--no-sam2-postprocess "
            f"--sam2-compile-mode max-autotune-no-cudagraphs "
            f"--folders_per_gpu 1 "
            f"--workers_per_gpu 1 "
            f"--mask_writer_threads 4 "
            f"--png_compression 0 "
            f"--compile "
            f"--no-compile-stage1 "
            f"--no-offload_video_to_cpu "
            f"--no-offload_state_to_cpu "
            f"--async_loading_frames "
            f"--no-progress",
            "sam3",
            gpu="multi",
        ),
    ]
def _object_recon_steps(job: Job, code_root: Path, data_root: Path,
                        da3_gpus: str, sam3d_gpu: str, depth_unit: str,
                        pixel_stride: int, mask_erode: int,
                        max_points_per_cam: int, max_size_mb: float,
                        save_debug_pointcloud: bool, start_sec: float = 0.0, generate_texture: bool = False) -> List[Step]:
    ds_root = _dataset_root(data_root, job)
    data_name = _ensure_pipeline_data_name(data_root, job)
    debug_flag = " --save_debug_pointcloud" if bool(save_debug_pointcloud) else ""
    debug_flag += " --generate_texture" if generate_texture else ""
    mask_helper = Path(__file__).resolve().parent / "mask_views.py"
    # Stage 2 can be started on a dataset whose images1/ + mask/ are missing (older preprocessing):
    # regenerate them (≈2 min) only in that case.
    refresh_refs = (
        f"if [ -d {_q(ds_root / 'mask')} ] && ls {_q(ds_root / 'images1')}/*.jpg >/dev/null 2>&1; then "
        f"echo '[ref-frames] images1/ and mask/ present; nothing to do'; else "
        f"{_ref_frames_cmd(ds_root, 0.0)} && "
        f"CUDA_VISIBLE_DEVICES=\"$GPU_PICK\" python sam3text_object.py --object_name {_q(data_name)} --text_input {_q(job.text_prompt)} && "
        f"{_q(sys.executable)} {_q(mask_helper)} check --data_dir {_q(ds_root)}; fi"
    )
    da3_step = Step("Depth-Anything-3 multiview metric depth", code_root / "Depth-Anything-3",
                    f"CUDA_VISIBLE_DEVICES=\"$GPU_LIST\" python da3_multiview_metric_align_uint16_v4_8gpu.py "
                    f"--folder {_q(ds_root)} --model-dir {_q(os.getenv('HOI_DA3_MODEL', str(MODEL_ROOT / 'da3')))} --all_da3 --gpus \"$GPU_LIST_LOGICAL\" "
                    f"--min_free_vram_mb \"$GPU_VRAM_MB\"", "depthanything3", gpu="multi")
    if not job.is_primary_take:
        # Same object as take01: depth is per recording, the mesh is not. Link take01's mesh
        # instead of reconstructing it again (the scheduler waits for take01's recon step).
        return [
            _equalize_step(job, data_root),
            _calib_sync_step(job, data_root),
            da3_step,
            _shared_mesh_link_step(job, data_root),
        ]
    return [
        _equalize_step(job, data_root),
        _calib_sync_step(job, data_root),
        Step("Reference frames + SAM3 object mask (only if missing)", code_root / "sam3", refresh_refs, "sam3", gpu="single"),
        da3_step,
        Step(MESH_RECON_STEP_TITLE, code_root / "MV-SAM3D",
             # Cameras whose first-frame mask is empty are swapped out of the mesh views and
             # dropped from depth fusion (MV-SAM3D refuses empty masks).
             f"eval \"$({_q(sys.executable)} {_q(Path(__file__).resolve().parent / 'mask_views.py')} select "
             f"--data_dir {_q(ds_root)})\" && "
             f"CUDA_VISIBLE_DEVICES=\"$GPU_PICK\" python run_mv_sam3d_metric_scale.py "
             f"--setting_name {_q(data_name)} --depth_unit {_q(depth_unit)} "
             f"--mesh_cam_ids \"$MESH_CAMS\" --depth_cam_ids \"$DEPTH_CAMS\" "
             f"--pixel_stride {int(pixel_stride)} --mask_erode {int(mask_erode)} "
             f"--max_points_per_cam {int(max_points_per_cam)} --max_size_mb {float(max_size_mb)}"
             f"{debug_flag}", "sam3d-objects", gpu="single"),
    ]


def _shared_mesh_link_step(job: Job, data_root: Path) -> Step:
    """Secondary takes: point meshes/ at the dataset-level mesh reconstructed from take01."""
    ds_folder = _dataset_folder(data_root, job)
    take_root = _dataset_root(data_root, job)
    script = Path(__file__).resolve().parent / "link_shared_take_resources.py"
    return Step(
        f"Link shared object mesh from {PRIMARY_TAKE}", take_root,
        f"{_q(sys.executable)} {_q(script)} --dataset_dir {_q(ds_folder)} --take_dir {_q(take_root)} "
        f"--require meshes/compressed.ply",
        None,
        after={"chain": job.dataset_logical_name, "title": MESH_RECON_STEP_TITLE},
    )


def _tracking_steps(job: Job, code_root: Path, data_root: Path,
                    tracking_gpu: str, postopt_gpu: str,
                    width: int, height: int, fps: float,
                    depth_scale: float, tracking_method: str = "foundationpose") -> List[Step]:
    if tracking_method not in TRACKING_METHODS:
        raise ValueError(f"Unknown pose tracking method: {tracking_method}")
    ds_root = _dataset_root(data_root, job)
    data_name = _ensure_pipeline_data_name(data_root, job)
    debug_dir = Path("..") / data_name / "debug_depth"
    helper = Path(__file__).resolve().parent / "tracking_run.py"
    state_command = f"{_q(sys.executable)} {_q(helper)}"
    state_args = f"--scene {_q(ds_root)} --method {_q(tracking_method)}"
    if tracking_method == "wapr":
        wapr_python = python_for("wapr")
        tracker = Step("WAPR object tracking", code_root,
             f"CUDA_VISIBLE_DEVICES=\"$GPU_PICK\" OPENBLAS_NUM_THREADS=2 {_q(wapr_python)} -u "
             f"{_q(Path(__file__).resolve().parent / 'wapr_tracking.py')} "
             f"--scene_dir {_q(ds_root)} --width {int(width)} --height {int(height)} "
             f"--fps {float(fps)} --depth_scale {float(depth_scale)} --backend torch", None, gpu="single")
        # Explicit raw input prevents stale FoundationPose smoothing outputs from
        # ever becoming WAPR's optimizer input.
        pose_input = f" --pose_dir {_q(ds_root / 'debug_depth' / 'ob_in_world')}"
    else:
        tracker = Step("FoundationPose object tracking", code_root,
             f"CUDA_VISIBLE_DEVICES=\"$GPU_PICK\" bash run_resize_camk_foundationpose_args_local.sh "
             f"--data_name {_q(data_name)} --w {int(width)} --h {int(height)} --use_depth 1 "
             f"--depth_dir {_q(ds_root / 'depth')} --no_force 1", None, gpu="single")
        pose_input = ""
    return [
        _calib_sync_step(job, data_root),
        Step("Preserve previous tracking results", code_root,
             f"{state_command} prepare {state_args}"),
        tracker,
        Step("Object geometry global pose post-optimization", code_root / "FoundationPose",
             f"CUDA_VISIBLE_DEVICES=\"$GPU_PICK\" python postopt_geometry_global_pose.py "
             f"--data_name {_q(data_name)} --data_root .. --debug_dir {_q(debug_dir)} "
             f"--depth_scale {float(depth_scale)} --fps {float(fps)}{pose_input}", "foundationpose", gpu="single"),
        Step("Render all-camera green mesh tracking video", code_root / "FoundationPose",
             f"CUDA_VISIBLE_DEVICES=\"$GPU_PICK\" python render_allcams_mesh_grid_fast_green.py "
             f"--data_dir {_q(ds_root)} --render_w 160 --render_h 90 --fps {float(fps)} "
             f"--pose_dir {_q(ds_root / 'debug_depth/geometry_global_opt/ob_in_world')} "
             f"--alpha 0.40 --grid_cols 6 --use_light 0 --outline 0 --save_frames 0 && "
             # OpenCV writes MPEG-4 part 2, which browsers cannot play: re-encode to H.264 in place.
             f"_v={_q(ds_root / 'mesh_overlay_allcams_fast_green' / 'mesh_overlay_allcams_fast_green.mp4')} && "
             f"ffmpeg -y -v error -i \"$_v\" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart -an "
             f"\"${{_v%.mp4}}_h264.mp4\" && mv -f \"${{_v%.mp4}}_h264.mp4\" \"$_v\"",
             "foundationpose", gpu="single"),
        Step("Record completed tracking method", code_root,
             f"{state_command} complete {state_args}"),
    ]


def _hand_steps(job: Job, code_root: Path, data_root: Path) -> List[Step]:
    data_name = _ensure_pipeline_data_name(data_root, job)
    hand_prior_name = _ensure_pipeline_prior_name(data_root, job)
    return [
        _calib_sync_step(job, data_root),
        Step(
            "Sapiens + EasyMocap hand reconstruction",
            code_root,
            f"CUDA_VISIBLE_DEVICES=\"$GPU_PICK\" bash run_extract_sapiens_easymocap_10fps_kalman.sh "
            f"--data_name {_q(data_name)} "
            f"--hand_prior_name {_q(hand_prior_name)} "
            f"--hand_side {_q(job.hand_side)}",
            None,
            gpu="single",
        ),
    ]


def _final_steps(
    job: Job,
    code_root: Path,
    data_root: Path,
    max_frames: int,
    fps: float,
) -> List[Step]:
    """
    Final Stage 5 pipeline:

    1. Contact-aware MANO refinement + Do-As-I-Do export
    2. Export standardized interaction data
    3. Fast multi-GPU hand-object grid rendering
    4. CPU encode grid JPGs into final MP4

    Final outputs:
        <dataset>/outputmano/grid/*.jpg
        <dataset>/outputmano/grid/final_hand_object_interaction.mp4
    """

    # ------------------------------------------------------------------
    # Stage 5 always runs from the EasyMocap checkout.
    # ------------------------------------------------------------------
    code_root = STAGE5_EASYMOCAP_ROOT

    if not code_root.is_dir():
        raise ValueError(
            f"Stage 5 EasyMocap root does not exist: {code_root}"
        )

    # Dataset name relative to:
    #
    #   <pipeline-root>/
    #
    # Example:
    #   rigid/collector01/cup_take01
    # ------------------------------------------------------------------
    data_name = _ensure_pipeline_data_name(
        data_root,
        job,
        root=code_root,
    )

    ds_root = _dataset_root(
        data_root,
        job,
    )

    # Python environment used for standardized export.
    easymocap_python = python_for("easymocap")

    # ------------------------------------------------------------------
    # Final visualization paths.
    # ------------------------------------------------------------------
    outputmano_dir = ds_root / "outputmano"
    grid_dir = outputmano_dir / "grid"

    final_video = (
        grid_dir
        / "final_hand_object_interaction.mp4"
    )

    grid_glob = (
        grid_dir
        / "*.jpg"
    )

    render_fps = max(
        1,
        int(round(float(fps))),
    )

    encode_fps = float(fps)

    # ------------------------------------------------------------------
    # Final CPU video encoding.
    #
    # H200 does not provide NVENC, so use libx264 directly.
    #
    # ultrafast:
    #   much faster than veryfast
    #
    # threads 0:
    #   let x264 automatically use available CPU threads
    # ------------------------------------------------------------------
    ffmpeg_command = (
        f"echo '[Final video] Encoding grid frames with CPU libx264 ultrafast' && "
        f"ffmpeg "
        f"-y "
        f"-hide_banner "
        f"-loglevel warning "
        f"-framerate {encode_fps} "
        f"-pattern_type glob "
        f"-i {_q(grid_glob)} "
        f"-vf {_q('pad=ceil(iw/2)*2:ceil(ih/2)*2')} "
        f"-c:v libx264 "
        f"-preset ultrafast "
        f"-crf 18 "
        f"-threads 0 "
        f"-pix_fmt yuv420p "
        f"-movflags +faststart "
        f"{_q(final_video)}"
    )

    return [
        # ==============================================================
        # STEP 1
        #
        # Joint hand/object optimization.
        #
        # This stage produces:
        #
        #   outputmano/
        #       smoothed_ob_in_world/
        #       contact_mano/
        #       object_mesh_scaled.ply
        #
        # Rendering is skipped in the optimizer itself.
        # ==============================================================
        Step(
            "Contact-aware MANO refinement + Do-As-I-Do export",
            code_root,
            (
                "bash "
                "run_joint_pose_solver_only_and_export_"
                "bimanual_10fps_8gpu_hocap_meshsdf_31cam.sh "
                f"--data_name {_q(data_name)} "
                f"--hand {_q(job.hand_side)} "
                f"--gpus \"$GPU_LIST\" "
                f"--min_free_vram_mb \"$GPU_VRAM_MB\""
            ),
            None,
            gpu="multi",
        ),

        # ==============================================================
        # STEP 2
        #
        # Export standardized interaction information.
        # ==============================================================
        Step(
            "Export standardized interaction data",
            code_root / "FoundationPose",
            (
                f"{_q(easymocap_python)} "
                "export_standardized_mano_object_table_"
                "4rings_first500.py "
                f"--data_name {_q(data_name)} "
                f"--max_frames {int(max_frames)} "
                "--overwrite"
            ),
            None,
        ),

        # ==============================================================
        # STEP 3
        #
        # FAST multi-GPU renderer.
        #
        # Your current run_render_outputmano_grid_8gpu.sh is already
        # calling the fast-grid renderer, as confirmed by:
        #
        #   [render-fast]
        #   persistent OffscreenRenderer
        #   grid_only=True
        #
        # 4 GPUs split the sequence by frames.
        #
        # Output:
        #
        #   outputmano/grid/000000.jpg
        #   outputmano/grid/000001.jpg
        #   ...
        #
        # No final MP4 is encoded here.
        # ==============================================================
        Step(
            "Render final hand-object interaction grids",
            code_root,
            (
                "bash run_render_outputmano_grid_8gpu.sh "
                f"--data_name {_q(data_name)} "
                f"--gpus \"$GPU_LIST\" "
                f"--grid_fps {render_fps}"
            ),
            None,
            gpu="multi",
        ),

        # ==============================================================
        # STEP 4
        #
        # CPU-only final MP4 encoding.
        #
        # IMPORTANT:
        # This is deliberately a separate CPU step.
        #
        # Therefore after Step 3 finishes:
        #
        #   GPU 4,5,6,7
        #
        # are immediately released back to the scheduler.
        # ==============================================================
        Step(
            "Encode final hand-object interaction video",
            code_root,
            ffmpeg_command,
            None,
        ),
    ]

def _steps_for(stage: str, job: Job, code_root: Path, data_root: Path,
               start_sec: float, end_sec: float, fps: float,
               width: int, height: int, ffmpeg_threads: int,
               da3_gpus: str, sam3d_gpu: str, tracking_gpu: str, postopt_gpu: str,
               depth_unit: str, pixel_stride: int, mask_erode: int,
               max_points_per_cam: int, max_size_mb: float,
               save_debug_pointcloud: bool, depth_scale: float,
               max_frames: int, auto_end_sec: bool = True, generate_texture: bool = False,
               tracking_method: str = "foundationpose") -> List[Step]:
    if stage == "preprocess":
        return _preprocess_steps(job, code_root, data_root, start_sec, end_sec, fps,
                                 width, height, ffmpeg_threads, auto_end_sec)
    if stage == "masks":
        return _sam3_mask_steps(job, code_root, data_root)
    if stage == "meshscale":
        # The mesh is shared with take01; rescaling it from another take's silhouettes
        # would silently change every take.
        return [_mesh_scale_step(job, data_root)] if job.is_primary_take else []
    if stage == "recon":
        return _object_recon_steps(job, code_root, data_root, da3_gpus, sam3d_gpu,
                                   depth_unit, pixel_stride, mask_erode,
                                   max_points_per_cam, max_size_mb, save_debug_pointcloud,
                                   float(start_sec), generate_texture)
    if stage == "tracking":
        return _tracking_steps(job, code_root, data_root, tracking_gpu, postopt_gpu,
                               width, height, fps, depth_scale, tracking_method)
    if stage == "hands":
        return _hand_steps(job, code_root, data_root)
    if stage == "final":
        return _final_steps(job, code_root, data_root, max_frames, fps)
    raise ValueError(stage)

# ---------------------------------------------------------------------------
# Detached, tmux-managed runs
# ---------------------------------------------------------------------------

def _read_json(path: Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _tmux(*args):
    # Drop $TMUX so launching works even when the web UI itself lives inside tmux.
    env = {k: v for k, v in os.environ.items() if k != "TMUX"}
    return subprocess.run(["tmux", "-L", TMUX_SOCKET, *args], env=env, capture_output=True, text=True)


def _tmux_session_alive(session: str) -> bool:
    if not shutil.which("tmux"):
        return False
    return _tmux("has-session", "-t", f"={session}").returncode == 0


def _attach_command(meta: dict) -> str:
    if meta.get("backend") != "tmux":
        return f"tail -f {_q(Path(meta['dir']) / 'log.txt')}"
    return f"tmux -L {TMUX_SOCKET} attach -t {meta['session']}"


def _run_state(run_dir: Path, meta: dict) -> dict:
    """Effective state; detects runners that died without writing a final state."""
    state = _read_json(run_dir / "state.json", {}) or {}
    status = state.get("status", "queued")
    if status == "running" and state.get("runner_pid") and not _pid_alive(state["runner_pid"]):
        state["status"] = "lost"
    elif status == "queued" and time.time() - float(meta.get("created", 0)) > 60:
        if meta.get("backend") != "tmux" or not _tmux_session_alive(meta.get("session", "")):
            state["status"] = "lost"
    return state


def _list_runs() -> List[dict]:
    runs: List[dict] = []
    if not RUNS_DIR.is_dir():
        return runs
    for d in RUNS_DIR.iterdir():
        meta = _read_json(d / "meta.json")
        if not meta or not d.is_dir():
            continue
        meta["dir"] = str(d)
        meta["state"] = _run_state(d, meta)
        runs.append(meta)
    runs.sort(key=lambda m: float(m.get("created", 0)), reverse=True)
    return runs


def _find_run(run_id) -> Optional[dict]:
    if not run_id:
        return None
    for m in _list_runs():
        if m["run_id"] == run_id:
            return m
    return None


def _active_runs() -> List[dict]:
    return [m for m in _list_runs() if m["state"].get("status") in ACTIVE_STATUSES]


def _run_label(meta: dict) -> str:
    st = meta["state"].get("status", "queued")
    stages = "+".join(meta.get("stage_sequence", []))
    ds = ", ".join(meta.get("datasets", []))
    if len(ds) > 48:
        ds = ds[:45] + "..."
    return f"{STATUS_ICON.get(st, '?')} {meta['run_id']} · {stages} · {ds}"


def _log_tail(run_dir: Path, limit: int = LOG_TAIL_CHARS) -> str:
    p = run_dir / "log.txt"
    if not p.exists():
        return ""
    size = p.stat().st_size
    with open(p, "rb") as f:
        if size > limit:
            f.seek(size - limit)
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _step_icon(st: dict) -> str:
    return {"pending": "·", "running": "▶️", "done": "✅", "failed": "❌",
            "stopped": "⏹️", "skipped": "⏭️"}.get(st.get("status", "pending"), "?")


def _fmt_gpus(gpus) -> str:
    if not gpus:
        return "CPU"
    return "GPU " + ",".join(str(g) for g in gpus)


def _run_status_md(meta: dict) -> str:
    st = meta["state"]
    status = st.get("status", "queued")
    steps = meta.get("steps", [])
    total = len(steps)
    step_states = st.get("steps", {})
    now = time.time()
    started = float(st.get("started") or meta.get("created") or now)
    finished = float(st.get("finished") or now)
    elapsed = (finished - started) / 60.0
    n_done = sum(1 for s in step_states.values() if s.get("status") == "done")
    n_failed = sum(1 for s in step_states.values() if s.get("status") == "failed")
    running = [(int(i), s) for i, s in step_states.items() if s.get("status") == "running"]
    running.sort()
    sched = meta.get("scheduler", {})

    if status == "queued":
        head = "### ⏳ Queued (runner starting in tmux)"
    elif status == "running":
        head = f"### ▶️ Running — {n_done}/{total} steps done, {len(running)} running"
    elif status == "done":
        head = f"### ✅ Completed\nProcessed **{len(meta.get('datasets', []))} dataset(s)**, **{total} step(s)** in {elapsed:.1f} min."
    elif status == "failed":
        failed = [steps[int(i)]["label"] + f" (rc {s.get('rc')})" for i, s in step_states.items() if s.get("status") == "failed"]
        head = "### ❌ Finished with failures\n" + "\n".join(f"- {x}" for x in failed)
    elif status == "stopped":
        head = f"### ⏹️ Stopped by user\n{n_done}/{total} steps had completed."
    else:
        head = "### ⚠️ Runner lost\nThe runner process disappeared (machine reboot / killed). Check the log below."

    lines = [
        head,
        f"**Run:** `{meta['run_id']}` &nbsp;&nbsp; **Stages:** `{'+'.join(meta.get('stage_sequence', []))}` &nbsp;&nbsp; "
        f"**Hand prior:** `{meta.get('hand_prior_name')}`  ",
        f"**Datasets:** `{', '.join(meta.get('datasets', []))}`  ",
        f"**Data root:** `{meta.get('data_root')}`  ",
        f"**Scheduler:** parallel datasets `{sched.get('parallel_datasets', 'auto') or 'auto'}` · "
        f"VRAM/step `{sched.get('vram_per_step_mb', GPU_MIN_FREE_VRAM_MB)} MiB` · "
        f"CPU steps `{sched.get('cpu_parallel', 3)}` · OOM retries `{sched.get('oom_max_retries', 0)}` · "
        f"GPUs `{','.join(map(str, sched.get('allowed_gpus') or [])) or 'all'}`  ",
    ]
    if status in ACTIVE_STATUSES:
        lines.append(f"**Elapsed:** {elapsed:.1f} min · **Failed so far:** {n_failed} · "
                     f"**OOM requeues:** {st.get('oom_requeued', 0)}  ")
    elif st.get("oom_requeued"):
        lines.append(f"**OOM requeues during this run:** {st['oom_requeued']}  ")
    if running:
        lines.append("")
        lines.append("| # | step | resources | running for |")
        lines.append("|---|---|---|---|")
        for i, s in running:
            mins = (now - float(s.get("started") or now)) / 60.0
            retry = f" (attempt {s['attempts']})" if int(s.get("attempts", 1)) > 1 else ""
            lines.append(f"| {i + 1} | {steps[i]['label']}{retry} | {_fmt_gpus(s.get('gpus'))} | {mins:.1f} min |")
    waiting = st.get("waiting_note")
    if waiting and status == "running":
        lines.append(f"\n⏳ {waiting}")
    return "\n".join(lines)


def _tail_lines(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > 64000:
            f.seek(size - 64000)
        data = f.read().decode("utf-8", errors="replace")
    return "\n".join(data.splitlines()[-n:])


def _run_log_view(meta: dict, step_choice=None) -> str:
    run_dir = Path(meta["dir"])
    steps = meta.get("steps", [])
    step_states = meta["state"].get("steps", {})
    try:
        choice = int(step_choice)
    except (TypeError, ValueError):
        choice = -1
    if 0 <= choice < len(steps):
        s = step_states.get(str(choice), {})
        log = s.get("log")
        if log and Path(log).exists():
            return f"── {steps[choice]['label']} · {_fmt_gpus(s.get('gpus'))} · {s.get('status')} ──\n" + _tail_lines(Path(log), 400)
        return f"── {steps[choice]['label']} ── (not started yet)"

    # Auto view: scheduler events + tail of every running step (or the failed ones).
    parts = [_tail_lines(run_dir / "log.txt", 60)]
    running = sorted(int(i) for i, s in step_states.items() if s.get("status") == "running")
    show = running or sorted(int(i) for i, s in step_states.items() if s.get("status") == "failed")[-2:]
    for i in show:
        s = step_states[str(i)]
        parts.append(f"\n──── [{s.get('status')}] {steps[i]['label']} · {_fmt_gpus(s.get('gpus'))} ────")
        if s.get("log"):
            parts.append(_tail_lines(Path(s["log"]), 40))
    return "\n".join(parts)


def _steps_dropdown(meta: Optional[dict], selected=None):
    if meta is None:
        return gr.Dropdown(choices=[("Auto (scheduler + running steps)", -1)], value=-1), -1
    step_states = meta["state"].get("steps", {})
    choices = [("Auto (scheduler + running steps)", -1)]
    for i, s in enumerate(meta.get("steps", [])):
        ss = step_states.get(str(i), {})
        gp = f" [{_fmt_gpus(ss.get('gpus'))}]" if ss.get("gpus") else ""
        n_rt = int(ss.get("oom_retries", 0) or 0) + int(ss.get("busy_retries", 0) or 0)
        rt = f" ↻{n_rt}" if n_rt else ""
        choices.append((f"{_step_icon(ss)} {i + 1:02d} · {s['label']}{gp}{rt}", i))
    try:
        selected = int(selected)
    except (TypeError, ValueError):
        selected = -1
    if selected not in [c[1] for c in choices]:
        selected = -1
    return gr.Dropdown(choices=choices, value=selected), selected


def _run_view(run_id, step_choice=None):
    """(status_md, log_text, attach_cmd, step_dropdown) for one run, or a neutral view."""
    meta = _find_run(run_id)
    if meta is None:
        dd, _ = _steps_dropdown(None)
        return "### Ready\nNo run selected. Submit a stage above, or pick a previous run from the list.", "", "", dd
    dd, sel = _steps_dropdown(meta, step_choice)
    return _run_status_md(meta), _run_log_view(meta, sel), _attach_command(meta), dd


def _runs_dropdown(selected=None, force: bool = True):
    """Dropdown update for the runs list; auto-selects the active or latest run."""
    runs = _list_runs()
    choices = [(_run_label(m), m["run_id"]) for m in runs]
    ids = [m["run_id"] for m in runs]
    if selected not in ids:
        active = [m["run_id"] for m in runs if m["state"].get("status") in ACTIVE_STATUSES]
        selected = active[0] if active else (ids[0] if ids else None)
    if not force and _LAST_RUN_CHOICES["choices"] == choices:
        # Avoid re-rendering the dropdown on every timer tick while nothing changed.
        return gr.skip(), selected
    _LAST_RUN_CHOICES["choices"] = choices
    return gr.Dropdown(choices=choices, value=selected), selected


def refresh_run_view(run_id=None, step_choice=None):
    dd, selected = _runs_dropdown(run_id, force=True)
    status_md, log, attach, step_dd = _run_view(selected, step_choice if selected == run_id else None)
    return status_md, log, dd, attach, step_dd


def tick_run_view(run_id=None, step_choice=None):
    dd, selected = _runs_dropdown(run_id, force=False)
    status_md, log, attach, step_dd = _run_view(selected, step_choice)
    # Only push the step list when its labels changed, so an open dropdown is not disturbed.
    key = selected
    labels = [c[0] for c in step_dd.constructor_args["choices"]] if hasattr(step_dd, "constructor_args") else None
    if _LAST_STEP_CHOICES.get(key) == labels:
        step_dd = gr.skip()
    else:
        _LAST_STEP_CHOICES[key] = labels
    return status_md, log, dd, attach, step_dd


def view_run(run_id=None):
    """User picked another run: reset the step selector to Auto."""
    status_md, log, attach, step_dd = _run_view(run_id, None)
    return status_md, log, attach, step_dd


def view_step(run_id=None, step_choice=None):
    meta = _find_run(run_id)
    if meta is None:
        return ""
    _, sel = _steps_dropdown(meta, step_choice)
    return _run_log_view(meta, sel)


def live_tracking_view(run_id=None, previous_key=None):
    """Read only published preview images; never render or run inference in Gradio."""
    meta = _find_run(run_id)
    if meta is None:
        return [], "Select a run to see its latest WAPR tracking frame.", None
    records, key, notes = [], [run_id], []
    seen = set()
    for index, step in enumerate(meta.get("steps", [])):
        if step.get("title") != "WAPR object tracking":
            continue
        scene = Path(step["dataset_path"])
        if scene in seen:
            continue
        seen.add(scene)
        folder = scene / "debug_depth/live_tracking"
        request = _read_json(folder / "request.json", {}) or {}
        latest = _read_json(folder / "latest.json", {}) or {}
        state = meta.get("state", {}).get("steps", {}).get(str(index), {})
        # A selected historical run must not show another run's new preview.
        started = float(state.get("started") or 0)
        finished = float(state.get("finished") or float("inf"))
        belongs = started > 0 and started <= float(request.get("started", 0)) <= finished
        label = step.get("logical_data_name", scene.name)
        if not belongs:
            notes.append(f"{label}: waiting for live preview.")
            key.append([str(scene), "waiting", state.get("status")])
            continue
        key.append([str(scene), request.get("session"), request.get("completed"), latest.get("frame"),
                    request.get("finished"), request.get("failed")])
        count, total = request.get("completed", 0), request.get("total", "?")
        if latest.get("session") != request.get("session") or not latest.get("image"):
            error = _read_json(folder / "error.json", {}) or {}
            detail = f"Preview unavailable: {error['error']}" if error.get("error") else "preparing preview"
            notes.append(f"{label}: {count}/{total} frames · {detail}.")
            continue
        image_path = folder / Path(latest["image"]).name
        if not image_path.is_file():
            continue
        phase = "stopped" if request.get("failed") else "tracking finished" if request.get("finished") else "tracking"
        caption = f"{label} · frame {latest['frame']} · {latest.get('mode', '')}"
        records.append((image_path, caption))
        notes.append(f"{label}: **{count}/{total}** frames · {phase} · preview frame **{latest['frame']}**.")
    if not seen:
        notes.append("This run has no WAPR tracking step.")
    if key == previous_key:
        return gr.skip(), gr.skip(), previous_key
    # Copy pixels now: the worker may rotate old files while Gradio serializes.
    from PIL import Image
    images = []
    for path, caption in records:
        try:
            with Image.open(path) as source:
                images.append((source.convert("RGB").copy(), caption))
        except (OSError, ValueError):
            pass
    return images, "  \n".join(notes), key


def _plan_steps(stage_sequence: Sequence[str], hand_prior_name, selected_datasets,
                prompt_table, hand_side, data_root_str, start_sec, end_sec, fps,
                width, height, ffmpeg_threads, da3_gpus, sam3d_gpu,
                tracking_gpu, postopt_gpu, depth_unit, pixel_stride, mask_erode,
                max_points_per_cam, max_size_mb, save_debug_pointcloud,
                depth_scale, max_frames, generate_texture=False, auto_end_sec=True, all_takes=True,
                skip_done=True, tracking_method="foundationpose"):
    if tracking_method not in TRACKING_METHODS:
        raise ValueError(f"Unknown pose tracking method: {tracking_method}")
    jobs = _build_jobs(data_root_str, hand_prior_name, selected_datasets, prompt_table, hand_side, bool(all_takes))
    data_root = _data_root_path(data_root_str)
    code_root = EASYMOCAP_ROOT
    if not code_root.is_dir():
        raise ValueError(f"Fixed EasyMocap root does not exist: {code_root}")
    if any(stage in stage_sequence for stage in ("hands", "final")) and float(fps) != 10.0:
        raise ValueError("Hand reconstruction and final refinement use the original 10 FPS preset; select FPS=10.")
    if float(end_sec) <= float(start_sec) and not auto_end_sec:
        raise ValueError("end_sec must be greater than start_sec (or enable auto end = shortest video)")

    # Pre-flight: fail before launching anything when a dataset cannot run the requested stages.
    problems = []
    notes: List[str] = []
    # Stages already completed on disk are not run again (per take). A take whose requested
    # stages are all done is dropped from the run; a partially processed one resumes at the
    # first incomplete stage.
    stages_by_job: Dict[str, List[str]] = {}
    kept_jobs: List[Job] = []
    for job in jobs:
        ds_root = _dataset_root(data_root, job)
        if skip_done:
            done = [st for st in stage_sequence if _stage_complete(ds_root, st, tracking_method)]
            if "tracking" in stage_sequence and "tracking" not in done and "final" in done:
                done.remove("final")  # New object poses require a new hand/object final export.
        else:
            done = []
        todo = [st for st in stage_sequence if st not in done]
        if not todo:
            notes.append(f"{job.display_name}: skipped — {', '.join(stage_sequence)} already completed on disk")
            continue
        if done:
            notes.append(f"{job.display_name}: {', '.join(done)} already completed on disk — running {', '.join(todo)} only")
        stages_by_job[job.logical_data_name] = todo
        kept_jobs.append(job)
    jobs = kept_jobs
    if not jobs:
        raise ValueError("Nothing to run: every selected take already has the requested stage outputs on disk.\n"
                         + "\n".join(f"- {n}" for n in notes)
                         + "\n(Untick 'Skip stages already completed' to force a re-run.)")

    if tracking_method == "wapr" and any("tracking" in stages for stages in stages_by_job.values()):
        for path in (python_for("wapr"), python_for("foundationpose"),
                     PROJECT_ROOT / "web_demos/third_party/WAPR/wapr/tracking.py"):
            if not path.is_file():
                raise ValueError(f"WAPR runtime missing: {path}. Run scripts/install.sh --component wapr and configure the FoundationPose environment.")

    primary_in_run = {j.dataset_logical_name for j in jobs if j.is_primary_take and "recon" in stages_by_job[j.logical_data_name]}
    for job in jobs:
        job_stages = stages_by_job[job.logical_data_name]
        ds_folder = _dataset_folder(data_root, job)
        flat_note = _ensure_flat(ds_folder)
        if flat_note:
            notes.append(f"{job.dataset}: {flat_note}")
        for n in _prepare_take_dir(data_root, job):
            notes.append(f"{job.display_name}: {n}")
        ds_root = _dataset_root(data_root, job)
        if "preprocess" in job_stages:
            vdir = _take_video_dir(data_root, job)
            if not any(vdir.glob("*.mp4")) and not any(vdir.glob("*.MP4")):
                problems.append(f"{job.display_name}: no .mp4 in videos/{job.take}/ (raw videos not uploaded yet?)")
        elif job_stages and job_stages[0] == "meshscale":
            if not (ds_root / "meshes" / "compressed.ply").is_file():
                problems.append(f"{job.display_name}: meshes/compressed.ply missing — run Stage 2 first")
        elif job_stages and job_stages[0] in ("masks", "recon", "tracking", "hands", "final"):
            if not (ds_root / "images").is_dir():
                problems.append(f"{job.display_name}: images/ missing — run Stage 1 (preprocess) first")
        # A later take needs take01's object mesh: it must exist already, or take01's recon
        # must be part of this run (the scheduler then orders the two chains).
        if (not job.is_primary_take and "recon" in job_stages
                and not (ds_folder / "meshes" / "compressed.ply").is_file()
                and job.dataset_logical_name not in primary_in_run):
            problems.append(f"{job.display_name}: shared object mesh missing — run Stage 2 on {job.dataset} ({PRIMARY_TAKE}) first "
                            f"or include {PRIMARY_TAKE} in this run")
        elif (not job.is_primary_take and "recon" not in job_stages
                and job_stages and job_stages[0] in ("tracking", "hands", "final")
                and not (ds_root / "meshes" / "compressed.ply").is_file()):
            problems.append(f"{job.display_name}: meshes/compressed.ply missing — run Stage 2 first")
    if problems:
        raise ValueError("Deselect these datasets or fix them first:\n- " + "\n- ".join(problems))

    records = []
    # One chain per dataset, steps in stage order. The runner executes chains
    # concurrently and places each step on GPUs as VRAM becomes available.
    for job in jobs:
        for stage in stages_by_job[job.logical_data_name]:
            steps = _steps_for(stage, job, code_root, data_root, start_sec, end_sec, fps,
                               width, height, ffmpeg_threads, str(da3_gpus), str(sam3d_gpu),
                               str(tracking_gpu), str(postopt_gpu), str(depth_unit),
                               int(pixel_stride), int(mask_erode), int(max_points_per_cam),
                               float(max_size_mb), bool(save_debug_pointcloud),
                               float(depth_scale), int(max_frames), bool(auto_end_sec), bool(generate_texture), tracking_method)
            for step in steps:
                records.append({
                    "stage": stage,
                    "tracking_method": tracking_method,
                    "label": f"{job.logical_data_name} · {step.title}",
                    "title": step.title,
                    "logical_data_name": job.logical_data_name,
                    "dataset_path": str(_dataset_root(data_root, job)),
                    "text_prompt": job.text_prompt,
                    "cwd": str(step.cwd),
                    "command": step.command,
                    "conda_env": step.conda_env,
                    "gpu": step.gpu,
                    "after": step.after,
                    "shell": _shell_for_step(step),
                })
    return jobs, data_root, records, notes


def _launch_detached(run_dir: Path, session: str) -> str:
    """Start the runner for run_dir; returns the backend used ('tmux' or 'nohup')."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run_dir)]
    if shutil.which("tmux"):
        res = _tmux("new-session", "-d", "-s", session, "-c", str(run_dir), "-x", "200", "-y", "50",
                    shlex.join(cmd))
        if res.returncode == 0:
            return "tmux"
        with open(run_dir / "log.txt", "ab") as f:
            f.write(f"[webui] tmux launch failed, falling back to nohup: {res.stderr.strip()}\n".encode())
    out = open(run_dir / "runner.out", "ab")
    subprocess.Popen(cmd, cwd=str(run_dir), stdout=out, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    return "nohup"


def submit_run(stage_sequence: Sequence[str], hand_prior_name, selected_datasets, *rest):
    """Write a run plan to RUNS_DIR and start it in tmux.

    `rest` = the pipeline parameters accepted by _plan_steps followed by the
    three scheduler knobs (parallel_datasets, vram_per_step_mb, cpu_parallel).
    Returns (status, log, runs dropdown, attach cmd, steps dropdown).
    """
    try:
        *plan_args, auto_end_sec, all_takes, skip_done, tracking_method, parallel_datasets, vram_per_step_mb, cpu_parallel, oom_max_retries, allowed_gpus = rest
        plan_args = plan_args + [bool(auto_end_sec), bool(all_takes), bool(skip_done), tracking_method]
        jobs, data_root, records, notes = _plan_steps(stage_sequence, hand_prior_name, selected_datasets, *plan_args)
        if not records:
            raise ValueError("Nothing to run for the selected stage(s).")

        # Refuse to run two pipelines on the same dataset at once.
        wanted = {j.logical_data_name for j in jobs}
        for m in _active_runs():
            busy = wanted & {s["logical_data_name"] for s in m.get("steps", [])}
            if busy:
                raise ValueError(
                    f"Run {m['run_id']} is still active on {', '.join(sorted(busy))}. "
                    "Stop it or wait for it to finish before submitting again."
                )

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        session = f"hoi_{run_id}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        meta = {
            "run_id": run_id,
            "session": session,
            "created": time.time(),
            "stage_sequence": list(stage_sequence),
            "hand_prior_name": jobs[0].hand_prior_name,
            "hand_side": jobs[0].hand_side,
            "tracking_method": tracking_method,
            "datasets": sorted({j.dataset for j in jobs}),
            "takes": [j.display_name for j in jobs],
            "data_root": str(data_root),
            "scheduler": {
                "parallel_datasets": max(0, int(parallel_datasets or 0)),
                "vram_per_step_mb": max(1024, int(vram_per_step_mb or GPU_MIN_FREE_VRAM_MB)),
                "cpu_parallel": max(1, int(cpu_parallel or 1)),
                "oom_max_retries": max(0, int(oom_max_retries or 0)),
                "allowed_gpus": _parse_gpu_list(allowed_gpus),
            },
            "steps": records,
        }
        _write_json_atomic(run_dir / "state.json", {"status": "queued", "total_steps": len(records)})
        _write_json_atomic(run_dir / "meta.json", meta)
        with open(run_dir / "log.txt", "w") as f:
            for n in notes:  # pre-flight fixes applied while planning (e.g. nested folders flattened)
                f.write(f"[webui] {n}\n")
        meta["backend"] = _launch_detached(run_dir, session)
        _write_json_atomic(run_dir / "meta.json", meta)
    except Exception as exc:
        dd, _ = _runs_dropdown(None, force=True)
        step_dd, _ = _steps_dropdown(None)
        return f"### ❌ Configuration error\n{exc}", str(exc), dd, "", step_dd

    return refresh_run_view(run_id)


def resubmit_run(run_id=None, step_choice=None):
    """Create a new run from the unfinished steps (failed / skipped / stopped / pending) of a finished run."""
    meta = _find_run(run_id)
    if meta is None:
        return refresh_run_view(run_id)
    st = meta["state"]
    if st.get("status") in ACTIVE_STATUSES:
        status_md, log, attach, step_dd = _run_view(run_id, step_choice)
        dd, _ = _runs_dropdown(run_id, force=True)
        return "### ⚠️ Run is still active — stop it or wait before resubmitting\n" + status_md, log, dd, attach, step_dd
    step_states = st.get("steps", {})
    unfinished = [i for i, s in enumerate(meta.get("steps", []))
                  if step_states.get(str(i), {}).get("status", "pending") != "done"]
    try:
        if not unfinished:
            raise ValueError("Every step of this run already finished successfully.")
        records = []
        dropped: List[str] = []
        for i in unfinished:
            rec = dict(meta["steps"][i])
            # The dataset (or take) folder may have been deleted since the original run.
            ds_path = rec.get("dataset_path")
            if ds_path and not Path(ds_path).is_dir():
                if rec["logical_data_name"] not in dropped:
                    dropped.append(rec["logical_data_name"])
                continue
            rec["resubmitted_from"] = f"{meta['run_id']}#{i + 1}"
            records.append(rec)
        if not records:
            raise ValueError("Nothing to resubmit: the dataset folder(s) of every unfinished step no longer exist: "
                             + ", ".join(dropped))
        wanted = {r["logical_data_name"] for r in records}
        for m in _active_runs():
            busy = wanted & {s["logical_data_name"] for s in m.get("steps", [])}
            if busy:
                raise ValueError(f"Run {m['run_id']} is still active on {', '.join(sorted(busy))}.")
        new_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        session = f"hoi_{new_id}"
        run_dir = RUNS_DIR / new_id
        run_dir.mkdir(parents=True, exist_ok=False)
        new_meta = {k: v for k, v in meta.items() if k not in ("dir", "state", "backend", "steps", "run_id", "session", "created")}
        new_meta.update({
            "run_id": new_id,
            "session": session,
            "created": time.time(),
            "datasets": sorted({r["logical_data_name"].split("/")[2] for r in records}),
            "takes": sorted({r["logical_data_name"].split(f"/{DATA_KIND}/", 1)[-1] for r in records}),
            "resubmitted_from": meta["run_id"],
            "steps": records,
        })
        _write_json_atomic(run_dir / "state.json", {"status": "queued", "total_steps": len(records)})
        _write_json_atomic(run_dir / "meta.json", new_meta)
        with open(run_dir / "log.txt", "w") as f:
            for name in dropped:
                f.write(f"[webui] {name}: dataset folder no longer exists — its steps were not resubmitted\n")
        new_meta["backend"] = _launch_detached(run_dir, session)
        _write_json_atomic(run_dir / "meta.json", new_meta)
    except Exception as exc:
        status_md, log, attach, step_dd = _run_view(run_id, step_choice)
        dd, _ = _runs_dropdown(run_id, force=True)
        return f"### ❌ Could not resubmit\n{exc}\n\n" + status_md, log, dd, attach, step_dd
    return refresh_run_view(new_id)


def stop_run(run_id=None, step_choice=None):
    meta = _find_run(run_id)
    if meta is None:
        return refresh_run_view(run_id)
    st = meta["state"]
    status = st.get("status")
    prefix = ""
    if status not in ACTIVE_STATUSES:
        prefix = "### ⏹️ No running process\n"
    else:
        try:
            runner_pid = st.get("runner_pid")
            if runner_pid and _pid_alive(runner_pid):
                os.kill(int(runner_pid), signal.SIGTERM)  # runner terminates every child group and records "stopped"
            else:
                for s in st.get("steps", {}).values():
                    if s.get("status") == "running" and s.get("pgid"):
                        try:
                            os.killpg(int(s["pgid"]), signal.SIGTERM)
                        except Exception:
                            pass
                if meta.get("backend") == "tmux":
                    _tmux("kill-session", "-t", f"={meta['session']}")
                _write_json_atomic(Path(meta["dir"]) / "state.json",
                                   {**st, "status": "stopped", "finished": time.time()})
            prefix = "### ⏹️ Stop requested\n"
            time.sleep(0.5)
        except Exception as exc:
            prefix = f"### ⚠️ Could not stop process\n{exc}\n\n"
    status_md, log, attach, step_dd = _run_view(run_id, step_choice)
    dd, _ = _runs_dropdown(run_id, force=True)
    return prefix + status_md, log, dd, attach, step_dd


# ---------------------------------------------------------------------------
# Runner: VRAM-aware scheduler executing one submitted run
# ---------------------------------------------------------------------------

GPU_WARMUP_SEC = 120.0   # a freshly launched GPU step keeps its VRAM reservation this long
SCHED_TICK_SEC = 2.0
OOM_RETRY_COOLDOWN_SEC = float(os.getenv("WEBUI_OOM_COOLDOWN_SEC", "30"))   # let the driver release memory before a requeued step is relaunched
OOM_PATTERNS = re.compile(
    r"CUDA out of memory|OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|"
    r"CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory|"
    r"RuntimeError: NVML|failed to allocate .*bytes|Out of memory|OOM when allocating|ResourceExhaustedError",
    re.IGNORECASE,
)


GPU_BUSY_MAX_RETRIES = int(os.getenv("WEBUI_GPU_BUSY_MAX_RETRIES", "5"))
GPU_BUSY_PATTERNS = re.compile(
    r"without enough free VRAM|No visible GPU has free VRAM|No CUDA-visible GPU has free VRAM|"
    r"no GPU satisfies|No GPU currently has >= |no GPU has free VRAM",
    re.IGNORECASE,
)


def _log_indicates_gpu_busy(log_path, tail_bytes: int = 64 * 1024) -> bool:
    """True when a step refused to start because its assigned GPU(s) lost their free VRAM."""
    try:
        p = Path(log_path)
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
            return bool(GPU_BUSY_PATTERNS.search(f.read().decode("utf-8", errors="replace")))
    except Exception:
        return False


def _log_indicates_oom(log_path, tail_bytes: int = 256 * 1024) -> bool:
    """True when the tail of a step log carries a CUDA/GPU out-of-memory signature."""
    try:
        p = Path(log_path)
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
            text = f.read().decode("utf-8", errors="replace")
        return bool(OOM_PATTERNS.search(text))
    except Exception:
        return False


def _nvidia_total_mb() -> Dict[int, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        res: Dict[int, int] = {}
        for line in out.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                res[int(parts[0])] = int(float(parts[1]))
        return res
    except Exception:
        return {}


def _nvidia_free_mb() -> Optional[Dict[int, int]]:
    """{gpu_index: free MiB} from nvidia-smi, or None when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        free: Dict[int, int] = {}
        for line in out.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                free[int(parts[0])] = int(float(parts[1]))
        return free
    except Exception:
        return None


def _run_worker(run_dir: Path) -> int:
    """Execute one submitted run. Runs inside tmux (or nohup).

    Steps are grouped into one chain per dataset (strict order inside a chain).
    Chains are executed concurrently; each step is started only when its GPU
    demand fits into (live free VRAM - reservations of just-launched steps).
    """
    meta = _read_json(run_dir / "meta.json")
    if not meta:
        print(f"[runner] no meta.json in {run_dir}", file=sys.stderr)
        return 2
    steps: List[dict] = meta.get("steps", [])
    sched = meta.get("scheduler", {})
    vram_mb = int(sched.get("vram_per_step_mb") or GPU_MIN_FREE_VRAM_MB)
    max_par = int(sched.get("parallel_datasets") or 0)
    max_cpu = int(sched.get("cpu_parallel") or 3)
    oom_max_retries = int(sched.get("oom_max_retries", 2) or 0)
    STARVE_SEC = 900.0  # a step that cannot be placed for this long while nothing runs is failed, not hung

    def req_mb(idx: int) -> int:
        """VRAM headroom a step needs: the run budget, raised after OOM retries."""
        return max(vram_mb, int(state["steps"][str(idx)].get("vram_mb") or 0))

    chains: Dict[str, List[int]] = {}
    for i, s in enumerate(steps):
        chains.setdefault(s["logical_data_name"], []).append(i)
    chain_pos: Dict[str, int] = {k: 0 for k in chains}
    chain_dead: Dict[str, bool] = {k: False for k in chains}

    now = time.time()
    state: dict = {
        "status": "running",
        "runner_pid": os.getpid(),
        "started": now,
        "total_steps": len(steps),
        "steps": {str(i): {"status": "pending"} for i in range(len(steps))},
        "running": [],
        "done": 0,
        "failed": 0,
        "oom_requeued": 0,
        "waiting_note": "",
    }

    def save():
        state["running"] = sorted(running.keys())
        _write_json_atomic(run_dir / "state.json", state)

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    events = open(run_dir / "log.txt", "ab", buffering=0)

    def ev(text: str):
        line = time.strftime("%H:%M:%S ") + text + "\n"
        events.write(line.encode("utf-8", errors="replace"))
        sys.stdout.write(line)
        sys.stdout.flush()

    running: Dict[int, dict] = {}
    stop = {"requested": False}

    def on_signal(signum, frame):
        stop["requested"] = True
        for r in running.values():
            p = r["proc"]
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)

    allowed_gpus = set(int(g) for g in (sched.get("allowed_gpus") or []))

    def free_mb() -> Optional[Dict[int, int]]:
        """Free VRAM per GPU, restricted to the GPUs this run may use."""
        free = _nvidia_free_mb()
        if free is None or not allowed_gpus:
            return free
        return {g: v for g, v in free.items() if g in allowed_gpus}

    gpu_free = free_mb()
    if allowed_gpus and gpu_free is not None and not gpu_free:
        ev(f"[sched] WARNING: none of the allowed GPUs {sorted(allowed_gpus)} exist on this machine")
    if max_par <= 0:
        max_par = max(1, len(gpu_free or {}))
    save()
    ev(f"[sched] run {meta['run_id']} · {len(chains)} dataset chain(s) · {len(steps)} step(s) · "
       f"parallel_datasets={max_par} · vram_per_step={vram_mb} MiB · cpu_parallel={max_cpu} · oom_retries={oom_max_retries} · "
       f"gpus={sorted(gpu_free) if gpu_free else 'none (nvidia-smi unavailable)'}"
       + (f" (restricted to {sorted(allowed_gpus)})" if allowed_gpus else ""))

    def launch(idx: int, gpus: List[int]):
        step = steps[idx]
        prev = state["steps"][str(idx)]
        attempt = int(prev.get("attempts", 0)) + 1
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", step["label"])[:80]
        log_path = logs_dir / f"{idx:02d}_{slug}.log"
        fh = open(log_path, "ab", buffering=0)
        fh.write((f"$ [{step['logical_data_name']}] {step['title']}\n"
                  f"[text prompt] {step.get('text_prompt', '')}\n"
                  f"[resources] {_fmt_gpus(gpus)} · attempt {attempt} · vram budget {req_mb(idx)} MiB\n"
                  f"{step['command']}\n\n").encode("utf-8", errors="replace"))
        env = os.environ.copy()
        env["GPU_PICK"] = str(gpus[0]) if gpus else ""
        env["GPU_LIST"] = ",".join(str(g) for g in gpus)
        env["GPU_VRAM_MB"] = str(req_mb(idx))
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", step["shell"]],
                cwd=step["cwd"],
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid,
                env=env,
            )
        except OSError as exc:
            # e.g. the working directory (dataset folder) was deleted since the plan was
            # made: fail this chain only, never the whole run.
            fh.write(f"[sched] cannot start step: {exc}\n".encode("utf-8", errors="replace"))
            fh.close()
            state["steps"][str(idx)] = {"status": "failed", "rc": -4, "finished": time.time(),
                                        "log": str(log_path), "attempts": attempt,
                                        "last_error": f"cannot start: {exc}"}
            state["failed"] += 1
            chain_dead[step["logical_data_name"]] = True
            for j in chains[step["logical_data_name"]]:
                if state["steps"][str(j)]["status"] == "pending":
                    state["steps"][str(j)]["status"] = "skipped"
            ev(f"[sched] FAILED #{idx + 1} {step['label']} — cannot start ({exc}) — remaining steps of this dataset are skipped")
            return
        running[idx] = {"proc": proc, "fh": fh, "gpus": gpus, "chain": step["logical_data_name"],
                        "reserve_until": time.time() + GPU_WARMUP_SEC}
        state["steps"][str(idx)] = {"status": "running", "gpus": gpus, "started": time.time(),
                                    "pgid": proc.pid, "log": str(log_path), "attempts": attempt,
                                    "oom_retries": int(prev.get("oom_retries", 0)),
                                    "busy_retries": int(prev.get("busy_retries", 0)),
                                    "vram_mb": int(prev.get("vram_mb") or 0)}
        retry_note = f" (attempt {attempt}, budget {req_mb(idx)} MiB)" if attempt > 1 else ""
        ev(f"[sched] START #{idx + 1} {step['label']} → {_fmt_gpus(gpus)}{retry_note}")

    def finish(idx: int, rc: int):
        r = running.pop(idx)
        r["fh"].close()
        step = steps[idx]
        s = state["steps"][str(idx)]
        s.update(finished=time.time(), rc=rc, pgid=None)
        if stop["requested"]:
            s["status"] = "stopped"
            ev(f"[sched] STOPPED #{idx + 1} {step['label']} (rc {rc})")
        elif rc != 0 and r["gpus"] and int(s.get("busy_retries", 0)) < GPU_BUSY_MAX_RETRIES and (rc == GPU_BUSY_RC or _log_indicates_gpu_busy(s.get("log"))):
            # The assigned GPU(s) were taken by someone else between scheduling and launch:
            # requeue at the same budget after a cooldown (does not count as an OOM retry).
            s.update(status="pending", busy_retries=int(s.get("busy_retries", 0)) + 1,
                     not_before=time.time() + OOM_RETRY_COOLDOWN_SEC, last_error="GPU busy")
            state["oom_requeued"] += 1
            with open(s["log"], "ab") as f:
                f.write(f"\n[sched] assigned GPU(s) no longer free — requeued (retry {s['busy_retries']}/{GPU_BUSY_MAX_RETRIES})\n\n".encode())
            ev(f"[sched] GPU BUSY #{idx + 1} {step['label']} on {_fmt_gpus(r['gpus'])} — requeued "
               f"(retry {s['busy_retries']}/{GPU_BUSY_MAX_RETRIES})")
        elif rc != 0 and r["gpus"] and int(s.get("oom_retries", 0)) < oom_max_retries and _log_indicates_oom(s.get("log")):
            # CUDA OOM: put the step back into the queue with a larger VRAM budget so the
            # scheduler waits for a GPU with more headroom (or fewer co-located steps).
            # Never ask for more than the emptiest GPU can offer right now, otherwise the step would wait forever.
            free_now = free_mb() or {}
            cap = max(free_now.values()) - 1024 if free_now else 10 ** 9
            new_budget = max(req_mb(idx), min(max(req_mb(idx) * 2, req_mb(idx) + 8192), cap))
            s.update(status="pending", oom_retries=int(s.get("oom_retries", 0)) + 1, vram_mb=new_budget,
                     not_before=time.time() + OOM_RETRY_COOLDOWN_SEC, last_error="CUDA OOM")
            state["oom_requeued"] += 1
            with open(s["log"], "ab") as f:
                f.write(f"\n[sched] CUDA OOM detected — requeued (retry {s['oom_retries']}/{oom_max_retries}, "
                        f"VRAM budget now {new_budget} MiB, relaunch after {OOM_RETRY_COOLDOWN_SEC:.0f}s)\n\n".encode())
            ev(f"[sched] OOM #{idx + 1} {step['label']} on {_fmt_gpus(r['gpus'])} — requeued "
               f"(retry {s['oom_retries']}/{oom_max_retries}, budget {new_budget} MiB)")
        elif rc != 0:
            s["status"] = "failed"
            state["failed"] += 1
            chain_dead[r["chain"]] = True
            why = " after CUDA OOM retries" if s.get("last_error") == "CUDA OOM" and _log_indicates_oom(s.get("log")) else ""
            ev(f"[sched] FAILED #{idx + 1} {step['label']} (rc {rc}){why} — remaining steps of this dataset are skipped")
            for j in chains[r["chain"]]:
                if state["steps"][str(j)]["status"] == "pending":
                    state["steps"][str(j)]["status"] = "skipped"
        else:
            s["status"] = "done"
            state["done"] += 1
            chain_pos[r["chain"]] += 1
            ev(f"[sched] OK #{idx + 1} {step['label']}")

    def dependency_state(idx: int) -> str:
        """'ok' (no dependency / satisfied), 'wait' (other step pending or running), 'dead' (it failed)."""
        dep = steps[idx].get("after")
        if not dep:
            return "ok"
        dep_idxs = [j for j in chains.get(dep.get("chain"), []) if steps[j]["title"] == dep.get("title")]
        if not dep_idxs:
            return "ok"  # not part of this run: the step itself verifies the shared file exists
        st_ = state["steps"][str(dep_idxs[-1])].get("status", "pending")
        if st_ == "done":
            return "ok"
        if st_ in ("failed", "skipped", "stopped") or chain_dead.get(dep.get("chain")):
            return "dead"
        return "wait"

    def fail_unlaunched(idx: int, reason: str):
        s_ = state["steps"][str(idx)]
        s_.update(status="failed", rc=-3, finished=time.time(), last_error=reason)
        state["failed"] += 1
        chain = steps[idx]["logical_data_name"]
        chain_dead[chain] = True
        for j in chains[chain]:
            if state["steps"][str(j)]["status"] == "pending":
                state["steps"][str(j)]["status"] = "skipped"
        ev(f"[sched] FAILED #{idx + 1} {steps[idx]['label']} — {reason} — remaining steps of this dataset are skipped")

    last_wait_note = 0.0
    try:
        while True:
            # 1) reap finished steps
            for idx in list(running):
                rc = running[idx]["proc"].poll()
                if rc is not None:
                    finish(idx, rc)

            if stop["requested"]:
                if not running:
                    break
                save()
                time.sleep(0.5)
                continue

            # 2) candidates: next step of each idle, healthy chain
            candidates = []
            cooling = []
            active_chains = {r["chain"] for r in running.values()}
            for name, idxs in chains.items():
                if chain_dead[name] or name in active_chains or chain_pos[name] >= len(idxs):
                    continue
                nxt = idxs[chain_pos[name]]
                if float(state["steps"][str(nxt)].get("not_before") or 0) > time.time():
                    cooling.append(nxt)
                    continue
                candidates.append(nxt)
            if not candidates and not running and not cooling:
                break

            # 3) place candidates: multi-GPU first (they benefit most from a free pool), then single, then CPU
            gpu_free = free_mb() if any(steps[i].get("gpu") for i in candidates) else gpu_free
            now = time.time()
            avail: Dict[int, int] = dict(gpu_free or {})
            for r in running.values():
                if now < r["reserve_until"]:
                    for g in r["gpus"]:
                        if g in avail:
                            avail[g] -= vram_mb
            order = {"multi": 0, "single": 1, None: 2}
            candidates.sort(key=lambda i: (order.get(steps[i].get("gpu"), 2), i))
            blocked = []
            for idx in candidates:
                # Cross-chain dependency (e.g. take02 waits for take01's object mesh): hold the
                # step while the other chain's step is pending/running, fail it if that failed.
                dep = dependency_state(idx)
                if dep == "wait":
                    d = steps[idx]["after"]
                    blocked.append(f"#{idx + 1} waits for {d['chain']} · {d['title']}")
                    continue
                if dep == "dead":
                    d = steps[idx]["after"]
                    fail_unlaunched(idx, f"depends on {d['chain']} · {d['title']}, which did not succeed")
                    continue
                if len(active_chains) >= max_par:
                    blocked.append("parallel-dataset limit reached")
                    break
                kind = steps[idx].get("gpu")
                if kind is None:
                    if sum(1 for r in running.values() if not r["gpus"]) >= max_cpu:
                        blocked.append("CPU step limit reached")
                        continue
                    launch(idx, [])
                elif gpu_free is None:
                    # No nvidia-smi: cannot schedule GPU work; fail the step instead of hanging forever.
                    launch(idx, [])
                    ev(f"[sched] WARNING nvidia-smi unavailable; #{idx + 1} launched without GPU assignment")
                elif kind == "single":
                    need = req_mb(idx)
                    ok = [g for g in avail if avail[g] > need]
                    if not ok:
                        blocked.append(f"no GPU with > {need} MiB free for #{idx + 1}")
                        continue
                    # Spread compute: fewest of our own steps on the GPU first, then most headroom
                    # (every candidate already satisfies the VRAM budget).
                    load = {g: sum(1 for r in running.values() if g in r["gpus"]) for g in ok}
                    g = max(ok, key=lambda x: (-load[x], avail[x] // 4096, -x))
                    launch(idx, [g])
                    avail[g] -= need
                else:  # multi
                    need = req_mb(idx)
                    ok = sorted(g for g in avail if avail[g] > need)
                    if not ok:
                        blocked.append(f"no GPU with > {need} MiB free for #{idx + 1}")
                        continue
                    launch(idx, ok)
                    for g in ok:
                        avail[g] -= need
                active_chains.add(steps[idx]["logical_data_name"])

            if cooling:
                blocked.append(f"{len(cooling)} step(s) cooling down before retry")
            # Starvation guard: a candidate that cannot be placed while nothing runs at all
            # (e.g. its VRAM budget exceeds every GPU) is failed instead of hanging the run.
            for idx in candidates:
                if idx in running or state["steps"][str(idx)].get("status") != "pending":
                    continue
                s_ = state["steps"][str(idx)]
                if not running and dependency_state(idx) == "ok":
                    s_.setdefault("blocked_since", time.time())
                    if time.time() - float(s_["blocked_since"]) > STARVE_SEC:
                        s_.update(status="failed", rc=-2, finished=time.time(),
                                  last_error=f"could not be scheduled for {STARVE_SEC / 60:.0f} min: {state['waiting_note']}")
                        state["failed"] += 1
                        chain_dead[steps[idx]["logical_data_name"]] = True
                        for j in chains[steps[idx]["logical_data_name"]]:
                            if state["steps"][str(j)]["status"] == "pending":
                                state["steps"][str(j)]["status"] = "skipped"
                        ev(f"[sched] FAILED #{idx + 1} {steps[idx]['label']} — {s_['last_error']}")
                else:
                    s_.pop("blocked_since", None)
            state["waiting_note"] = "; ".join(dict.fromkeys(blocked)) if blocked else ""
            if blocked and not running:
                if now - last_wait_note > 60:
                    ev(f"[sched] waiting: {state['waiting_note']}")
                    last_wait_note = now
            save()
            time.sleep(SCHED_TICK_SEC)
    except Exception as exc:  # never leave the state as "running" on an internal error
        ev(f"[sched] INTERNAL ERROR: {exc!r}")
        on_signal(signal.SIGTERM, None)
        for idx in list(running):
            finish(idx, running[idx]["proc"].wait())
        state.update(status="failed", finished=time.time(), error=repr(exc))
        save()
        return 1

    elapsed = (time.time() - state["started"]) / 60.0
    if stop["requested"]:
        state.update(status="stopped", finished=time.time())
        ev(f"[sched] run stopped by user after {elapsed:.1f} min")
        save()
        return 130
    if state["failed"]:
        state.update(status="failed", finished=time.time())
        ev(f"[sched] run finished with {state['failed']} failed step(s) in {elapsed:.1f} min")
        save()
        return 1
    state.update(status="done", finished=time.time(), rc=0)
    ev(f"[sched] DONE: {len(steps)} step(s) in {elapsed:.1f} min")
    save()
    return 0


def _latest_media(base: Path, keywords: Sequence[str], extensions: Sequence[str], limit_files=4000):
    if not base.exists():
        return None
    keywords = tuple(k.lower() for k in keywords)
    exts = {e.lower() for e in extensions}
    skip_dirs = {"images", "images_ori", "masks", "masks_ori", "depth", "videos", ".git", "checkpoints"}
    best = None
    best_mtime = -1.0
    seen = 0
    for current, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        pcur = Path(current)
        rel = str(pcur.relative_to(base)).lower() if pcur != base else ""
        for fn in files:
            seen += 1
            if seen > limit_files:
                return str(best) if best else None
            p = pcur / fn
            if p.suffix.lower() not in exts:
                continue
            hay = rel + "/" + fn.lower()
            if keywords and not any(k in hay for k in keywords):
                continue
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if mt > best_mtime:
                best_mtime = mt
                best = p
    return str(best) if best else None


def _raw_preview_dataset_update(selected_datasets, current_value=None):
    choices = [str(x) for x in (selected_datasets or [])]
    cur = str(current_value or "")
    value = cur if cur in choices else (choices[0] if choices else None)
    return gr.Dropdown(choices=choices, value=value)


def preview_first_raw_video(root_str: str, hand_prior_name: str, dataset: str):
    try:
        hp = _validate_name(str(hand_prior_name or ""), "hand_prior_name")
        ds = _validate_name(str(dataset or ""), "dataset")
        root = _data_root_path(root_str)
        take01 = root / DATA_KIND / hp / ds / "videos" / "take01"
        if not take01.is_dir():
            return f"⚠️ Folder not found: `{take01}`", None

        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
        videos = sorted(
            (p for p in take01.iterdir() if p.is_file() and p.suffix.lower() in video_exts),
            key=lambda p: p.name.lower(),
        )
        if not videos:
            return f"⚠️ No video found in `{take01}`", None

        first = videos[0]
        return f"▶ Previewing first video: `{first}`", str(first)
    except Exception as exc:
        return f"Raw video preview error: {exc}", None


EASYMOCAP_PYTHON = python_for("easymocap")


def _video_codec(path: Path) -> Optional[str]:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def _browser_video(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """(playable path, note). OpenCV writes MPEG-4 part 2 ('mp4v'), which browsers cannot play;
    transcode once to H.264 next to the file (<stem>_h264.mp4) and reuse it afterwards."""
    if not path.is_file():
        return None, None
    codec = _video_codec(path)
    if codec in (None, "h264", "av1"):
        return str(path), None
    h264 = path.with_name(path.stem + "_h264.mp4")
    if h264.is_file() and h264.stat().st_mtime >= path.stat().st_mtime:
        return str(h264), None
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(h264)], check=True, timeout=600)
        return str(h264), f"transcoded `{path.name}` ({codec}) to H.264 for the browser"
    except Exception as exc:
        return str(path), f"⚠️ could not transcode `{path.name}` ({codec}): {exc}"


def _viewer_mesh(ply: Path) -> Tuple[Optional[str], Optional[str]]:
    """(path for gr.Model3D, note). The browser viewer is only reliable with GLB, so convert the
    binary PLY once (<stem>_viewer.glb, cached by mtime) using trimesh from the easymocap env."""
    if not ply.is_file():
        return None, None
    textured = _current_textured_model(ply)
    if textured:
        return textured, "Showing UV-textured model"
    glb = ply.with_name(ply.stem + "_viewer.glb")
    if glb.is_file() and glb.stat().st_mtime >= ply.stat().st_mtime:
        return str(glb), None
    py = EASYMOCAP_PYTHON if EASYMOCAP_PYTHON.exists() else Path(sys.executable)
    code = ("import sys, trimesh; m = trimesh.load(sys.argv[1], process=False); "
            "m = trimesh.util.concatenate(m.dump()) if isinstance(m, trimesh.Scene) else m; m.export(sys.argv[2])")
    try:
        subprocess.run([str(py), "-c", code, str(ply), str(glb)], check=True, capture_output=True, text=True, timeout=300)
        return str(glb), f"converted `{ply.name}` to GLB for the viewer"
    except Exception as exc:
        err = getattr(exc, "stderr", "") or str(exc)
        return str(ply), f"⚠️ PLY→GLB conversion failed, showing the PLY directly: {str(err).strip()[-200:]}"


def _current_textured_model(ply: Path):
    glb = ply.with_name("textured.glb")
    try:
        meta = json.loads(glb.with_suffix(".json").read_text())
        if (meta.get("status") == "complete" and
                meta["tracking_ply_sha256"] == hashlib.sha256(ply.read_bytes()).hexdigest() and
                meta["glb_sha256"] == hashlib.sha256(glb.read_bytes()).hexdigest()):
            return str(glb)
    except (OSError, ValueError, KeyError):
        pass
    return None


def refresh_visuals(hand_prior_name: str, dataset: str, root_str: str):
    try:
        hp = _validate_name(hand_prior_name, "hand_prior_name")
        ds = _validate_name(dataset, "dataset")
        root = _data_root_path(root_str)
        base = root / DATA_KIND / hp / ds
        if not base.is_dir():
            raise ValueError(f"dataset directory does not exist: {base}")
        notes: List[str] = []

        mesh_path, n = _viewer_mesh(base / "meshes" / "compressed.ply")
        if n:
            notes.append(n)

        track_vid, n = _browser_video(base / "mesh_overlay_allcams_fast_green" / "mesh_overlay_allcams_fast_green.mp4")
        if n:
            notes.append(n)

        grid_dir = base / "outputmano" / "grid"
        final_video = grid_dir / "final_hand_object_interaction.mp4"
        final_vid, n = _browser_video(final_video)
        if n:
            notes.append(n)

        final_frames = sorted(grid_dir.glob("*.jpg")) if grid_dir.is_dir() else []
        final_img = str(final_frames[-1]) if final_frames else None

        final_file = _latest_media(
            base,
            ("standard", "table", "export"),
            (".html", ".json", ".npz", ".pkl", ".csv", ".parquet"),
        )

        found = sum(
            x is not None
            for x in (mesh_path, track_vid, final_img, final_vid, final_file)
        )
        missing = [name for name, x in (("object mesh (Stage 2)", mesh_path), ("tracking video (Stage 3)", track_vid),
                                        ("final frames (Stage 5)", final_img), ("final video (Stage 5)", final_vid)) if x is None]
        status = (
            f"Found {found} preview artifact(s) under `{base}`."
            + (f"  \nNot produced yet: {', '.join(missing)}." if missing else "")
            + ("  \n" + "  \n".join(notes) if notes else "")
        )
        return status, mesh_path, track_vid, final_img, final_vid, final_file
    except Exception as exc:
        return (
            f"Visualization refresh error: {exc}",
            None,
            None,
            None,
            None,
            None,
        )

def _vis_dataset_update(root_str, hand_prior_name, selected_datasets, current_value=None):
    """All datasets under the hand prior (selected ones first) so results can be viewed without selecting."""
    try:
        names = [c[1] for c in _dataset_choices(root_str, hand_prior_name)]
    except Exception:
        names = []
    selected = [str(x) for x in (selected_datasets or []) if str(x) in names]
    choices = selected + [n for n in names if n not in selected]
    cur = str(current_value or "")
    value = cur if cur in choices else (choices[0] if choices else None)
    return gr.Dropdown(choices=choices, value=value)


def make_ui(default_data_root: str):
    css = """
    .app-shell {max-width: 1500px !important; margin: 0 auto;}
    .stage-card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 4px;}
    .mono textarea {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important; font-size: 12px !important;}
    """
    with gr.Blocks(title=APP_TITLE, ) as demo:
        gr.Markdown(
            "# Hand–Object Interaction Pipeline v2\n"
            f"**Pipeline code root:** `{EASYMOCAP_ROOT}`  \n"
            "**Data hierarchy:** `<Data Root>/rigid/<hand_prior_name>/<dataset_name>` → preprocessing/SAM3 → Depth-Anything-3 + MV-SAM3D reconstruction → FoundationPose / WAPR tracking + geometry post-opt → hand reconstruction → refinement/export."
        )

        with gr.Row():
            data_root = gr.Textbox(value=default_data_root, label="Data Root (contains rigid/)", scale=5, info="A directory containing rigid/; configure additional permitted roots before launch")
            data_kind = gr.Textbox(value=DATA_KIND, label="Data type (fixed)", interactive=False, scale=1)
            refresh_structure_btn = gr.Button("↻ Scan folders", scale=1)
        structure_status = gr.Markdown(f"Scanning `{DATA_KIND}/...` folders is available from the button above.")

        with gr.Row():
            hand_prior = gr.Dropdown(
                choices=[],
                value=None,
                label="Hand prior name",
                info="Folders directly under rigid/, e.g. collector01",
                scale=4,
            )
            hand_side = gr.Dropdown(
                choices=["bimanual", "right", "left"],
                value="bimanual",
                label="Hand side",
                scale=2,
            )

        with gr.Accordion("Hand calibration — separate per-person setup", open=False):
            gr.Markdown("Calibrate the selected **Hand prior name** once before processing captures. "
                        "Input: `<Hand Calibration Root>/<capture>/videos/`, `intri.yml`, `extri.yml`. "
                        "Provide all 31 camera videos as `nodeNN_camN.mp4` (following `CALIB_CAMERA_ORDER`), "
                        "or a complete numbered/legacy rpizero set. Videos are copied and mapped to `1..31.mp4` "
                        "in a temporary workspace before frame extraction; originals are unchanged. "
                        "The requested interval is extracted at **10 FPS**. This action is never part of the normal processing chain.")
            with gr.Row():
                calibration_dataset = gr.Textbox(value="hands_collector01", label="Calibration capture name")
                calibration_side = gr.Dropdown(choices=["right", "left"], value="right", label="Calibration hand")
                calibration_gpus = gr.Textbox(value=DEFAULT_ALLOWED_GPUS, label="Calibration GPUs (physical IDs or auto)")
            with gr.Row():
                calibration_start = gr.Number(value=0, label="Calibration start (seconds)")
                calibration_end = gr.Number(value=0.1, label="Calibration end (seconds, exclusive)")
                calibration_replace = gr.Checkbox(value=False, label="Replace existing prior (retain backup)")
                calibration_dry = gr.Checkbox(value=True, label="Preview commands only")
            calibration_button = gr.Button("Run Hand Calibration for selected person")
            calibration_status = gr.Markdown("Select a person above. Preview commands before running calibration.")
            calibration_logs = gr.Textbox(label="Calibration log", lines=12, interactive=False)
            from calibration_panel import run as run_person_calibration
            calibration_button.click(run_person_calibration,
                inputs=[data_root, hand_prior, calibration_dataset, calibration_start, calibration_end,
                        calibration_side, calibration_gpus, calibration_replace, calibration_dry],
                outputs=[calibration_status, calibration_logs], concurrency_limit=1)

        gr.Markdown("### Dataset selection")
        datasets = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Datasets under selected hand prior (single or multiple)",
        )
        selection_count = gr.Markdown("**Selected: 0**")
        with gr.Row():
            select_all_btn = gr.Button("Select all datasets")
            select_good_btn = gr.Button("Select ★ good (from screening)", variant="secondary")
            clear_all_btn = gr.Button("Clear selection")
            refresh_datasets_btn = gr.Button("↻ Refresh dataset list")

        with gr.Accordion("Quick screening — 31-camera contact sheets (first frame + 30 / 50 / 70 %)", open=False):
            gr.Markdown(
                "Flip through datasets and judge clarity / coverage at a glance. Sheets come from `images/` when the dataset "
                "is preprocessed, otherwise straight from the raw videos (cached after the first build). Ratings are stored in "
                "`rigid/<hand_prior>/.webui_screening.json`, show up in the dataset list above, and **Select ★ good** picks them."
            )
            with gr.Row():
                screen_prev_btn = gr.Button("◀ Prev", scale=1)
                screen_dataset = gr.Dropdown(choices=[], value=None, label="Dataset", scale=6)
                screen_next_btn = gr.Button("Next ▶", scale=1)
            with gr.Row():
                screen_show_btn = gr.Button("Show / refresh sheets", variant="primary", scale=2)
                screen_force = gr.Checkbox(value=False, label="Regenerate (ignore cache)", scale=1)
                screen_all_btn = gr.Button("Pre-generate sheets for all datasets", scale=2)
                flatten_btn = gr.Button("Flatten nested dataset folders (<ds>/<ds>/ from zip)", scale=2)
            screen_status = gr.Markdown("Pick a dataset and click **Show / refresh sheets**.")
            screen_gallery = gr.Gallery(label="Contact sheets (click to enlarge)", columns=2, height=780, object_fit="contain")
            with gr.Row():
                screen_rating = gr.Radio(choices=RATING_CHOICES, value="unrated", label="Rating (saved immediately)", scale=3)
                screen_notes = gr.Textbox(value="", label="Notes (optional)", scale=4)
                screen_save_btn = gr.Button("Save notes", scale=1)
            screen_save_status = gr.Markdown("")

        gr.Markdown("### Raw input video preview")
        with gr.Row():
            raw_preview_dataset = gr.Dropdown(
                choices=[],
                value=None,
                label="Dataset to preview",
                info="Choose from the datasets selected above",
                scale=4,
            )
            raw_preview_btn = gr.Button("▶ Preview first raw video", scale=2)
        raw_preview_status = gr.Markdown(
            "Select a dataset above, then click the preview button. "
            "The first video in `videos/take01/` will be shown."
        )
        raw_preview_video = gr.Video(
            label="First video in videos/take01",
            height=420,
        )
        raw_preview_btn.click(
            preview_first_raw_video,
            inputs=[data_root, hand_prior, raw_preview_dataset],
            outputs=[raw_preview_status, raw_preview_video],
            concurrency_limit=None,
        )

        gr.Markdown(
            "### Optional SAM3 text prompt per dataset\n"
            f"Leave a prompt empty to use the default: `{DEFAULT_PROMPT}`"
        )
        prompts = gr.Dataframe(
            headers=["dataset", "text_prompt"],
            datatype=["str", "str"],
            value=[],
            row_count=1,
            column_count=(2, "fixed"),
            label="Per-dataset prompts (dataset column is populated from the selection above)",
        )

        with gr.Row():
            start_sec = gr.Number(value=0.0, label="Start sec",
                                  info="Keep 0: the object mesh/scale use the t=0 reference frame, and depth/masks must start at the same instant.")
            end_sec = gr.Number(value=0.0, label="End sec (0 = no cap when auto)",
                                info="With auto end enabled this is only an upper cap.")
            fps = gr.Number(value=10.0, label="FPS")
            auto_end_sec = gr.Checkbox(
                value=True,
                label="Auto end = shortest video of each dataset",
                info="Per dataset, ffprobe the shortest camera video and snap the end to the FPS grid so all cameras get equal frame counts.",
            )
            all_takes = gr.Checkbox(
                value=True,
                label="Process all takes (videos/take01, take02, …)",
                info="Every take becomes its own chain; take02+ work in <dataset>/takes/<take>/ and reuse take01's "
                     "calibration and object mesh (Stage 2 reconstruction runs once per dataset).",
            )
            skip_done = gr.Checkbox(
                value=True,
                label="Skip stages already completed (per take)",
                info="A take whose requested stages already have their final outputs on disk is left out; a partially "
                     "processed take resumes at its first incomplete stage. Untick to force a full re-run.",
            )

        tracking_method = gr.Dropdown(
            choices=[("FoundationPose tracking", "foundationpose"), ("WAPR tracking", "wapr")],
            value="foundationpose", label="Pose tracking method",
            info="Used by Stage 3. WAPR tracks from the previous pose with multi-view consistency checks. "
                 "Switching methods reruns tracking and preserves previous results in tracking_history/.",
        )

        with gr.Accordion("Advanced parameters", open=False):
            with gr.Row():
                width = gr.Number(value=640, precision=0, label="Width")
                height = gr.Number(value=360, precision=0, label="Height")
                ffmpeg_threads = gr.Number(value=2, precision=0, label="FFmpeg threads")
                da3_gpus = gr.Textbox(value="auto", label="Depth-Anything-3 GPUs (auto: all >32 GiB)", interactive=False)
            with gr.Row():
                sam3d_gpu = gr.Textbox(value="auto", label="MV-SAM3D GPU (auto: most-free >32 GiB)", interactive=False)
                tracking_gpu = gr.Textbox(value="auto", label="Pose tracking GPU (auto: most-free >32 GiB)", interactive=False)
                postopt_gpu = gr.Textbox(value="auto", label="Pose post-opt GPU (auto: most-free >32 GiB)", interactive=False)
                depth_unit = gr.Dropdown(choices=["mm", "m"], value="mm", label="MV-SAM3D depth unit")
            with gr.Row():
                pixel_stride = gr.Number(value=2, precision=0, label="MV-SAM3D pixel stride")
                mask_erode = gr.Number(value=1, precision=0, label="MV-SAM3D mask erode")
                max_points_per_cam = gr.Number(value=50000, precision=0, label="Max points / camera")
                max_size_mb = gr.Number(value=9.8, label="Max compressed mesh size (MB)")
            with gr.Row():
                save_debug_pointcloud = gr.Checkbox(value=True, label="Save MV-SAM3D debug point cloud")
                generate_texture = gr.Checkbox(value=False, label="生成贴图模型（可选）", info="额外生成带 UV 和贴图的 GLB，来自模型重建外观；增加生成时间。")
                depth_scale = gr.Number(value=1000, label="WAPR / pose post-opt integer depth divisor")
                max_frames = gr.Number(value=500, precision=0, label="Export max frames")
            gr.Markdown(
                "**GPU / VRAM scheduling (multi-dataset runs).** Each selected dataset becomes an independent chain; "
                "chains run concurrently and every step is placed on GPUs whose free VRAM (minus reservations of "
                "just-launched steps) exceeds the per-step budget. Multi-GPU steps take every qualifying GPU."
            )
            with gr.Row():
                parallel_datasets = gr.Number(value=int(os.getenv("WEBUI_PARALLEL_DATASETS", "0")), precision=0, label="Parallel datasets (0 = one per GPU)")
                vram_per_step_mb = gr.Number(value=GPU_MIN_FREE_VRAM_MB, precision=0, label="VRAM budget per GPU step (MiB)")
                cpu_parallel = gr.Number(value=int(os.getenv("WEBUI_CPU_PARALLEL", "3")), precision=0, label="Max concurrent CPU-only steps")
                oom_max_retries = gr.Number(
                    value=2, precision=0, label="Auto-requeue CUDA OOM steps (max retries, 0 = off)",
                    info="A GPU step that dies with CUDA out-of-memory goes back into the queue with a doubled VRAM budget after a 30 s cooldown.",
                )
            allowed_gpus = gr.Textbox(
                value=DEFAULT_ALLOWED_GPUS, label="Allowed GPUs (physical ids, e.g. 4,5,6,7 or 4-7; 'all' = no restriction)",
                info="The scheduler never places a step on any other GPU. Default comes from WEBUI_ALLOWED_GPUS in the launcher.",
            )

        gr.Markdown(
            "## Pipeline control\n"
            "Every stage button runs that stage for **all selected datasets**. "
            "Full pipeline runs stage-by-stage across all selected datasets."
        )
        with gr.Row():
            run_pre = gr.Button("1 · Preprocess selected", variant="secondary")
            run_masks = gr.Button("1b · SAM3 masks only (re-prompt)", variant="secondary")
            run_recon = gr.Button("2 · Object recon selected", variant="secondary")
            run_meshscale = gr.Button("2b · Mesh scale check (silhouettes, optional)", variant="secondary")
            run_track = gr.Button("3 · Object tracking selected", variant="secondary")
            run_hand = gr.Button("4 · Hand recon selected", variant="secondary")
            run_final = gr.Button("5 · Final refine/export selected", variant="secondary")
        with gr.Row():
            run_all = gr.Button("▶ Run ALL stages on selected datasets", variant="primary")
            stop_btn = gr.Button("■ Stop selected run", variant="stop")
            resubmit_btn = gr.Button("↻ Resubmit unfinished steps of selected run", variant="secondary")

        with gr.Row():
            run_select = gr.Dropdown(
                choices=[],
                value=None,
                label="Runs (tmux-managed — keep running after the page closes)",
                info="Reopening the page re-attaches to the active run automatically.",
                scale=5,
            )
            refresh_runs_btn = gr.Button("↻ Refresh runs", scale=1)
        step_select = gr.Dropdown(
            choices=[("Auto (scheduler + running steps)", -1)],
            value=-1,
            label="Log view",
            info="Auto shows scheduler events plus the tail of every running step; pick a step for its full log.",
        )
        attach_cmd = gr.Textbox(
            label="Attach from a terminal",
            value="",
            interactive=False,
            elem_classes=["mono"],
        )
        status = gr.Markdown("### Ready")
        logs = gr.Textbox(label="Live logs", lines=18, max_lines=30, elem_classes=["mono"], interactive=False)
        run_timer = gr.Timer(3.0)
        with gr.Accordion("Live WAPR tracking preview", open=True):
            live_status = gr.Markdown("The latest tracked pose appears here automatically while WAPR is running.")
            live_gallery = gr.Gallery(label="Latest tracking visualization", columns=1, rows=1, height=480,
                                      preview=True, selected_index=0, object_fit="contain",
                                      format="jpeg", elem_id="tracking-live-preview")
            gr.Markdown("Up to six camera views: green = tracked mesh, red = shared point cloud fused from all valid cameras' masked depths. Updates automatically; the full video is generated after tracking.")
        live_key = gr.State(None)
        live_timer = gr.Timer(1.0)
        live_timer.tick(live_tracking_view, inputs=[run_select, live_key],
                        outputs=[live_gallery, live_status, live_key], queue=False, concurrency_limit=1)
        run_select.change(live_tracking_view, inputs=[run_select, live_key],
                          outputs=[live_gallery, live_status, live_key], queue=False, concurrency_limit=1)

        args = [hand_prior, datasets, prompts, hand_side, data_root, start_sec, end_sec, fps,
                width, height, ffmpeg_threads, da3_gpus, sam3d_gpu, tracking_gpu, postopt_gpu,
                depth_unit, pixel_stride, mask_erode, max_points_per_cam, max_size_mb,
                save_debug_pointcloud, depth_scale, max_frames, generate_texture,
                auto_end_sec, all_takes, skip_done, tracking_method, parallel_datasets, vram_per_step_mb, cpu_parallel, oom_max_retries, allowed_gpus]
        run_outputs = [status, logs, run_select, attach_cmd, step_select]

        def _submit(stages):
            def fn(*a):
                return submit_run(stages, *a)
            return fn

        submit_kwargs = dict(inputs=args, outputs=run_outputs, concurrency_limit=None)
        for _btn, _stages in ((run_pre, ["preprocess"]), (run_masks, ["masks"]), (run_recon, ["recon"]),
                              (run_meshscale, ["meshscale"]), (run_track, ["tracking"]),
                              (run_hand, ["hands"]), (run_final, ["final"]),
                              (run_all, ["preprocess", "recon", "tracking", "hands", "final"])):
            _btn.click(_submit(_stages), **submit_kwargs).then(
                # Update the "latest run" markers in the dataset list once the run is queued.
                refresh_datasets, inputs=[data_root, hand_prior, datasets], outputs=[structure_status, datasets],
                concurrency_limit=None,
            )

        stop_btn.click(stop_run, inputs=[run_select, step_select], outputs=run_outputs, concurrency_limit=None)
        resubmit_btn.click(resubmit_run, inputs=[run_select, step_select], outputs=run_outputs, concurrency_limit=None)
        refresh_runs_btn.click(refresh_run_view, inputs=[run_select, step_select], outputs=run_outputs, concurrency_limit=None)
        run_select.input(view_run, inputs=[run_select], outputs=[status, logs, attach_cmd, step_select], concurrency_limit=None)
        step_select.input(view_step, inputs=[run_select, step_select], outputs=[logs], concurrency_limit=None)
        run_timer.tick(tick_run_view, inputs=[run_select, step_select], outputs=run_outputs, concurrency_limit=None)

        gr.Markdown("## Visualizations")
        with gr.Accordion("Intermediate / final results (optional — open only when needed)", open=False):
            with gr.Row():
                vis_dataset = gr.Dropdown(choices=[], value=None, label="Selected dataset to visualize")
                refresh = gr.Button("↻ Refresh current results")
            vis_status = gr.Markdown("Select a dataset, then refresh while a stage is running or after it finishes.")

            gr.Markdown("### Object reconstruction")
            mesh = gr.Model3D(label="Object model (textured GLB when available)", clear_color=[0.04, 0.04, 0.05, 1.0])

            gr.Markdown(
                "### Object tracking visualization\n"
                "Only shows `mesh_overlay_allcams_fast_green/mesh_overlay_allcams_fast_green.mp4`."
            )
            track_vid = gr.Video(
                label="mesh_overlay_allcams_fast_green.mp4",
                height=520,
            )

            gr.Markdown(
                "### Final exported hand–object interaction\n"
                "Rendered from the optimized `outputmano/` result (`outputmano/grid/*.jpg`) after Stage 5."
            )
            with gr.Row():
                final_img = gr.Image(label="Latest final visualization frame", height=360)
                final_vid = gr.Video(
                    label="final_hand_object_interaction.mp4",
                    height=520,
                )
            final_file = gr.File(label="Latest exported standardized data artifact")

            refresh.click(
                refresh_visuals,
                inputs=[hand_prior, vis_dataset, data_root],
                outputs=[vis_status, mesh, track_vid, final_img, final_vid, final_file],
                concurrency_limit=None,
            )


        # Shared single-sequence media panel; uses the detached main-UI scheduler.
        import sys as _media_sys
        _media_scripts = Path(__file__).resolve().parents[1] / "scripts" / "hoi_qualitative"
        if str(_media_scripts) not in _media_sys.path:
            _media_sys.path.insert(0, str(_media_scripts))
        from web_panel import mount_media_panel
        mount_media_panel(data_root, hand_prior, datasets, globals(), visualization_dataset=vis_dataset)


        gr.Markdown("## Export results")
        with gr.Accordion("Export Stage 5 outputs (outputmano/*.mp4 + standardized_hoi_30fps/) of the selected datasets", open=True):
            gr.Markdown(
                "The list below follows the dataset selection at the top of the page. Export copies, for each dataset, "
                "every `outputmano/**/*.mp4` and the whole `standardized_hoi_30fps/` folder into "
                "`<export folder>/<dataset>/` (unchanged files are skipped, so re-exporting is cheap) and can bundle "
                "everything into one `.zip` for download."
            )
            export_sel = gr.CheckboxGroup(choices=[], value=[], label="Datasets to export (synced from selection above)")
            with gr.Row():
                export_dir = gr.Textbox(
                    value="",
                    label="Export folder",
                    placeholder=f"empty = <Data Root>/exports/<hand_prior_name>",
                    info="Absolute path, or relative to Data Root.",
                    scale=5,
                )
                export_zip_chk = gr.Checkbox(value=True, label="Also build a .zip for download", scale=2)
                export_auto_chk = gr.Checkbox(value=False, label="Auto-export whenever the selection changes", scale=2)
            with gr.Row():
                export_btn = gr.Button("⬇ Export selected datasets", variant="primary", scale=2)
                export_sync_btn = gr.Button("↻ Re-sync list from selection", scale=1)
            export_status = gr.Markdown("Select datasets above, then click Export.")
            export_file = gr.File(label="Download export .zip")

            export_btn.click(
                export_datasets,
                inputs=[data_root, hand_prior, export_sel, export_dir, export_zip_chk],
                outputs=[export_status, export_file],
                concurrency_limit=None,
            )
            export_sync_btn.click(
                _export_dataset_update,
                inputs=[data_root, hand_prior, datasets, export_sel],
                outputs=export_sel,
                concurrency_limit=None,
            )

        gr.Markdown("## Dataset File Manager")
        with gr.Accordion("Delete selected files / folders from one dataset", open=False):
            gr.Markdown(
                "Browse inside one dataset and permanently remove selected direct children. "
                "Deletion is blocked while a pipeline step is running, and paths are restricted "
                "to the selected dataset root."
            )
            with gr.Row():
                fm_dataset = gr.Dropdown(choices=[], value=None, label="Dataset to manage", scale=3)
                fm_subdir = gr.Textbox(value=".", label="Current folder (relative to dataset)", interactive=False, scale=4)
                fm_scan = gr.Button("↻ Scan", scale=1)
            fm_status = gr.Markdown("Choose a dataset and scan its contents.")

            with gr.Row():
                fm_open_folder = gr.Dropdown(choices=[], value=None, label="Open child folder", scale=4)
                fm_open_btn = gr.Button("Open folder", scale=1)
                fm_up_btn = gr.Button("↑ Up", scale=1)

            fm_entries = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Select direct files / folders to permanently delete",
            )
            with gr.Row():
                fm_confirm = gr.Textbox(
                    value="",
                    label='Confirmation — type DELETE',
                    placeholder="DELETE",
                    scale=4,
                )
                fm_delete = gr.Button("Delete selected permanently", variant="stop", scale=2)

            fm_scan.click(
                scan_dataset_directory,
                inputs=[data_root, hand_prior, fm_dataset, fm_subdir],
                outputs=[fm_status, fm_entries, fm_open_folder],
                concurrency_limit=None,
            )
            fm_open_btn.click(
                file_manager_open_folder,
                inputs=[data_root, hand_prior, fm_dataset, fm_subdir, fm_open_folder],
                outputs=fm_subdir,
                concurrency_limit=None,
            ).then(
                scan_dataset_directory,
                inputs=[data_root, hand_prior, fm_dataset, fm_subdir],
                outputs=[fm_status, fm_entries, fm_open_folder],
                concurrency_limit=None,
            )
            fm_up_btn.click(
                file_manager_up,
                inputs=[data_root, hand_prior, fm_dataset, fm_subdir],
                outputs=fm_subdir,
                concurrency_limit=None,
            ).then(
                scan_dataset_directory,
                inputs=[data_root, hand_prior, fm_dataset, fm_subdir],
                outputs=[fm_status, fm_entries, fm_open_folder],
                concurrency_limit=None,
            )
            fm_delete.click(
                delete_dataset_entries,
                inputs=[data_root, hand_prior, fm_dataset, fm_subdir, fm_entries, fm_confirm],
                outputs=[fm_status, fm_entries, fm_open_folder, fm_confirm],
                concurrency_limit=None,
            )

            fm_dataset.change(
                lambda: ".",
                outputs=fm_subdir,
                concurrency_limit=None,
            ).then(
                scan_dataset_directory,
                inputs=[data_root, hand_prior, fm_dataset, fm_subdir],
                outputs=[fm_status, fm_entries, fm_open_folder],
                concurrency_limit=None,
            )

        # Folder discovery / selection synchronization.
        refresh_structure_btn.click(
            refresh_hand_priors,
            inputs=[data_root, hand_prior],
            outputs=[structure_status, hand_prior],
        ).then(
            refresh_datasets,
            inputs=[data_root, hand_prior, datasets],
            outputs=[structure_status, datasets],
        ).then(
            sync_prompt_rows,
            inputs=[datasets, prompts],
            outputs=prompts,
        ).then(
            _vis_dataset_update,
            inputs=[data_root, hand_prior, datasets, vis_dataset],
            outputs=vis_dataset,
        ).then(
            _export_dataset_update,
            inputs=[data_root, hand_prior, datasets, export_sel],
            outputs=export_sel,
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        ).then(
            _file_manager_dataset_update,
            inputs=[data_root, hand_prior, fm_dataset],
            outputs=fm_dataset,
        )

        data_root.submit(
            refresh_hand_priors,
            inputs=[data_root, hand_prior],
            outputs=[structure_status, hand_prior],
        ).then(
            refresh_datasets,
            inputs=[data_root, hand_prior, datasets],
            outputs=[structure_status, datasets],
        ).then(
            sync_prompt_rows,
            inputs=[datasets, prompts],
            outputs=prompts,
        ).then(
            _vis_dataset_update,
            inputs=[data_root, hand_prior, datasets, vis_dataset],
            outputs=vis_dataset,
        ).then(
            _export_dataset_update,
            inputs=[data_root, hand_prior, datasets, export_sel],
            outputs=export_sel,
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        ).then(
            _file_manager_dataset_update,
            inputs=[data_root, hand_prior, fm_dataset],
            outputs=fm_dataset,
        )

        hand_prior.change(
            refresh_datasets,
            inputs=[data_root, hand_prior, datasets],
            outputs=[structure_status, datasets],
        ).then(
            sync_prompt_rows,
            inputs=[datasets, prompts],
            outputs=prompts,
        ).then(
            _vis_dataset_update,
            inputs=[data_root, hand_prior, datasets, vis_dataset],
            outputs=vis_dataset,
        ).then(
            _export_dataset_update,
            inputs=[data_root, hand_prior, datasets, export_sel],
            outputs=export_sel,
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        ).then(
            _file_manager_dataset_update,
            inputs=[data_root, hand_prior, fm_dataset],
            outputs=fm_dataset,
        )

        refresh_datasets_btn.click(
            refresh_datasets,
            inputs=[data_root, hand_prior, datasets],
            outputs=[structure_status, datasets],
        ).then(
            sync_prompt_rows,
            inputs=[datasets, prompts],
            outputs=prompts,
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        ).then(
            _file_manager_dataset_update,
            inputs=[data_root, hand_prior, fm_dataset],
            outputs=fm_dataset,
        )

        # Fires for user clicks and for every programmatic update (select all / clear / ★ good / refresh).
        datasets.change(
            selection_summary,
            inputs=[data_root, hand_prior, datasets],
            outputs=selection_count,
            concurrency_limit=None,
        )
        datasets.change(
            sync_prompt_rows,
            inputs=[datasets, prompts],
            outputs=prompts,
        ).then(
            _vis_dataset_update,
            inputs=[data_root, hand_prior, datasets, vis_dataset],
            outputs=vis_dataset,
        ).then(
            _export_dataset_update,
            inputs=[data_root, hand_prior, datasets, export_sel],
            outputs=export_sel,
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        ).then(
            maybe_auto_export,
            inputs=[export_auto_chk, data_root, hand_prior, export_sel, export_dir, export_zip_chk],
            outputs=[export_status, export_file],
            concurrency_limit=None,
        )

        select_all_btn.click(
            select_all_datasets,
            inputs=[data_root, hand_prior],
            outputs=[datasets, prompts],
        ).then(
            _vis_dataset_update,
            inputs=[data_root, hand_prior, datasets, vis_dataset],
            outputs=vis_dataset,
        ).then(
            _export_dataset_update,
            inputs=[data_root, hand_prior, datasets, export_sel],
            outputs=export_sel,
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        )

        clear_all_btn.click(
            clear_all_datasets,
            inputs=[data_root, hand_prior],
            outputs=[datasets, prompts],
        ).then(
            _vis_dataset_update,
            inputs=[data_root, hand_prior, datasets, vis_dataset],
            outputs=vis_dataset,
        ).then(
            _export_dataset_update,
            inputs=[data_root, hand_prior, datasets, export_sel],
            outputs=export_sel,
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        )

        select_good_btn.click(
            select_good_datasets,
            inputs=[data_root, hand_prior, prompts],
            outputs=[datasets, prompts],
        ).then(
            _vis_dataset_update,
            inputs=[data_root, hand_prior, datasets, vis_dataset],
            outputs=vis_dataset,
        ).then(
            _export_dataset_update,
            inputs=[data_root, hand_prior, datasets, export_sel],
            outputs=export_sel,
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        )

        screen_outputs = [screen_status, screen_gallery, screen_rating, screen_notes]
        screen_show_btn.click(show_dataset_sheets, inputs=[data_root, hand_prior, screen_dataset, screen_force],
                              outputs=screen_outputs, concurrency_limit=None)
        screen_dataset.input(show_dataset_sheets, inputs=[data_root, hand_prior, screen_dataset],
                             outputs=screen_outputs, concurrency_limit=None)
        screen_prev_btn.click(lambda r, h, c: screen_step(r, h, c, -1), inputs=[data_root, hand_prior, screen_dataset],
                              outputs=[screen_dataset] + screen_outputs, concurrency_limit=None)
        screen_next_btn.click(lambda r, h, c: screen_step(r, h, c, +1), inputs=[data_root, hand_prior, screen_dataset],
                              outputs=[screen_dataset] + screen_outputs, concurrency_limit=None)
        flatten_btn.click(flatten_nested_datasets, inputs=[data_root, hand_prior, datasets],
                          outputs=[screen_status, datasets, screen_dataset], concurrency_limit=None)
        screen_all_btn.click(generate_all_sheets, inputs=[data_root, hand_prior, screen_force],
                             outputs=screen_status, concurrency_limit=None)
        rating_kwargs = dict(inputs=[data_root, hand_prior, screen_dataset, screen_rating, screen_notes, datasets],
                             outputs=[screen_save_status, datasets, screen_dataset], concurrency_limit=None)
        screen_rating.input(save_screening_rating, **rating_kwargs)
        screen_save_btn.click(save_screening_rating, **rating_kwargs)
        for _ev in (hand_prior.change, refresh_datasets_btn.click):
            _ev(_screen_dataset_update, inputs=[data_root, hand_prior, screen_dataset], outputs=screen_dataset)

        # Auto-scan once on app load.
        demo.load(
            refresh_hand_priors,
            inputs=[data_root, hand_prior],
            outputs=[structure_status, hand_prior],
        ).then(
            refresh_datasets,
            inputs=[data_root, hand_prior, datasets],
            outputs=[structure_status, datasets],
        ).then(
            _raw_preview_dataset_update,
            inputs=[datasets, raw_preview_dataset],
            outputs=raw_preview_dataset,
        ).then(
            _file_manager_dataset_update,
            inputs=[data_root, hand_prior, fm_dataset],
            outputs=fm_dataset,
        ).then(
            _screen_dataset_update,
            inputs=[data_root, hand_prior, screen_dataset],
            outputs=screen_dataset,
        ).then(
            _export_dataset_update,
            inputs=[data_root, hand_prior, datasets, export_sel],
            outputs=export_sel,
        ).then(
            # Re-attach the page to the active (or most recent) tmux-managed run.
            refresh_run_view,
            inputs=[run_select, step_select],
            outputs=[status, logs, run_select, attach_cmd, step_select],
            concurrency_limit=None,
        )

        gr.Markdown(
            "**Safety:** folder names are strictly validated; arbitrary shell commands are not accepted. "
            "Keep authentication enabled when exposing the app publicly."
        )
    demo._hoi_css = css
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.getenv("HOI_DATA_ROOT", str(EASYMOCAP_ROOT)), help="Directory that contains rigid/")
    p.add_argument("--host", default=os.getenv("WEBUI_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Ask Gradio to create a public tunnel URL")
    p.add_argument("--no-auth", action="store_true", help="Disable login (not recommended for a public URL)")
    p.add_argument("--auth-user", default=os.getenv("WEBUI_USER", "operator"))
    p.add_argument("--auth-password", default=os.getenv("WEBUI_PASSWORD"))
    p.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)  # internal: execute a submitted run
    args = p.parse_args()

    if args.run_dir:
        sys.exit(_run_worker(Path(args.run_dir)))

    if args.no_auth and (args.share or args.host not in ("127.0.0.1", "localhost", "::1")):
        p.error("Public access requires authentication; remove --no-auth.")

    auth = None
    if not args.no_auth:
        password = args.auth_password or secrets.token_urlsafe(12)
        auth = (args.auth_user, password)
        print("\n[webui] Login credentials")
        print(f"[webui] username: {args.auth_user}")
        print(f"[webui] password: {password}")
        print("[webui] Set WEBUI_PASSWORD to keep a stable password across restarts.\n")

    demo = make_ui(args.data_root)
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=auth,
        show_error=True,
        css=getattr(demo, "_hoi_css", None),
        allowed_paths=[str(_data_root_path(args.data_root)), str(RUNS_DIR), *[str(Path(p).expanduser().resolve()) for p in os.getenv("HOI_ALLOWED_DATA_ROOTS", "").split(os.pathsep) if p]],
    )


if __name__ == "__main__":
    main()
